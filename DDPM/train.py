import torch
import os
from tqdm import tqdm

from default_configs import accumulation_steps, loss_fn, cls_loss_fn, sim_loss_fn, device, checkpoints_path
from utils import define_new_models, get_mni152_mask, q_sample, get_v_target, extract_seqs, analyze_loss

def load_trained_models(model_name_at_epoch: str, **kwargs):
    load_path = os.path.join(checkpoints_path, f'{model_name_at_epoch}.tar')
    try:
        checkpoint = torch.load(load_path)
    except:
        raise Exception("No checkpoint file")
    
    model_type = checkpoint.get('model_type', 'ViT')
    settings = checkpoint['settings']
    loss_list, valid_loss_list = checkpoint['loss'], checkpoint['valid_loss']
    
    model_name = kwargs.get('model_name', checkpoint['model_name'])
    settings = {**settings, **kwargs.get('settings', {})}
    models_dict = define_new_models(model_type, model_name, **settings)
    settings = models_dict['settings']
    
    transformer = models_dict['transformer']
    unet = models_dict['unet']
    classifier = models_dict['classifier']
    simmatbuilder = models_dict['simmatbuilder']

    optimizer = models_dict['optimizer']
    scheduler = models_dict['scheduler']
    
    unet.load_state_dict(checkpoint.get('unet_state_dict', checkpoint.get('denoise_model_state_dict')))
    unet = unet.to(device)
    params = list(unet.parameters())

    if settings['use_vit']:
        transformer.load_state_dict(checkpoint.get('transformer_state_dict', checkpoint.get('vit_model_state_dict')))
        transformer = transformer.to(device)
        params += list(transformer.parameters())
    else:
        assert transformer is None
    
    if settings['use_cls_deviation']:
        classifier.load_state_dict(checkpoint.get('classifier_state_dict', checkpoint.get('group_model_state_dict')))
        classifier = classifier.to(device)
        params += list(classifier.parameters())
    else:
        assert classifier is None
    
    if settings['use_sim_matrix']:
        simmatbuilder = simmatbuilder.to(device)
        # params += list(simmatbuilder.parameters()) # zero
    else:
        assert simmatbuilder is None
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    models_dict = {
        
        'model_type' : model_type,
        'model_name' : model_name,
        'settings' : settings,

        'transformer' : transformer,
        'unet' : unet,
        'classifier' : classifier,
        'simmatbuilder' : simmatbuilder,

        'optimizer' : optimizer,
        'scheduler' : scheduler,

        'loss_list' : loss_list,
        'valid_loss_list' : valid_loss_list,

    }
    return models_dict

def train_one_epoch(models_dict, trainloader, validloader,
                    grad_accumulation = False, accumulation_steps = accumulation_steps,
                    valid_cycle = 1, save_cycle = 1, **kwargs):
    
    model_type, model_name, settings = models_dict['model_type'], models_dict['model_name'], models_dict['settings']
    use_vit, use_cls_deviation, use_sim_matrix = settings['use_vit'], settings['use_cls_deviation'], settings['use_sim_matrix']
    cls_loss_coef, sim_loss_coef = settings['cls_loss_coef'], settings['sim_loss_coef']
    timesteps, p_uncond = settings['timesteps'], settings['p_uncond']

    transformer = models_dict['transformer']
    unet = models_dict['unet']
    classifier = models_dict['classifier']
    simmatbuilder = models_dict['simmatbuilder']

    optimizer = models_dict['optimizer']
    scheduler = models_dict['scheduler']

    loss_list, valid_loss_list = models_dict['loss_list'], models_dict['valid_loss_list']
    training_epoch = loss_list[-1][0] if len(loss_list) != 0 else 0

    template_mask = torch.tensor(get_mni152_mask(), dtype = torch.float32, device = device)

    unet.train()
    if use_vit:
        transformer.train()
    if use_cls_deviation:
        classifier.train()
    if use_sim_matrix:
        vector_buffer = []
    
    if not grad_accumulation:
        accumulation_steps = 1

    optimizer.zero_grad()
    avg_loss, avg_cls_loss, avg_sim_loss = 0., 0., 0.
    given_accumulation_steps = accumulation_steps

    for batch, data in tqdm(enumerate(trainloader), desc = f'Epoch {training_epoch + 1}', total = len(trainloader)):
        
        img_seq, age_seq, group_seq = data['image'].to(device), data['age'].to(device), data['group'].to(device)
        del data
        given_size = img_seq.size(0)
        timestep = torch.randint(0, timesteps, (given_size,), dtype = torch.long, device = device)
        
        extracted_seqs = extract_seqs(img_seq, age_seq, is_last = True, use_age_diff = False)
        cond_last_img_seq = extracted_seqs['cond_last_img_seq'].unsqueeze(1)
        cond_img_seq, cond_age_seq = extracted_seqs['cond_img_seq'], extracted_seqs['cond_age_seq']
        tgt_img_seq, tgt_age_diff_seq = extracted_seqs['tgt_img_seq'], extracted_seqs['tgt_age_diff_seq']
        del img_seq, age_seq, extracted_seqs

        tgt_img_seq = tgt_img_seq.unsqueeze(1)
        noise = torch.randn_like(tgt_img_seq, device = device)
        noised_img_seq = q_sample(tgt_img_seq, timestep, noise)
        noised_img_seq = noised_img_seq * template_mask[None, None, :, :, :]
        
        if use_vit:
            cond_signal = transformer(cond_img_seq, cond_age_seq, template_mask)
            for k in cond_signal.keys():
                assert cond_signal[k].requires_grad
        else:
            tgt_age_diff_seq = torch.zeros((given_size,), device = device)
            cond_signal = {# 'img_cond' : torch.zeros_like(tgt_img_seq, device = device),
                           'img_cond' : cond_last_img_seq,
                           'age_cond' : torch.zeros_like(tgt_age_diff_seq, device = device)}

        cond_signal_mask = torch.bernoulli(torch.tensor([1. - p_uncond] * given_size, device = device))
        
        v_theta = unet(noised_img_seq, tgt_age_diff_seq, cond_signal, cond_signal_mask, timestep)
        v_target = get_v_target(tgt_img_seq, timestep, noise)
        v_theta = v_theta * template_mask[None, None, :, :, :]
        v_target = v_target * template_mask[None, None, :, :, :]
        loss = loss_fn(v_theta, v_target) / given_accumulation_steps
        assert loss.grad_fn is not None
        avg_loss += loss.item()
        del cond_img_seq, cond_age_seq, tgt_img_seq, tgt_age_diff_seq, noised_img_seq, v_theta, v_target
        
        if use_cls_deviation:
            cls_pred = classifier(cond_signal['img_vector'])
            cls_target = group_seq
            cls_loss = cls_loss_fn(cls_pred, cls_target) * cls_loss_coef / given_accumulation_steps
            assert cls_loss.grad_fn is not None
            loss = loss + cls_loss
            avg_cls_loss += cls_loss.item()
        
        if use_sim_matrix:
            vector_buffer.append(cond_signal['img_vector'])

            if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(trainloader):
                img_vectors = torch.cat(vector_buffer, 0)
                vector_buffer = []
                sim_loss = simmatbuilder(img_vectors) * sim_loss_coef
                assert sim_loss.grad_fn is not None
                loss = loss + sim_loss
                avg_sim_loss += sim_loss.item()
        
        loss.backward()
        
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
            
            loss_tuple = (avg_loss,)
            loss_tuple += (avg_cls_loss,) if use_cls_deviation else ()
            loss_tuple += (avg_sim_loss,) if use_sim_matrix else ()
            loss_list.append((training_epoch + 1, batch + 1, *loss_tuple))
            
            avg_loss, avg_cls_loss, avg_sim_loss = 0., 0., 0.
            if grad_accumulation and (batch + 1) == len(trainloader) - len(trainloader) % accumulation_steps:
                given_accumulation_steps = len(trainloader) % accumulation_steps
    
    print('Train Loss :', tuple(analyze_loss(loss_list).iloc[-1, 1:-1]))
    
    scheduler.step()

    if (training_epoch + 1) % valid_cycle == 0:

        unet.eval()
        if use_vit:
            transformer.eval()
        if use_cls_deviation:
            classifier.eval()
        if use_sim_matrix:
            vector_buffer = []
        
        avg_valid_loss, avg_valid_cls_loss, avg_valid_sim_loss = 0., 0., 0.
        valid_cls_correct = 0

        with torch.no_grad():
            for batch, data in tqdm(enumerate(validloader), desc = f'Epoch {training_epoch + 1}', total = len(validloader)):
                
                img_seq, age_seq, group_seq = data['image'].to(device), data['age'].to(device), data['group'].to(device)
                del data
                given_size = img_seq.size(0)
                timestep = torch.randint(0, timesteps, (given_size,), dtype = torch.long, device = device)

                extracted_seqs = extract_seqs(img_seq, age_seq, is_last = True, use_age_diff = False)
                cond_last_img_seq = extracted_seqs['cond_last_img_seq'].unsqueeze(1)
                cond_img_seq, cond_age_seq = extracted_seqs['cond_img_seq'], extracted_seqs['cond_age_seq']
                tgt_img_seq, tgt_age_diff_seq = extracted_seqs['tgt_img_seq'], extracted_seqs['tgt_age_diff_seq']
                del img_seq, age_seq, extracted_seqs

                tgt_img_seq = tgt_img_seq.unsqueeze(1)
                noise = torch.randn_like(tgt_img_seq, device = device)
                noised_img_seq = q_sample(tgt_img_seq, timestep, noise)
                noised_img_seq = noised_img_seq * template_mask[None, None, :, :, :]

                if use_vit:
                    cond_signal = transformer(cond_img_seq, cond_age_seq, template_mask)
                else:
                    tgt_age_diff_seq = torch.zeros((given_size,), device = device)
                    cond_signal = {# 'img_cond' : torch.zeros_like(tgt_img_seq, device = device),
                                   'img_cond' : cond_last_img_seq,
                                   'age_cond' : torch.zeros_like(tgt_age_diff_seq, device = device)}
                
                one_mask = torch.tensor([1.] * given_size, device = device)
                
                v_theta = unet(noised_img_seq, tgt_age_diff_seq, cond_signal, one_mask, timestep)
                v_target = get_v_target(tgt_img_seq, timestep, noise)
                v_theta = v_theta * template_mask[None, None, :, :, :]
                v_target = v_target * template_mask[None, None, :, :, :]
                loss = loss_fn(v_theta, v_target)
                avg_valid_loss += loss.item() / len(validloader)
                del cond_img_seq, cond_age_seq, tgt_img_seq, tgt_age_diff_seq, noised_img_seq, v_theta, v_target
                
                if use_cls_deviation:
                    cls_pred = classifier(cond_signal['img_vector'])
                    cls_target = group_seq
                    cls_loss = cls_loss_fn(cls_pred, cls_target) * cls_loss_coef
                    avg_valid_cls_loss += cls_loss.item() / len(validloader)
                    valid_cls_correct += (cls_pred.argmax(1) == cls_target).sum().item()
                
                if use_sim_matrix:
                    vector_buffer.append(cond_signal['img_vector'])

                    if (batch + 1) == len(validloader):
                        img_vectors = torch.cat(vector_buffer, 0)
                        vector_buffer = []
                        sim_loss = simmatbuilder(img_vectors) * sim_loss_coef
                        avg_valid_sim_loss += sim_loss.item()

            valid_loss_tuple = (avg_valid_loss,)
            valid_loss_tuple += (avg_valid_cls_loss,) if use_cls_deviation else ()
            valid_loss_tuple += (avg_valid_sim_loss,) if use_sim_matrix else ()
            valid_loss_list.append((training_epoch + 1, batch + 1, *valid_loss_tuple))

            print('Valid Loss :', valid_loss_tuple)
            if use_cls_deviation:
                print(f'Cls Acc : {valid_cls_correct} / {len(validloader.dataset)} ({100 * valid_cls_correct / len(validloader.dataset):>.2f}%)')
    
    save_cond1 = (training_epoch + 1) % save_cycle == 0
    save_cond2 = (training_epoch + 1) > 50 and training_epoch == analyze_loss(valid_loss_list).iloc[:, -1].argmin(0)
    save_cond = save_cond1 or save_cond2
    if save_cond:

        save_path = os.path.join(checkpoints_path, f'{model_name}_e{training_epoch + 1}.tar')
        save_dict = {

            'model_type' : model_type,
            'model_name' : model_name,
            'settings' : settings,

            'unet_state_dict' : unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),

            'loss' : loss_list,
            'valid_loss' : valid_loss_list,

        }
        if use_vit:
            save_dict['transformer_state_dict'] = transformer.state_dict()
        if use_cls_deviation:
            save_dict['classifier_state_dict'] = classifier.state_dict()
        
        print('Saving model states... Do not disturb...')
        torch.save(save_dict, save_path)
        print(f'Trained model states at <<< Epoch {training_epoch + 1} >>> saved...!')

    models_dict = {
        
        'model_type' : model_type,
        'model_name' : model_name,
        'settings' : settings,

        'transformer' : transformer,
        'unet' : unet,
        'classifier' : classifier,
        'simmatbuilder' : simmatbuilder,

        'optimizer' : optimizer,
        'scheduler' : scheduler,

        'loss_list' : loss_list,
        'valid_loss_list' : valid_loss_list,

    }
    return models_dict