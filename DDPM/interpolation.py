from data.data_loader import validation_set
from utils import *
from sample import *

def get_interpolated_ages(age: torch.Tensor, tol_diff: float = 1.0) -> torch.Tensor:
    '''
    * age : age tensor from data['age'].
    * tol_diff : tolerable difference between age.
                 less than or same as tol_diff age interval needs no sampling.
    '''
    assert age.ndim == 1
    length = age.ne(0).sum(0).item()

    age_diff = torch.diff(age)[:(length - 1)]

    floor_divided = torch.div(age_diff, tol_diff, rounding_mode = 'floor').long()
    is_divisible = (age_diff % tol_diff == 0).long()
    sampling_needed_num = floor_divided - is_divisible

    new_ages = [torch.linspace(age[i], age[i + 1], num)[1:-1] for i, num in enumerate(sampling_needed_num + 2)]
    new_ages = torch.cat(new_ages)

    interpolated_ages = torch.cat((age[:length], new_ages))
    sampling_needed_bool = torch.cat((torch.zeros_like(age[:length]), torch.ones_like(new_ages))).bool()
    interpolated_ages, sorting_indices = torch.sort(interpolated_ages)
    sampling_needed_bool = sampling_needed_bool[sorting_indices]

    ## Additional interpolation rule:
    # If no interpolation occurred for a given age and tol_diff input,
    # do interpolation where the age interval (age_diff) is largest and closest to last.
    # This ensures interpolation always occurs at least once for the given sequence.
    # This rule was only applied to 'one' piece of data in the validation dataset (1 out of 59).
    if not sampling_needed_bool.any():
        max_idx_at_last = (len(age_diff) - torch.flip(age_diff, [0]).argmax(0) - 1).item()
        new_age = torch.mean(age[max_idx_at_last:(max_idx_at_last + 2)]).unsqueeze(0)
        interpolated_ages = torch.cat((age[:(max_idx_at_last + 1)], new_age, age[(max_idx_at_last + 1):length]))
        sampling_needed_bool = torch.zeros_like(interpolated_ages).bool()
        sampling_needed_bool[max_idx_at_last + 1] = True

    return interpolated_ages, sampling_needed_bool


def get_interpolated_images(data, eval_models_dict, tol_age_diff, sampling_per_img, **kwargs) -> Tuple:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    updated_img_seq = data['image'][0:1]
    updated_age_seq = data['age'][0:1]

    interpolated_ages, sampling_needed_bool = get_interpolated_ages(data['age'], tol_age_diff)
    assert updated_age_seq[0] == interpolated_ages[0]

    for i, interpolated_age in enumerate(interpolated_ages[1:]):
        # sampling = sampling_needed_bool[1:][i]
        sampling = True
        if sampling:
            last_sampled_img_list = []

            frame_num = 9
            cond_start_idx = max(0, len(updated_age_seq) - frame_num)
            cond_img_seq = image_padding(updated_img_seq[cond_start_idx:], frame_num).unsqueeze(0)
            cond_age_seq = age_padding(updated_age_seq[cond_start_idx:], frame_num).unsqueeze(0)

            for seed in range(0, sampling_per_img):
                sampled_imgs, _ = DDIM_sample(
                    cond_img_seq, cond_age_seq, [interpolated_age],
                    eval_models_dict,
                    guide_w = kwargs.get('guide_w', 1.3),
                    eta = kwargs.get('eta', 0.0),
                    steps = kwargs.get('steps', 30),
                    seed = seed,
                    memory_cycle = None,
                    device = device,
                )
                last_sampled_img_list.append(sampled_imgs[-1, 0, 0])
            
            last_sampled_img = np.stack(last_sampled_img_list, 0).mean(0)
            interpolated_img = torch.tensor(last_sampled_img)
        else:
            idx_to_extract = torch.nonzero(data['age'] == interpolated_age).item()
            interpolated_img = data['image'][idx_to_extract]
        
        updated_img_seq = torch.cat((updated_img_seq, interpolated_img.unsqueeze(0)))
        updated_age_seq = torch.cat((updated_age_seq, interpolated_age.unsqueeze(0)))
    
    assert (updated_age_seq == interpolated_ages).all()
    return updated_img_seq, updated_age_seq


def make_sample_able_dataset(dataset: List) -> List:

    def truncate_data(data: dict) -> Tuple:
        frame_num = 9
        truncated_idx = max(0, len(data['age']) - frame_num)
        truncated_img_seq = image_padding(data['image'][truncated_idx:], frame_num)
        truncated_age_seq = age_padding(data['age'][truncated_idx:], frame_num)
        return {
            'subject_id' : data.get('subject_id'),
            'image' : truncated_img_seq,
            'age' : truncated_age_seq,
        }
    
    return list(map(truncate_data, dataset))


load_model_name = 'STAge_v1_bs2_ga2_ps6_max_ndiff_d(108_11_32)_FF_e114'
eval_models_dict = load_eval_models(load_model_name)


# The 1st code block here is for autoregressive generation for interpolation.
# Rules for image/age interpolation follow the function get_interpolated_images.
# After interpolated_validation_set was completed, I ran the 2nd code block below.


try:
    interpolated_validation_set = torch.load(f'/projects1/pi/hkjung/DDPM/data/interpolated_validation_set_v1_all.tar')
except:
    interpolated_validation_set = []

for i in range(len(validation_set)):

    if i < len(interpolated_validation_set):
        continue
    else:
        print(f'{i}-th validation data ({validation_set[i]["subject_id"]}) interpolation...')
    
    updated_img_seq, updated_age_seq = get_interpolated_images(
        validation_set[i], eval_models_dict,
        tol_age_diff = 1.0, sampling_per_img = 5,
        guide_w = 1.3, steps = 30
    )
    updated_dir = {
        'subject_id' : validation_set[i]['subject_id'],
        'image' : updated_img_seq,
        'age' : updated_age_seq,
    }
    interpolated_validation_set.append(updated_dir)

    save_path = f'/projects1/pi/hkjung/DDPM/data/interpolated_validation_set_v1_all.tar'
    torch.save(interpolated_validation_set, save_path)




# The 2nd code block here is for trivial transformation of interpolated_validation_set into
# the sample_able_dataset, each data of which has length 9 so that DDIM_sample can operate.
# By save_DDIM_sample, the last image within each data sequence is deemed as the GT target image to generate,
# harnessing the rest of the (interpolated) sequence elements as the conditional.

"""
interpolated_validation_set = torch.load(f'/projects1/pi/hkjung/DDPM/data/interpolated_validation_set_v1.tar')
print(len(interpolated_validation_set))  # 59

interpolated_validation_set = make_sample_able_dataset(interpolated_validation_set)

for i in range(len(interpolated_validation_set)):
    print('STAge_max on interpolated_validation_set :', i)
    save_DDIM_sample(
        eval_models_dict, i, interpolated_validation_set,
        guide_w_list = [1.3], eta_list = [0.0], steps_list = [30], seed_avg = 5,
        folder_suffix = 'samples_after_interpolation_v1',
    )

"""