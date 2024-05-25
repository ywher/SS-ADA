
### acdc
# dataset_name='acdc'
# ratio=1
# for ratio in 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022
# do
#     python gen_label_unlabel_samples.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_path '../../data/acdc' \
#         --img_folder 'rgb_anon/train' \
#         --img_suffix '_rgb_anon.png' \
#         --gt_folder 'gt/train' \
#         --gt_suffix '_gt_labelTrainIds.png' \
#         --sample_ratio ${ratio} \
#         --output_path "../../splits/${dataset_name}" \
#         --random_seed 0
# done


### surround_school
# dataset_name='surround_school'
# ratio=1
# for ratio in 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022
# do
#     python gen_label_unlabel_samples.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_path "../../data/${dataset_name}" \
#         --img_folder "leftImg8bit/train" \
#         --img_suffix "_Img8bit.png" \
#         --gt_folder "gtFine/train" \
#         --gt_suffix "_gtFine_labelTrainIds.png" \
#         --sample_ratio ${ratio} \
#         --output_path "../../splits/${dataset_name}" \
#         --random_seed 0
# done

### cityscapes
# python gen_label_unlabel_samples.py \
#     --dataset_name "${dataset_name}" \
#     --dataset_path '../../data/acdc' \
#     --img_folder 'rgb_anon/train' \
#     --img_suffix '_rgb_anon.png' \
#     --gt_folder 'gt/train' \
#     --gt_suffix '_gt_labelTrainIds.png' \
#     --sample_ratio ${ratio} \
#     --output_path "../../splits/${dataset_name}" \
#     --random_seed 0


### kyxz
# dataset_name='kyxz'
# ratio=1
# for ratio in 1.0 #0.5 0.25 0.125 0.0625 0.05 0.022 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022
# do
#     python gen_label_unlabel_samples.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_path "../../data/${dataset_name}" \
#         --img_folder "image/train" \
#         --img_suffix ".png" \
#         --gt_folder "label/train" \
#         --gt_suffix ".png" \
#         --sample_ratio ${ratio} \
#         --output_path "../../splits/${dataset_name}" \
#         --random_seed 0
# done


### bev_2023
dataset_name='bev_2024'
ratio=1
for ratio in 0.5 0.25 0.125 0.0625 0.05 0.022 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022 #
do
    python gen_label_unlabel_samples.py \
        --dataset_name "${dataset_name}" \
        --dataset_path "../../data/${dataset_name}" \
        --img_folder "image/train" \
        --img_suffix ".png" \
        --gt_folder "label/train" \
        --gt_suffix ".png" \
        --sample_ratio ${ratio} \
        --output_path "../../splits/${dataset_name}" \
        --random_seed 0
done

# dataset_name='bev_20234_6cls'
# ratio=1
# for ratio in 1.0 #0.5 0.25 0.125 0.0625 0.05 0.022 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022
# do
#     python gen_label_unlabel_samples.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_path "../../data/${dataset_name}" \
#         --img_folder "image/train" \
#         --img_suffix ".png" \
#         --gt_folder "label/train" \
#         --gt_suffix ".png" \
#         --sample_ratio ${ratio} \
#         --output_path "../../splits/${dataset_name}" \
#         --random_seed 0
# done

# dataset_name='avm_seg'
# ratio=1
# for ratio in 1.0 #0.5 0.25 0.125 0.0625 0.05 0.022 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022
# do
#     python gen_label_unlabel_samples.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_path "../../data/${dataset_name}" \
#         --img_folder "image_rotate_and_resize2/train" \
#         --img_suffix ".png" \
#         --gt_folder "label_rotate_and_resize2/train" \
#         --gt_suffix ".png" \
#         --sample_ratio ${ratio} \
#         --output_path "../../splits/${dataset_name}" \
#         --random_seed 0
# done

### 
# dataset_name='HYRoad_3cls'
# ratio=1
# for ratio in 1.0 #0.5 0.25 0.125 0.0625 0.05 0.022 0.01 # 0.5 0.25 0.125 0.0625 0.05 0.022
# do
#     python gen_label_unlabel_samples.py \
#         --dataset_name "${dataset_name}" \
#         --dataset_path "../../data/${dataset_name}" \
#         --img_folder "image/train" \
#         --img_suffix ".png" \
#         --gt_folder "label/train" \
#         --gt_suffix ".png" \
#         --sample_ratio ${ratio} \
#         --output_path "../../splits/${dataset_name}" \
#         --random_seed 0
# done