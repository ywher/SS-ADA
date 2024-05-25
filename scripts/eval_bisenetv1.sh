### cityscapes
# config_path="configs/cityscapes_acda_bisenetv1.yaml"
# # config_path="configs/cityscapes_acda_syn_bisenetv1.yaml"
# config_path="configs/cityscapes_acda_bisenetv1_single.yaml"
# # config_path="configs/cityscapes_da_bisenetv1.yaml"
# # config_path="configs/cityscapes_ac_bisenetv1_single.yaml"
# # config_path="configs/cityscapes_bisenetv1.yaml"
# # config_path="configs/cityscapes_syn_bisenetv1.yaml"
# # config_path="configs/surround_school_bisenetv1_source.yaml"
# # exp_root="exp/cityscapes/unimatch_acda_bisenetv1/bisenetv1/1488"
# # exp_root="exp/syn_city/supervised_bisenetv1/bisenetv1/2975"
# # exp_root="exp/cityscapes/unimatch_bisenetv1/bisenetv1/149_eden"
# # exp_root="exp/syn_city/unimatch_bisenetv1/bisenetv1/149_fx"
# # exp_root="exp/cityscapes/supervised_bisenetv1_tar/bisenetv1/2975_17"
# # exp_root="exp/syn_city/unimatch_acda_bisenetv1/bisenetv1/1488_fx"
# # exp_root="exp/cityscapes/unimatch_acda_bisenetv1_single/bisenetv1/186_random"
# # exp_root="exp/cityscapes/unimatch_da_bisenetv1/bisenetv1/1488"
# # exp_root="exp/cityscapes/unimatch_ac_bisenetv1_single/bisenetv1/1488_entropy"
# eval_mode="sliding_window"

# for model_type in "latest" "best"
# do
#     model_path="${exp_root}/${model_type}.pth"
#     CUDA_VISIBLE_DEVICES=1 python eval_bisenetv1.py \
#     --config ${config_path} \
#     --model_path ${model_path} \
#     --exp_root ${exp_root} \
#     --eval_mode ${eval_mode} \
#     --model_type ${model_type} \
#     --save_pred \
#     --save_color \
#     --show_bar
# done

### acdc
# config_path="configs/acdc_acda_bisenetv1_single.yaml"
# # config_path="configs/acdc_bisenetv1.yaml"
# eval_mode="sliding_window"
# exp_folder="unimatch_acda_bisenetv1_single"
# # exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_bisenetv1"
# for split in 800 # 800_class_iou_weight 400_class_iou_weight # 800 400 200 100
# do
#     exp_root="exp/acdc/${exp_folder}/bisenetv1/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=1 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done


### surround_school
# dataset="surround_school"
# # config_path="configs/surround_school_bisenetv1.yaml"
# # config_path="configs/surround_school_acda_bisenetv1_single.yaml"
# config_path="configs/surround_school_bisenetv1_source.yaml"
# eval_mode="original"
# # exp_folder="unimatch_bisenetv1"
# exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_acda_bisenetv1_single"
# for split in 2975_17 # 56_entropy_3_60 #450 # 225 113 56 28 23 10 113_debug
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
#     for model_type in "best" # "latest"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done

### parking
# dataset="parking_fisheye"
# config_path="configs/parking_fisheye_bisenetv1.yaml"
# eval_mode="original"
# resize_ratio=0.5
# exp_folder="supervised_bisenetv1_tar"
# for split in 140 #450 # 225 113 56 28 23 10 113_debug
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done

### kyxz
# dataset="kyxz"
# config_path="configs/kyxz_bisenetv1.yaml"
# eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_bisenetv1"
# for split in 1850 925 463 231 116 93 41 # 1850 925 463 231 116 93 41
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/100_epoch/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done


### bev_2023
# dataset="bev_2023"
# config_path="configs/parking_bev_bisenetv1.yaml"
# eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_bisenetv1"
# for split in 110_200epoch  # 110_150epoch 110_200epoch
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done

### bev_2024
dataset="bev_2024"
# config_path="configs/parking_bev2024_bisenetv1.yaml"
config_path="configs/parking_bev2024_acda_bisenetv1_single.yaml"
eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
exp_folder="unimatch_acda_bisenetv1_single"
# exp_folder="unimatch_bisenetv1"
for split in 70_entropy_200epoch_iouclass 70_entropy_200epoch 35_entropy_200epoch # 110_150epoch 110_200epoch
do
    exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
    for model_type in "latest" "best"
    do
        model_path="${exp_root}/${model_type}.pth"
        CUDA_VISIBLE_DEVICES=1 python eval_bisenetv1.py \
        --config ${config_path} \
        --model_path ${model_path} \
        --exp_root ${exp_root} \
        --eval_mode ${eval_mode} \
        --model_type ${model_type} \
        --save_pred \
        --save_color \
        --show_bar
    done
done

###bev_20234
# dataset="bev_20234"
# config_path="configs/parking_bev20234_bisenetv1.yaml"
# eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_bisenetv1"
# for split in 250_200epoch  # 110_150epoch 110_200epoch
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done

###bev_20234
# dataset="bev_20234_6cls"
# config_path="configs/parking_bev20234_6cls_bisenetv1.yaml"
# eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_bisenetv1"
# for split in 240_200epoch_resize3_pretrain2lr0.005 # 110_150epoch 110_200epoch
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done

### HYRoad
# dataset="HYRoad"  #HYRoad_3cls
# config_path="configs/HYRoad_bisenetv1.yaml"  #HYRoad_3cls_bisenetv1
# eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
# # exp_folder="unimatch_bisenetv1"
# for split in 80_200epoch_pretrain  # 110_150epoch 110_200epoch
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
#     for model_type in "latest" "best"
#     do
#         model_path="${exp_root}/${model_type}.pth"
#         CUDA_VISIBLE_DEVICES=0 python eval_bisenetv1.py \
#         --config ${config_path} \
#         --model_path ${model_path} \
#         --exp_root ${exp_root} \
#         --eval_mode ${eval_mode} \
#         --model_type ${model_type} \
#         --save_pred \
#         --save_color \
#         --show_bar
#     done
# done