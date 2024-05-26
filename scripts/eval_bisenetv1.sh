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

### bev_2023
# dataset="bev_2023"
# config_path="configs/parking_bev2023_bisenetv1.yaml"
# eval_mode="original"
# exp_folder="supervised_bisenetv1_tar"
# for split in 110
# do
#     exp_root="exp/${dataset}/${exp_folder}/bisenetv1/${split}"
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

### bev_2024
dataset="bev_2024"
config_path="configs/parking_bev2024_bisenetv1.yaml"
eval_mode="original"
exp_folder="supervised_bisenetv1_both"
exp_folder="ss_ada_bisenetv1_single"
for split in 70_entropy_200epoch_iou # 110_150epoch 110_200epoch
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
