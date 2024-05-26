# python infer_bisenetv1.py \
# --config "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/cityscapes_acda_bisenetv1_single.yaml" \
# --img_fodler "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/images" \
# --save_folder "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/outputs" \
# --exp_root "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/cityscapes/unimatch_acda_bisenetv1/bisenetv1/entropy/1488_[20,40,60]_reweight" \
# --model_type "latest" \
# --model_path "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/cityscapes/unimatch_acda_bisenetv1/bisenetv1/entropy/1488_[20,40,60]_reweight/latest.pth" \

### 
# split='back'
# CUDA_VISIBLE_DEVICES=1 python infer_bisenetv1.py \
# --config "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/parking_fisheye_bisenetv1.yaml" \
# --img_fodler "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/parking_longbin/${split}" \
# --save_folder "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/parking_longbin/${split}_pred_480" \
# --exp_root "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/parking_fisheye/supervised_bisenetv1_tar/bisenetv1/140" \
# --model_type "best" \
# --model_path "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/parking_fisheye/supervised_bisenetv1_tar/bisenetv1/140/best.pth" \
# --resize_ratio 0.476

# python infer_bisenetv1.py \
# --config "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/parking_fisheye_bisenetv1.yaml" \
# --img_fodler "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/parking_fisheye/leftImg8bit/val" \
# --save_folder "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/parking_fisheye/leftImg8bit/val_0.5_pred" \
# --exp_root "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/parking_fisheye/supervised_bisenetv1_tar/bisenetv1/140" \
# --model_type "best" \
# --model_path "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/parking_fisheye/supervised_bisenetv1_tar/bisenetv1/140/best.pth" \
# --resize_ratio 0.5

### parking_bev
# python infer_bisenetv1.py \
# --config "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/parking_bev_bisenetv1.yaml" \
# --img_fodler "/media/ywh/pool2/datasets/ros_bag/longbin_parking/2024-04-17-20-14-46/birdview" \
# --save_folder "/media/ywh/pool2/datasets/ros_bag/longbin_parking/2024-04-17-20-14-46/birdview_pred_resize2" \
# --exp_root "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/bev_2023/supervised_bisenetv1_tar/bisenetv1/110_200epoch" \
# --model_type "best" \
# --model_path "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/bev_2023/supervised_bisenetv1_tar/bisenetv1/110_200epoch/best.pth" \
# --resize_ratio 0.5

### bev2023 model opredict on bev2024

config_root="/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs"
# config_name="parking_bev_bisenetv1"
config_name="parking_bev2024_acda_bisenetv1_single"
data_root="/media/ywh/pool1/yanweihao/dataset/ros_bag/longbin_parking"
data_folder="2024-04-27-19-01-09"
# exp_root="/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/bev_2023/supervised_bisenetv1_tar/bisenetv1/110_200epoch"
exp_root="/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/bev_2024/supervised_bisenetv1_tar/bisenetv1/140_200epoch"
# exp_root="/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/bev_2024/unimatch_acda_bisenetv1_single/bisenetv1/35_entropy_200epoch"
CUDA_VISIBLE_DEVICES=1 python infer_bisenetv1.py \
--config "${config_root}/${config_name}.yaml" \
--img_fodler "${data_root}/${data_folder}/avm" \
--save_folder "${data_root}/${data_folder}/avm_bev2024_pred" \
--exp_root "${exp_root}" \
--model_type "best" \
--model_path "${exp_root}/best.pth" \
# --resize_ratio 0.5


