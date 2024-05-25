#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco', 'syn_city', 'gtav_ist', 'synthia_ist', 'acdc', 'parking_fisheye', 'surround_school', 'kyxz'， 'bev_2023', 'bev_2024', 'bev_20234', 'bev_20234']
# method: ['unimatch', 'unimatch_bisenetv1', 'fixmatch', 'supervised', 'supervised_bisenetv1', 'supervised_bisenetv1_tar', 'supervised_bisenetv1_both']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
# dataset='bev_20234_6cls'
dataset='bev_2023'
# dataset='avm_seg'
# method='unimatch_bisenetv1'  # semi supervised training
method='supervised_bisenetv1_tar'  # sup training
exp='bisenetv1'
split='80'
# split='6000'
# config_name='parking_bev20234_6cls_bisenetv1'
# config_name='avm_seg_bisenetv1'
config_name='HYRoad_3cls_bisenetv1'
# config_name='HYRoad_bisenetv1'
# config_name='kyxz_bisenetv1'
# config_name='surround_school_bisenetv1_source'
# config_name='surround_school_bisenetv1'
# config_name='acdc_bisenetv1'

config=configs/${config_name}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/${split}_200epoch_pretrain

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log