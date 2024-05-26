#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['bev_2023', 'bev_2024']
# method: ['ss_ada_bisenetv1_single']
# exp: just for specifying the 'save_path'
# split: ['140', '70', '35', ...]. Please check directory './splits/$dataset' for concrete splits
# config_name: [parking_bev2024_acda_bisenetv1_single] Please check directory './configs' for specific filename

### parking bev
dataset='bev_2024'
method='ss_ada_bisenetv1_single'
exp='bisenetv1'
split='70'
config_name='parking_bev2024_acda_bisenetv1_single'
init_split=1

config=configs/${config_name}.yaml
labeled_id_path=splits/$dataset/$init_split/labeled.txt
unlabeled_id_path=splits/$dataset/$init_split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/${split}_entropy_200epoch  #_iouclass

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=1 python $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path 2>&1 | tee $save_path/$now.log