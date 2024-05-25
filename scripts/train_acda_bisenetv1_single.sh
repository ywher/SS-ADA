#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco']
# method: ['unimatch', 'unimatch_bisenetv1', 'fixmatch', 'supervised', 'supervised_bisenetv1']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits

### cityscapes
# dataset='cityscapes'
# method='unimatch_acda_bisenetv1_single'
# exp='bisenetv1'
# split='744'
# config_name='cityscapes_acda_bisenetv1_single'
# init_split=30

# dataset='syn_city'
# method='unimatch_acda_bisenetv1_single'
# exp='bisenetv1'
# split='744'
# config_name='cityscapes_acda_syn_bisenetv1_single'
# init_split=30

### acdc
# dataset='acdc'
# method='unimatch_acda_bisenetv1_single'
# exp='bisenetv1'
# split='200'
# config_name='acdc_acda_bisenetv1_single'
# init_split=16

### surround_school
# dataset='surround_school'
# method='unimatch_acda_bisenetv1_single'
# exp='bisenetv1'
# split='56'
# config_name='surround_school_acda_bisenetv1_single'
# init_split=5


# dataset='kyxz'
# method='unimatch_acda_bisenetv1_aux_single'
# exp='bisenetv1'
# split='925'
# config_name='kyxz_acda_bisenetv1_single'
# init_split=19


### parking bev
dataset='bev_2024'
method='unimatch_acda_bisenetv1_single'
exp='bisenetv1'
split='28'
config_name='parking_bev2024_acda_bisenetv1_single'
init_split=1

config=configs/${config_name}.yaml
labeled_id_path=splits/$dataset/$init_split/labeled.txt
unlabeled_id_path=splits/$dataset/$init_split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/${split}_entropy_200epoch  #_iouclass

mkdir -p $save_path

CUDA_VISIBLE_DEVICES=0 python $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path 2>&1 | tee $save_path/$now.log