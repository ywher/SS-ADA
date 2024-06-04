#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['bev_2023', 'bev_2024']
# method: ['supervised_bisenetv1_tar', 'supervised_bisenetv1_both']
# exp: just for specifying the 'save_path'
# split: ['140', '70', '35', ...]. Please check directory './splits/$dataset' for concrete splits
# config_name: [parking_bev2024_bisenetv1, parking_bev2023_bisenetv1] Please check directory './configs' for specific filename

# source only
# dataset='bev_2023'
# method='supervised_bisenetv1_tar'  # sup training
# exp='bisenetv1'
# split='110'
# config_name='parking_bev2023_bisenetv1'

# bev2024 supervised learning
dataset='bev_2024'
method='supervised_bisenetv1_tar'
exp='bisenetv1'
split='140'
config_name='parking_bev2024_bisenetv1'

# bev2024 joint training
# dataset='bev_2024'
# method='supervised_bisenetv1_both'
# exp='bisenetv1'
# split='140'
# config_name='parking_bev2024_bisenetv1'

config=configs/${config_name}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/${split}_100epoch_lr0.01

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log