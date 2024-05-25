### SYNTHIA
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/synthia" \
# --img_dir "RGB" \
# --img_suffix ".png" \
# --gt_dir "GT/LABELS_trainid_16" \
# --gt_suffix "_labelTrainIds.png" \
# --txt_name "synthia_train_list.txt"

### acdc
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/acdc" \
# --img_dir "rgb_anon/val" \
# --img_suffix "_rgb_anon.png" \
# --gt_dir "gt/val" \
# --gt_suffix "_gt_labelTrainIds.png" \
# --txt_name "acdc_val_list.txt" \

### cityscapes
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/cityscapes" \
# --img_dir "leftImg8bit/train" \
# --img_suffix "_leftImg8bit.png" \
# --gt_dir "gtFine/train" \
# --gt_suffix "_gtFine_labelTrainIds.png" \
# --txt_name "cityscapes_train_list.txt" \
# --sub_folder

### cityscapes_17 for surround_school
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/cityscapes" \
# --img_dir "leftImg8bit/train_512" \
# --img_suffix "_leftImg8bit.png" \
# --gt_dir "gtFine/train_17_512" \
# --gt_suffix "_gtFine_labelTrainIds.png" \
# --txt_name "cityscapes_train_17_512_list.txt" \
# --sub_folder


### surround_school
# split="val"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/surround_school" \
# --img_dir "leftImg8bit/${split}" \
# --img_suffix "_Img8bit.png" \
# --gt_dir "gtFine/${split}" \
# --gt_suffix "_gtFine_labelTrainIds.png" \
# --txt_name "surround_school_${split}_list.txt" \


### parking_fisheye
# split="val"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/parking_fisheye" \
# --img_dir "leftImg8bit/${split}" \
# --img_suffix "_Img8bit.png" \
# --gt_dir "gtFine/${split}" \
# --gt_suffix "_gtFine_labelTrainIds.png" \
# --txt_name "parking_fisheye_${split}_list.txt" \

### kyxz
# split="train"
# dataset="kyxz"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/kyxz" \
# --img_dir "image/${split}" \
# --img_suffix ".png" \
# --gt_dir "label/${split}" \
# --gt_suffix ".png" \
# --txt_name "${dataset}_${split}_list.txt" \

### rellis3d
# split="train"
# dataset="rellis3d"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/rellis3d" \
# --img_dir "image" \
# --img_suffix ".jpg" \
# --gt_dir "label" \
# --gt_suffix ".png" \
# --txt_name "${dataset}_${split}_list.txt" \
# --sub_folder

###parking bev
# split="train"
# dataset="bev_20234_6cls"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/${dataset}" \
# --img_dir "image/${split}" \
# --img_suffix ".png" \
# --gt_dir "label/${split}" \
# --gt_suffix ".png" \
# --txt_name "${dataset}_${split}_list.txt" \

# split="val"
# dataset="HYRoad_3cls"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/${dataset}" \
# --img_dir "image/${split}" \
# --img_suffix ".png" \
# --gt_dir "label/${split}" \
# --gt_suffix ".png" \
# --txt_name "${dataset}_${split}_list.txt" \

# avm_seg
# split="val"
# dataset="avm_seg"
# python gen_dataset_txt.py \
# --data_root "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/data/${dataset}" \
# --img_dir "image_rotate_and_resize2/${split}" \
# --img_suffix ".png" \
# --gt_dir "label_rotate_and_resize2/${split}" \
# --gt_suffix ".png" \
# --txt_name "${dataset}_${split}_list.txt" \