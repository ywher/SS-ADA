import os
from tqdm import tqdm
import argparse

def gen_dataset_txt(args):
    img_suffix = args.img_suffix
    gt_suffix = args.gt_suffix
    txt_name = args.txt_name
    # output_txt_path = os.path.join(args.data_root.split('/')[0], txt_name)
    output_txt_path = os.path.join(txt_name)
    print('Output txt path: {}'.format(output_txt_path))
    
    image_names = os.listdir(os.path.join(args.data_root, args.img_dir))
    image_names = [img_name for img_name in image_names if img_name.endswith(img_suffix)]
    gt_names = os.listdir(os.path.join(args.data_root, args.gt_dir))
    gt_names = [gt_name for gt_name in gt_names if gt_name.endswith(gt_suffix)]
    print('#images: {}, #gt: {}'.format(len(image_names), len(gt_names)))
    assert len(image_names) == len(gt_names), "The number of images and ground truth is not equal."
    
    with open(output_txt_path, 'w') as f:
        for img_name, gt_name in zip(sorted(image_names), sorted(gt_names)):
            img_item = os.path.join(args.img_dir, img_name)
            gt_item = os.path.join(args.gt_dir, gt_name)
            f.write('{} {}\n'.format(img_item, gt_item))
    f.close()


def gen_dataset_txt_subfolder(args):
    img_suffix = args.img_suffix
    gt_suffix = args.gt_suffix
    txt_name = args.txt_name
    # output_txt_path = os.path.join(args.data_root.split('/')[0], txt_name)
    output_txt_path = os.path.join(txt_name)
    print('Output txt path: {}'.format(output_txt_path))
    
    image_names = []
    gt_names = []
    sub_folders = os.listdir(os.path.join(args.data_root, args.img_dir))
    sub_folders.sort()
    for sub_folder in sub_folders:
        sub_image_names = os.listdir(os.path.join(args.data_root, args.img_dir, sub_folder))
        sub_image_names = [os.path.join(sub_folder, img_name) for img_name in sub_image_names if img_name.endswith(img_suffix)]
        sub_gt_names = os.listdir(os.path.join(args.data_root, args.gt_dir, sub_folder))
        sub_gt_names = [os.path.join(sub_folder, gt_name) for gt_name in sub_gt_names if gt_name.endswith(gt_suffix)]
        print('Subfolder: {}, #images: {}, #gt: {}'.format(sub_folder, len(sub_image_names), len(sub_gt_names)))
        assert len(sub_image_names) == len(sub_gt_names), "The number of images and ground truth is not equal."
        # direct extend the list
        image_names.extend(sub_image_names)
        gt_names.extend(sub_gt_names)
    
    with open(output_txt_path, 'w') as f:
        for img_name, gt_name in zip(sorted(image_names), sorted(gt_names)):
            img_item = os.path.join(args.img_dir, img_name)
            gt_item = os.path.join(args.gt_dir, gt_name)
            f.write('{} {}\n'.format(img_item, gt_item))
    f.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/data/synthia')
    parser.add_argument('--img_dir', type=str, default='RGB')
    parser.add_argument('--img_suffix', type=str, default='.png')
    parser.add_argument('--gt_dir', type=str, default='GT/LABELS_trainid_16')
    parser.add_argument('--gt_suffix', type=str, default='_labelTrainIds.png')
    parser.add_argument('--txt_name', type=str, default='synthia_train_list.txt')
    parser.add_argument('--sub_folder', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.sub_folder:
        gen_dataset_txt_subfolder(args)
    else:
        gen_dataset_txt(args)