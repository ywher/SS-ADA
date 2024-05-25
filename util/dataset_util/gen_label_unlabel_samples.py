import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Sampling")
    parser.add_argument("--dataset_name", type=str, default='cityscapes', choices=['cityscapes', 'acdc', 'surround_school', 'kyxz', 'bev_2023', 'bev_2024', 'bev_20234', 'bev_20234_6cls', 'HYRoad', 'HYRoad_3cls', 'avm_seg'], help="Name of the dataset (e.g., cityscapes)")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--img_folder", type=str, default="rgb_anon/train", help="Path to the image folder")
    parser.add_argument("--img_suffix", type=str, default=".png", help="Suffix of the image files")
    parser.add_argument("--gt_folder", type=str, default="gt/train", help="Path to the ground truth folder")
    parser.add_argument("--gt_suffix", type=str, default="_gt_labelTrainIds.png", help="Suffix of the ground truth files")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="Sampling ratio")
    parser.add_argument("--output_path", type=str, help="Path to the output folder")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")

    return parser.parse_args()

def sample_dataset(args):
    dataset_path = args.dataset_path
    img_folder = args.img_folder
    img_suffix = args.img_suffix
    gt_folder = args.gt_folder
    gt_suffix = args.gt_suffix
    sample_ratio = args.sample_ratio
    random_seed = args.random_seed
    random.seed(random_seed)

    # 统计数据集路径下所有的图片和真值路径 
    image_gt_pathes = []
    img_folder_path = os.path.join(dataset_path, img_folder)
    gt_folder_path = os.path.join(dataset_path, gt_folder)
    img_names = os.listdir(img_folder_path)
    img_names = [img_name for img_name in img_names if img_name.endswith(img_suffix)]
    gt_names = os.listdir(gt_folder_path)
    gt_names = [gt_name for gt_name in gt_names if gt_name.endswith(gt_suffix)]
    assert len(img_names) == len(gt_names), "The number of images and ground truth is not equal."
    for img_name, gt_name in zip(sorted(img_names), sorted(gt_names)):
        img_path = os.path.join(img_folder, img_name)
        gt_path = os.path.join(gt_folder, gt_name)
        image_gt_pathes.append([img_path, gt_path])
    
    
    # city_folders = os.listdir(dataset_root)
    # for city_folder in city_folders:
    #     city_folder_path = os.path.join(dataset_root, city_folder)
    #     image_names = os.listdir(city_folder_path)
    #     for image_name in image_names:
    #         image_path = os.path.join("leftImg8bit", 'train', city_folder, image_name)
    #         image_paths.append(image_path)
            
    # 打乱图片真值路径
    random.shuffle(image_gt_pathes)

    # 随机采样一定数量的图片名称
    sample_num = int(len(image_gt_pathes) * sample_ratio + 0.5)
    sampled_image_gt_paths = random.sample(image_gt_pathes, sample_num)
    remained_image_gt_paths = [path for path in image_gt_pathes if path not in sampled_image_gt_paths]

    return sampled_image_gt_paths, remained_image_gt_paths  # in list format

def main():
    args = parse_args()

    # 采样数据集
    sampled_image_gt_paths, remained_image_gt_paths = sample_dataset(args)
    num_sampled_images = len(sampled_image_gt_paths)

    # 创建输出文件夹
    output_folder = os.path.join(args.output_path, str(num_sampled_images))
    os.makedirs(output_folder, exist_ok=True)

    # 将采样的图片路径保存为 labeled.txt
    with open(os.path.join(output_folder, "labeled.txt"), "w") as f:
        for path in sampled_image_gt_paths:
            f.write("{} {}\n".format(path[0], path[1]))

    # 将未被采样到的图片路径保存为 unlabeled.txt
    with open(os.path.join(output_folder, "unlabeled.txt"), "w") as f:
        for path in remained_image_gt_paths:
            f.write("{} {}\n".format(path[0], path[1]))

    print(f"Sampling completed. Sampled {num_sampled_images} images.")
    print(f"Sampled images are saved to {os.path.join(output_folder, 'labeled.txt')}")
    print(f"Unsampled images are saved to {os.path.join(output_folder, 'unlabeled.txt')}")

if __name__ == "__main__":
    main()
