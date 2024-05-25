import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import yaml
from tqdm import tqdm
import logging
import random
from util.functions import calculate_entropy, calculate_confidence, save_prediction_results, save_entropy_results, save_confidence_results
from util.get_acda_iters import AC_iters
from util.utils import AverageMeter, intersectionAndUnion_gpu
from model.semseg.bisenetv1 import BiSeNetV1  # 导入分割模型
# from u2pl.dataset.builder import get_loader_single_gpu
# from u2pl.dataset import augmentation as psp_trsform

class AC_Sample:
    def __init__(self, config, ac_iters, output_root):
        self.config = config
        self.ac_iters = ac_iters
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        self.sample_times = 0
        self.output_dir = os.path.join(self.output_root, f'AC_round_{self.sample_times}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger("global")
        self.image_name_2_path, self.image_name_2_txt_content, self.labeled_names, self.unlabeled_names = self.get_image_name_2_image_path()
        self.img_transform = self.build_transform()
        self.active_strategy = self.config['active']['strategy']
        assert self.active_strategy in ['entropy', 'confidence', 'iou_score', 'random'], "Invalid active strategy, not implemented. Must be one of: 'entropy', 'confidence' or 'random'."
        self.calculate_function_dict = {
            'entropy': calculate_entropy,
            'confidence': calculate_confidence,
            'iou_score': self.calculate_iou_score,
            'random': None,
        }
        self.save_function_dict = {
            'entropy': save_entropy_results,
            'confidence': save_confidence_results,
            'iou_score': save_confidence_results,
            'random': None,
        }
        self.calculate_criteria = self.calculate_function_dict[self.active_strategy]
        self.save_criteria = self.save_function_dict[self.active_strategy]
        
        # for calculating class weights
        self.num_classes = self.config["nclass"]
        self.ignore_label=self.config['criterion']['kwargs']['ignore_index']
        self.total_pixels = 0
        self.class_img_counts = {cls:0 for cls in range(self.num_classes)}
        self.class_pixel_counts = {cls:0 for cls in range(self.num_classes)}
        self.class_img_counts_tmp = {cls:0 for cls in range(self.num_classes)}
        self.class_pixel_counts_tmp = {cls:0 for cls in range(self.num_classes)}
        self.iou_classes = torch.zeros(self.num_classes)
    
    # checked * 2
    def normalize(self, img, mask=None):
        img = self.img_transform(img)
        if mask is not None:
            mask = torch.from_numpy(np.array(mask)).long()
            return img, mask
        return img
    
    # checked * 2
    def build_transform(self):
        img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return img_transform
    
    # get the unlabeled data {image_name: image_path}
    def load_unlabeled(self, rank=0):
        unlabeled_txt = self.config["data_list"].replace('labeled', 'unlabeled')
        if rank == 0:
            self.logger.info(f"Loading unlabeled data from {unlabeled_txt}")
        with open(unlabeled_txt, 'r') as f:
            file_list = f.readlines()
            file_list = [file.strip().split(' ')[0] for file in file_list]
        f.close()
        unlabeled_path_dict = {}
        data_root = self.config["data_root"]
        # bar = tqdm(total=len(file_list))
        for file in file_list:
            image_name = file.split('/')[-1]
            image_path = os.path.join(data_root, file)
            unlabeled_path_dict[image_name] = image_path
            # bar.update(1)
        # bar.close()
        if rank == 0:
            self.logger.info(f"Unlabeled data loaded: {len(unlabeled_path_dict)} images")
        
        return unlabeled_path_dict
    
    
    # load labeled ground truth
    def load_labeled_gt(self, labeled_txt, rank=0):
        if rank == 0:
            self.logger.info(f"Loading labeled data from {labeled_txt}")
        with open(labeled_txt, 'r') as f:
            file_list = f.readlines()
            file_list = [file.strip().split(' ')[1] for file in file_list]
        f.close()
        labeled_gt_path_dict = {}
        data_root = self.config["data_root"]
        for file in file_list:
            image_name = file.split('/')[-1]
            image_path = os.path.join(data_root, file)
            labeled_gt_path_dict[image_name] = image_path
        if rank == 0:
            self.logger.info(f"Labeled gt loaded: {len(labeled_gt_path_dict)} gt images")

        return labeled_gt_path_dict
        
    
    def inference_labeled(self, model, labeled_img_lb_path_dict):
        prediction_dir = os.path.join(self.output_dir, 'pred_labeled')
        os.makedirs(prediction_dir, exist_ok=True)
        bar = tqdm(total=len(labeled_img_lb_path_dict))
        self.logger.info(f"Start inference on {len(labeled_img_lb_path_dict)} labeled images")
        five_one = len(labeled_img_lb_path_dict) // 5
        count = 0
        for img_name, (img_path, _) in labeled_img_lb_path_dict.items():
            img = Image.open(img_path).convert('RGB')
            img = self.normalize(img)
            img = img.cuda()
            img = torch.unsqueeze(img, 0)
            with torch.no_grad():
                outputs = model(img)
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    outputs = outputs[0]
                if len(outputs.size()) == 4:
                    outputs = outputs.squeeze(0)
                save_prediction_results({img_name: outputs}, prediction_dir)
            if (five_one > 0) and (count % five_one == 0):
                bar.update(five_one)
            count += 1
        bar.close()
    
    # checked
    def inference_unlabeled(self, model, unlabeled_path_dict):
        criteria_img_dict = {}
        prediction_dir = os.path.join(self.output_dir, 'pred')
        criteria_dir = os.path.join(self.output_dir, self.active_strategy)
        os.makedirs(prediction_dir, exist_ok=True)
        os.makedirs(criteria_dir, exist_ok=True)
        bar = tqdm(total=len(unlabeled_path_dict))
        self.logger.info(f"Start inference on {len(unlabeled_path_dict)} unlabeled images")
        five_one = len(unlabeled_path_dict) // 5
        count = 0
        for img_name, img_path in unlabeled_path_dict.items():
            img = Image.open(img_path).convert('RGB')
            img = self.normalize(img)  # (1,3,1024,2048)
            img = img.cuda()
            img = torch.unsqueeze(img, 0)
            with torch.no_grad():
                outputs = model(img)
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    outputs = outputs[0]  # (1, C, H, W)
                if len(outputs.size()) == 4:
                    outputs = outputs.squeeze(0)  # (C, H, W)
                save_prediction_results({img_name: outputs}, prediction_dir)
                # get the criteria for active learning
                criteria = self.calculate_criteria(outputs)
                # save the criteria result
                self.save_criteria({img_name: criteria}, criteria_dir)
                criteria_img_dict[img_name] = np.mean(criteria.cpu().numpy())
            count += 1
            if (five_one > 0) and (count % five_one == 0):
                bar.update(five_one)
        bar.close()
        
        # save the criteria_img_dict to a csv file
        with open(os.path.join(self.output_dir, '{}.csv'.format(self.active_strategy)), 'w') as f:
            f.write("image_name, {}\n".format(self.active_strategy))
            for img_name, criteria in criteria_img_dict.items():
                f.write(f"{img_name},{criteria}\n")
        return criteria_img_dict

    # checked
    def update_labeled_unlabeled(self, criteria_img_dict, rank=0):
        selected_nums = self.ac_iters.sample_increment[self.sample_times]
        if self.active_strategy == 'random':
            selected_image_names = random.sample(self.unlabeled_names, selected_nums)
        else:
            if self.active_strategy == 'entropy':  # choose higher entropy
                # choose higher entropy
                sorted_criteria = sorted(criteria_img_dict.items(), key=lambda x: x[1], reverse=True)
            elif self.active_strategy == 'confidence' or self.active_strategy == 'iou_score':  # choose lower entropy or iou score
                # choose lower entropy
                sorted_criteria = sorted(criteria_img_dict.items(), key=lambda x: x[1], reverse=False)
            selected_image_names = [item[0] for item in sorted_criteria[:selected_nums]]
        
        # add the selected_image_names to self.labeled_names
        self.labeled_names = self.labeled_names + selected_image_names
        # remove the selected_image_names from self.unlabeled_names
        self.unlabeled_names = list(set(self.unlabeled_names) - set(selected_image_names))

        if rank == 0:
            labeled_txt_path = os.path.join(self.output_dir, 'labeled.txt')
            labeled_files = open(labeled_txt_path, 'a')
            unlabeled_files = open(labeled_txt_path.replace('labeled.txt', 'unlabeled.txt'), 'w')
            with open(self.config["data_list"], 'r') as f:
                labeled_files.write(f.read())
                # if f.read not end with \n then add \n
                # if not f.read().endswith('\n'):
                #     labeled_files.write('\n')
            f.close()
            
            # write selected image names to selected.txt and labeled.txt
            selected_txt_path = os.path.join(self.output_dir, 'selected.txt')
            with open(selected_txt_path, 'w') as f:
                for img_name in selected_image_names:
                    # image_path = self.image_name_2_path[img_name]
                    # gt_path = image_path.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png').replace('leftImg8bit', 'gtFine')
                    # one_row = image_path + ' ' + gt_path + '\n'
                    one_row = self.image_name_2_txt_content[img_name]
                    f.write(one_row)
                    labeled_files.write(one_row)
            f.close()
            
            # load the selected gt and update the class_img_counts_tmp and class_pixel_counts_tmp
            selected_gt_path_dict = self.load_labeled_gt(selected_txt_path, rank)
            for _, gt_path in selected_gt_path_dict.items():
                # 读取语义分割真值并转换为 NumPy 数组
                gt_image = Image.open(gt_path)
                gt_data = np.array(gt_image, dtype=np.uint8)
                if len(gt_data.shape) == 3:
                    gt_data = gt_data[:, :, 0]
                self.update_counts(gt_data)
                
            # save the class_img_counts_tmp and class_pixel_counts_tmp into three col csv file
            with open(os.path.join(self.output_dir, 'class_img_pixel_counts_selected.csv'), 'w') as f:
                f.write("class, img_counts, pixel_counts\n")
                for cls in range(self.num_classes):
                    f.write(f"{cls}, {self.class_img_counts_tmp[cls]}, {self.class_pixel_counts_tmp[cls]}\n")
                    
            # add the class_img_counts_tmp and class_pixel_counts_tmp to class_img_counts and class_pixel_counts
            for cls in range(self.num_classes):
                self.class_img_counts[cls] += self.class_img_counts_tmp[cls]
                self.class_pixel_counts[cls] += self.class_pixel_counts_tmp[cls]
                
            # save the class_img_counts and class_pixel_counts into three col csv file
            with open(os.path.join(self.output_dir, 'class_img_pixel_counts_labeled.csv'), 'w') as f:
                f.write("class, img_counts, pixel_counts\n")
                for cls in range(self.num_classes):
                    f.write(f"{cls}, {self.class_img_counts[cls]}, {self.class_pixel_counts[cls]}\n")
            
            # reset the class_img_counts_tmp and class_pixel_counts_tmp
            self.class_img_counts_tmp = {cls:0 for cls in range(self.num_classes)}
            self.class_pixel_counts_tmp = {cls:0 for cls in range(self.num_classes)}
            
            # write the unlabeled image names to unlabeled.txt
            for img_name in self.unlabeled_names:
                # image_path = self.image_name_2_path[img_name]
                # gt_path = image_path.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png').replace('leftImg8bit', 'gtFine')
                # unlabeled_files.write(image_path + ' ' + gt_path + '\n')
                content = self.image_name_2_txt_content[img_name]
                # if not content.endswith('\n'):
                #     content += '\n'
                unlabeled_files.write(content)
            
            labeled_files.close()
            unlabeled_files.close()
            self.config['data_list'] = labeled_txt_path
            self.config['n_sup'] = len(self.labeled_names)

    # checked * 2
    def update_image_name_2_image_path(self):
        self.image_name_2_path, self.image_name_2_txt_content, self.labeled_names, self.unlabeled_names = self.get_image_name_2_image_path()
    
    # checked * 2
    def get_image_name_2_image_path(self):
        '''
        
        '''
        image_name_2_image_path = {}
        image_name_2_txt_content = {}
        labeled_names = []
        unlabeled_names = []
        labeled_txt = self.config["data_list"]
        unlabeled_txt = labeled_txt.replace('labeled', 'unlabeled')
        with open(labeled_txt, 'r') as f:
            file_list = f.readlines()
            image_name_2_txt_content.update({file.strip().split(' ')[0].split('/')[-1]: file for file in file_list})
            file_list = [file.strip().split(' ')[0] for file in file_list]
            image_name_2_image_path.update({file.split('/')[-1]: file for file in file_list})
            labeled_names = [file.split('/')[-1] for file in file_list]
        f.close()
        with open(unlabeled_txt, 'r') as f:
            file_list = f.readlines()
            image_name_2_txt_content.update({file.strip().split(' ')[0].split('/')[-1]: file for file in file_list})
            file_list = [file.strip().split(' ')[0] for file in file_list]
            image_name_2_image_path.update({file.split('/')[-1]: file for file in file_list})
            unlabeled_names = [file.split('/')[-1] for file in file_list]
        f.close()
        return image_name_2_image_path, image_name_2_txt_content, labeled_names, unlabeled_names

    # checked
    def update_counts(self, gt_data):
        unique_labels, counts = np.unique(gt_data, return_counts=True)

        for label, count in zip(unique_labels, counts):
            if label != self.ignore_label:
                self.total_pixels += count
                self.class_pixel_counts_tmp[label] += count
                self.class_img_counts_tmp[label] += 1


    def generate_class_weights(self, rank=0, fre_cfg=None, model=None):
        if fre_cfg is not None:
            if fre_cfg['strategy'] == 'num_class':
                class_weight = self.generate_class_weights_1(rank, fre_cfg)
            elif fre_cfg['strategy'] == 'iou_class':
                class_weight = self.generate_class_weights_2(rank, fre_cfg, model)
            else:
                raise ValueError("Invalid strategy for generating class weights")
            return class_weight
        else:
            raise ValueError("fre_cfg should not be None.")
    
    # generate class weight according to the number of pixels in each class in the labeled target data
    def generate_class_weights_1(self, rank=0, fre_cfg=None):
        # if self.sample_times == 1:
        #     labeled_txt = self.config["data_list"]
        # else:
        #     labeled_txt = self.config["data_list"].replace('labeled.txt', 'selected.txt')
        # labeled_gt_path_dict = self.load_labeled_gt(labeled_txt, rank)
        # for _, gt_path in labeled_gt_path_dict.items():
        #     # 读取语义分割真值并转换为 NumPy 数组
        #     gt_image = Image.open(gt_path)
        #     gt_data = np.array(gt_image, dtype=np.uint8)
        #     if len(gt_data.shape) == 3:
        #         gt_data = gt_data[:, :, 0]
        #     self.update_counts(gt_data)
        
        # cal class frequencies
        # class_frequencies = {label: count / self.total_pixels for label, count in self.class_pixel_counts.items()}
        # get class frequencies in ascending order of label id
        # class_frequencies = np.array([class_frequencies[label] for label in range(max(class_frequencies.keys()) + 1)])
        
        class_frequencies = [self.class_pixel_counts[cls] / self.total_pixels for cls in range(self.num_classes)]
        class_frequencies = np.array(class_frequencies)

        if fre_cfg is not None:
            normalize_frequencies = fre_cfg.get('normalize_frequencies', True)
            upper_value = fre_cfg.get('upper_value', 2.0)
            if normalize_frequencies:
                class_frequencies = np.median(class_frequencies) / class_frequencies
                max_frequency = max(class_frequencies)
                class_frequencies = class_frequencies / max_frequency  # to [0, 1]
            
            # from [0, 1] to [1, upper_value]
            class_weights = class_frequencies * (upper_value - 1) + 1
        else:
            # 根据类别频次生成类别权重
            class_weights = np.median(class_frequencies) / class_frequencies
        class_weights = torch.FloatTensor(class_weights)
        return class_weights
    
    # generate class weight according the iou of classes in the labeled target data
    def generate_class_weights_2(self, rank=0, fre_cfg=None, model=None):
        # calculate class weight
        class_weight = np.ones(self.num_classes) - self.iou_classes.detach().numpy()  # 1 - iou, [0, 1]
        # from [0, 1] to [1, upper_value]
        if fre_cfg is not None:
            class_weight = class_weight * (fre_cfg['upper_value'] - 1) + 1
        class_weight = torch.FloatTensor(class_weight)
        return class_weight
    
    # calculate class ious for labeled data
    def calculate_labeled_iou(self, rank=0, model=None):
        model.eval()
        # read the labeled.txt content
        labeled_txt = self.config["data_list"]
        data_root = self.config["data_root"]
        num_classes = self.num_classes
        with open(labeled_txt, 'r') as f:
            file_list = f.readlines()
            image_path = [file.strip().split(' ')[0] for file in file_list]
            lb_path = [file.strip().split(' ')[1] for file in file_list]
        f.close()
        
        labeled_img_lb_path_dict = {img.split('/')[-1]: [os.path.join(data_root, img), os.path.join(data_root, lb)] for img, lb in zip(image_path, lb_path)}
        if len(labeled_img_lb_path_dict) == 0:
            return
        
        # AC_round_1/pred_labeled
        prediction_dir = os.path.join(self.output_root, 'AC_round_{}'.format(self.sample_times), 'pred_labeled')
        os.makedirs(prediction_dir, exist_ok=True)
        bar = tqdm(total=len(labeled_img_lb_path_dict))
        self.logger.info(f"Start inference on {len(labeled_img_lb_path_dict)} labeled images")
        five_one = len(labeled_img_lb_path_dict) // 5
        count = 0
        # pred the image and cal the ious
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        for img_name, (img_path, gt_path) in labeled_img_lb_path_dict.items():
            # self.logger.info(f"Processing {img_name}, img_path: {img_path}, gt_path: {gt_path}")
            img = Image.open(img_path).convert('RGB')
            img = self.normalize(img)
            img = img.cuda()
            img = torch.unsqueeze(img, 0)
            lb = Image.open(gt_path).convert('L')
            lb = torch.from_numpy(np.array(lb, dtype=np.uint8)).long()
            lb = lb.cuda()
            with torch.no_grad():
                outputs = model(img)
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    outputs = outputs[0]
                if len(outputs.size()) == 4:
                    outputs = outputs.squeeze(0)  # (C, H, W)
            pred = save_prediction_results({img_name: outputs}, prediction_dir, return_pred=True)  # (H, W)
            intersection, union, _ = intersectionAndUnion_gpu(pred, lb, num_classes, ignore_index=self.ignore_label)
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            count += 1
            if (five_one > 0) and (count % five_one == 0):
                bar.update(five_one)
        bar.close()
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        self.iou_classes = torch.from_numpy(iou_class)
        self.logger.info(f"iou_class: {self.iou_classes}")
        # mIoU = np.mean(iou_class)

    # 
    def calculate_iou_score(self, predictions):
        # (B, C, H, W)
        if len(predictions.size()) == 4:
            preds = torch.argmax(predictions, dim=1)  # (B, H, W)
        # (C, H, W)
        elif len(predictions.size()) == 3:
            preds = torch.argmax(predictions, dim=0)  # (H, W)
        iou_score = self.iou_classes[preds]
            
        return iou_score
    
    # checked
    def run_active_learning(self, model, rank=0):
        # calculate class ious for labeled data
        self.calculate_labeled_iou(rank, model)
        
        # calculate criteria for unlabeled data
        self.output_dir = os.path.join(self.output_root, f'AC_round_{self.sample_times}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load unlabeled data
        if rank == 0:
            self.logger.info("Loading unlabeled data")
        unlabeled_data_dict = self.load_unlabeled(rank)
        if rank == 0:
            self.logger.info("Unlabeled data loaded")
        # if rank == 0:
        #     self.logger.info("Loading unlabeled data")
        #     unlabeled_data_dict = self.load_unlabeled(rank)
        #     self.logger.info("Unlabeled data loaded")

        # Inference on unlabeled data
        criteria_img_dict = {}
        if self.active_strategy != 'random':  #  and len(unlabeled_data_dict) > 0
            if rank == 0:
                self.logger.info("Inference on unlabeled data")
            criteria_img_dict = self.inference_unlabeled(model, unlabeled_data_dict)
            if rank == 0:
                self.logger.info("Inference on unlabeled data finished")
            # if rank == 0:
            #     self.logger.info("Inference on unlabeled data")
            #     criteria_img_dict = self.inference_unlabeled(model, unlabeled_data_dict)
            #     self.logger.info("Inference on unlabeled data finished")

        # Update labeled and unlabeled datasets
        # if rank == 0:
        #     self.logger.info("Updating labeled and unlabeled datasets")
        # self.update_labeled_unlabeled(criteria_img_dict, rank)
        # if rank == 0:
        #     self.logger.info("Labeled and unlabeled datasets updated")
        if rank == 0:
            self.logger.info("Updating labeled and unlabeled datasets")
            self.update_labeled_unlabeled(criteria_img_dict, rank)
            self.logger.info("Labeled and unlabeled datasets updated")

        # Increment sample count
        self.sample_times += 1

if __name__ == "__main__":
    # Usage example:
    yaml_path = "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/configs/cityscapes_acda_bisenetv1_single.yaml"
    cfg = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)

    # Initialize model
    model = BiSeNetV1(cfg)  # Initialize your segmentation model
    pretrained_path = "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/exp/cityscapes/supervised_bisenetv1/bisenetv1/1488/latest.pth"
    state_dict = torch.load(pretrained_path, map_location='cpu')
    # remove the prefix 'module.' from the keys
    state_dict['model_state'] = {k[7:]: v for k, v in state_dict['model'].items() if k.startswith('module')}
    # print(state_dict.keys())
    model.load_state_dict(state_dict['model_state'], strict=True)
    model.cuda()
    model.eval()
    
    # initialize active learning parameters
    ac_iters = AC_iters(cfg)  # Define sample rounds and increments
    
    # initialize dataloader
    # train_loader_sup, train_loader_unsup, val_loader, source_loader = get_loader_single_gpu(cfg, seed=0, ac_iters=ac_iters, epoch=0)  # Initialize your supervised training data loader
    output_root = "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/exp/cityscapes/unimatch_acda_bisenetv1_single/bisenetv1/1488/acda_log"  # Define your output directory

    # Create AC_Sample instance
    acda_semi = AC_Sample(cfg, ac_iters, output_root)

    # Run active learning
    for i in range(len(cfg['active']['sample_epoch'])):
        acda_semi.run_active_learning(model)
        cfg["data_list"] = os.path.join(acda_semi.output_dir, 'labeled.txt')
        cfg["n_sup"] = len(acda_semi.labeled_names)
        print(cfg["data_list"])
        print(acda_semi.config["data_list"])
        class_weight = acda_semi.generate_class_weights(rank=0, fre_cfg=cfg['criterion']['fre_cfg'], model=model)
        print(class_weight, '\n')
