from copy import deepcopy
import math
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from dataset.transform import *

from PIL import Image
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.semseg.bisenetv1 import BiSeNetV1
# from util.get_acda_iters import AC_iters
from util.ohem import ProbOhemCrossEntropy2d

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None, ac_iters=None, n_class=19, resize_ratio=1.0):
        '''
        SemiDataset class represents a dataset for semi-supervised learning.

        Args:
            name (str): The name of the dataset.
            root (str): The root directory of the dataset.
            mode (str): The mode of the dataset, e.g., 'train_l', 'train_u', 'source','val'.
            size (tuple, optional): The desired size of the images. Defaults to None. h w 
            id_path (str, optional): The path to the file containing the image IDs. Defaults to None.
            nsample (int, optional): The number of samples to use. Defaults to None.
            ac_iters (object, optional): The object representing the active learning iterations. Defaults to None.
        '''
        self.name = name
        self.root = root
        self.mode = mode
        self.resize_ratio = resize_ratio
        self.size = size
        self.n_class = n_class
        self.ac_iters = ac_iters
        if 'gtav' in self.name:
            self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        elif 'syn' in self.name:
            # TODO
            self.id_to_trainid = {}
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle"
        }
        if self.n_class == 16:
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle"
            }
        if self.n_class == 17:
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "guard rail",
                4: "pole",
                5: "light",
                6: "sign",
                7: "tree",
                8: "terrain",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "truck",
                14: "bus",
                15: "motocycle",
                16: "bicycle",
            }
        if self.n_class == 8:
            self.trainid2name = {
                0: "background",
                1: "parking line",
                2: "white line",
                3: "yellow line",
                4: "zebra crossing",
                5: "arrow",
                6: "slope",
                7: "other",
            }
        if self.n_class == 19 and self.name == 'rellis3d':
             self.trainid2name = {
                0: "dirt",
                1: "grass",
                2: "tree",
                3: "pole",
                4: "water",
                5: "sky",
                6: "vehicle",
                7: "object",
                8: "asphalt",
                9: "building",
                10: "log",
                11: "person",
                12: "fence",
                13: "bush",
                14: "concrete",
                15: "barrier",
                16: "puddle",
                17: "mud",
                18: "rubble",
            }
        if self.n_class == 6:
            self.trainid2name = {
                0: "background",
                1: "white line",
                2: "yellow line",
                3: "zebra crossing",
                4: "arrow",
                5: "slope",
            }
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            # if n_class == 16 and self.name == 'cityscapes':
            #     self.ids = [id.replace('gtFine/train', 'gtFine/train_16') for id in self.ids]
            # train_l and train_u should have the same number of samples
            if nsample is not None and len(self.ids) < nsample:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
            # if mode == 'train_l' and nsample is not None:
            #     self.ids *= math.ceil(nsample / len(self.ids))
            #     self.ids = self.ids[:nsample]
        elif mode == 'source':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            #active learning
            if self.ac_iters is not None:
                total_samples = self.ac_iters.total_samples
            # domain adaptation
            else:
                total_samples = nsample
            repeats = total_samples // len(self.ids) + 1
            self.ids = self.ids * repeats
            # self.ids = (self.ids * repeats)[:total_samples]
        else:  # val
            if id_path is not None:
                with open(id_path, 'r') as f:
                    self.ids = f.read().splitlines()
            else:
                with open('splits/%s/val.txt' % name, 'r') as f:
                    self.ids = f.read().splitlines()
            # if n_class == 16 and self.name == 'cityscapes':
            #     self.ids = [id.replace('gtFine/val', 'gtFine/val_16') for id in self.ids]

    def __getitem__(self, item):
        '''
        Retrieves the item at the specified index.

        Args:
            item (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image, label, and ID.
        '''
        id = self.ids[item]
        img_path = os.path.join(self.root, id.split(' ')[0])            
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train_u':
            label = np.ones((img.size[1], img.size[0]), dtype=np.uint8) + 254  # all 255
            label = Image.fromarray(label)
        else:
            lb_path = os.path.join(self.root, id.split(' ')[1])
            label = Image.open(lb_path).convert('L')

        if self.mode == 'val':
            img, label = normalize(img, label)
            return img, label, id

        if self.resize_ratio != 1.0:
            img, label = resize_img_lb(img, label, self.resize_ratio)
        # random resize (0.5, 2.0)
        img, label = resize(img, label, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        # crop the image and label (768, 768)
        #if type of self.size if list or tple
        if isinstance(self.size, (list, tuple)):
            img, label = crop_h_w(img, label, self.size, ignore_value)
        else:
            img, label = crop(img, label, self.size, ignore_value)
        # random horizontal flip
        img, label = hflip(img, label, p=0.5)
        # print('img.size:', img.size)

        if self.mode in ['train_l', 'source']:
            return normalize(img, label)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)  # size [w, h]

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((label.size[1], label.size[0])))  # [h, w]

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        label = torch.from_numpy(np.array(label)).long()
        # the region for padding the cropped image
        ignore_mask[label == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        '''
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        '''
        return len(self.ids)
    
if __name__ =='__main__':
    yaml_path = '/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/configs/cityscapes_acda_bisenetv1.yaml'
    cfg = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)

    # Initialize model
    model = BiSeNetV1(cfg)  # Initialize your segmentation model
    pretrained_path = "/media/ywh/pool1/yanweihao/projects/active_learning/SS-ADA/exp/cityscapes/unimatch_acda_bisenetv1/bisenetv1/entropy/1488/latest.pth"
    state_dict = torch.load(pretrained_path, map_location='cpu')
    # remove the prefix 'module.' from the keys
    state_dict['model_state'] = {k[7:]: v for k, v in state_dict['model'].items() if k.startswith('module')}
    # print(state_dict.keys())
    model.load_state_dict(state_dict['model_state'], strict=True)
    model.cuda()
    model.eval()
    
    ac_iters = AC_iters(cfg)
    print('ac_iters.total_iters:', ac_iters.total_iters)
    print('ac_iters.total_samples:', ac_iters.total_samples)
    trainset_s = SemiDataset(cfg['source']['type'], cfg['source']['data_root'], 'source', cfg['crop_size'], cfg['source']['data_list'], ac_iters=ac_iters)
    trainloader_s = DataLoader(trainset_s, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
    trainloader_s_iter = iter(trainloader_s)
    print(len(trainset_s))
    print(len(trainloader_s))
    
    # 
    # criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    # for i in range(5):
    #     img_s, mask_s = trainloader_s_iter.next()
    #     img_s = img_s.cuda()
    #     mask_s = mask_s.cuda
    #     preds, preds_aux1, preds_aux2, preds_fp = model(img_s, True)
    #     loss_src = criterion_l(preds, mask_s)
