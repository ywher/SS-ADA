# from supervised import evaluate_single_gpu
from supervised_bisenetv1_tar import evaluate_single_gpu
import torch
import argparse
import datetime
import numpy as np
import yaml
import os
import csv
from torch.utils.data import DataLoader
from dataset.semi import SemiDataset
from model.semseg.bisenetv1 import BiSeNetV1


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cityscapes_bisenetv1.yaml', help='the path to config file')
    parser.add_argument('--exp_root', type=str, default='exp/cityscapes/unimatch_bisenetv1/bisenetv1/744', help='the path to the exp folder')
    parser.add_argument('--eval_mode', type=str, default='sliding_window', choices=['original', 'center_crop', 'sliding_window'], help='the evaluation mode')
    parser.add_argument('--resize_ratio', type=float, default=1.0, help='the ratio of resize the image')
    parser.add_argument('--model_type', type=str, default='last', choices=['latest', 'best'], help='the type of model for evaluation, last or best')
    parser.add_argument('--model_path', type=str,
                        default='exp/cityscapes/unimatch_bisenetv1/bisenetv1/744/latest.pth', help='the path to model checkpoint')
    parser.add_argument('--save_pred', action='store_true', default=False, help='whether to store the prediction result')
    parser.add_argument('--save_color', action='store_true', default=False, help='whether to store the color prediction result')
    parser.add_argument('--show_bar', action='store_true', default=False, help='whether to show the progress bar')
    
    return parser.parse_args()

def load_pretrained(model, model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    new_state_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            new_state_dict[key.replace("module.", "")] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict, strict=True)
    return model

def save_mious_2_csv(iou_class, save_path):
    '''
    ### save the iou information in csv file
    '''
    mean_iou = np.mean(iou_class)
    with open(save_path, "w") as f:
        writer = csv.writer(f)
        first_row = ["mean_iou"] + [str(i) for i in range(len(iou_class))]
        second_row = ["{:.3f}".format(mean_iou)] + ["{:.3f}".format(iou) for iou in iou_class]  # 保留三位小数
        writer.writerow(first_row)
        writer.writerow(second_row)
    f.close()
    
def main():
    args = get_parse()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model = BiSeNetV1(cfg)
    model = load_pretrained(model, args.model_path)
    model.cuda()
    if cfg.get('val', False):
        if cfg.get('val').get('data_list', False):
            data_list = cfg['val']['data_list']
            val_dataset = SemiDataset(cfg['val']['dataset'], cfg['val']['data_root'], 'val', id_path=data_list, n_class=cfg['nclass'], resize_ratio=args.resize_ratio)
        else:
            val_dataset = SemiDataset(cfg['val']['dataset'], cfg['val']['data_root'], 'val', n_class=cfg['nclass'], resize_ratio=args.resize_ratio)
    else:
        val_dataset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', n_class=cfg['nclass'], resize_ratio=args.resize_ratio)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
    _, iou_class = evaluate_single_gpu(cfg['dataset'], model, args.model_type, val_loader, args.eval_mode, cfg, args.save_pred, args.save_color, args.exp_root, args.show_bar)
    save_path = os.path.join(args.exp_root, '{}_iou.csv'.format(args.model_type))
    save_mious_2_csv(iou_class, save_path)


if __name__ == '__main__':
    main()
