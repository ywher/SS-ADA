# from supervised import evaluate_single_gpu
import torch
import argparse
import numpy as np
import yaml
import os
import time
from tqdm import tqdm
from torchvision import transforms
from util.utils import AverageMeter
from util.color_map import color_map as color_maps
from util.color_prediction import colorize_prediction
from util.functions import save_entropy_results, save_confidence_results
from model.semseg.bisenetv1 import BiSeNetV1
from PIL import Image


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/cityscapes_bisenetv1.yaml', help='the path to config file')
    parser.add_argument('--img_fodler', type=str, default='data/cityscapes/leftImg8bit/val', help='the path to the image folder')
    parser.add_argument('--save_folder', type=str, default='data/cityscapes/leftImg8bit/val_pred', help='the path to the save folder')
    parser.add_argument('--exp_root', type=str, default='exp/cityscapes/unimatch_bisenetv1/bisenetv1/744', help='the path to the exp folder')
    parser.add_argument('--model_type', type=str, default='last', choices=['latest', 'best'], help='the type of model for evaluation, last or best')
    parser.add_argument('--model_path', type=str, default='exp/cityscapes/unimatch_bisenetv1/bisenetv1/744/latest.pth', help='the path to model checkpoint')
    parser.add_argument('--resize_ratio', type=float, default=1.0, help='the ratio of resize the image')
    parser.add_argument('--batch_size', type=int, default=1, help='the batch size of the inference')
    parser.add_argument('--need_fp', action='store_true', default=False, help='whether to use feature perturbation pred')
    parser.add_argument('--save_entropy', action='store_true', default=False, help='whether to save the entropy result')
    # parser.add_argument('--save_pred', action='store_true', default=False, help='whether to store the prediction result')
    # parser.add_argument('--save_color', action='store_true', default=False, help='whether to store the color prediction result')
    # parser.add_argument('--show_bar', action='store_true', default=False, help='whether to show the progress bar')
    
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
    model.load_state_dict(new_state_dict, strict=False)
    return model

def inference(model, iamge_folder, save_folder, need_fp, cfg, args):
    resize_time = AverageMeter()
    trans_time = AverageMeter()
    infer_time = AverageMeter()
    g2cpu_time = AverageMeter()
    # img_transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    color_map = color_maps[cfg['dataset']]
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    os.makedirs(os.path.join(save_folder, 'pred_labelid'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'pred_color'), exist_ok=True)
    if args.save_entropy:
        os.makedirs(os.path.join(save_folder, 'pred_confidence'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'pred_entropy'), exist_ok=True)
    # os.makedirs
    if need_fp:
        os.makedirs(os.path.join(save_folder, 'pred_fp_labelid'), exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'pred_fp_color'), exist_ok=True)
    
    # load the image list
    img_list = os.listdir(iamge_folder)
    img_list.sort()
    bar = tqdm(total=len(img_list))
    for img_name in img_list:
        img_path = os.path.join(iamge_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        if args.resize_ratio != 1.0:
            resize_start = time.time()
            img = img.resize((int(img.width * args.resize_ratio), int(img.height * args.resize_ratio)))
            resize_time.update(time.time() - resize_start)
        trans_start = time.time()
        img_data = transform(img).unsqueeze(0).cuda()
        if args.batch_size > 1:
            img_data = img_data.repeat(args.batch_size, 1, 1, 1)  # repeat the image data to the batch size
        trans_time.update(time.time() - trans_start)
        
        infer_start = time.time()
        with torch.no_grad():
            if need_fp:
                pred, pred_fp = model(img_data, need_fp=need_fp)
                if pred_fp.shape[0] == 1:
                    pred_fp = pred_fp[0]
            else:
                pred = model(img_data)
            # print(pred.shape)
            if pred.shape[0] == 1:  # [1, C, H, W] -> [C, H, W]
                pred = pred[0]
        if args.batch_size > 1:
            pred = torch.softmax(pred, dim=1)
            pred_confidence, pred_id = torch.max(pred, dim=1)
        else:
            pred = torch.softmax(pred, dim=0)
            pred_confidence, pred_id = torch.max(pred, dim=0)
        infer_time.update(time.time() - infer_start)
        
        if args.batch_size > 1:
            pred_id = pred_id[0]
            pred_confidence = pred_confidence[0]
            pred = pred[0]
        
        g2c_start = time.time()
        pred_img = pred_id.cpu().numpy().astype(np.uint8)
        g2cpu_time.update(time.time() - g2c_start)
        pred_img = Image.fromarray(pred_img)
        
        pred_img.save(os.path.join(save_folder, 'pred_labelid', img_name))
        color_pred_img = colorize_prediction(pred_img, color_map)
        color_pred_img.save(os.path.join(save_folder, 'pred_color', img_name))
        
        if args.save_entropy:
            save_confidence_results({img_name: pred_confidence}, os.path.join(save_folder, 'pred_confidence'))
            pred_entropy = torch.sum(-pred * torch.log2(pred + 1e-10), dim=0) / np.log2(pred.shape[0])
            save_entropy_results({img_name: pred_entropy}, os.path.join(save_folder, 'pred_entropy'))
        if need_fp:
            pred_fp = torch.argmax(pred_fp, dim=0)
            pred_fp_img = Image.fromarray(pred_fp.cpu().numpy().astype(np.uint8))
            pred_fp_img.save(os.path.join(save_folder, 'pred_fp_labelid', img_name))
            color_pred_fp_img = colorize_prediction(pred_fp_img, color_map)
            color_pred_fp_img.save(os.path.join(save_folder, 'pred_fp_color', img_name))
        print('save the prediction result of {}'.format(img_name))
        if args.resize_ratio != 1.0:
            print('Total_time: {:.4f}, Avg resize_time: {:.4f}, Avg trans_time: {:.4f}, Avg infer_time: {:.4f}, Avg g2cpu_time: {:.4f}'.format(resize_time.avg+trans_time.avg+infer_time.avg+g2cpu_time.avg, resize_time.avg, trans_time.avg, infer_time.avg, g2cpu_time.avg))
        else:
            print('Total_time: {:.4f}, Avg trans_time: {:.4f}, Avg infer_time: {:.4f}, Avg g2cpu_time: {:.4f}'.format(trans_time.avg+infer_time.avg+g2cpu_time.avg, trans_time.avg, infer_time.avg, g2cpu_time.avg))
        bar.update(1)
    bar.close()
    
    
def main():
    args = get_parse()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model = BiSeNetV1(cfg, aux_mode='pred')
    model = load_pretrained(model, args.model_path)
    model.cuda()
    model.eval()
    inference(model, args.img_fodler, args.save_folder, args.need_fp, cfg, args)


if __name__ == '__main__':
    main()
