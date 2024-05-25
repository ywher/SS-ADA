import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import yaml
from model.semseg.bisenetv1 import BiSeNetV1



def show_model_params(model_path):
    state_dict = torch.load(model_path)['model']
    for key in state_dict.keys():
        print(key)
        # print(state_dict[key].shape)
        # print(state_dict[key].dtype)
        # print(state_dict[key].requires_grad)
        # print("=====================================")



if __name__ == '__main__':
    model_path = "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/exp/avm_seg/supervised_bisenetv1_tar/bisenetv1/6000_200epoch/best.pth"
    config_apth = "/media/ywh/pool1/yanweihao/projects/active_learning/UniMatch/configs/parking_bev20234_6cls_bisenetv1.yaml"
    cfg = yaml.load(open(config_apth, "r"), Loader=yaml.Loader)
    model = BiSeNetV1(cfg=cfg)
    show_model_params(model_path)
    print("=====================================")
    
    for k,v in model.state_dict().items():
        print(k)
        # print(v.shape)
        # print(v.dtype)
        # print(v.requires_grad)
    print("=====================================")
    model.load_pretrained_model(model_path)
    