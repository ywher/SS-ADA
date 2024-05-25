#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from model.backbone.resnet18 import Resnet18

# from torch.nn import BatchNorm2d


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
    

class PresentationHead(nn.Module):
    
    def __init__(self, in_chan, out_chan, drop_rate=0.1, *args, **kwargs):
        super(PresentationHead, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv2 = ConvBNReLU(out_chan, out_chan, ks=3, stride=1, padding=1) 
        self.conv_out = nn.Conv2d(out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=True)
        self.drop = nn.Dropout2d(drop_rate)
        self.init_weight()
        
    def forward(self, x):
        rep = self.conv1(x)
        rep = self.drop(rep)
        rep = self.conv2(rep)
        rep = self.drop(rep)
        rep = self.conv_out(rep)
        return rep
    
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight()

    def forward(self, x, print_size=False):
        feat8, feat16, feat32 = self.resnet(x)
        if print_size:
            print("feat8 size: {} feat16 size: {} feat32 size: {}".format(feat8.size(), feat16.size(), feat32.size()))

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        if print_size:
            print("avg size: {}".format(avg.size()))

        feat32_arm = self.arm32(feat32)
        if print_size:
            print("feat32_arm size: {}".format(feat32_arm.size()))
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        if print_size:
            print("feat32_up size: {}".format(feat32_up.size()))
        feat32_up = self.conv_head32(feat32_up)
        if print_size:
            print("feat32_up size: {}".format(feat32_up.size()))

        feat16_arm = self.arm16(feat16)
        if print_size:
            print("feat16_arm size: {}".format(feat16_arm.size()))
        if not feat16_arm.size()[2:] == feat32_up.size()[2:]:
            feat16_arm = F.interpolate(feat16_arm, size=feat32_up.size()[2:], mode='bilinear', align_corners=False)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        if print_size:
            print("feat16_up size: {}".format(feat16_up.size()))
        feat16_up = self.conv_head16(feat16_up)
        if print_size:
            print("feat16_up size: {}".format(feat16_up.size()))

        return feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        ## use conv-bn instead of 2 layer mlp, so that tensorrt 7.2.3.4 can work for fp16
        self.conv = nn.Conv2d(out_chan,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        #  self.conv1 = nn.Conv2d(out_chan,
        #          out_chan//4,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.conv2 = nn.Conv2d(out_chan//4,
        #          out_chan,
        #          kernel_size = 1,
        #          stride = 1,
        #          padding = 0,
        #          bias = False)
        #  self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        #  atten = self.conv1(atten)
        #  atten = self.relu(atten)
        #  atten = self.conv2(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetV1(nn.Module):

    def __init__(self, cfg, aux_mode='train', *args, **kwargs):
        super(BiSeNetV1, self).__init__()
        self.logger = logging.getLogger("global")
        self.n_classes = cfg["nclass"]
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, self.n_classes, up_factor=8)
        self.aux_mode = aux_mode
        self.batch_size = cfg["batch_size"]
        self.aux_sup = False  # whether to use auxiliary supervision from source domain with different number of classes
        if self.aux_mode == 'train':
            self.conv_out16 = BiSeNetOutput(128, 64, self.n_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, self.n_classes, up_factor=16)
            
        
        # aux seg head
        if cfg["source"].get("nclass", False) and cfg["source"]["nclass"] != self.n_classes:
            self.aux_sup = True
            self.aux_classes = cfg["source"]["nclass"]
            self.aux_conv_out = BiSeNetOutput(256, 256, self.aux_classes, up_factor=8)
            if self.aux_mode == 'train':
                self.aux_conv_out16 = BiSeNetOutput(128, 64, self.aux_classes, up_factor=8)
                self.aux_conv_out32 = BiSeNetOutput(128, 64, self.aux_classes, up_factor=16)
        
        self.init_weight()

    def forward(self, x, need_fp=False, print_size=False, rank=0):
        H, W = x.size()[2:]
        feat_cp8, feat_cp16 = self.cp(x, print_size)
        feat_sp = self.sp(x)
        if need_fp:
            feat_fuse = self.ffm(torch.cat((feat_sp, nn.Dropout(0.5)(feat_sp))), torch.cat((feat_cp8, nn.Dropout2d(0.5)(feat_cp8))))
            # print("feat_fuse size: {}".format(feat_fuse.size()))
            if self.aux_sup:
                feat_src, feat_tar, feat_src_fp, feat_tar_fp = feat_fuse.split([self.batch_size, 2*self.batch_size, self.batch_size, 2*self.batch_size], dim=0)
                feat_src_out = self.aux_conv_out(torch.cat((feat_src, feat_src_fp), dim=0))
                feat_src_out, feat_src_out_fp = feat_src_out.chunk(2)
                feat_tar_out = self.conv_out(torch.cat((feat_tar, feat_tar_fp), dim=0))
                feat_tar_out, feat_tar_out_fp = feat_tar_out.chunk(2)
                if self.aux_mode == 'train':
                    feat_src_out16 = self.aux_conv_out16(feat_cp8[:self.batch_size])
                    feat_src_out32 = self.aux_conv_out32(feat_cp16[:self.batch_size])
                    feat_tar_out16 = self.conv_out16(feat_cp8[self.batch_size:])
                    feat_tar_out32 = self.conv_out32(feat_cp16[self.batch_size:])
                    return feat_src_out, feat_src_out16, feat_src_out32, feat_src_out_fp, feat_tar_out, feat_tar_out16, feat_tar_out32, feat_tar_out_fp
                elif self.aux_mode == 'eval':
                    return feat_src_out, feat_src_out_fp, feat_tar_out, feat_tar_out_fp
                elif self.aux_mode == 'pred':
                    return feat_src_out, feat_src_out_fp, feat_tar_out, feat_tar_out_fp
            else:
                feat_out = self.conv_out(feat_fuse)
                feat_out, feat_out_fp = feat_out.chunk(2)
                
                if self.aux_mode == 'train':
                    feat_out16 = self.conv_out16(feat_cp8)
                    feat_out32 = self.conv_out32(feat_cp16)
                    return feat_out, feat_out16, feat_out32, feat_out_fp
                elif self.aux_mode == 'eval':
                    return feat_out, feat_out_fp
                elif self.aux_mode == 'pred':
                    # feat_out = feat_out.argmax(dim=1)
                    # feat_out_fp = feat_out_fp.argmax(dim=1)
                    return feat_out, feat_out_fp
                else:
                    raise NotImplementedError
        else:
            if print_size:
                print("rank {}: feat_cp8 size: {} feat_sp.size: {}".format(rank, feat_cp8.size(), feat_sp.size()))
            if feat_cp8.size()[2:] != feat_sp.size()[2:]:
                # feat_sp has the right resolution for cropping, feat_cp8 is upsampled whose resolution is even
                feat_cp8 = F.interpolate(feat_cp8, size=feat_sp.size()[2:], mode='bilinear', align_corners=False)
            feat_fuse = self.ffm(feat_sp, feat_cp8)  # [2, 128, 96, 96]
            feat_out = self.conv_out(feat_fuse)

            if self.aux_mode == 'train':
                feat_out16 = self.conv_out16(feat_cp8)
                feat_out32 = self.conv_out32(feat_cp16)
                if feat_out32.size()[2:] != feat_out.size()[2:]:
                    feat_out32 = F.interpolate(feat_out32, size=feat_out.size()[2:], mode='bilinear', align_corners=False)
                return feat_out, feat_out16, feat_out32
            elif self.aux_mode == 'eval':
                return feat_out
            elif self.aux_mode == 'pred':
                # feat_out = feat_out.argmax(dim=1)
                return feat_out
            else:
                raise NotImplementedError

            
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def load_pretrained_model(self, pretrained_model, rm_layer_names=None):
        state_dict = torch.load(pretrained_model)['model']
        if rm_layer_names is None:
            self.load_state_dict(state_dict, strict=True)
        else:
            new_state_dict = {}
            for key in state_dict.keys():
                save_flag = True
                for rm_layer_name in rm_layer_names:
                    if rm_layer_name in key:
                        save_flag = False
                        break
                if save_flag:
                    new_state_dict[key.replace('module.', '')] = state_dict[key]
            # for k, _ in new_state_dict.items():
            #     self.logger.info(k)
            # self.logger.info("==========")
            self.load_state_dict(new_state_dict, strict=False)

if __name__ == "__main__":
    net = BiSeNetV1(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(16, 3, 640, 480).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

    net.get_params()
