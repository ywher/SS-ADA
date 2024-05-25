import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import copy
import datetime

from dataset.semi import SemiDataset
from model.semseg.bisenetv1 import BiSeNetV1
from supervised import evaluate, evaluate_single_gpu
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.optimizer import set_optimizer_bisenet
from util.scheduler import get_scheduler
from util.get_acda_iters import AC_iters
from util.active_sample import AC_Sample
from util.functions import generate_confidence_mask

# cuda_id = 1
# torch.cuda.set_device(cuda_id)

# the work root
work_root=os.path.abspath(os.path.join(os.path.dirname(__file__)))

def get_parse():
    parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default=os.path.join(work_root, 'configs/parking_bev2024_acda_bisenetv1_single.yaml'))
    parser.add_argument('--labeled-id-path', type=str, default=os.path.join(work_root, 'splits/bev_2024/1/labeled.txt'))
    parser.add_argument('--unlabeled-id-path', type=str, default=os.path.join(work_root, 'splits/bev_2024/1/unlabeled.txt'))
    parser.add_argument('--save-path', type=str, default=os.path.join(work_root, 'exp/bev_2024/unimatch_acda_bisenetv1_single/bisenetv1/28_entropy_200epoch'))
    return parser.parse_args()

def main():
    # get args and config
    args = get_parse()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg["n_gpus"] = 1
    
    os.makedirs(args.save_path, exist_ok=True)

    # initialize the logger
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    logger = init_log('global', logging.INFO, log_file=os.path.join(args.save_path, '${}.log'.format(now)))
    logger.propagate = 0

    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
        
    # tensorbaord
    writer = SummaryWriter(args.save_path)
    # save the config file to the output directory
    with open(os.path.join(args.save_path, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    ### set model
    model = BiSeNetV1(cfg)
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    model.cuda()
    
    ### set optimizer
    optimizer = set_optimizer_bisenet(model, cfg["optim"])

    ### set criterion
    # labeled loss
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    # unlabeled loss
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    ### set active_iters
    ac_iters = AC_iters(cfg)
    
    ### set dataset
    # source dataset
    trainset_s = SemiDataset(cfg['source']['type'], cfg['source']['data_root'], 'source', cfg['crop_size'], cfg['source']['data_list'], ac_iters=ac_iters, n_class=cfg['nclass'])
    # unlabeled target dataset and labeled target dataset, len(train_u) is more than len(train_l) at the begining
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path, n_class=cfg['nclass'])
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids), n_class=cfg['nclass'])
    # val dataset
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', n_class=cfg['nclass'])

    trainloader_s = DataLoader(trainset_s, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
    trainloader_s_iter = iter(trainloader_s)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    logger.info("len(trainset_src_sup) {}".format(len(trainset_s)))
    logger.info("len(trainloader_src_sup) {}".format(len(trainloader_s)))
    logger.info("len(trainset_tar_sup) {}".format(len(trainset_l)))
    logger.info("len(trainloader_tar_sup) {}".format(len(trainloader_l)))
    logger.info("len(trainset_tar_unsup) {}".format(len(trainset_u)))
    logger.info("len(trainloader_tar_unsup) {}".format(len(trainloader_u)))
    logger.info("num of n_sup {}".format(cfg["n_sup"]))
    logger.info("the path of data list {}".format(cfg["data_list"]))

    total_iters = ac_iters.total_iters
    previous_best = 0.0
    epoch = -1
    
    ### resume training
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    ### set lr scheduler
    optimizer_start = set_optimizer_bisenet(model, cfg["optim"])
    lr_scheduler = get_scheduler(cfg, len(trainloader_u), optimizer_start, start_epoch=epoch + 1, ac_iters=ac_iters)
    
    ### set AC_Sampler
    AC_Sampler = AC_Sample(config=cfg, ac_iters=ac_iters, output_root=os.path.join(args.save_path, 'acda_log'))
    n_unsup = len(AC_Sampler.unlabeled_names)
    
    ### 
    global_start_time = time.time()
    total_iters = ac_iters.total_iters
    start_iters = ac_iters.get_cur_epoch_init_iters(epoch + 1)
    iters = copy.deepcopy(start_iters)
    for epoch in range(epoch + 1, cfg['epochs']):
        ### triggle the active sampling
        if epoch in cfg["active"]["sample_epoch"]:
            ### run active samlping
            logger.info("Start active sampling in epoch {}".format(epoch))
            model.eval()
            AC_Sampler.run_active_learning(model)
            # calculate the class weight based on target labeled gt, defalut not used
            if cfg["criterion"]["fre_cfg"]["update_fre"]:
                class_weight = AC_Sampler.generate_class_weights(rank=0, fre_cfg=cfg['criterion']['fre_cfg'], model=model).cuda()
            model.train()
            
            ### update dateset info
            cfg["data_list"] = os.path.join(AC_Sampler.output_dir, 'labeled.txt')
            cfg["n_sup"] = len(AC_Sampler.labeled_names)
            n_unsup = len(AC_Sampler.unlabeled_names)
            n_sample = max(n_unsup, cfg["n_sup"])
            # update the labeled dataset
            del trainset_l
            del trainloader_l
            trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], cfg["data_list"], nsample=n_sample, n_class=cfg['nclass'])
            trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
            logger.info("iter for epoch {}".format(n_sample//cfg['batch_size']))
            logger.info("len(train_set_tar_sup) {}".format(len(trainset_l)))
            logger.info("len(train_loader_tar_sup) {}".format(len(trainloader_l)))
            
            ### update criteria if update the class weight
            if cfg["criterion"]["fre_cfg"]["update_fre"]:
                logger.info("class weight {}".format(class_weight))
                if cfg['criterion']['name'] == 'CELoss':
                    criterion_l = nn.CrossEntropyLoss(weight=class_weight, **cfg['criterion']['kwargs']).cuda()
                elif cfg['criterion']['name'] == 'OHEM':
                    criterion_l = ProbOhemCrossEntropy2d(use_weight=True, weight=class_weight, **cfg['criterion']['kwargs']).cuda()
                else:
                    raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
            
            ### update the unlabeld dataset
            if n_unsup > 0:
                del trainset_u
                del trainloader_u
                trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], cfg["data_list"].replace('labeled', 'unlabeled'), nsample=n_sample, n_class=cfg['nclass'])
                trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True)
                logger.info("len(train_set_tar_unsup) {}".format(len(trainset_u)))
                logger.info("len(train_loader_tar_unsup) {}".format(len(trainloader_u)))
        
        
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_losses = AverageMeter()
        total_losses_src = AverageMeter()
        total_losses_tar = AverageMeter()  # target labeled loss
        total_losses_tar_s = AverageMeter()  # target unlabeled loss, for two strong perturbations
        total_losses_tar_w_fp = AverageMeter()  # target unlabeled loss, for weak feature perturbation
        total_mask_ratio = AverageMeter()  # mask ratio for pseudo label
        total_mask_u_s1_ratio = AverageMeter()
        total_mask_u_s2_ratio = AverageMeter()
        total_mask_u_fp_ratio = AverageMeter()
        
        if n_unsup > 0:
            loader = zip(trainloader_l, trainloader_u, trainloader_u)

            for i, ((img_x, mask_x),
                    (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                    (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
                img_s, mask_s = trainloader_s_iter.next()
                img_s, mask_s = img_s.cuda(), mask_s.cuda()
                img_x, mask_x = img_x.cuda(), mask_x.cuda()
                img_u_w = img_u_w.cuda()
                img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
                cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
                img_u_w_mix = img_u_w_mix.cuda()
                img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
                ignore_mask_mix = ignore_mask_mix.cuda()

                with torch.no_grad():
                    model.eval()

                    pred_u_w_mix = model(img_u_w_mix)[0].detach()
                    conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                    mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

                img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                    img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
                img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                    img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

                model.train()

                num_sr, num_lb, num_ulb = img_s.shape[0], img_x.shape[0], img_u_w.shape[0]

                preds, preds_aux_1, preds_aux_2, preds_fp = model(torch.cat((img_s, img_x, img_u_w)), True)
                pred_s, pred_x, pred_u_w = preds.split([num_sr, num_lb, num_ulb])
                pred_s_aux_1, pred_x_aux_1, _ = preds_aux_1.split([num_sr, num_lb, num_ulb])
                pred_s_aux_2, pred_x_aux_2, _ = preds_aux_2.split([num_sr, num_lb, num_ulb])
                _, _, pred_u_w_fp = preds_fp.split([num_sr, num_lb, num_ulb])
                # pred_u_w_fp = preds_fp[num_sr + num_lb:]

                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)))[0].chunk(2)

                pred_u_w = pred_u_w.detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]  # [B,H,W]
                mask_u_w = pred_u_w.argmax(dim=1)  # [B,H,W]

                mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                    mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

                mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
                conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
                ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

                mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
                conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
                ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
                
                # source loss
                loss_src = criterion_l(pred_s, mask_s)
                loss_src_aux1 = criterion_l(pred_s_aux_1, mask_s)
                loss_src_aux2 = criterion_l(pred_s_aux_2, mask_s)
                
                # target labeled loss
                loss_x = criterion_l(pred_x, mask_x)
                loss_x_aux1 = criterion_l(pred_x_aux_1, mask_x)
                loss_x_aux2 = criterion_l(pred_x_aux_2, mask_x)

                # target unlabeled loss for two stong perturbations
                if cfg.get('mask_ratio', False):
                    threshold = cfg['mask_ratio']['threshold']
                    if cfg['mask_ratio']['dynamic']:
                        threshold_ratio = (1.0 - iters / total_iters) * threshold
                    else:
                        threshold_ratio = threshold
                loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
                if cfg.get('mask_ratio', False) and cfg['mask_ratio']['use_class_threshold']:
                    mask_u_s1 = generate_confidence_mask(mask_u_w_cutmixed1, conf_u_w_cutmixed1, AC_Sampler.iou_classes, threshold_ratio=threshold_ratio)
                else:
                    mask_u_s1 = (conf_u_w_cutmixed1 >= cfg['conf_thresh'])
                loss_u_s1 = loss_u_s1 * (mask_u_s1 & (ignore_mask_cutmixed1 != 255))
                loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

                loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
                if cfg.get('mask_ratio', False) and cfg['mask_ratio']['use_class_threshold']:
                    mask_u_s2 = generate_confidence_mask(mask_u_w_cutmixed2, conf_u_w_cutmixed2, AC_Sampler.iou_classes, threshold_ratio=threshold_ratio)
                else:
                    mask_u_s2 = (conf_u_w_cutmixed2 >= cfg['conf_thresh'])
                loss_u_s2 = loss_u_s2 * (mask_u_s2 & (ignore_mask_cutmixed2 != 255))
                loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

                # target unlabeled loss for weak feature perturbation
                loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
                if cfg.get('mask_ratio', False) and cfg['mask_ratio']['use_class_threshold']:
                    mask_u_w_fp = generate_confidence_mask(mask_u_w, conf_u_w, AC_Sampler.iou_classes, threshold_ratio=threshold_ratio)
                else:
                    mask_u_w_fp = (conf_u_w >= cfg['conf_thresh'])
                loss_u_w_fp = loss_u_w_fp * (mask_u_w_fp & (ignore_mask != 255))
                loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

                # cal loss
                total_loss_src = (loss_src + loss_src_aux1 + loss_src_aux2) / 3.0
                total_loss_tlb = (loss_x + loss_x_aux1 + loss_x_aux2) / 3.0
                loss = (total_loss_src + total_loss_tlb + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 3.0
                
                optimizer.zero_grad()
                loss.backward()
            
                optimizer.step()
                lr_scheduler.step()
                
                iters += 1

                total_losses.update(loss.item())
                total_losses_src.update(total_loss_src.item())
                total_losses_tar.update(total_loss_tlb.item())
                total_losses_tar_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
                total_losses_tar_w_fp.update(loss_u_w_fp.item())
                
                # mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
                # total_mask_ratio.update(mask_ratio.item())
                
                mask_u_s1_ratio = (mask_u_s1 & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
                mask_u_s2_ratio = (mask_u_s2 & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
                mask_u_fp_ratio = (mask_u_w_fp & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
                total_mask_u_s1_ratio.update(mask_u_s1_ratio.item())
                total_mask_u_s2_ratio.update(mask_u_s2_ratio.item())
                total_mask_u_fp_ratio.update(mask_u_fp_ratio.item())
                
            
                writer.add_scalar('train/loss_all',     loss.item(), iters)
                writer.add_scalar('train/loss_src_sup', total_loss_src.item(), iters)
                writer.add_scalar('train/loss_src',     loss_src.item(), iters)
                writer.add_scalar('train/loss_src_aux1',    loss_src_aux1.item(), iters)
                writer.add_scalar('train/loss_src_aux2',    loss_src_aux2.item(), iters)
                writer.add_scalar('train/loss_tar_sup', total_loss_tlb.item(), iters)
                writer.add_scalar('train/loss_x',       loss_x.item(), iters)
                writer.add_scalar('train/loss_x_aux1',  loss_x_aux1.item(), iters)
                writer.add_scalar('train/loss_x_aux2',  loss_x_aux2.item(), iters)
                writer.add_scalar('train/loss_s',       (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp',    loss_u_w_fp.item(), iters)
                # mask ratio
                # writer.add_scalar('train/mask_ratio',   mask_ratio, iters)
                writer.add_scalar('train/mask_u_s1_ratio',   mask_u_s1_ratio, iters)
                writer.add_scalar('train/mask_u_s2_ratio',   mask_u_s2_ratio, iters)
                writer.add_scalar('train/mask_u_fp_ratio',   mask_u_fp_ratio, iters)
                
                if (i % (max(len(trainloader_u), len(trainloader_l)) // 8) == 0):
                    cur_time = time.time()
                    eta = (cur_time - global_start_time) / (iters - start_iters + 1) * (total_iters - iters - 1)
                    eta = str(datetime.timedelta(seconds=int(eta)))
                    cur_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3) # from bytes to GB
                    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) # from bytes to GB
                    logger.info('Iters: {:}, {}/{}\t Eta: {}, Total loss: {:.3f}, Loss src: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask s1 ratio: {:.3f}, Mask s2 ratio: {:.3f}, Mask fp ratio: {:.3f}, Cur Mem: {:.2f}, Max Mem: {:.2f}'.format(i, iters, total_iters, eta, total_losses.avg, total_losses_src.avg, total_losses_tar.avg, total_losses_tar_s.avg, total_losses_tar_w_fp.avg, total_mask_u_s1_ratio.avg, total_mask_u_s2_ratio.avg, total_mask_u_fp_ratio.avg, cur_memory_allocated, max_memory_allocated))
                    # logger.info('Iters: {:}, {}/{}\t Eta: {}, Total loss: {:.3f}, Loss src: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}, Cur Mem: {:.2f}, Max Mem: {:.2f}'.format(i, iters, total_iters, eta, total_losses.avg, total_losses_src.avg, total_losses_tar.avg, total_losses_tar_s.avg, total_losses_tar_w_fp.avg, total_mask_ratio.avg, cur_memory_allocated, max_memory_allocated))
        else:
            trainloader_l.sampler.set_epoch(epoch)

            for i, (img_x, mask_x) in enumerate(trainloader_l):
                iters += 1
                img_s, mask_s = trainloader_s_iter.next()
                img_s, mask_s = img_s.cuda(), mask_s.cuda()
                img_x, mask_x = img_x.cuda(), mask_x.cuda()

                num_sr, num_lb = img_s.shape[0], img_x.shape[0]

                preds, preds_aux_1, preds_aux_2, preds_fp = model(torch.cat((img_s, img_x)), True)
                pred_s, pred_x = preds.split([num_sr, num_lb])
                
                pred_s_aux_1, pred_x_aux_1 = preds_aux_1.split([num_sr, num_lb])
                pred_s_aux_2, pred_x_aux_2 = preds_aux_2.split([num_sr, num_lb])

                loss_src = criterion_l(pred_s, mask_s)
                loss_src_aux1 = criterion_l(pred_s_aux_1, mask_s)
                loss_src_aux2 = criterion_l(pred_s_aux_2, mask_s)

                loss_x = criterion_l(pred_x, mask_x)
                loss_x_aux1 = criterion_l(pred_x_aux_1, mask_x)
                loss_x_aux2 = criterion_l(pred_x_aux_2, mask_x)

                # cal loss
                total_loss_src = (loss_src + loss_src_aux1 + loss_src_aux2) / 3.0
                total_loss_tlb = (loss_x + loss_x_aux1 + loss_x_aux2) / 3.0
                loss = (total_loss_src + total_loss_tlb) / 2.0
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()

                total_losses.update(loss.item())
                total_losses_src.update(total_loss_src.item())
                total_losses_tar.update(total_loss_tlb.item())
            
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_src_sup', total_loss_src.item(), iters)
                writer.add_scalar('train/loss_src', loss_src.item(), iters)
                writer.add_scalar('train/loss_src_aux1', loss_src_aux1.item(), iters)
                writer.add_scalar('train/loss_src_aux2', loss_src_aux2.item(), iters)
                writer.add_scalar('train/loss_tar_sup', total_loss_tlb.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_x_aux1', loss_x_aux1.item(), iters)
                writer.add_scalar('train/loss_x_aux2', loss_x_aux2.item(), iters)
                
            
                if (i % (len(trainloader_l) // 8) == 0):
                    cur_time = time.time()
                    eta = (cur_time - global_start_time) / (iters - start_iters + 1) * (total_iters - iters - 1)
                    eta = str(datetime.timedelta(seconds=int(eta)))
                    cur_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3) # from bytes to GB
                    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) # from bytes to GB
                    logger.info('Iters: {:}, {}/{}\t Eta: {}, Total loss: {:.3f}, Loss src: {:.3f}, Loss x: {:.3f}, Cur Mem: {:.2f}, Max Mem: {:.2f}'.format(i, iters, total_iters, eta, total_losses.avg, total_losses_src.avg, total_losses_tar.avg, cur_memory_allocated, max_memory_allocated))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate_single_gpu(model, valloader, eval_mode, cfg)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
        
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()