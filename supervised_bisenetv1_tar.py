import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
from PIL import Image
import datetime
from tqdm import tqdm


from dataset.semi import SemiDataset
from model.semseg.bisenetv1 import BiSeNetV1
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from util.optimizer import set_optimizer_bisenet
from util.scheduler import get_scheduler
from util.color_map import color_map
from util.color_prediction import colorize_prediction

def get_args():
    parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--labeled-id-path', type=str, required=False)
    parser.add_argument('--unlabeled-id-path', type=str, default=None)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    return parser.parse_args()


def evaluate(model, loader, mode, cfg, rank):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    

    # bar = tqdm(total=len(loader))
    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                # print('grid:', grid)
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
                row_s = 0
                while row_s < h:
                    col_s = 0
                    while col_s < w:
                        # pring the region of the image
                        # print('rank {}, row: min(h, row + grid); col: min(w, col + grid)'.format(rank), row, min(h, row + grid), col, min(w, col + grid))
                        row_e, col_e = min(h, row_s + grid), min(w, col_s + grid)
                        if row_e == h:
                            row_s = h - grid
                        if col_e == w:
                            col_s = w - grid
                        pred = model(img[:, :, row_s: row_e, col_s: col_e])
                        if len(pred) > 1:
                            pred = pred[0]
                        # print('rannk {}, pred shape {}'.format(rank, pred.size()))
                        final[:, :, row_s: row_e, col_s: col_e] += pred.softmax(dim=1)
                        col_s += grid
                        # col += int(grid * 2 / 3)
                    row_s += grid
                    # row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img)
                if len(pred) > 1:
                    pred = pred[0]
                pred = pred.argmax(dim=1)
            if pred.size() != mask.size():
                pred = nn.functional.interpolate(pred.unsqueeze(1).float(), size=mask.size()[1:], mode='nearest').squeeze(1).long()
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
    #         bar.update(1)
    # bar.close()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    model.train()
    return mIOU, iou_class


def evaluate_single_gpu(dataset_name, model, model_type, loader, mode, cfg, save_pred=False, save_color=False, exp_root='', show_bar=False):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    
    if save_pred:
        os.makedirs(os.path.join(exp_root, model_type, 'pred_trainid'), exist_ok=True)
    if save_color:
        os.makedirs(os.path.join(exp_root, model_type, 'pred_color'), exist_ok=True)

    if show_bar:
        bar = tqdm(total=len(loader))
    with torch.no_grad():
        for img, mask, id in loader:
            
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b,cfg['nclass'], h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])  # , print_size=True
                        if len(pred) > 1:
                            pred = pred[0]
                        pred_logits = pred.softmax(dim=1)
                        crop_h, crop_w = min(h, row + grid) - row, min(w, col + grid) - col
                        if pred_logits.size()[2] != crop_h or pred_logits.size()[3] != crop_w:
                            pred_logits = nn.functional.interpolate(pred_logits, (crop_h, crop_w), mode='bilinear', align_corners=True)
                        # print('row: min(h, row + grid); col: min(w, col + grid), pred.size', row, min(h, row + grid), col, min(w, col + grid), pred.size())
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred_logits
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img)
                if len(pred) > 1:
                    pred = pred[0]
                pred = pred.argmax(dim=1)
            
            if pred.size() != mask.size():
                pred = nn.functional.interpolate(pred.unsqueeze(1).float(), size=mask.size()[1:], mode='nearest').squeeze(1).long()  
            if save_pred:
                # save the prediction
                pred = pred.cpu().numpy()
                for i in range(len(id)):
                    pred_i = pred[i]
                    pred_i = np.uint8(pred_i)
                    pred_i = Image.fromarray(pred_i)
                    img_name = id[i].split(' ')[0].split('/')[-1]
                    pred_i.save(os.path.join(exp_root, model_type, 'pred_trainid', img_name))
                    if save_color:
                        color_pred_i = colorize_prediction(pred_i, color_map[dataset_name])
                        color_pred_i.save(os.path.join(exp_root, model_type, 'pred_color', img_name))
                        
                intersection, union, target = \
                intersectionAndUnion(pred, mask.numpy(), cfg['nclass'], 255)
            else:
                intersection, union, target = \
                    intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            # dist.all_reduce(reduced_intersection)
            # dist.all_reduce(reduced_union)
            # dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            if show_bar:
                bar.update(1)
    if show_bar:
        bar.close()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    model.train()

    return mIOU, iou_class


def main():
    args = get_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    local_rank = int(os.environ["LOCAL_RANK"])
    model = BiSeNetV1(cfg)
    if cfg.get('pretrain', False):
        # for k, v in model.state_dict().items():
        #     logger.info(k)
        # logger.info('=======')
        model.load_pretrained_model(cfg['pretrain'], rm_layer_names=cfg.get('rm_layer_names', None))
        logger.info('Pretrained model has been loaded')
        
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    
    optimizer = set_optimizer_bisenet(model, cfg["optim"])
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, n_class=cfg['nclass'])
    if cfg.get('val', False):
        valset = SemiDataset(cfg['val']['dataset'], cfg['val']['data_root'], 'val', n_class=cfg['nclass'])
    else:
        valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', n_class=cfg['nclass'])
    logger.info('Trainset: %d, Valset: %d' % (len(trainset), len(valset)))

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)
    logger.info('Trainloader: %d, Valloader: %d' % (len(trainloader), len(valloader)))

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    # resume training
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    # set lr scheduler
    # optimizer_start = set_optimizer_bisenet(model, cfg["optim"])
    lr_scheduler = get_scheduler(cfg, len(trainloader), optimizer, start_epoch=epoch + 1)
    
    global_start_time = time.time()
    start_iters = (epoch + 1) * len(trainloader)
    iters = start_iters
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        sup_total_loss = AverageMeter()
        sup_loss = AverageMeter()
        sup_loss_aux1 = AverageMeter()
        sup_loss_aux2 = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred, pred_aux1, pred_aux2 = model(img)

            # print('pred pred_aux1, pred_aux2 and mask shape', pred.size(), pred_aux1.size(), pred_aux2.size(), mask.size())
            loss = criterion(pred, mask)
            loss_aux1 = criterion(pred_aux1, mask)
            loss_aux2 = criterion(pred_aux2, mask)
            
            total_loss = (loss + loss_aux1 + loss_aux2) / 3.0
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            sup_total_loss.update(total_loss.item())
            sup_loss.update(loss.item())
            sup_loss_aux1.update(loss_aux1.item())
            sup_loss_aux2.update(loss_aux2.item())

            iters += 1

            if rank == 0:
                writer.add_scalar('train/loss_all', total_loss.item(), iters)
                writer.add_scalar('train/loss_main', loss.item(), iters)
                writer.add_scalar('train/loss_aux1', loss_aux1.item(), iters)
                writer.add_scalar('train/loss_aux2', loss_aux2.item(), iters)
            
            if (i % (max(min(200, len(trainloader) // 10), 1)) == 0) and (rank == 0):
                cur_time = time.time()
                eta = (cur_time - global_start_time) / (iters - start_iters + 1) * (total_iters - iters - 1)
                eta = str(datetime.timedelta(seconds=int(eta)))
                cur_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3) # from bytes to GB
                max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) # from bytes to GB
                logger.info('Iters: {}/{}\t, Eta: {}, Total loss: {:.3f}, Sup loss: {:.3f}, Aux loss1: {:.3f}, Aux loss2: {:.3f}, Cur Mem: {:.2f}, Max Mem: {:.2f}'.format(iters, total_iters, eta, sup_total_loss.avg, sup_loss.avg, sup_loss_aux1.avg, sup_loss_aux2.avg, cur_memory_allocated, max_memory_allocated))

        if (epoch + 1) % cfg['eval_interval'] == 0:
            eval_mode = 'sliding_window' if cfg['dataset'] in ['cityscapes', 'syn_city', 'acdc'] else 'original'
            mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, rank)

            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            if rank == 0:
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
