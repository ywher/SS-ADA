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

def get_args():
    parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, default=None)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    return parser.parse_args()


def evaluate(model, loader, mode, cfg):
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
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        if len(pred) > 1:
                            pred = pred[0]
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
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
            # bar.update(1)
    # bar.close()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    model.train()
    return mIOU, iou_class


def evaluate_single_gpu(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

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
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        if len(pred) > 1:
                            pred = pred[0]
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
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
            bar.update(1)
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
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', n_class=cfg['nclass'])

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=cfg['workers'], drop_last=False, sampler=valsampler)
    
    # source dataset
    if cfg['source']['use_source']:
        sourceset = SemiDataset(cfg['source']['type'], cfg['source']['data_root'], 'source', cfg['crop_size'], cfg['source']['data_list'], n_class=cfg['nclass'], nsample=len(trainloader) * cfg['epochs'] * cfg['batch_size'] * cfg['n_gpus'])
        sourcesampler = torch.utils.data.distributed.DistributedSampler(sourceset)
        sourceloader = DataLoader(sourceset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['workers'], drop_last=True, sampler=sourcesampler)
        sourceloader_iter = iter(sourceloader)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if rank == 0:
        logger.info("len of trainloader: %d" % len(trainloader))
        logger.info("len of valloader: %d" % len(valloader))
        logger.info("len of sourceloader: %d" % len(sourceloader))
        logger.info("total iters: %d" % total_iters)

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
        sup_loss_src = AverageMeter()
        sup_loss_src_aux1 = AverageMeter()
        sup_loss_src_aux2 = AverageMeter()
        sup_loss_tgt = AverageMeter()
        sup_loss_tgt_aux1 = AverageMeter()
        sup_loss_tgt_aux2 = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):
            
            src_img, src_mask = sourceloader_iter.next()
            src_img, src_mask = src_img.cuda(), src_mask.cuda()

            img, mask = img.cuda(), mask.cuda()
            
            num_src, num_tgt = src_img.size(0), img.size(0)
            pred, pred_aux1, pred_aux2 = model(torch.cat([src_img, img], dim=0))

            pred_src, pred_tgt = pred.split([num_src, num_tgt], dim=0)
            pred_aux1_src, pred_aux1_tgt = pred_aux1.split([num_src, num_tgt], dim=0)
            pred_aux2_src, pred_aux2_tgt = pred_aux2.split([num_src, num_tgt], dim=0)
            
            loss_src = criterion(pred_src, src_mask)
            loss_src_aux1 = criterion(pred_aux1_src, src_mask)
            loss_src_aux2 = criterion(pred_aux2_src, src_mask)
            
            loss_tgt = criterion(pred_tgt, mask)
            loss_tgt_aux1 = criterion(pred_aux1_tgt, mask)
            loss_tgt_aux2 = criterion(pred_aux2_tgt, mask)
            
            total_loss = (loss_src + loss_src_aux1 + loss_src_aux2) / 3.0 + (loss_tgt + loss_tgt_aux1 + loss_tgt_aux2) / 3.0
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            sup_total_loss.update(total_loss.item())
            sup_loss_src.update(loss_src.item())
            sup_loss_src_aux1.update(loss_src_aux1.item())
            sup_loss_src_aux2.update(loss_src_aux2.item())
            sup_loss_tgt.update(loss_tgt.item())
            sup_loss_tgt_aux1.update(loss_tgt_aux1.item())
            sup_loss_tgt_aux2.update(loss_tgt_aux2.item())

            iters += 1

            if rank == 0:
                writer.add_scalar('train/loss_all', total_loss.item(), iters)
                writer.add_scalar('train/loss_src', loss_src.item(), iters)
                writer.add_scalar('train/loss_src_aux1', loss_src_aux1.item(), iters)
                writer.add_scalar('train/loss_src_aux2', loss_src_aux2.item(), iters)
                writer.add_scalar('train/loss_tgt', loss_tgt.item(), iters)
                writer.add_scalar('train/loss_tgt_aux1', loss_tgt_aux1.item(), iters)
                writer.add_scalar('train/loss_tgt_aux2', loss_tgt_aux2.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                cur_time = time.time()
                eta = (cur_time - global_start_time) / (iters - start_iters + 1) * (total_iters - iters - 1)
                eta = str(datetime.timedelta(seconds=int(eta)))
                cur_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3) # from bytes to GB
                max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) # from bytes to GB
                logger.info('Iters: {}/{}\t, Eta: {}, Total loss: {:.3f}, Src loss: {:.3f}, Src aux loss1: {:.3f}, Src aux loss2: {:.3f},  Tar loss: {:.3f}, Tar aux loss1: {:.3f}, Tar aux loss2: {:.3f}, Cur Mem: {:.2f}, Max Mem: {:.2f}'.format(iters, total_iters, eta, sup_total_loss.avg, sup_loss_src.avg, sup_loss_src_aux1.avg, sup_loss_src_aux2.avg, sup_loss_tgt.avg, sup_loss_tgt_aux1.avg, sup_loss_tgt_aux2.avg, cur_memory_allocated, max_memory_allocated))

        eval_mode = 'sliding_window' if cfg['dataset'] in ['cityscapes', 'syn_city']else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
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