from __future__ import division
import os
import shutil
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import config as config
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
from logg import logg_init_new
from dataset_itk import DatasetFromList
from ema import EMA
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from get_items import *
from torchvision import transforms
from loss import *
from gpu_jitter import build_gpu_jitter
from cal_metrics.spesen import tuning_threshold
from utils.mio import save_string_list
from tensorboardX import SummaryWriter
# import encoding
# from sync_batchnorm import patch_replication_callback
import sys
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
sys.path.insert(0, '../')
import config
import random

seed = config.seed

random.seed(seed)


# torch.backends.cudnn.deterministic=True
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark =True
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def cal_accuracy_class(pred, gt):
    pred_labels = np.argmax(pred, axis=1)
    accuracy = (pred_labels == gt)
    return np.sum(accuracy) * 100 / np.shape(accuracy)[0], pred


def cal_accuracy_multiclass(pred, gt):
    pred_labels = (pred > 0.5).astype(int)
    mask = (gt != -1)  # b * c
    accuracy = (pred_labels == gt) * mask
    sum_eachcls = np.zeros((config.n_classes))
    count_cls = 0
    for cls_idx in range(config.n_classes):
        if np.sum(mask[:, cls_idx]) != 0:
            sum_eachcls[cls_idx] = np.sum(accuracy[:, cls_idx]) / np.sum(mask[:, cls_idx])
            count_cls += 1
        else:
            sum_eachcls[cls_idx] = 0
    acc = np.sum(sum_eachcls) / count_cls
    return acc, pred


def train(train_loader, model, criterion, optimizer, epoch, logger, gpu_jitter=None, ema=None, lr_scheduler=None):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    classification = AverageMeter()
    model.train()
    end = time.time()
    all_pred = np.zeros(((len(train_loader) - 1) * config.batch_size, config.n_classes))
    if 'attrcls' in config.data_mode:
        all_labels = np.zeros(((len(train_loader)) * config.batch_size, config.n_classes))
    else:
        all_labels = np.zeros(((len(train_loader)) * config.batch_size))
    batch_start = 0

    for i, (input, label_mbs, label_lb, label_sp, label_dy, img_dir, another_input) in enumerate(train_loader):

        with autocast():
            label_mb = label_mbs.long().view(-1)
            label_lb = label_lb.long().view(-1)
            label_sp = label_sp.long().view(-1)
            label_dy = label_dy.long().view(-1)

            input = input.cuda(non_blocking=True)
            label_mb = label_mb.cuda(non_blocking=True)
            label_lb = label_lb.cuda(non_blocking=True)
            label_sp = label_sp.cuda(non_blocking=True)
            label_dy = label_dy.cuda(non_blocking=True)
            if gpu_jitter:
                input = gpu_jitter(input)
            input_var = torch.autograd.Variable(input)
            label_mb_var = torch.autograd.Variable(label_mb)
            label_lb_var = torch.autograd.Variable(label_lb)
            label_sp_var = torch.autograd.Variable(label_sp)
            label_dy_var = torch.autograd.Variable(label_dy)
            gcn_cla_mb, vit_cls_mb, all_cls_mb, hf0, hf1, hf2 = model(input_var)
            del input_var
            loss_vit = criterion(vit_cls_mb, label_mb_var)
            loss_all = criterion(all_cls_mb, label_mb_var)
            loss_gcn = criterion(gcn_cla_mb, label_mb_var)
            loss_lb = criterion(hf0, label_lb_var)
            loss_sp = criterion(hf1, label_sp_var)
            loss_dy = criterion(hf2, label_dy_var)
            loss_tol = 0.2 * loss_vit + 0.2 * loss_all + 0.4 * loss_gcn + 0. * (loss_lb + loss_sp + loss_dy)
            one_epoch_pred = F.softmax(gcn_cla_mb, dim=1)

        if config.confusion and feat.size(0) % 2 == 0:
            loss_tol += config.confusion * PairwiseConfusion_nolim(feat)
        if config.optimizer_name == 'SAM':
            loss_tol.backward()
            optimizer.first_step(zero_grad=True)
            if 'fgpn' in config.model_name:
                pass
            else:
                second_loss = criterion(model(input_var)[0], target_var)
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
        else:

            scaler.scale(loss_tol).backward()
            if (i + 1) % 3 == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        if config.ema and epoch > config.ema[0]:
            ema.update()
        batch_time.update(time.time() - end)
        end = time.time()
        pred_class_mb, pred_all = cal_accuracy_class(one_epoch_pred.data.cpu().numpy(), np.array(label_mbs))

        losses.update(loss_tol.item(), input.size(0))
        classification.update(pred_class_mb, input.size(0))
        if i % config.print_freq == 0:
            # print (min(iteration/ 20000.0, 0.1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Class {classification.val:.3f} ({classification.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                                     batch_time=batch_time, loss=losses,
                                                                                     classification=classification))
        all_pred[batch_start: batch_start + input.size(0), :] = pred_all

        all_labels[batch_start: batch_start + input.size(0)] = np.array(label_mbs)
        batch_start += input.size(0)

        if config.lr_scheduler_name == 'OneCycleLR':
            lr_scheduler.step()

        if i == len(train_loader) - 2:
            break

    all_pred = all_pred[:batch_start]
    all_labels = all_labels[:batch_start]
    logger.info('Train: Epoch {},  Classification {classification.avg:.3f}, Loss {loss.avg:.4f}'.format(epoch,
                                                                                                        classification=classification,
                                                                                                        loss=losses))
    return all_pred, all_labels, classification.avg, losses.avg


def validation(val_loader, model, criterion, epoch, logger, optimizer, save_file):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    classification = AverageMeter()
    batch_start = 0
    model.eval()
    # torch.set_grad_enabled(False)
    end = time.time()
    img_list = []
    all_pred = np.zeros((len(val_loader) * config.val_batch_size, config.n_classes))
    all_labels = np.zeros(((len(val_loader)) * config.val_batch_size))

    with torch.no_grad():
        for i, (input, label_mbs, label_lb, label_sp, label_dy, img_dir, another_input) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            labels = label_mbs.long().view(-1)
            target = labels.cuda(non_blocking=True)
            label_lb = label_lb.long().view(-1)
            label_lb = label_lb.cuda(non_blocking=True)
            label_sp = label_sp.long().view(-1)
            label_sp = label_sp.cuda(non_blocking=True)
            label_dy = label_dy.long().view(-1)
            label_dy = label_dy.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            label_lb_var = torch.autograd.Variable(label_lb)
            label_sp_var = torch.autograd.Variable(label_sp)
            label_dy_var = torch.autograd.Variable(label_dy)

            gcn_cla_mb, vit_cls_mb, all_cls_mb, hf0, hf1, hf2 = model(input_var)
            loss_vit = criterion(vit_cls_mb, target_var)
            loss_all = criterion(all_cls_mb, target_var)
            loss_gcn = criterion(gcn_cla_mb, target_var)
            loss_lb = criterion(hf0, label_lb_var)
            loss_sp = criterion(hf1, label_sp_var)
            loss_dy = criterion(hf2, label_dy_var)
            loss_tol = 0.2 * loss_vit + 0.2 * loss_all + 0.4 * loss_gcn + 0.2*(loss_lb + loss_sp + loss_dy)
            one_epoch_pred = F.softmax(gcn_cla_mb, dim=1)

            optimizer.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            pred_class, pred_all = cal_accuracy_class(one_epoch_pred.data.cpu().numpy(), np.array(labels))

            losses.update(loss_tol.item(), input.size(0))
            classification.update(pred_class, input.size(0))
            if i % config.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Class {classification.val:.3f} ({classification.avg:.3f})'
                      .format(epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                              classification=classification))
            all_pred[batch_start: batch_start + input.size(0), :] = pred_all
            all_labels[batch_start: batch_start + input.size(0)] = np.array(labels)
            img_list.extend(img_dir)
            batch_start += input.size(0)

    all_pred = all_pred[:batch_start]
    all_labels = all_labels[:batch_start]
    logger.info('Valid: Epoch {}, Classification {classification.avg:.3f}, Loss {loss.avg:.4f}'.format(epoch,
                                                                                                       classification=classification,
                                                                                                       loss=losses))
    dat = []
    for i in range(len(all_pred[0])):
        dat.append("p_" + str(i + 1))
    data = [dat + ['label', 'file_name']]
    for i in range(len(all_pred)):
        pre_value = [all_pred[i][j] for j in range(len(all_pred[0]))]
        data.append(pre_value + [all_labels[i], img_list[i]])
    data = pd.DataFrame(data)
    data.to_csv(save_file, index=False, header=False)
    return all_pred, all_labels, img_list, classification.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = config.lr * (0.1 ** (epoch // config.lr_controler)) #* (0.1 ** (epoch // 35))
    # lr = config.lr * (1 - float(epoch) / config.epochs) ** 0.9
    if not (config.optimizer_name == 'Adam'):
        lr = config.lr * (0.1 ** (epoch // config.lr_controler))
        if config.face_lr:
            if epoch // config.lr_controler > 2:
                lr = config.lr * (0.1 ** 2) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def cal_metrics(all_pred, all_labels, epoch, logger):
    if 'attrcls' in config.data_mode:
        all_gt = all_labels
        try:
            auc = np.zeros((config.n_classes))
            for one_cls in range(config.n_classes):
                auc[one_cls] = roc_auc_score(all_gt[:, one_cls][all_gt[:, one_cls] != -1],
                                             all_pred[:, one_cls][all_gt[:, one_cls] != -1], average=None)
        except ValueError:
            auc = np.zeros((config.n_classes))

        all_pred_labels = (all_pred > 0.5).astype(int)
        precision = np.zeros((config.n_classes))
        recall = np.zeros((config.n_classes))
        acc = np.zeros((config.n_classes))
        for one_cls in range(config.n_classes):
            precision[one_cls] = precision_score(all_gt[:, one_cls][all_gt[:, one_cls] != -1],
                                                 all_pred_labels[:, one_cls][all_gt[:, one_cls] != -1], average=None)[1]
            recall[one_cls] = recall_score(all_gt[:, one_cls][all_gt[:, one_cls] != -1],
                                           all_pred_labels[:, one_cls][all_gt[:, one_cls] != -1], average=None)[1]
            acc[one_cls] = np.sum((all_gt[:, one_cls][all_gt[:, one_cls] != -1] == all_pred_labels[:, one_cls][
                all_gt[:, one_cls] != -1]).astype(int)) / np.sum(np.array([all_gt[:, one_cls] != -1]).astype(int))
    else:
        all_labels = all_labels.astype('int')
        all_gt = np.zeros((len(all_labels), config.n_classes))  # b * c
        for idx, one_label in enumerate(all_labels):
            all_gt[idx, one_label] = 1
            # all_pred_auc[idx, int(all_pred[idx])] =1
        try:
            auc = roc_auc_score(all_gt, all_pred, average=None)

        except ValueError:
            auc = np.zeros((config.n_classes))
        all_pred_labels = np.argmax(all_pred, axis=1)
        precision = precision_score(all_labels, all_pred_labels, average=None)
        recall = recall_score(all_labels, all_pred_labels, average=None)
        acc = np.zeros((config.n_classes))

    for i in range(config.n_classes):
        logger.info('AUC:class_' + str(i) + ':' + str(auc[i]))
    for i in range(config.n_classes):
        logger.info('Precision:class_' + str(i) + ':' + str(precision[i]))
    for i in range(config.n_classes):
        logger.info('Recall:class_' + str(i) + ':' + str(recall[i]))
    for i in range(config.n_classes):
        logger.info('Acc:class_' + str(i) + ':' + str(acc[i]))

    average_recall = np.mean(recall)
    average_auc = np.mean(auc)
    return average_recall, average_auc, auc, recall, acc


def cal_prob(all_pred, all_labels, epoch, logger):
    prob_pos = np.mean(all_pred[all_labels == 1][:, 1])
    prob_neg = np.mean(all_pred[all_labels == 0][:, 0])
    logger.info('Prob_pos:' + str(prob_pos))
    logger.info('Prob_neg:' + str(prob_neg))
    return prob_neg, prob_pos


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = config.mode_save_base_dir + "%s/" % (config.model_save_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)


def main():
    if not os.path.exists(config.model_save_logg_dir):
        os.makedirs(config.model_save_logg_dir)
    logger = logg_init_new(config.model_save_logg_dir,
                           config.model_save_name,
                           config.model_mode_record,
                           config.model_load_record,
                           config.model_save_record,
                           config.dataset_record,
                           config.dataloader_record,
                           config.lr_scheduler_record,
                           config.optimizer_record,
                           config.loss_record,
                           config.train_dataaug_opt,
                           config.val_dataaug_opts
                           )

    writer = SummaryWriter(config.model_save_logg_dir + 'tfboard.log')
    ###dataset

    if not config.inference_mode:
        train_set = DatasetFromList(config.train_set_dir, config.train_set_roi_dir, config.train_dataaug_opt)
    #    if config.inference_mode:
    #        config.val_set_dir = config.val_set_dir_inf
    val_sets = []
    for idx in range(len(config.val_set_dirs)):
        val_sets.append(
            DatasetFromList(config.val_set_dirs[idx], None, config.val_dataaug_opts[idx]))
    ###dataloader
    if not config.inference_mode:
        if config.use_sampler:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.batch_size, sampler=config.sampler, **config.dataloaer_settings)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.batch_size, shuffle=True, **config.dataloaer_settings)
    val_loaders = []
    for idx in range(len(config.val_set_dirs)):
        val_loaders.append(torch.utils.data.DataLoader(
            val_sets[idx],
            batch_size=config.val_batch_size, shuffle=False, **config.dataloaer_settings))
    ###model set
    model = config.model

    ###optimizer##########################
    if 'custom' not in config.optimizer_name:
        optimizer = get_optimizer(config.optimizer_name, config.optimizer_opt, config.lr, model)
    elif config.optimizer_name == 'custom_SGD':
        params = list(
            map(lambda x: x[1], list(filter(lambda kv: '_dfl' in kv[0], model.named_parameters()))))
        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: '_dfl' not in kv[0], model.named_parameters()))))

        optimizer = torch.optim.SGD([{'params': base_params}, {'params': params, 'lr': config.lr / 100.}],
                                    lr=config.lr,
                                    momentum=config.optimizer_opt['momentum'],
                                    nesterov=True,
                                    weight_decay=config.optimizer_opt['weight_decay'])

    # get the number of model parameterss
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    #############use_lr_scheduler
    if config.use_lr_scheduler:
        lr_scheduler = get_lr_scheduler(config.lr_scheduler_name, optimizer, config.lr_scheduler_opt)
    else:
        lr_scheduler = None
    print('-3', torch.cuda.memory_allocated(0) / (1024 ** 3),
          torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) - torch.cuda.memory_allocated(0) / (
                  1024 ** 3))
    ###model load
    if config.resume and os.path.isfile(config.resume):
        print("=> loading checkpoint '{}'".format(config.resume))
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        if config.inherit_epoch and 'epoch' in checkpoint.keys():
            config.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        if 'state_dict' in checkpoint.keys():
            ft_state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint.keys():
            ft_state_dict = checkpoint['model']


        ###remove `module.`
        pretrained_dict = OrderedDict()
        count = 0
        current_state = model.state_dict()
        for k, v in ft_state_dict.items():
            k = k.replace('norm1', 'norm1')
            k = k.replace('relu1', 'relu1')
            k = k.replace('conv1', 'conv1')
            k = k.replace('norm2', 'norm2')
            k = k.replace('relu2', 'relu2')
            k = k.replace('conv2', 'conv2')

            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            pretrained_dict[name] = v
            count += 1

        current_state.update(pretrained_dict)
        model.load_state_dict(current_state)

        if 'optimizer_state_dict' in checkpoint.keys() and config.inherit_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # optimizer.state = defaultdict(dict, optimizer.state)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            print('optimizer parameters loaded')

        if 'scheduler_state_dict' in checkpoint.keys() and config.inherit_lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(config.resume))
    ###load model to GPU
    cudnn.benchmark = True
    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
        # patch_replication_callback(model)
        model = model.cuda()
    else:
        # torch.cuda.set_device(config.gpu)
        model = model.cuda()
    #######ema#########
    if config.ema:
        ema = EMA(model, config.ema[1])
    else:
        ema = None

    ###loss func
    criterion = config.loss_function
    val_criterion = config.val_loss_function

    ### gpu jitter##########################
    gpu_jitter_fn = build_gpu_jitter() if config.gpu_aug else None

    if config.inference_mode:
        for idx in range(len(config.val_set_dirs)):

            all_pred, all_labels, img_list, _, _ = validation(val_loaders[idx], model, val_criterion,
                                                              config.start_epoch + 1, logger, optimizer,
                                                              config.save_csv[idx])
            val_recall_one_epoch, val_auc_one_epoch, val_allauc_one_epoch, val_allrecall_one_epoch, val_allacc_one_epoch = cal_metrics(
                all_pred, all_labels, 0, logger)
            logger.info('AUC_' + str(idx) + ':' + str(val_auc_one_epoch))
            logger.info('recall_' + str(idx) + ':' + str(val_recall_one_epoch))

            # if config.n_classes <=2:
            #     sen_and_spe = tuning_threshold(all_labels, all_pred, threshold_find = 0.999)
            #     sen_test = sen_and_spe['sen']
            #     spe_test = sen_and_spe['spe']
            #     logger.info('Sen_th0.99:{}'.format(sen_test))
            #     logger.info('Spe_th0.99:{}'.format(spe_test))
            # print all_pred
            info_2record = []
            for img_idx, one_img in enumerate(img_list):
                one_row = one_img.replace('\n', '') + ' ' + str(all_labels[img_idx])
                for one_pred in all_pred[img_idx]:
                    one_row = one_row + ' ' + str(one_pred)
                # print one_row
                info_2record.append(one_row)
            save_string_list(config.model_save_logg_dir + 'val_result_' + str(idx) + '.txt', info_2record)
            torch.cuda.empty_cache()

    ###start training
    if not config.inference_mode:
        lowerest_loss = [10000] * len(config.val_set_dirs)
        best_auc = [0] * len(config.val_set_dirs)
        best_acc = [0] * len(config.val_set_dirs)
        best_recall = [0] * len(config.val_set_dirs)
        best_auc_cls = np.zeros((len(config.val_set_dirs), config.n_classes))
        best_acc_cls = np.zeros((len(config.val_set_dirs), config.n_classes))
        best_recall_cls = np.zeros((len(config.val_set_dirs), config.n_classes))

        ###lr_scheduler
        if config.use_lr_scheduler:
            # lr_scheduler = get_lr_scheduler(config.lr_scheduler_name, optimizer, config.lr_scheduler_opt)
            if config.start_epoch != 0:
                lr_scheduler.step(config.start_epoch)
            elif 'Warm' in config.lr_scheduler_name:
                lr_scheduler.step()

        for epoch in range(config.start_epoch, config.epochs):
            logger.info('Epoch:{}'.format(epoch))
            logger.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            writer.add_scalar('train_lr/lr', optimizer.param_groups[0]['lr'], epoch)
            all_pred, all_labels, train_acc_one_epoch, train_losses = \
                train(train_loader,
                      model,
                      criterion,
                      optimizer,
                      epoch,
                      logger,
                      gpu_jitter_fn,
                      ema,
                      lr_scheduler,
                      )
            train_recall_one_epoch, train_auc_one_epoch, train_allauc_one_epoch, train_allrecall_one_epoch, _ = cal_metrics(
                all_pred, all_labels, epoch, logger)
            logger.info('AUC:' + str(train_auc_one_epoch))
            logger.info('recall:' + str(train_recall_one_epoch))
            logger.info('acc:' + str(train_acc_one_epoch))
            writer.add_scalar('train_loss/train', train_losses, epoch)
            writer.add_scalar('train_pred/train', train_acc_one_epoch, epoch)
            writer.add_scalar('train_auc/train', train_auc_one_epoch, epoch)
            writer.add_scalar('train_recall/train', train_recall_one_epoch, epoch)

            prob_neg, prob_pos = cal_prob(all_pred, all_labels, epoch, logger)
            writer.add_scalar('prob_pos/train', prob_pos, epoch)
            writer.add_scalar('prob_neg/train', prob_neg, epoch)

            # if config.n_classes <=2:
            #     sen_and_spe = tuning_threshold(all_labels, all_pred, threshold_find = 0.999)
            #     sen = sen_and_spe['sen']
            #     spe = sen_and_spe['spe']
            #     logger.info('Sen_th0.99:{}'.format(sen))
            #     logger.info('Spe_th0.99:{}'.format(spe))
            if config.ema and epoch > config.ema[0]:
                ema.apply_shadow()
                print('ema used!')
            elif config.ema and epoch == config.ema[0]:
                ema.register()

            for idx in range(len(config.val_set_dirs)):
                all_pred, all_labels, _, val_acc_one_epoch, val_losses = \
                    validation(
                        val_loaders[idx],
                        model,
                        val_criterion,
                        epoch,
                        logger,
                        optimizer, config.save_csv[idx])
                val_recall_one_epoch, val_auc_one_epoch, val_allauc_one_epoch, val_allrecall_one_epoch, val_allacc_one_epoch = cal_metrics(
                    all_pred, all_labels, epoch, logger)
                # if config.n_classes <=2:
                #     sen_and_spe = tuning_threshold(all_labels, all_pred, threshold_find = 0.999)
                #     sen = sen_and_spe['sen']
                #     spe = sen_and_spe['spe']
                #     logger.info('Sen_th0.99:{}'.format(sen))
                #     logger.info('Spe_th0.99:{}'.format(spe))
                print('all_pred', len(all_pred))
                logger.info('AUC_' + str(idx) + ':' + str(val_auc_one_epoch))
                logger.info('recall_' + str(idx) + ':' + str(val_recall_one_epoch))
                logger.info('acc_' + str(idx) + ':' + str(val_acc_one_epoch))
                writer.add_scalar('val_loss_' + str(idx) + '/val_' + str(idx), val_losses, epoch)
                writer.add_scalar('val_pred_' + str(idx) + '/val_' + str(idx), val_acc_one_epoch, epoch)
                writer.add_scalar('val_auc_' + str(idx) + '/val_' + str(idx), val_auc_one_epoch, epoch)
                writer.add_scalar('val_recall_' + str(idx) + '/val_' + str(idx), val_recall_one_epoch, epoch)

                prob_neg, prob_pos = cal_prob(all_pred, all_labels, epoch, logger)
                writer.add_scalar('prob_pos' + '/val_' + str(idx), prob_pos, epoch)
                writer.add_scalar('prob_neg' + '/val_' + str(idx), prob_neg, epoch)

                for cls_idx in range(config.n_classes):
                    writer.add_scalar('val_auc_cls_' + str(cls_idx) + '_' + str(idx) + '/val_' + str(idx),
                                      val_allauc_one_epoch[cls_idx], epoch)
                    writer.add_scalar('val_recall_cls_' + str(cls_idx) + '_' + str(idx) + '/val_' + str(idx),
                                      val_allrecall_one_epoch[cls_idx], epoch)
                    writer.add_scalar('val_acc_cls_' + str(cls_idx) + '_' + str(idx) + '/val_' + str(idx),
                                      val_allacc_one_epoch[cls_idx], epoch)

                    if val_allrecall_one_epoch[cls_idx] >= best_recall_cls[idx, cls_idx]:
                        best_recall_cls[idx, cls_idx] = val_allrecall_one_epoch[cls_idx]
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_recall_cls[idx, cls_idx],
                            'dataset_idx': idx,
                            'cls_idx': cls_idx,
                        }, 'best_recall_' + 'cls_' + str(cls_idx) + '_' + str(idx) + '.pth.tar')

                    if val_allauc_one_epoch[cls_idx] >= best_auc_cls[idx, cls_idx]:
                        best_auc_cls[idx, cls_idx] = val_allauc_one_epoch[cls_idx]
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_auc_cls[idx, cls_idx],
                            'dataset_idx': idx,
                            'cls_idx': cls_idx,
                        }, 'best_auc_' + 'cls_' + str(cls_idx) + '_' + str(idx) + '.pth.tar')

                    if val_allacc_one_epoch[cls_idx] >= best_acc_cls[idx, cls_idx]:
                        best_acc_cls[idx, cls_idx] = val_allacc_one_epoch[cls_idx]
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_acc_cls[idx, cls_idx],
                            'dataset_idx': idx,
                            'cls_idx': cls_idx,
                        }, 'best_acc_' + 'cls_' + str(cls_idx) + '_' + str(idx) + '.pth.tar')

                if val_losses < lowerest_loss[idx]:
                    lowerest_loss[idx] = val_losses
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': lowerest_loss[idx],
                    }, 'lowerest_loss_' + str(idx) + '.pth.tar')

                if val_recall_one_epoch >= best_recall[idx]:
                    best_recall[idx] = val_recall_one_epoch
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_recall[idx],
                    }, 'best_recall_' + str(idx) + '.pth.tar')

                if val_auc_one_epoch >= best_auc[idx]:
                    best_auc[idx] = val_auc_one_epoch
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_auc[idx],
                    }, 'best_auc_' + str(idx) + '.pth.tar')

                if val_acc_one_epoch >= best_acc[idx]:
                    best_acc[idx] = val_acc_one_epoch
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_acc[idx],
                    }, 'best_acc_' + str(idx) + '.pth.tar')

                if epoch % config.model_save_freq == 0:
                    directory = config.mode_save_base_dir + "%s/" % (config.model_save_name)
                    shutil.copyfile(directory + 'best_auc_' + str(idx) + '.pth.tar',
                                    directory + 'best_auc_' + str(idx) + '_epoch_' + str(epoch) + '.pth.tar')
                    shutil.copyfile(directory + 'best_acc_' + str(idx) + '.pth.tar',
                                    directory + 'best_acc_' + str(idx) + '_epoch_' + str(epoch) + '.pth.tar')
                    shutil.copyfile(directory + 'best_recall_' + str(idx) + '.pth.tar',
                                    directory + 'best_recall_' + str(idx) + '_epoch_' + str(epoch) + '.pth.tar')

            if config.use_lr_scheduler:
                scheduler_state_dict_2record = lr_scheduler.state_dict()
            else:
                scheduler_state_dict_2record = None

            if config.ema and epoch > config.ema[0]:
                if epoch % config.model_save_freq == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_auc': best_auc,
                        'best_recall': best_recall,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler_state_dict_2record,
                    }, 'checkpoint_ema_' + str(epoch) + '.pth.tar')

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_auc': best_auc,
                    'best_recall': best_recall,
                    'best_acc': best_acc,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_state_dict_2record,
                }, filename='checkpoint_ema.pth.tar')
                ema.restore()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_auc': best_auc,
                'best_recall': best_recall,
                'best_acc': best_acc,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict_2record,
            })

            if epoch % config.model_save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_auc': best_auc,
                    'best_recall': best_recall,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_state_dict_2record,
                }, 'checkpoint_' + str(epoch) + '.pth.tar')

            # logger.info('Best_recall:{}'.format(best_recall))
            # logger.info('Best_auc:{}'.format(best_auc))
            # logger.info('Best_sen:{}'.format(best_sen))
            # logger.info('Best_spe:{}'.format(best_spe))
            # logger.info('Best_test_auc:{}'.format(best_test_auc))
            # logger.info('Best_test_auc_1:{}'.format(best_test_auc_1))
            for idx in range(len(config.val_set_dirs)):
                for cls_idx in range(config.n_classes):
                    logger.info('Best_auc_cls{}_{}:{}'.format(cls_idx, idx, best_auc_cls[idx, cls_idx]))
                for cls_idx in range(config.n_classes):
                    logger.info('Best_recall_cls{}_{}:{}'.format(cls_idx, idx, best_recall_cls[idx, cls_idx]))
                for cls_idx in range(config.n_classes):
                    logger.info('Best_acc_cls{}_{}:{}'.format(cls_idx, idx, best_acc_cls[idx, cls_idx]))
                logger.info('Best_auc_{}:{}'.format(idx, best_auc[idx]))
                logger.info('Best_recall_{}:{}'.format(idx, best_recall[idx]))
                logger.info('Best_acc_{}:{}'.format(idx, best_acc[idx]))
            if config.use_lr_scheduler and config.lr_scheduler_name != 'OneCycleLR':
                lr_scheduler.step()


if __name__ == '__main__':
    main()
