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
import numpy as np

from losses.triplet_loss_soft_margin import TripletLoss
from loss import dice_loss_3d_, Max_Entropy_loss_2d
from losses.asl import ASLSingleLabel, AsymmetricLossOptimized

from samplers.ct_sampler import CTSampler
from sampler import weight_load

from optimizer.sam import SAM

from model.fusion import FusionModel


def get_model(model_name):
    if model_name == 'fusion':
        net = FusionModel()
    return net


def get_loss(loss_name, loss_dict = None):
    if loss_name == 'dice_loss':
        loss = dice_loss_3d_
    elif loss_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss().cuda()
    elif loss_name == 'focal':
        loss = FocalLoss(class_num=config.n_classes, alpha=config.focal_alpha, gamma = config.focal_gamma).cuda()
    elif loss_name == 'max_entropy':
        loss = Max_Entropy_loss_2d
    elif loss_name == 'BCE':
        def bce_forward_ignoring(self, x, target):
            tw = self.weight.cuda()
            # tpw = self.pos_weight
            # import pdb; pdb.set_trace()
            if tw is not None and len(target) < len(tw):
                tw = tw[:len(target)]
                # tpw = tpw[:len(target)]
            # print(tw, (target != -1).type(x.dtype))
            xw = (target != -1).type(x.dtype) * tw
            return F.binary_cross_entropy_with_logits(
                x, target,
                weight=xw.detach(),
                pos_weight=None,
                reduction=self.reduction)
        nn.BCEWithLogitsLoss.forward = bce_forward_ignoring

        if loss_dict['weight'] is not None:
            weight_2trans = torch.tensor(loss_dict['weight'] * loss_dict['batch_size']).view(-1, len(loss_dict['weight']))
            loss = nn.BCEWithLogitsLoss(weight = weight_2trans).cuda()
        else:
            loss = nn.BCEWithLogitsLoss().cuda()
    elif loss_name == 'triplet_loss_soft_margin_batch_soft':
        loss1 = TripletLoss(margin = loss_dict['margin'], sample= loss_dict['sample']).cuda()
        loss2 = nn.CrossEntropyLoss().cuda()
        loss = [loss1, loss2]
    elif loss_name == 'Asl_multilabel':
        loss = AsymmetricLossOptimized(
            gamma_neg = loss_dict['gamma_neg'],
            gamma_pos= loss_dict['gamma_pos'],
            clip = loss_dict['clip'],
            eps= loss_dict['eps'],
            disable_torch_grad_focal_loss = loss_dict['disable_torch_grad_focal_loss'],
            label_smooth = loss_dict['label_smooth'],
        )
    elif loss_name == 'Asl_singlelabel':
        loss = ASLSingleLabel(gamma_pos = loss_dict['gamma_pos'], gamma_neg = loss_dict['gamma_neg'], eps = loss_dict['eps'], reduction = loss_dict['reduction'])
    else:
        loss = None
        print ('Error: Can not find loss func')
    return loss


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_optimizer(optimizer_name, optimizer_opt, lr, model):
    filter_bias_and_bn = optimizer_opt['filter_bias_and_bn']
    if optimizer_opt['weight_decay'] and filter_bias_and_bn:
        parameters = add_weight_decay(model, optimizer_opt['weight_decay'])
        optimizer_opt['weight_decay'] = 0.
    else:
        parameters = model.parameters()

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr,
                            momentum=optimizer_opt['momentum'],
                            nesterov=True,
                            weight_decay=optimizer_opt['weight_decay'])
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(parameters,lr=lr,
                                     weight_decay=optimizer_opt['weight_decay'])
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.Adam(parameters,lr=lr,
                                     weight_decay=optimizer_opt['weight_decay'])
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(parameters, lr=lr,
                                        alpha=optimizer_opt['alpha'],
                                        eps=1e-08,
                                        weight_decay=optimizer_opt['weight_decay'],
                                        momentum=optimizer_opt['momentum'],
                                        centered=False)
    elif optimizer_name == 'SAM':
        if optimizer_opt['base_optimizer'] == 'SGD':
            base_optimizer = torch.optim.SGD
            optimizer = SAM(parameters, base_optimizer, lr=lr,
                            momentum=optimizer_opt['momentum'],
                            nesterov=True,
                            weight_decay=optimizer_opt['weight_decay'])
    else:
        optimizer = None
        print ('Error: Can not find optimizer')
    return optimizer

def get_lr_scheduler(lr_scheduler_name, optimizer, lr_scheduler_opt = {
    #'milestones': [60-64, 90-64],
    'milestones': [40, 80, 120],
    'gamma': 0.1,}
                     ):
    if lr_scheduler_name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones= lr_scheduler_opt['milestones'],
            gamma= lr_scheduler_opt['gamma']
        )
    if lr_scheduler_name == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr= lr_scheduler_opt['max_lr'],
            steps_per_epoch= lr_scheduler_opt['steps_per_epoch'],
            epochs= lr_scheduler_opt['epochs'],
            pct_start= lr_scheduler_opt['pct_start'])

    elif lr_scheduler_name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = lr_scheduler_opt['T_0'], T_mult= lr_scheduler_opt['T_mult'], eta_min= lr_scheduler_opt['eta_min'], last_epoch= lr_scheduler_opt['last_epoch'])


def get_sampler(sampler_name, sampler_setting = {}):

    if sampler_name == 'WeightedRandomSampler':
        if 'sampler_list_dir' in sampler_setting:
            weights = weight_load(sampler_setting['sampler_list_dir'])
        if 'num_samples' not in sampler_setting or sampler_setting['num_samples'] is None:
            sampler_setting['num_samples'] = len(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples = sampler_setting['num_samples'], replacement = sampler_setting['replacement'])
    elif sampler_name == 'RandomSampler':
        sampler = torch.utils.data.sampler.RandomSampler(
                                data_source = sampler_setting['data_source'],
                                replacement= sampler_setting['replacement'],
                                num_samples = sampler_setting['num_samples'])
    elif sampler_name == 'ct_sampler':

        sampler = CTSampler(train_patch_path = sampler_setting['train_patch_path'],
                            num_per_ct = sampler_setting['num_per_ct'],
                            pos_fraction = sampler_setting['pos_fraction'],
                            shuffle_ct= sampler_setting['shuffle_ct'],
                            numct_perbatch = sampler_setting['numct_perbatch'])
    elif sampler_name == 'DistributedSampler':
        sampler = sampler_setting
        # sampler = DistributedSampler(data_source = sampler_setting['data_source'],
        #                              num_replicas= sampler_setting['num_replicas'],
        #                              rank = sampler_setting['rank'])

    else:
        raise RuntimeError('Error: Can not find sampler')
    return sampler

