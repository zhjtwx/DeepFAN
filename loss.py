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


def dice_loss_3d_(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    target = target.view(target.size(0), target.size(1), target.size(2), -1)
    input = input.view(input.size(0), input.size(1), input.size(2), -1)
    probs = F.softmax(input, dim=1)

    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)  #

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2 * (num / (den1 + den2 + 0.0000001))
    dice = dice[:, 1:]  # we ignore bg dice val, and take the fg
    dice = torch.sum(dice, dim=1)
    dice = dice / (target.size(1) - 1)
    dice_total = -1 * torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return dice_total


def PairwiseConfusion(features, target_var):
    batch_size = features.size(0)
    # print batch_size
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')
    batch_left = features[:int(0.5 * batch_size)]
    batch_right = features[int(0.5 * batch_size):]
    target_var = target_var.float()
    target_var = target_var - 0.5
    target_var *= 2
    target_var_left = target_var[:int(0.5 * batch_size)]
    target_var_right = target_var[int(0.5 * batch_size):]

    target_var_diff = target_var_left * target_var_right
    target_var_diff[target_var_diff == 1] = 0
    target_var_diff[target_var_diff == -1] = 1

    loss = torch.norm((batch_left - batch_right).abs(), 2, 1) / float(batch_size)
    # print target_var_diff,loss.size()
    loss = (loss * target_var_diff).sum()
    return loss


def PairwiseConfusion_nolim(features):
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        print(batch_size)
        raise Exception('Incorrect batch size provided')
    batch_left = features[:int(0.5 * batch_size)]
    batch_right = features[int(0.5 * batch_size):]
    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    return loss


def EntropicConfusion(features):
    batch_size = features.size(0)
    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)


def Max_Entropy_loss_2d(input, target, alpha=1.0):
    batch_size = input.size(0)
    num_class = input.size(1)
    target_0 = 1 - target

    target = torch.cat((target_0.view(-1, 1), target.view(-1, 1)), dim=1).type(torch.cuda.FloatTensor)
    kl_div = nn.KLDivLoss(size_average=False).cuda()
    loss1 = kl_div(F.softmax(input, 1), target) * (1.0 / batch_size)

    ####
    #     kl_div = nn.CrossEntropyLoss().cuda()
    #     loss1 = kl_div(input, target)
    ####

    loss2 = (F.softmax(input, 1) * torch.log(F.softmax(input, 1))).sum() * ((-1.0 / batch_size) / 1)

    # loss2 = kl_div(input, F.log_softmax(input, 1))
    # print loss1, loss2
    loss = loss1 - alpha * loss2
    return loss
