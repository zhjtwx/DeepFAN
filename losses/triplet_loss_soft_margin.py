import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math


def pdist(v):
    dist = torch.norm(v[:, None] - v, dim=2, p=2)
    return dist


class TripletLoss(nn.Module):
    def __init__(self, margin='soft', sample= 'weighted'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sample = sample

    def forward(self, inputs, targets):

        n = inputs.size(0)
        # import pdb
        # pdb.set_trace()
        # pairwise distances
        dist = pdist(inputs)

        # find the hardest positive and negative
        mask_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_neg = ~mask_pos
        mask_pos[torch.eye(n).bool().cuda()] = 0
        if self.sample == 'sample':
            # weighted sample pos and negative to avoid outliers causing collapse
            posw = (dist + 1e-12) * mask_pos.float()
            posi = torch.multinomial(posw, 1)

            dist_p = dist.gather(0, posi.view(1, -1))
            # There is likely a much better way of sampling negatives in proportion their difficulty, based on distance
            # this was a quick hack that ended up working better for some datasets than hard negative
            negw = (1 / (dist + 1e-12)) * mask_neg.float()
            negi = torch.multinomial(negw, 1)

            dist_n = dist.gather(0, negi.view(1, -1))


        elif self.sample == 'weighted':
            weight_p = (torch.exp(dist) * mask_pos) / (torch.exp(dist) * mask_pos).sum(1).unsqueeze(1)
            #print (dist_p.size())
            dist_p = dist * mask_pos * weight_p
            dist_p = dist_p.sum(1)

            weight_n = (torch.exp(-dist) * mask_neg) / (torch.exp(-dist) * mask_neg).sum(1).unsqueeze(1)
            dist_n = dist * mask_neg * weight_n
            #print (dist_n.size())
            dist_n = dist_n.sum(1)

        else:
            # hard negative
            ninf = torch.ones_like(dist) * float('-inf')
            dist_p = torch.max(dist * mask_pos.float(), dim=1)[0]
            nindex = torch.max(torch.where(mask_neg, -dist, ninf), dim=1)[1]
            dist_n = dist.gather(0, nindex.unsqueeze(0))

        # calc loss
        diff = dist_p - dist_n
        if isinstance(self.margin, str) and self.margin == 'soft':
            diff = F.softplus(diff)
        else:
            diff = torch.clamp(diff + self.margin, min=0.)

        loss = diff.mean()



        # calculate metrics, no impact on loss
        #metrics = OrderedDict()
        # with torch.no_grad():
        #     _, top_idx = torch.topk(dist, k=2, largest=False)
        #     top_idx = top_idx[:, 1:]
        #     flat_idx = top_idx.squeeze() + n * torch.arange(n, out=torch.LongTensor()).cuda()
        #     top1_is_same = torch.take(mask_pos, flat_idx)
        #     metrics['prec'] = top1_is_same.float().mean().item()
        #     metrics['dist_acc'] = (dist_n > dist_p).float().mean().item()
        #     if not isinstance(self.margin, str):
        #         metrics['dist_sm'] = (dist_n > dist_p + self.margin).float().mean().item()
        #         metrics['nonzero_count'] = torch.nonzero(diff).size(0)
        #     metrics['dist_p'] = dist_p.mean().item()
        #     metrics['dist_n'] = dist_n.mean().item()
        #     metrics['rel_dist'] = ((dist_n - dist_p) / torch.max(dist_p, dist_n)).mean().item()

        return loss#, metrics