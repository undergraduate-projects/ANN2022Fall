from jittor import nn
import jittor as jt
import math
import numpy as np
from .loss import OhemCrossEntropy2d
from .lovasz_losses import lovasz_softmax
import scipy.ndimage as nd


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def execute(self, output, target):
        return nn.cross_entropy_loss(output, target, self.weight, self.ignore_index, self.reduction)


class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = nn.interpolate(preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss

class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = nn.interpolate(preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2*0.4


class CriterionOhemDSN2(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def execute(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = nn.interpolate(preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        loss2 = lovasz_softmax(nn.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)

        return loss1 + loss2