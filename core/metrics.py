import numpy as np
import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return 1 - dice.mean()

def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc

def dice_coef(pred, target, smooth=1e-6):
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray): target = torch.from_numpy(target)
    pred, target = pred.float(), target.float()
    if pred.device != target.device: target = target.to(pred.device)
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()

def iou(pred, target, smooth=1e-6):
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray): target = torch.from_numpy(target)
    pred, target = pred.float(), target.float()
    if pred.device != target.device: target = target.to(pred.device)
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).clamp(0, 1).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def accuracy_score(pred, target):
    if isinstance(pred, np.ndarray): pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray): target = torch.from_numpy(target)
    pred, target = pred.float(), target.float()
    if pred.device != target.device: target = target.to(pred.device)
    pred = (torch.sigmoid(pred) > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()
