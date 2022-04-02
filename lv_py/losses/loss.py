import torch
import torch.nn.functional as F

def sigmoid_focal_loss(data, label, alpha=0.25, gamma=2):
    pt = data.sigmoid() * label + (-data).sigmoid() * (1 - label)
    alpha_t = alpha * label + (1 - alpha) * (1 - label)

    loss = F.binary_cross_entropy_with_logits(data, label, reduction='none')
    weight = (1 - pt).pow(gamma) * alpha_t
    loss = loss * weight
    loss = loss.view(loss.shape[0], -1).sum(1).mean()
    return loss

def dice_loss(data, label, weight=None, eps=1e-3):
    a = data.sigmoid().flatten(1)
    b = label.flatten(1)

    inter = (a * b).sum(1)
    union = (a**2).sum(1) + (b**2).sum(1)
    loss = 1 - 2 * inter / (union + eps)
    if weight is not None:
        loss = (loss * weight).sum() / weight.sum()
    else:
        loss = loss.mean()
    return loss

def cross_entropy_with_logits(data, label):
    log_prob = data.log_softmax(1)
    loss = (label * log_prob).sum(1)
    loss = - loss.sum() / label.sum().clip(min=1)
    return loss
