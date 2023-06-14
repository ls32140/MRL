import torch
from torch import nn
from torch.nn import functional as F

eps = 1e-7

class SCELoss(nn.Module):
    def __init__(self, num_classes=10, a=0.1, b=1, onMean=None):
        super(SCELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a #两个超参数
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.onMean = onMean

    def forward(self, pred, labels):
        # CE 部分，正常的交叉熵损失
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0) #最小设为 1e-4，即 A 取 -4
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce
        if self.onMean is not None:
            return loss
        return loss.mean()

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce

class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, onMean=None):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.onMean = onMean
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        loss = self.nce(pred, labels) + self.rce(pred, labels)
        if self.onMean is not None:
            return loss
        return loss.mean()

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7, onMean=None):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.onMean = onMean

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        if self.onMean is not None:
            return gce
        return gce.mean()

class NormalizedGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7, onMean=None):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.scale = scale
        self.onMean = onMean

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        loss = self.scale * ngce
        if self.onMean is not None:
            return loss
        return loss.mean()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0.5, num_classes=10, alpha=None):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor
        return loss


class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5, onMean=None):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)
        self.onMean = onMean

    def forward(self, pred, labels):
        loss = self.nfl(pred, labels) + self.rce(pred, labels)
        if self.onMean is not None:
            return loss
        return loss.mean()

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae

class NGCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7, onMean=None):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)
        self.onMean = onMean

    def forward(self, pred, labels):
        loss = self.ngce(pred, labels) + self.mae(pred, labels)
        if self.onMean is not None:
            return loss
        return loss.mean()
class AUELoss(nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9, eps=eps, scale=1.0):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q
        self.eps = eps
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss * self.scale

class NCEandAUE(torch.nn.Module):
    def __init__(self, alpha=1., beta=1., num_classes=10, a=6, q=1.5, onMean=None):
        super(NCEandAUE, self).__init__()
        self.num_classes = num_classes
        self.onMean = onMean
        self.nce = NormalizedCrossEntropy(num_classes=num_classes, scale=alpha)
        self.aue = AUELoss(num_classes=num_classes, a=a, q=q, scale=beta)

    def forward(self, pred, labels):
        loss = self.nce(pred, labels) + self.aue(pred, labels)
        if self.onMean is not None:
            return loss
        return loss.mean()