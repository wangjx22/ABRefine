"""Code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import Logger

logger = Logger.logger


class InteractionLoss(object):
    """Define Class InteractionLoss."""

    def __init__(
            self,
            enable_margin_loss=False,
            margin=0
    ):
        """Run __init__ method."""
        # code.
        super(InteractionLoss, self).__init__()
        self.enable_margin_loss = enable_margin_loss
        if self.enable_margin_loss:
            self.MarginRankinger = torch.nn.MarginRankingLoss(margin=margin)
            self.MarginRankingDister = torch.nn.PairwiseDistance()
        self.BCEer = torch.nn.BCELoss(reduce=False, reduction='none')
        # self.CateLoss = torch.nn.CrossEntropyLoss()

    def forward(self, binder_true, binder_logits, fake):
        """Run forward method."""
        # code.
        loss = self.BCEer(binder_logits, binder_true)
        fake = torch.reciprocal(fake)
        loss = loss * fake
        # loss = loss.mean()
        return loss


class KLLoss(nn.Module):
    """Define Class KLLoss."""

    def __init__(
            self,
            reverse=False,
    ):
        """Run __init__ method."""
        # code.
        super(KLLoss, self).__init__()
        self.reverse = reverse
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        # self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def forward(self, z, y):
        """Run forward method."""
        # code.
        Q_model = z[..., None]
        Q_model = torch.cat([Q_model, 1 - Q_model], dim=-1)
        P_model = y[..., None]
        P_model = torch.cat([P_model, 1 - P_model], dim=-1)
        if self.reverse:
            return self.kl_loss((P_model+1e-8).log(), Q_model)
        else:
            return self.kl_loss((Q_model+1e-8).log(), P_model)


class HingeRankingLoss(nn.Module):
    """Define Class HingeRankingLoss."""

    def __init__(self, margin=1, reduction='mean'):
        """Run __init__ method."""
        # code.
        super(HingeRankingLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, score_dev):
        """Run forward method."""
        # code.
        l = torch.max(torch.zeros_like(score_dev), self.margin - score_dev)
        if self.reduction == 'mean':
            return torch.mean(l)
        elif self.reduction == 'none':
            return l
        else:
            raise NotImplementedError(f"reduce={self.reduction} not supported yet.")

class RegressionHingeRankingLoss(nn.Module):
    """Define Class RegressionHingeRankingLoss."""

    def __init__(self,reduction='mean'):
        """Run __init__ method."""
        # code.
        super(RegressionHingeRankingLoss, self).__init__()
        self.reduction = reduction

    def forward(self, label_dev, score_dev):
        """Run forward method."""
        # code.
        l = torch.max(torch.zeros_like(score_dev), label_dev - score_dev)
        if self.reduction == 'mean':
            return torch.mean(l)
        elif self.reduction == 'none':
            return l
        else:
            raise NotImplementedError(f"reduce={self.reduction} not supported yet.")


class LogisticRankingLoss(nn.Module):
    """Define Class LogisticRankingLoss."""

    def __init__(self, reduction='mean'):
        """Run __init__ method."""
        # code.
        super(LogisticRankingLoss, self).__init__()
        self.reduction = reduction

    def forward(self, score_dev):
        """Run forward method."""
        # code.
        l = - F.logsigmoid(-score_dev)

        if self.reduction == 'mean':
            return torch.mean(l)
        elif self.reduction == 'none':
            return l
        else:
            raise NotImplementedError(f"reduce={self.reduction} not supported yet.")

class RankNetLoss(nn.Module):
    """Define Class RankNetLoss."""

    def __init__(self, reduction='mean'):
        """Run __init__ method."""
        # code.
        super(RankNetLoss, self).__init__()
        self.reduction = reduction

    def forward(self, score_dev, label):
        """Run forward method."""
        # code.
        s = F.sigmoid(score_dev)
        l = - (label * torch.log(s + 1e-8) + (1-label) * torch.log(1-s + 1e-8))

        if self.reduction == 'mean':
            return torch.mean(l)
        elif self.reduction == 'none':
            return l
        else:
            raise NotImplementedError(f"reduce={self.reduction} not supported yet.")
