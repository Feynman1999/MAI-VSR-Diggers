import megengine.module as M
import megengine.functional as F
from ..builder import LOSSES

@LOSSES.register_module()
class L1Loss(M.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, label):
        return F.loss.l1_loss(pred = pred, label = label)

@LOSSES.register_module()
class L2Loss(M.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred, label, Mask = None):
        return F.loss.square_loss(pred= pred, label= label)

@LOSSES.register_module()
class CharbonnierLoss(M.Module):
    def __init__(self, reduction="mean"):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-8
        self.reduction = reduction
        assert self.reduction in ("sum", "mean")

    def calc(self, X, Y, mask = None):
        diff = X - Y
        error = F.sqrt(diff * diff + self.eps)
        if mask is not None:
            error = error * mask
        if self.reduction == "mean":
            loss = F.mean(error)
        else:
            loss = F.sum(error)
        return loss

    def forward(self, X, Y, mask = None):
        if isinstance(X, list):
            ans =  [self.calc(x,y) for x,y in zip(X,Y)]
            if self.reduction == "mean":
                loss = sum(ans) / len(ans)
            else:
                loss = sum(ans)
            return loss
        else:
            return self.calc(X,Y, mask)

@LOSSES.register_module()
class RSDNLoss(M.Module):
    def __init__(self, a = 1.0, b=1.0, c=1.0):
        super(RSDNLoss, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.charbonnierloss = CharbonnierLoss()

    def forward(self, HR_G, HR_D, HR_S, label, label_D, label_S):
            return self.a * self.charbonnierloss(HR_S, label_S) + \
                self.b * self.charbonnierloss(HR_D, label_D) + \
                self.c * self.charbonnierloss(HR_G, label)


@LOSSES.register_module()
class HeavisideLoss(M.Module):
    def __init__(self, t = 1):
        super(HeavisideLoss, self).__init__()
        self.t = t
        self.eps = 1e-8

    def forward(self, x, y):
        # mask
        ab = F.abs(x -y)
        mask = (ab > self.t).astype("float32")
        return F.sum(mask * ab) / (F.sum(mask) + self.eps)