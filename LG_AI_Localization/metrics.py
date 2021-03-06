from sklearn.metrics import f1_score
import torch.nn as nn
import torch

def score_function(real, pred):
    score = f1_score(real, pred, average = 'macro')
    return score

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 1, reduction = 'mean', ignore_index = -1000):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = ignore_index, reduction = 'none')

    @torch.cuda.amp.autocast()
    def forward(self, inputs, targets, mixup = None):
        loss = self.loss_fn(inputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt) ** self.gamma * loss

        if mixup is not None:
            F_loss *= mixup
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        elif self.reduction == 'none':
            return F_loss


