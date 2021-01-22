import torch
import torch.nn as nn
import torch.nn.functional as F


class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        input_ = torch.where(input_ > 1, torch.ones_like(input_), input_)
        target = target.float()

        return self.bce(input_, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
            ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


class SimplerFocalLoss(nn.Modeule):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = loss_fct(logits, targets)
        probas = torch.sigmoid(logits)
        loss = torch.where(targets >= 0.5, self.alpha * (1. - probas)
                           ** self.gamma * bce_loss, probas**self.gamma * bce_loss)
        return loss.mean()


class ImprovedPANNsLoss(nn.Module):
    def __init__(self,
                 output_key="logit",
                 framewise_output_key="framewise_output",
                 weights=[1, 1]):

        super().__init__()

        self.output_key = output_key
        self.framewise_output_key = framewise_output_key
        if output_key == "logit":
            self.normal_loss = nn.BCEWithLogitsLoss()
        else:
            self.normal_loss = nn.BCELoss()

        self.bce = nn.BCELoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]

        # for RuntimeError: CUDA error: device-side assert triggered
        input_ = torch.where(input_ > 1, torch.ones_like(input_), input_)

        target = target.float()

        framewise_output = input[self.framewise_output_key]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        normal_loss = self.normal_loss(input_, target)
        auxiliary_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss


class ImprovedFocalLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super().__init__()

        # self.focal = FocalLoss()
        self.focal = SimplerFocalLoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        normal_loss = self.focal(input_, target)
        auxiliary_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * normal_loss + self.weights[1] * auxiliary_loss
