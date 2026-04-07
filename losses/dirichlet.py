import numpy as np
import torch


class PriorNetMixedLoss:
    def __init__(self, losses, mixing_params):
        assert isinstance(losses, (list, tuple))
        assert isinstance(mixing_params, (list, tuple, np.ndarray))
        assert len(losses) == len(mixing_params)
        self.losses = losses
        self.mixing_params = mixing_params if mixing_params is not None else [1.0] * len(self.losses)

    def __call__(self, logits_list, labels_list):
        return self.forward(logits_list, labels_list)

    def forward(self, logits_list, labels_list):
        total_loss = []
        target_concentration = 0.0
        for i, loss in enumerate(self.losses):
            if loss.target_concentration > target_concentration:
                target_concentration = loss.target_concentration
            total_loss.append(loss(logits_list[i], labels_list[i]) * self.mixing_params[i])
        total_loss = torch.stack(total_loss, dim=0)
        return torch.sum(total_loss) / target_concentration


class DirichletKLLoss:
    def __init__(self, target_concentration=1e3, concentration=1.0, reverse=True):
        self.target_concentration = float(target_concentration)
        self.concentration = concentration
        self.reverse = reverse

    def __call__(self, logits, labels, reduction="mean"):
        alphas = torch.exp(logits)
        return self.forward(alphas, labels, reduction=reduction)

    def forward(self, alphas, labels, reduction="mean"):
        loss = self.compute_loss(alphas, labels)

        if reduction == "mean":
            return torch.mean(loss)
        if reduction == "none":
            return loss
        raise NotImplementedError

    def compute_loss(self, alphas, labels=None):
        target_alphas = torch.ones_like(alphas) * self.concentration
        if labels is not None:
            target_alphas += \
                torch.zeros_like(alphas).scatter_(1, labels[:, None], self.target_concentration)
        
        if self.reverse:
            loss = dirichlet_reverse_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        else:
            loss = dirichlet_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        return loss


def dirichlet_kl_divergence(alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8):
    if precision is None:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if target_precision is None:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)

    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    assert torch.all(torch.isfinite(precision_term)).item()
    alphas_term = torch.sum(
        torch.lgamma(alphas + epsilon) - torch.lgamma(target_alphas + epsilon)
        + (target_alphas - alphas)
        * (torch.digamma(target_alphas + epsilon) - torch.digamma(target_precision + epsilon)),
        dim=1,
        keepdim=True,
    )

    assert torch.all(torch.isfinite(alphas_term)).item()

    cost = torch.squeeze(precision_term + alphas_term)
    return cost


def dirichlet_reverse_kl_divergence(alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8):
    return dirichlet_kl_divergence(
        alphas=target_alphas,
        target_alphas=alphas,
        precision=target_precision,
        target_precision=precision,
        epsilon=epsilon,
    )

