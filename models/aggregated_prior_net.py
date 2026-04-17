from abc import ABC
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatedPriorNet(nn.Module, ABC):
    def __init__(self, client_models, eps=1e-8):
        super().__init__()
        self.client_models = nn.ModuleList(client_models)
        self.num_clients = len(client_models)
        self.eps = eps

    @abc.abstractmethod
    def aggregated_alphas(self, x):
        pass

    def forward(self, x):
        aggregated_alphas = self.aggregated_alphas(x)
        return aggregated_alphas

    def alphas(self, x):
        return self.aggregated_alphas(x)
    
class SimpleAggregatedPriorNet(AggregatedPriorNet):
    def aggregated_alphas(self, x):
        client_alphas = [model(x) for model in self.client_models]
        stacked = torch.stack(client_alphas, dim=0)

        aggregated_alphas = stacked.sum(dim=0) - (self.num_clients - 1)
        aggregated_alphas = torch.clamp(aggregated_alphas, min=self.eps)
        return aggregated_alphas


class UncertaintyAggregatedPriorNet(AggregatedPriorNet):
    def __init__(self, client_models, uncertainty_metric="none", eps=1e-8):
        super().__init__(client_models, eps)
        self.uncertainty_metric = uncertainty_metric

    def aggregated_alphas(self, x):
        clients_uncertainty = [model.uncertainty_metrics(x)[self.uncertainty_metric] for model in self.client_models]
        clients_alphas = [model(x) for model in self.client_models]

        stacked_uncertainty = torch.stack(clients_uncertainty, dim=0)
        stacked_alphas = torch.stack(clients_alphas, dim=0)

        weights = F.softmin(stacked_uncertainty, dim=0).unsqueeze(-1)
        
        aggregated_alphas = (weights * stacked_alphas).sum(dim=0)
        aggregated_alphas = torch.clamp(aggregated_alphas, min=self.eps)
        return aggregated_alphas
    

class MajorityVoteAggregatedPriorNet(AggregatedPriorNet):
    def aggregated_alphas(self, x):
        clients_alphas = [model(x) for model in self.client_models]
        stacked_alphas = torch.stack(clients_alphas, dim=0)
        probs = stacked_alphas / stacked_alphas.sum(dim=1, keepdim=True)

        votes = torch.argmax(probs, dim=-1)
        vote_counts = F.one_hot(votes, num_classes=probs.shape[-1]).sum(dim=0)

        aggregated_alphas = 1.0 + vote_counts.float()
        return aggregated_alphas
    

class HardMajorityVoteAggregatedPriorNet(AggregatedPriorNet):
    def aggregated_alphas(self, x):
        clients_alphas = [model(x) for model in self.client_models]
        stacked_alphas = torch.stack(clients_alphas, dim=0)
        probs = stacked_alphas / stacked_alphas.sum(dim=1, keepdim=True)

        votes = torch.argmax(probs, dim=-1)
        vote_counts = F.one_hot(votes, num_classes=probs.shape[-1]).sum(dim=0)
        winners = torch.argmax(vote_counts, dim=1)

        aggregated_alphas = torch.zeros_like(vote_counts).float().scatter_(1, winners.unsqueeze(1), 9) + 1
        return aggregated_alphas


class MostCertainAggregatedPriorNet(AggregatedPriorNet):
    def __init__(self, client_models, uncertainty_metric="mutual_information", eps=1e-8):
        super().__init__(client_models, eps)
        self.uncertainty_metric = uncertainty_metric

    def aggregated_alphas(self, x):
        clients_uncertainty = [model.uncertainty_metrics(x)[self.uncertainty_metric] for model in self.client_models]
        clients_alphas = [model(x) for model in self.client_models]

        stacked_uncertainty = torch.stack(clients_uncertainty, dim=0)
        stacked_alphas = torch.stack(clients_alphas, dim=0)

        most_certain_prediction = torch.argmin(stacked_uncertainty, dim=0)
        batch_indices = torch.arange(x.shape[0], device=stacked_alphas.device)

        aggregated_alphas = stacked_alphas[most_certain_prediction, batch_indices, :]
        aggregated_alphas = torch.clamp(aggregated_alphas, min=self.eps)
        return aggregated_alphas