import torch

from models.prior_net import AggregatedPriorNet


def dpn_aggregate(client_models, device=torch.device("cuda")):
    return AggregatedPriorNet(client_models).to(device)

