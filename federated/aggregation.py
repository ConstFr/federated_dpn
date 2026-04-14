import logging

import torch
from models.aggregated_prior_net import (
    SimpleAggregatedPriorNet, 
    UncertaintyAggregatedPriorNet, 
    MajorityVoteAggregatedPriorNet,
    HardMajorityVoteAggregatedPriorNet,
    MostCertainAggregatedPriorNet
)

logger = logging.getLogger(__name__)


def dpn_aggregate(client_models, aggregation_type="simple", aggregation_uncertainty_measure="none", device=torch.device("cuda")):
    logger.info(
        f"Aggregating {len(client_models)} client models "
        f"with aggregation_type={aggregation_type} "
        f"and aggregation_uncertainty_measure={aggregation_uncertainty_measure}\n"
    )
    if aggregation_type == "simple":
        return SimpleAggregatedPriorNet(client_models).to(device)
    elif aggregation_type == "uncertainty":
        return UncertaintyAggregatedPriorNet(client_models, aggregation_uncertainty_measure).to(device)
    elif aggregation_type == "majority_voting":
        return MajorityVoteAggregatedPriorNet(client_models).to(device)
    elif aggregation_type == "hard_majority_voting":
        return HardMajorityVoteAggregatedPriorNet(client_models).to(device)
    elif aggregation_type == "most_certain":
        return MostCertainAggregatedPriorNet(client_models).to(device)
    raise NotImplementedError(f"Not implemented aggregation type: {aggregation_type}")