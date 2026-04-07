from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np

from evaluation.metrics import evaluate_accuracy
from federated.aggregation import dpn_aggregate
from models.prior_net import PriorNet, SimpleCNN
from training.train_prior_net import train_dpn


def partition_data_iid(dataset, num_clients):
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    return [list(idx) for idx in split_indices]


def local_train(
    model: nn.Module,
    in_dataset,
    val_in_dataset,
    ood_dataset,
    val_ood_dataset,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
):
    model.train()
    return train_dpn(
        model,
        in_dataset,
        val_in_dataset,
        ood_dataset,
        val_ood_dataset,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )


def run_one_shot_federated_learning(
    in_dataset,
    ood_dataset,
    val_in_dataset,
    val_ood_dataset,
    num_clients: int = 10,
    local_epochs: int = 10,
    device: torch.device = torch.device("cuda"),
):
    client_indices = partition_data_iid(in_dataset, num_clients)

    client_models = []

    for client_id in range(num_clients):
        print(f"Training client {client_id + 1}/{num_clients}...")

        local_cnn = SimpleCNN().to(device)
        local_model = PriorNet(local_cnn).to(device)

        client_in_dataset = Subset(in_dataset, client_indices[client_id])
        client_ood_dataset = Subset(ood_dataset, client_indices[client_id])

        new_state = local_train(
            local_model,
            client_in_dataset,
            val_in_dataset,
            client_ood_dataset,
            val_ood_dataset,
            epochs=local_epochs,
            device=device,
        )

        client_models.append(new_state)

    aggregated_model = dpn_aggregate(client_models, device=device)
    print("Evaluating aggregated model")
    test_accuracy = evaluate_accuracy(aggregated_model, val_in_dataset, val_ood_dataset, device=device)
    print(f"Validation Accuracy = {test_accuracy:.1f}%")
