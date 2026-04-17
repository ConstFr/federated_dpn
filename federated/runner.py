import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np

from evaluation.metrics import evaluate_accuracy, form_report
from federated.aggregation import dpn_aggregate
from models.prior_net import PriorNet, MNISTSimpleCNN, CIFAR10SimpleCNN
from training.train_prior_net import train_dpn

logger = logging.getLogger(__name__)


def partition_data_iid(dataset, num_clients):
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    return [list(idx) for idx in split_indices]


def partition_data_label_skew_simple(dataset, num_clients=5, num_labels=10, main_ratio=0.9):
    # thanks chatgpt
    if isinstance(dataset, Subset):
        base_targets = np.array(dataset.dataset.targets)
        subset_indices = np.array(dataset.indices)
        targets = base_targets[subset_indices]

        # local indices: 0 ... len(dataset)-1
        usable_indices = np.arange(len(dataset))
    else:
        targets = np.array(dataset.targets)
        usable_indices = np.arange(len(dataset))

    # group LOCAL dataset indices by label
    label_indices = {
        label: usable_indices[targets == label].tolist()
        for label in range(num_labels)
    }

    for label in range(num_labels):
        np.random.shuffle(label_indices[label])

    samples_per_client = len(dataset) // num_clients
    labels_per_client = num_labels // num_clients
    all_labels = list(range(num_labels))

    client_indices = []

    for client_id in range(num_clients):
        main_labels = list(range(client_id * labels_per_client, (client_id + 1) * labels_per_client))
        other_labels = [l for l in all_labels if l not in main_labels]

        client_data = []
        for _ in range(samples_per_client):
            candidate_labels = main_labels if np.random.rand() < main_ratio else other_labels
            available_labels = [l for l in candidate_labels if len(label_indices[l]) > 0]

            if not available_labels:
                available_labels = [l for l in all_labels if len(label_indices[l]) > 0]

            if not available_labels:
                break

            chosen_label = np.random.choice(available_labels)
            client_data.append(label_indices[chosen_label].pop())

        client_indices.append(client_data)

    for i, idxs in enumerate(client_indices):
        labels, counts = np.unique(targets[idxs], return_counts=True)
        logger.info(f"Client {i}: {dict(zip(labels, counts))}")

    return client_indices


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
    test: bool = False
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
        test=test
    )


def run_one_shot_federated_learning(
    in_dataset,
    ood_dataset,
    val_in_dataset,
    val_ood_dataset,
    model,
    num_clients: int = 10,
    local_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    aggregation_type: str = "simple",
    aggregation_uncertainty_measure: str = "none",
    partition_strategy: str = "iid", 
    device: torch.device = torch.device("cuda"),
):
    logger.info(
        f"Starting one-shot federated learning with num_clients={num_clients} "
        f"local_epochs={local_epochs} batch_size={batch_size} lr={lr} "
        f"aggregation_type={aggregation_type} aggregation_uncertainty_metric={aggregation_uncertainty_measure}\n"
    )
    if partition_strategy == "iid":
        client_indices = partition_data_iid(in_dataset, num_clients)
    elif partition_strategy == "label_skew":
        client_indices = partition_data_label_skew_simple(in_dataset, num_clients=num_clients, num_labels=len(in_dataset.dataset.classes), main_ratio=0.9)
    else:
        raise RuntimeError(f"Unsupported partition strategy: {partition_strategy}")

    client_models = []
    local_metrics = []

    for client_id in range(num_clients):
        logger.info(
            f"Training client {client_id + 1}/{num_clients} with "
            f"{len(client_indices[client_id])} in-distribution samples and "
            f"{len(client_indices[client_id])} OOD samples"
        )

        local_cnn = model().to(device)
        local_model = PriorNet(local_cnn).to(device)

        client_in_dataset = Subset(in_dataset, client_indices[client_id])
        client_ood_dataset = Subset(ood_dataset, client_indices[client_id])

        new_state, metrics = local_train(
            local_model,
            client_in_dataset,
            val_in_dataset,
            client_ood_dataset,
            val_ood_dataset,
            epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            test=True
        )

        client_models.append(new_state)
        local_metrics.append(metrics)

    if aggregation_type == "all":
        aggregation_config = [
            ("uncertainty", "confidence"), 
            ("uncertainty", "expected_entropy"), 
            ("hard_majority_voting", ""), 
            ("majority_voting", ""), 
            ("most_certain", ""), 
            ("uncertainty", "mutual_information"), 
            ("simple", "none")
        ]
    else:
        aggregation_config = [(aggregation_type, aggregation_uncertainty_measure)]

    aggregated_test_accuracies = {}
    for setting in aggregation_config:
        aggregation_type_, aggregation_uncertainty_measure_ = setting
        aggregated_model = dpn_aggregate(client_models, aggregation_type_, aggregation_uncertainty_measure_, device=device)
        logger.info(f"Evaluating aggregated model with {aggregation_type_} aggregation and {aggregation_uncertainty_measure_} uncertainty measure")
        aggregated_test_accuracy = evaluate_accuracy(aggregated_model, val_in_dataset, val_ood_dataset, device=device)
        logger.info(f"Validation accuracy = {aggregated_test_accuracy:.1f}%")
        aggregated_test_accuracies[setting] = aggregated_test_accuracy

    aggregated_test_accuracies["best_local_test_accuracy"] = max(m["accuracy"] for m in local_metrics)

    return aggregated_test_accuracies


def run_fedavg_federated_learning(
    in_dataset,
    ood_dataset,
    val_in_dataset,
    val_ood_dataset,
    model,
    num_clients: int = 10,
    num_rounds: int = 10,
    local_epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    partition_strategy: str = "iid",
    device: torch.device = torch.device("cuda"),
):
    logger.info(
        f"Starting FedAvg federated learning with num_clients={num_clients} "
        f"num_rounds={num_rounds} local_epochs={local_epochs} "
        f"batch_size={batch_size} lr={lr}\n"
    )

    if partition_strategy == "iid":
        client_indices = partition_data_iid(in_dataset, num_clients)
    elif partition_strategy == "label_skew":
        client_indices = partition_data_label_skew_simple(in_dataset, num_clients=num_clients, num_labels=in_dataset.classes, main_ratio=0.9)
    else:   
        raise RuntimeError(f"Unsupported partition strategy: {partition_strategy}")
    global_cnn = model().to(device)
    global_model = PriorNet(global_cnn).to(device)

    for round_id in range(num_rounds):
        logger.info(f"Starting round {round_id + 1}/{num_rounds}")

        client_states = []
        local_metrics = []

        for client_id in range(num_clients):
            logger.info(
                f"Training client {client_id + 1}/{num_clients} with "
                f"{len(client_indices[client_id])} in-distribution samples and "
                f"{len(client_indices[client_id])} OOD samples"
            )

            client_in_dataset = Subset(in_dataset, client_indices[client_id])
            client_ood_dataset = Subset(ood_dataset, client_indices[client_id])

            local_cnn = model().to(device)
            local_model = PriorNet(local_cnn).to(device)

            # Initialize local model from current global model
            local_model.load_state_dict(global_model.state_dict())

            new_state, metrics = local_train(
                local_model,
                client_in_dataset,
                val_in_dataset,
                client_ood_dataset,
                val_ood_dataset,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                test=True
            )
            
            local_metrics.append(metrics)
            client_states.append(new_state.state_dict())

        # Standard FedAvg: uniform averaging of client parameters
        avg_state = {}
        for key in client_states[0].keys():
            avg_state[key] = torch.stack(
                [client_state[key].float().to(device) for client_state in client_states],
                dim=0,
            ).mean(dim=0)

        global_model.load_state_dict(avg_state)

    logger.info("Evaluating global model")
    test_accuracy = evaluate_accuracy(
        global_model,
        val_in_dataset,
        val_ood_dataset,
        device=device,
    )
    logger.info(
        f"Round {round_id + 1}/{num_rounds} validation accuracy = {test_accuracy:.1f}%"
    )
    report = form_report(local_metrics, test_accuracy)
    return report