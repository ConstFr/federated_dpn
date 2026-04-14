import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from evaluation.uncertainty import dirichlet_prior_network_uncertainty
from losses.dirichlet import DirichletKLLoss
from training.trainer import calc_accuracy_torch

logger = logging.getLogger(__name__)


def evaluate_accuracy(
    model: nn.Module,
    val_in_dataset,
    val_out_dataset,
    device: torch.device = torch.device("cuda"),
) -> float:
    id_loss, ood_loss, accuracy = 0.0, 0.0, 0.0
    id_alphas_all, ood_alphas_all = [], []
    model.eval()

    testloader = DataLoader(val_in_dataset, batch_size=64, shuffle=False)
    test_oodloader = DataLoader(val_out_dataset, batch_size=64, shuffle=False)

    id_criterion = DirichletKLLoss(target_concentration=100, concentration=1.0, reverse=True)
    ood_criterion = DirichletKLLoss(target_concentration=0.0, concentration=1.0, reverse=True)

    id_alpha_0, ood_alpha_0 = 0.0, 0.0
    with torch.no_grad():
        for data, ood_data in zip(testloader, test_oodloader):
            id_inputs, labels = data
            ood_inputs, _ = ood_data
            if device is not None:
                id_inputs, labels, ood_inputs = map(lambda x: x.to(device), (id_inputs, labels, ood_inputs))
            id_outputs = model(id_inputs)
            ood_outputs = model(ood_inputs)

            id_outputs_0 = torch.sum(id_outputs, dim=1, keepdim=True)
            probs = id_outputs / id_outputs_0

            accuracy += calc_accuracy_torch(probs, labels).item()
            id_loss += id_criterion(id_outputs, labels).item()
            ood_loss += ood_criterion(ood_outputs, None).item()
            id_alpha_0 += torch.mean(torch.sum(id_outputs, dim=1)).item()
            ood_alpha_0 += torch.mean(torch.sum(ood_outputs, dim=1)).item()
            id_alphas_all.append(id_outputs.cpu().numpy())
            ood_alphas_all.append(ood_outputs.cpu().numpy())

    id_alpha_0 = id_alpha_0 / len(testloader)
    ood_alpha_0 = ood_alpha_0 / len(test_oodloader)
    id_loss = id_loss / len(testloader)
    ood_loss = ood_loss / len(test_oodloader)
    accuracy = accuracy / len(testloader)

    id_alphas_all = np.concatenate(id_alphas_all, axis=0)
    ood_alphas_all = np.concatenate(ood_alphas_all, axis=0)
    alphas_all = np.concatenate([id_alphas_all, ood_alphas_all], axis=0)
    uncertainties = dirichlet_prior_network_uncertainty(alphas_all)["mutual_information"]

    in_domain = np.zeros(shape=[id_alphas_all.shape[0]], dtype=np.int32)
    ood_domain = np.ones(shape=[ood_alphas_all.shape[0]], dtype=np.int32)
    domain_labels = np.concatenate([in_domain, ood_domain], axis=0)
    auc = roc_auc_score(domain_labels, uncertainties)

    logger.info(
        f"Test ID Loss: {np.round(id_loss, 1)}; "
        f"Test OOD Loss: {np.round(ood_loss, 1)}; "
        f"Test Error: {np.round(100.0 * (1.0 - accuracy), 1)}%; "
        f"Test ID precision: {np.round(id_alpha_0, 1)}; "
        f"Test OOD precision: {np.round(ood_alpha_0, 1)}; "
        f"Test AUROC: {np.round(100.0 * auc, 1)}"
    )
    return accuracy * 100.0
