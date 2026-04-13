import math
import os
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from evaluation.uncertainty import dirichlet_prior_network_uncertainty


class TrainerWithOOD:
    def __init__(
        self,
        model,
        criterion,
        id_criterion,
        ood_criterion,
        train_dataset,
        ood_dataset,
        test_dataset,
        test_ood_dataset,
        optimizer,
        scheduler=None,
        optimizer_params: Dict[str, Any] = None,
        scheduler_params: Dict[str, Any] = None,
        batch_size=50,
        device=None,
        log_interval: int = 100,
        test_criterion=None,
        clip_norm=10.0,
        num_workers=2,
        pin_memory=False,
        checkpoint_path="./",
        checkpoint_steps=0,
    ):
        assert isinstance(model, nn.Module)
        print(f"{len(train_dataset)=}, {len(ood_dataset)=}, {len(test_dataset)=}, {len(test_ood_dataset)=}")
        assert len(train_dataset) == len(ood_dataset)
        assert len(test_dataset) == len(test_ood_dataset)

        self.model = model
        self.criterion = criterion
        self.id_criterion = id_criterion
        self.ood_criterion = ood_criterion
        self.device = device
        self.log_interval = log_interval
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.checkpoint_path = checkpoint_path
        self.checkpoint_steps = checkpoint_steps
        self.batch_size = batch_size
        self.clip_norm = clip_norm
        self.test_criterion = test_criterion if test_criterion is not None else nn.CrossEntropyLoss()

        if optimizer_params is None:
            optimizer_params = {}
        self.optimizer = optimizer(self.model.parameters(), **optimizer_params)

        if scheduler_params is None:
            scheduler_params = {}
        self.scheduler = scheduler(self.optimizer, **scheduler_params)

        self.trainloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.testloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.oodloader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=self.pin_memory,
        )
        self.test_oodloader = DataLoader(
            test_ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=self.pin_memory,
        )

        self.train_loss, self.train_accuracy, self.train_eval_steps = [], [], []
        self.test_loss, self.test_accuracy, self.test_eval_steps = [], [], []
        self.steps: int = 0

    def load_checkpoint(self, load_opt_state=False, load_scheduler_state=False, map_location=None):
        checkpoint_path = os.path.join(self.checkpoint_path, "checkpoint.tar")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.steps = checkpoint["steps"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.test_loss = checkpoint["test_loss"]

        if load_opt_state:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler_state:
            self.scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        print(f"Model restored from checkpoint {checkpoint_path}")

    def train(self, n_epochs=None, n_iter=None, resume=False):
        init_epoch = 0
        if n_epochs is None:
            assert isinstance(n_iter, int)
            n_epochs = math.ceil(n_iter / len(self.trainloader))
        else:
            assert isinstance(n_epochs, int)

        if resume:
            init_epoch = math.floor(self.steps / len(self.trainloader))

        for epoch in range(init_epoch, n_epochs):
            print(f"Training epoch: {epoch + 1} / {n_epochs}")
            start = time.time()
            self._train_single_epoch()
            self.test(time=time.time() - start)
            self.scheduler.step()

    def _train_single_epoch(self):
        self.model.train()

        accuracies = 0.0
        id_loss, ood_loss = 0.0, 0.0
        id_alpha_0, ood_alpha_0 = 0.0, 0.0
        for data, ood_data in zip(self.trainloader, self.oodloader):
            inputs, labels = data
            ood_inputs, _ = ood_data
            if self.device is not None:
                inputs, labels, ood_inputs = map(
                    lambda x: x.to(self.device, non_blocking=self.pin_memory),
                    (inputs, labels, ood_inputs),
                )

            self.optimizer.zero_grad()

            cat_inputs = torch.cat([inputs, ood_inputs], dim=1).view(
                torch.Size([2 * inputs.size()[0]]) + inputs.size()[1:]
            )
            alphas = self.model(cat_inputs).view([inputs.size()[0], -1])
            id_outputs, ood_outputs = torch.chunk(alphas, 2, dim=1)

            loss = self.criterion((id_outputs, ood_outputs), (labels, None))
            assert torch.all(torch.isfinite(loss)).item()

            id_loss += self.id_criterion(id_outputs, labels).item()
            ood_loss += self.ood_criterion(ood_outputs, None).item()

            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            self.steps += 1

            id_alpha_0 += torch.mean(torch.sum(id_outputs, dim=1)).item()
            ood_alpha_0 += torch.mean(torch.sum(ood_outputs, dim=1)).item()

            id_outputs_0 = torch.sum(id_outputs, dim=1, keepdim=True)
            probs = id_outputs / id_outputs_0
            accuracy = calc_accuracy_torch(probs, labels, self.device).item()
            accuracies += accuracy

            if self.steps % self.log_interval == 0:
                self.train_accuracy.append(accuracy)
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

        accuracies /= len(self.trainloader)
        id_loss /= len(self.trainloader)
        ood_loss /= len(self.trainloader)
        id_alpha_0 /= len(self.trainloader)
        ood_alpha_0 /= len(self.trainloader)

        print(
            f"Train ID Loss: {np.round(id_loss, 1)}; "
            f"Train OOD Loss: {np.round(ood_loss, 1)}; "
            f"Train Error: {np.round(100.0 * (1.0 - accuracies), 1)}; "
            f"Train ID precision: {np.round(id_alpha_0, 1)}; "
            f"Train OOD precision: {np.round(ood_alpha_0, 1)}"
        )

        with open("./LOG.txt", "a") as f:
            f.write(
                f"Train ID Loss: {np.round(id_loss, 1)}; "
                f"Train OOD Loss: {np.round(ood_loss, 1)}; "
                f"Train Error: {np.round(100.0 * (1.0 - accuracies), 1)}; "
                f"Train ID precision: {np.round(id_alpha_0, 1)}; "
                f"Train OOD precision: {np.round(ood_alpha_0, 1)}; "
            )

    def test(self, time):
        id_loss, ood_loss, accuracy = 0.0, 0.0, 0.0
        id_alphas_all, ood_alphas_all = [], []
        self.model.eval()
        id_alpha_0, ood_alpha_0 = 0.0, 0.0
        with torch.no_grad():
            for data, ood_data in zip(self.testloader, self.test_oodloader):
                id_inputs, labels = data
                ood_inputs, _ = ood_data
                if self.device is not None:
                    id_inputs, labels, ood_inputs = map(
                        lambda x: x.to(self.device, non_blocking=self.pin_memory),
                        (id_inputs, labels, ood_inputs),
                    )
                id_outputs = self.model(id_inputs)
                ood_outputs = self.model(ood_inputs)

                id_alpha_0 += torch.mean(torch.sum(id_outputs, dim=1)).item()
                ood_alpha_0 += torch.mean(torch.sum(ood_outputs, dim=1)).item()

                id_outputs_0 = torch.sum(id_outputs, dim=1, keepdim=True)
                probs = id_outputs / id_outputs_0

                accuracy += calc_accuracy_torch(probs, labels).item()
                id_loss += self.id_criterion(id_outputs, labels).item()
                ood_loss += self.ood_criterion(ood_outputs, None).item()

                id_alphas_all.append(id_outputs.cpu().numpy())
                ood_alphas_all.append(ood_outputs.cpu().numpy())

        id_alpha_0 = id_alpha_0 / len(self.testloader)
        ood_alpha_0 = ood_alpha_0 / len(self.test_oodloader)
        id_loss = id_loss / len(self.testloader)
        ood_loss = ood_loss / len(self.testloader)
        accuracy = accuracy / len(self.testloader)

        id_alphas_all = np.concatenate(id_alphas_all, axis=0)
        ood_alphas_all = np.concatenate(ood_alphas_all, axis=0)
        alphas_all = np.concatenate([id_alphas_all, ood_alphas_all], axis=0)
        
        uncertainties = dirichlet_prior_network_uncertainty(alphas_all)["mutual_information"]

        in_domain = np.zeros(shape=[id_alphas_all.shape[0]], dtype=np.int32)
        ood_domain = np.ones(shape=[ood_alphas_all.shape[0]], dtype=np.int32)
        domain_labels = np.concatenate([in_domain, ood_domain], axis=0)
        auc = roc_auc_score(domain_labels, uncertainties)

        print(
            f"Test ID Loss: {np.round(id_loss, 1)}; "
            f"Test OOD Loss: {np.round(ood_loss, 1)}; "
            f"Test Error: {np.round(100.0 * (1.0 - accuracy), 1)}%; "
            f"Test ID precision: {np.round(id_alpha_0, 1)}; "
            f"Test OOD precision: {np.round(ood_alpha_0, 1)}; "
            f"Test AUROC: {np.round(100.0 * auc, 1)}; "
            f"Time Per Epoch: {np.round(time / 60.0, 1)} min"
        )

        with open("./LOG.txt", "a") as f:
            f.write(
                f"Test ID Loss: {np.round(id_loss, 1)}; "
                f"Test OOD Loss: {np.round(ood_loss, 1)}; "
                f"Test Error: {np.round(100.0 * (1.0 - accuracy), 1)}; "
                f"Test ID precision: {np.round(id_alpha_0, 1)}; "
                f"Test OOD precision: {np.round(ood_alpha_0, 1)}; "
                f"Test AUROC: {np.round(100.0 * auc, 1)}; "
                f"Time Per Epoch: {np.round(time / 60.0, 1)} min.\n"
            )
        self.test_loss.append(id_loss)
        self.test_accuracy.append(accuracy)
        self.test_eval_steps.append(self.steps)


def calc_accuracy_torch(y_probs, y_true, device=None, weights=None):
    if weights is None:
        if device is None:
            return torch.mean((torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float32))
        return torch.mean((torch.argmax(y_probs, dim=1) == y_true).to(device, torch.float32))

    if device is None:
        weights.to(dtype=torch.float32)
        return torch.mean(weights * (torch.argmax(y_probs, dim=1) == y_true).to(dtype=torch.float32))

    weights.to(device=device, dtype=torch.float32)
    return torch.mean(
        weights * (torch.argmax(y_probs, dim=1) == y_true).to(device=device, dtype=torch.float32)
    )

