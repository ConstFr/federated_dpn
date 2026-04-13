from pathlib import Path

import torch
from torch import optim

from losses.dirichlet import DirichletKLLoss, PriorNetMixedLoss
from training.trainer import TrainerWithOOD


def train_dpn(
    model,
    train_dataset,
    val_dataset,
    ood_train_dataset,
    ood_val_dataset,
    data_path="./data",
    n_epochs=100,
    lr=1e-3,
    model_dir="./model",
    lr_decay=0.2,
    lrc=None,
    target_concentration=1e2,
    concentration=1.0,
    gamma=1.0,
    weight_decay=0.0,
    batch_size=128,
    reverse_KL=True,
    gpu=None,
    optimizer="SGD",
    clip_norm=10.0,
    checkpoint_path=None,
    id_ratio=1.0,
    device="cuda",
):
    del data_path, optimizer

    if gpu is None:
        gpu = [0]
    if lrc is None:
        lrc = []

    model_dir = Path(model_dir)
    if checkpoint_path is None:
        checkpoint_path = model_dir / "model"

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        raise RuntimeError("CUDA or MPS is required but no GPU is available.")

    model.to(device)

    id_criterion = DirichletKLLoss(
        target_concentration=target_concentration,
        concentration=concentration,
        reverse=reverse_KL,
    )
    ood_criterion = DirichletKLLoss(
        target_concentration=0.0,
        concentration=concentration,
        reverse=reverse_KL,
    )
    criterion = PriorNetMixedLoss([id_criterion, ood_criterion], mixing_params=[1.0, gamma])

    optimizer_cls = optim.SGD
    optimizer_params = {"lr": lr, "momentum": 0.9, "weight_decay": weight_decay}
    adjusted_lrc = [int(m / id_ratio) for m in lrc]

    trainer = TrainerWithOOD(
        model=model,
        criterion=criterion,
        id_criterion=id_criterion,
        ood_criterion=ood_criterion,
        test_criterion=criterion,
        ood_dataset=ood_train_dataset,
        test_ood_dataset=ood_val_dataset,
        train_dataset=train_dataset,
        test_dataset=val_dataset,
        optimizer=optimizer_cls,
        device=device,
        checkpoint_path=checkpoint_path,
        scheduler=optim.lr_scheduler.MultiStepLR,
        optimizer_params=optimizer_params,
        scheduler_params={"milestones": adjusted_lrc, "gamma": lr_decay},
        batch_size=batch_size,
        clip_norm=clip_norm,
    )

    trainer.train(n_epochs)
    return trainer.model

