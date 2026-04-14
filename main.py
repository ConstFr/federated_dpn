import logging
from datetime import datetime
import os
from pathlib import Path
import random

import torch
import numpy as np
import hydra

from datasets.mnist_fashionmnist import make_mnist_fashionmnist_datasets
from federated.runner import run_one_shot_federated_learning
from federated.runner import run_fedavg_federated_learning



def resolve_device(device_config: str) -> str:
    if device_config != "auto":
        return device_config
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_logging() -> Path:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    return log_path


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg):
    log_path = configure_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")

    random_seed = cfg.experiment.random_seed
    seed_everything(random_seed)

    # Load datasets
    in_train_dataset, in_val_dataset, ood_train_dataset, ood_val_dataset = \
        make_mnist_fashionmnist_datasets(train_subset_size=cfg.experiment.train_subset_size)

    device = resolve_device(cfg.device)
    logger.info(f"Resolved execution device={device}")

    if getattr(cfg.experiment, "federated_learning_type", "one_shot") == "one_shot":
        # Run federated learning
        run_one_shot_federated_learning(
            in_train_dataset,
            ood_train_dataset,
            in_val_dataset,
            ood_val_dataset,
            num_clients=cfg.experiment.num_clients,
            local_epochs=cfg.experiment.local_epochs,
            batch_size=cfg.experiment.batch_size,
            lr=cfg.experiment.lr,
            aggregation_type=cfg.experiment.aggregation_type,
            aggregation_uncertainty_measure=getattr(cfg.experiment, "aggregation_uncertainty_measure", "none"),
            device=device,
        )
    elif getattr(cfg.experiment, "federated_learning_type", "one_shot") == "fed_avg":
        if getattr(cfg.experiment, "num_rounds", "none") == "none":
            raise RuntimeError((
                    "Variable num_rounds describing number of rounds for federated learning is missing. "
                    "Precise num_rounds in the configuration."))

        run_fedavg_federated_learning(
            in_train_dataset,
            ood_train_dataset,
            in_val_dataset,
            ood_val_dataset,
            num_clients=cfg.experiment.num_clients,
            num_rounds=cfg.experiment.num_rounds,
            local_epochs=cfg.experiment.local_epochs,
            batch_size=cfg.experiment.batch_size,
            lr=cfg.experiment.lr,
            device=device,
        )


if __name__ == "__main__":
    main()
