import logging
from datetime import datetime
from pathlib import Path

import torch
import hydra

from datasets.mnist_fashionmnist import make_mnist_fashionmnist_datasets
from federated.runner import run_one_shot_federated_learning


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


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg):
    log_path = configure_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_path}")

    # Load datasets
    in_train_dataset, in_val_dataset, ood_train_dataset, ood_val_dataset = \
        make_mnist_fashionmnist_datasets(train_subset_size=cfg.experiment.train_subset_size)

    device = resolve_device(cfg.device)
    logger.info(f"Resolved execution device={device}")

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


if __name__ == "__main__":
    main()
