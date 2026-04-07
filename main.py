from data.mnist_fashionmnist import make_mnist_fashionmnist_datasets
from federated.runner import run_one_shot_federated_learning
import torch

def main():
    # Load datasets
    in_train_dataset, in_val_dataset, ood_train_dataset, ood_val_dataset = make_mnist_fashionmnist_datasets()

    # Run federated learning
    run_one_shot_federated_learning(
        in_train_dataset,
        ood_train_dataset,
        in_val_dataset,
        ood_val_dataset,
        num_clients=3,
        local_epochs=2,
        device=torch.device("cuda"),
    )

if __name__ == "__main__":
    main()
