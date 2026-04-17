import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTSimpleCNN(nn.Module):
    """CNN backbone for MNIST-like classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(4 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 4 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CIFAR10SimpleCNN(nn.Module):
    """CNN backbone for CIFAR-10-like classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PriorNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return 1 + F.softplus(self.model(x))

    def alphas(self, x):
        return self.forward(x)

    def mutual_information(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        expected_entropy = self.expected_entropy_from_alphas(alphas, alpha0)
        entropy_of_exp = categorical_entropy_torch(probs)
        return entropy_of_exp - expected_entropy

    def entropy_of_expected(self, x):
        # probs = F.softmax(self.model(x), dim=1)
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        return categorical_entropy_torch(probs)

    def expected_entropy(self, x):
        alphas = self.alphas(x)
        return self.expected_entropy_from_alphas(alphas)

    def differential_entropy(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=1,
        ) - torch.lgamma(alpha0)

    def epkl(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return torch.squeeze((alphas.shape[1] - 1.0) / alpha0)

    def confidence(self, x):
        return torch.max(F.softmax(self.forward(x), dim=-1), dim=1).values

    @staticmethod
    def expected_entropy_from_alphas(alphas, alpha0=None):
        if alpha0 is None:
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return -torch.sum(
            (alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
            dim=1,
        )

    def uncertainty_metrics(self, x):
        return {
            "confidence": self.confidence(x),
            "entropy_of_expected": self.entropy_of_expected(x),
            "expected_entropy": self.expected_entropy(x),
            "mutual_information": self.mutual_information(x),
            "EPKL": self.epkl(x),
            "differential_entropy": self.differential_entropy(x).squeeze(),
        }


def categorical_entropy_torch(probs, dim=1, keepdim=False):
    log_probs = torch.log(probs)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    return -torch.sum(probs * log_probs, dim=dim, keepdim=keepdim)
