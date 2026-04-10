import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Prior network with CNN backbone for MNIST-like classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 64 * 3 * 3)
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

    def diffenrential_entropy(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=1,
        ) - torch.lgamma(alpha0)

    def epkl(self, x):
        alphas = self.alphas(x)
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return alphas.size()[1] / alpha0

    def confidence(self, x):
        return torch.max(F.softmax(self.forward(x)), dim=1)

    @staticmethod
    def expected_entropy_from_alphas(alphas, alpha0=None):
        if alpha0 is None:
            alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        return -torch.sum(
            (alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
            dim=1,
        )

    @staticmethod
    def uncertainty_metrics(alphas):
        alpha0 = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / alpha0
        epkl = (alphas.size()[1] - 1.0) / alphas
        dentropy = torch.sum(
            torch.lgamma(alphas) - (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0)),
            dim=1,
        ) - torch.lgamma(alpha0)
        conf = torch.max(probs, dim=1)
        expected_entropy = -torch.sum(
            (alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1)),
            dim=1,
        )
        entropy_of_exp = categorical_entropy_torch(probs)
        mutual_info = entropy_of_exp - expected_entropy
        return {
            "confidence": conf,
            "entropy_of_expected": entropy_of_exp,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_info,
            "EPKL": epkl,
            "differential_entropy": torch.squeeze(dentropy),
        }


class AggregatedPriorNet(nn.Module):
    def __init__(self, client_models, eps=1e-8):
        super().__init__()
        self.client_models = nn.ModuleList(client_models)
        self.num_clients = len(client_models)
        self.eps = eps

    def aggregated_alpha(self, x):
        client_alphas = [model(x) for model in self.client_models]
        stacked = torch.stack(client_alphas, dim=0)

        aggregated_alpha = stacked.sum(dim=0) - (self.num_clients - 1)
        aggregated_alpha = torch.clamp(aggregated_alpha, min=self.eps)
        return aggregated_alpha

    def forward(self, x):
        aggregated_alpha = self.aggregated_alpha(x)
        return aggregated_alpha

    def alphas(self, x):
        return self.aggregated_alpha(x)



def categorical_entropy_torch(probs, dim=1, keepdim=False):
    log_probs = torch.log(probs)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    return -torch.sum(probs * log_probs, dim=dim, keepdim=keepdim)
