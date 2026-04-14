import numpy as np
from scipy.special import digamma, gammaln


def dirichlet_prior_network_uncertainty(alphas, epsilon=1e-10):
    alpha0 = np.sum(alphas, axis=1, keepdims=True)
    probs = alphas / alpha0

    conf = np.max(probs, axis=1)

    entropy_of_exp = -np.sum(probs * np.log(probs + epsilon), axis=1)
    expected_entropy = \
        -np.sum((alphas / alpha0) * (digamma(alphas + 1) - digamma(alpha0 + 1.0)), axis=1)

    mutual_info = entropy_of_exp - expected_entropy

    epkl = np.squeeze((alphas.shape[1] - 1.0) / alpha0)
    dentropy = np.sum(
        gammaln(alphas) - (alphas - 1.0) * (digamma(alphas) - digamma(alpha0)),
        axis=1,
        keepdims=True,
    ) - gammaln(alpha0)
    
    return {
        "confidence": conf,
        "entropy_of_expected": entropy_of_exp,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_info,
        "EPKL": epkl,
        "differential_entropy": np.squeeze(dentropy),
    }

