import torch
import numpy as np


def compute_confidence(probs):
    """
    Confidence = max probability.
    probs: torch.Tensor or numpy array
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    return float(np.max(probs))


def compute_entropy(probs):
    """
    Shannon entropy of the probability distribution.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    # Avoid log(0)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)

    return float(-np.sum(probs * np.log(probs)))


def compute_brier_score(probs, true_idx):
    """
    Brier Score: sum((p_i - y_i)^2)
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    y = np.zeros_like(probs)
    y[true_idx] = 1.0

    return float(np.sum((probs - y) ** 2))
