"""Feature composition (Appendix D.1): mean over features of the maximum cosine similarity between a
decoder vector and any other decoder vector. Higher values mean several features encode the same
information."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..models import BaseSAE


@torch.no_grad()
def max_cosine_similarities(sae: BaseSAE, verbose: bool = False) -> list[float]:
    """Maximum cosine similarity of every decoder vector with any other decoder vector."""
    W_dec = sae.W_dec
    results = []
    for feature in tqdm(range(W_dec.shape[0]), disable=not verbose, desc="composition"):
        similarities = F.cosine_similarity(W_dec[feature].unsqueeze(0), W_dec, dim=1)
        similarities[feature] = -float("inf")
        results.append(similarities.max().item())
    return results


def composition_score(sae: BaseSAE, verbose: bool = False) -> float:
    similarities = max_cosine_similarities(sae, verbose)
    return sum(similarities) / len(similarities)
