"""Чистые функции метрик — отдельно от пайплайна, чтобы покрыть тестами."""

import numpy as np
from scipy import sparse


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not recommended:
        return 0.0
    hits = sum(1 for g in recommended[:k] if g in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for g in recommended[:k] if g in relevant)
    return hits / len(relevant)


def catalog_coverage(all_recommended: set[int], catalog_size: int) -> float:
    if catalog_size == 0:
        return 0.0
    return len(all_recommended) / catalog_size


def intra_list_diversity(vectors: sparse.csr_matrix) -> float:
    """1 - средняя попарная косинусная близость; вектора должны быть l2-нормированы."""
    n = vectors.shape[0]
    if n < 2:
        return 0.0
    sim = (vectors @ vectors.T).toarray()
    pair_sum = (sim.sum() - np.trace(sim)) / 2
    n_pairs = n * (n - 1) / 2
    return float(1 - pair_sum / n_pairs)
