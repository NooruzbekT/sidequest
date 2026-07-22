import numpy as np
from scipy import sparse

from metrics import catalog_coverage, intra_list_diversity, precision_at_k, recall_at_k


def test_precision_counts_hits_in_top_k():
    assert precision_at_k([1, 2, 3, 4], relevant={2, 4, 9}, k=4) == 0.5


def test_precision_ignores_items_beyond_k():
    assert precision_at_k([1, 2, 3], relevant={3}, k=2) == 0.0


def test_precision_empty_recommendations():
    assert precision_at_k([], relevant={1}, k=10) == 0.0


def test_recall_is_share_of_relevant_found():
    assert recall_at_k([1, 2, 3], relevant={1, 2, 8, 9}, k=10) == 0.5


def test_recall_empty_relevant_set():
    assert recall_at_k([1, 2], relevant=set(), k=10) == 0.0


def test_coverage():
    assert catalog_coverage({1, 2, 3}, catalog_size=10) == 0.3
    assert catalog_coverage(set(), catalog_size=0) == 0.0


def test_diversity_identical_items_is_zero():
    v = sparse.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert intra_list_diversity(v) == 0.0


def test_diversity_orthogonal_items_is_one():
    v = sparse.csr_matrix(np.eye(2))
    assert intra_list_diversity(v) == 1.0
