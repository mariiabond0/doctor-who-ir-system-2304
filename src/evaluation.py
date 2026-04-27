"""Evaluation utilities for Doctor Who IR metrics."""

from typing import Iterable


def compute_metrics(retrieved: Iterable[str], relevant: Iterable[str], top_k: int = 5):
    relevant_set = set(relevant)
    retrieved = list(retrieved)[:top_k]
    # retrieved_set = set(retrieved)
    overlap = sum(1 for doc in retrieved if doc in relevant_set)

    # overlap = len(retrieved_set & relevant_set)
    p_at_k = overlap / len(retrieved) if retrieved else 0.0
    r_at_k = overlap / len(relevant_set) if relevant_set else 0.0

    ap = 0.0
    num_rel = 0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            num_rel += 1
            ap += num_rel / rank
    ap /= len(relevant_set) if relevant_set else 0.0

    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            mrr = 1.0 / rank
            break

    return {
        "P@5": p_at_k,
        "R@5": r_at_k,
        "AP": ap,
        "MRR": mrr,
        "overlap": overlap,
    }
