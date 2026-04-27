"""Doctor Who Information Retrieval System - Main evaluation script."""

import json
import logging
import sqlite3

import faiss
from sentence_transformers import SentenceTransformer

import config
from src.bm_25 import bm25_search_sqlite
from src.boolean_search import boolean_search_sqlite
from src.evaluation import compute_metrics
from src.semantic_search import semantic_search_sqlite

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

with open(config.QUERIES_PATH, "r", encoding="utf-8") as f:
    second_example_17_queries = json.load(f)

QUERIES = second_example_17_queries["queries"]
ANSWERS = second_example_17_queries["answers"]


def load_faiss_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}


def faiss_query(query, index, mapping, model, top_k=config.DEFAULT_TOP_K):
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_emb.reshape(1, -1), top_k)
    return [mapping[i] for i in indices[0] if int(i) in mapping]


def fused_query(query, conn, top_k=config.DEFAULT_TOP_K, k=60, candidate_k=50):
    # Get results from BM25 (sparse) and semantic (dense)
    bm25_results = bm25_search_sqlite(query, conn, top_n=candidate_k)
    semantic_results = semantic_search_sqlite(query, conn, top_n=candidate_k)

    # Create rank dictionaries
    bm25_ranks = {doc: rank + 1 for rank, doc in enumerate(bm25_results)}
    semantic_ranks = {doc: rank + 1 for rank, doc in enumerate(semantic_results)}
    
    # Combine using Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    all_docs = set(bm25_results) | set(semantic_results)
    for doc in all_docs:
        rrf_sparse = 1 / (k + bm25_ranks.get(doc, candidate_k + 1))
        rrf_dense = 1 / (k + semantic_ranks.get(doc, candidate_k + 1))
        fused_scores[doc] = rrf_sparse + rrf_dense
    
    # Sort by fused score descending
    sorted_docs = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return sorted_docs[:top_k]


def evaluate_method(name, query_fn):
    print(f"\n--- {name} ---")
    metrics = []
    for i, query in enumerate(QUERIES):
        retrieved = query_fn(query)
        result = compute_metrics(retrieved, ANSWERS[i], top_k=config.DEFAULT_TOP_K)
        metrics.append(result)
        print(
            f"Query {i+1}: Overlap {result['overlap']}/{len(ANSWERS[i])}, "
            f"P@5 {result['P@5']:.2f}, R@5 {result['R@5']:.2f}, "
            f"AP {result['AP']:.2f}, MRR {result['MRR']:.2f}"
        )

    mean_p5 = sum(m["P@5"] for m in metrics) / len(metrics)
    mean_r5 = sum(m["R@5"] for m in metrics) / len(metrics)
    mean_ap = sum(m["AP"] for m in metrics) / len(metrics)
    mean_mrr = sum(m["MRR"] for m in metrics) / len(metrics)
    print(f"Mean P@5: {mean_p5:.2f}, Mean R@5: {mean_r5:.2f}, MAP: {mean_ap:.2f}, MRR: {mean_mrr:.2f}")
    return metrics


def save_results(summary):
    import pandas as pd

    df = pd.DataFrame(summary)
    df.to_csv(config.RESULTS_CSV_PATH, index=False)
    logger.info(f"Saved evaluation results to {config.RESULTS_CSV_PATH}")


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    logger.info(f"Connected to database: {config.DB_PATH}")

    try:
        faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        faiss_index.hnsw.efSearch = config.FAISS_EF_SEARCH
        faiss_mapping = load_faiss_mapping(config.FAISS_MAPPING_PATH)
        faiss_model = SentenceTransformer(config.MODEL_NAME)
    except Exception as error:
        logger.error("Failed to load FAISS resources: %s", error)
        raise

    results = []
    for name, fn in [
        ("Boolean Search", lambda q: boolean_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("BM25 Search", lambda q: bm25_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("Semantic Search", lambda q: semantic_search_sqlite(q, conn, top_n=config.DEFAULT_TOP_K)),
        ("FAISS Semantic Search", lambda q: faiss_query(q, faiss_index, faiss_mapping, faiss_model, top_k=config.DEFAULT_TOP_K)),
        ("Fused Search", lambda q: fused_query(q, conn, top_k=config.DEFAULT_TOP_K)),
    ]:
        method_metrics = evaluate_method(name, fn)
        for i, metrics in enumerate(method_metrics):
            row = {
                # "query": QUERIES[i],
                "method": name,
                # "expected": ";".join(ANSWERS[i]),
                "overlap": metrics["overlap"],
                "P@5": metrics["P@5"],
                "R@5": metrics["R@5"],
                "AP": metrics["AP"],
                "MRR": metrics["MRR"],
            }
            results.append(row)

    save_results(results)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
