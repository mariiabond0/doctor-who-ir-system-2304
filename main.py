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

QUERIES = [
    "Doctor fights with Weeping Angels and wants to save Amy",
    "Doctor and Clara in nineteenth century",
    "Doctor meets River Song for the first time",
    "Doctor and Donna encounter Daleks and Davros",
    "Doctor and Rose end up in a parallel universe",
    "Doctor meets van Gogh",
    "Doctor and Martha face time paradox creatures or similar threats",
    "Doctor and Bill encounter Cybermen",
    "Paternoster Gang and Doctor and Vastra and Strax",
    "Doctor and Rose meets her father Pete Tyler",
    "Rory dies",
    "Rose manequins aliens",
    "neighbor upstairs Amy not there",
    "cat nuns",
    "Doctor regenerates",
    "Doctor encounters Silence",
    "Doctor in Italy"
]

ANSWERS = [
    ["5x4", "5x5", "7x5", "6x11", "3x10"],
    ["7x6", "7x12", "7x8", "1x3", "7x10"],
    ["4x8", "4x9", "5x4", "5x5", "6x1"],
    ["4x12", "4x13", "2x12", "2x13", "9x1"],
    ["2x5", "2x6", "2x12", "2x13", "4x11"],
    ["5x10", "5x1", "5x12", "5x13", "1x3"],
    ["3x10", "3x8", "3x9", "3x11", "3x12"],
    ["10x11", "10x12", "2x5", "2x6", "8x12"],
    ["7x6", "7x11", "7x13", "8x1", "6x7"],
    ["1x8", "2x5", "2x6", "1x13", "4x11"],
    ["5x9", "5x12", "7x5", "5x13", "5x10"],
    ["1x1", "1x2", "2x0", "2x7", "1x4"],
    ["5x11", "6x12", "6x13", "6x1", "6x2"],
    ["2x1", "2x2", "2x3", "2x4", "2x7"],
    ["1x13", "4x13", "7x14", "10x12", "2x0"],
    ["6x1", "6x2", "6x13", "6x11", "5x1"],
    ["5x6", "4x2", "10x6", "5x12", "5x13"],
]


def load_faiss_mapping(mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}


def faiss_query(query, index, mapping, model, top_k=config.DEFAULT_TOP_K):
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    distances, indices = index.search(query_emb.reshape(1, -1), top_k)
    return [mapping[i] for i in indices[0] if int(i) in mapping]


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
    ]:
        method_metrics = evaluate_method(name, fn)
        for i, metrics in enumerate(method_metrics):
            row = {
                "query": QUERIES[i],
                "method": name,
                "expected": ";".join(ANSWERS[i]),
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
