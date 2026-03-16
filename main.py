import os
import json
from src.search import boolean_search
from src.bm_25 import build_bm25_corpus, bm25_search
from src.semantic import encode_corpus, semantic_search

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "dw_data")
CORPUS_PATH = os.path.join(DATA_DIR, "document_corpus_dw.json")
INDEX_PATH = os.path.join(DATA_DIR, "inverted_index.json")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    document_corpus = json.load(f)

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

# Queries and expected answers
queries = [
    "Fights with Weeping Angels and wants to save Amy",
    "Clara in nineteenth century",
    "Doctor meet River Song first time",
    "Donna encounter Daleks and Davros",
    "Rose end up in a parallel universe",
    "Doctor meets van Gogh",
    "Martha face time paradox creatures or similar threats",
    "Bill encounter Cybermen",
    "Paternoster Gang and Doctor and Vastra and Strax",
    "Rose meets her father Pete Tyler"
]

answers = [
    ["5x4", "5x5", "7x5", "6x11", "3x10"],
    ["7x6", "7x12", "7x8", "1x3", "7x10"],
    ["4x8", "4x9", "5x4", "5x5", "6x1"],
    ["4x12", "4x13", "2x12", "2x13", "9x1"],
    ["2x5", "2x6", "2x12", "2x13", "4x11"],
    ["5x10", "5x1", "5x12", "5x13", "1x3"],
    ["3x10", "3x8", "3x9", "3x11", "3x12"],
    ["10x11", "10x12", "2x5", "2x6", "8x12"],
    ["7x6", "7x11", "7x13", "8x1", "6x7"],
    ["1x8", "2x5", "2x6", "1x13", "4x11"]
]

# Prepare BM25 and Semantic embeddings
texts, doc_ids = build_bm25_corpus(document_corpus)
corpus_embeddings = encode_corpus(document_corpus)

def evaluate_method(method_name, query_func):
    print(f"\n--- {method_name} ---")
    for i, query in enumerate(queries):
        results = query_func(query)
        print(f"Query {i+1}: {query}")
        print("Predicted:", results)
        print("Expected: ", answers[i])
        print("Overlap: ", len(set(results) & set(answers[i])), "/", len(answers[i]))
        print("")

# Boolean search wrapper
def boolean_query(q):
    return boolean_search(q, inverted_index)

# BM25 search wrapper
def bm25_query(q):
    return bm25_search(q, texts, doc_ids)

# Semantic search wrapper
def semantic_query(q):
    return semantic_search(q, document_corpus, corpus_embeddings)

# Run evaluation
evaluate_method("Boolean Search", boolean_query)
evaluate_method("BM25 Search", bm25_query)
evaluate_method("Semantic Search", semantic_query)