import os
import json
import pandas as pd
import sqlite3

from src.boolean_search import boolean_search, boolean_search_sqlite
from src.bm_25 import build_bm25_corpus, build_bm25_corpus_sqlite, bm25_search, bm25_search_sqlite
from src.sentence_transformers import encode_corpus, load_embeddings_from_db, semantic_search, semantic_search_sqlite

# Paths & DB setup

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DW_DATA = os.path.join(PROJECT_ROOT, "dw_data")
DB_PATH = os.path.join(DW_DATA, "doctor_who.db")
CORPUS_PATH = os.path.join(DW_DATA, "document_corpus_dw.json")
INDEX_PATH = os.path.join(DW_DATA, "inverted_index.json")
FAISS_INDEX_PATH = os.path.join(DW_DATA, "faiss.index")
FAISS_MAPPING_PATH = os.path.join(DW_DATA, "faiss_mapping.json")

conn = sqlite3.connect(DB_PATH)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    document_corpus = json.load(f)
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)
with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
    index_to_doc_id = json.load(f)
index_to_doc_id = {int(k): v for k, v in index_to_doc_id.items()}

# Queries and expected answers

queries = [
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
    ["1x8", "2x5", "2x6", "1x13", "4x11"],
    ["5x9", "5x12", "7x5", "5x13", "5x10"],
    ["1x1", "1x2", "2x0", "2x7", "1x4"],
    ["5x11", "6x12", "6x13", "6x1", "6x2"],
    ["2x1", "2x2", "2x3", "2x4", "2x7"],
    ["1x13", "4x13", "7x14", "10x12", "2x0"],
    ["6x1", "6x2", "6x13", "6x11", "5x1"],
    ["5x6", "4x2", "10x6", "5x12", "5x13"]
]

# Prepare BM25 and Semantic embeddings
texts, doc_ids = build_bm25_corpus_sqlite(conn)
corpus_embeddings = load_embeddings_from_db(conn)

# Boolean search wrapper
def boolean_query(q):
    return boolean_search_sqlite(q, conn)

# BM25 search wrapper
def bm25_query(q):
    return bm25_search_sqlite(q, conn)

# Semantic search wrapper
def semantic_query(q):
    return semantic_search_sqlite(q, conn)

# FAISS Setup

import faiss
from sentence_transformers import SentenceTransformer

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
model = SentenceTransformer('all-MiniLM-L6-v2')

def faiss_query(q, top_k=5):
    query_emb = model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(query_emb.reshape(1, -1), top_k)
    return [index_to_doc_id[i] for i in I[0]]

def evaluate_method(method_name, query_func):
    print(f"\n--- {method_name} ---")
    for i, query in enumerate(queries):
        results = query_func(query)
        print(f"Query {i+1}: {len(set(results) & set(answers[i]))}/{len(answers[i])}")

# Run evaluation
evaluate_method("Boolean Search", boolean_query)
evaluate_method("BM25 Search", bm25_query)
evaluate_method("Semantic Search", semantic_query)
evaluate_method("FAISS Semantic Search", lambda q: faiss_query(q, top_k=5))


results_dict = {
    "Query": queries,
    "Boolean Correct": [],
    "BM25 Correct": [],
    "ST Correct": [],
    "FAISS Correct": []
}

for i, query in enumerate(queries):
    results_dict["Boolean Correct"].append(len(set(boolean_query(query)) & set(answers[i])))
    results_dict["BM25 Correct"].append(len(set(bm25_query(query)) & set(answers[i])))
    results_dict["ST Correct"].append(len(set(semantic_query(query)) & set(answers[i])))
    results_dict["FAISS Correct"].append(len(set(faiss_query(query)) & set(answers[i])))

df = pd.DataFrame(results_dict)
df.to_csv("dw_data/search_results_summary.csv", index=False)

