import json
import os
import pickle
import logging
import sqlite3
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from src.preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)
MODEL_NAME = config.MODEL_NAME


def load_episode_data() -> pd.DataFrame:
    """Load episode metadata and IMDb details from CSV files."""
    df_details = pd.read_csv(config.EPISODES_CSV)
    df_imdb = pd.read_csv(config.IMDB_CSV)
    merged = pd.merge(df_details, df_imdb, on="title", how="inner")
    merged.to_csv(config.MERGED_DATASET_PATH, index=False)
    return merged


def filter_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out seasons that are excluded by configuration."""
    if config.EXCLUDE_SEASONS:
        return df[~df["season"].astype(str).isin(config.EXCLUDE_SEASONS)]
    return df


def document_id(season: int, number: int) -> str:
    """Return a canonical document identifier for a season/episode."""
    return f"{season}x{number}"


def build_corpus(df: pd.DataFrame):
    """Build a document corpus, inverted index, and embeddings for the dataset."""
    document_corpus = {}
    inverted_index = defaultdict(set)
    embeddings_dict = {}
    model = SentenceTransformer(MODEL_NAME)

    for _, row in df.iterrows():
        doc_id = document_id(int(row["season"]), int(row["number"]))
        title = str(row.get("title", "")).strip()
        description = str(row.get("description", "")).strip()
        text = f"{title} {description}".strip()

        document_corpus[doc_id] = {
            "id": doc_id,
            "season": int(row["season"]),
            "number": int(row["number"]),
            "title": title,
            "description": description,
        }

        preprocessed = preprocess_text(text)
        for token in preprocessed:
            inverted_index[token].add(doc_id)

        embeddings_dict[doc_id] = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    return document_corpus, inverted_index, embeddings_dict


def sort_corpus(document_corpus):
    """Sort corpus entries by season and episode number."""
    return dict(
        sorted(
            document_corpus.items(),
            key=lambda item: (item[1]["season"], item[1]["number"]),
        )
    )


def save_json_corpus(document_corpus, inverted_index):
    """Persist the corpus and inverted index to JSON files."""
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as output:
        json.dump(document_corpus, output, ensure_ascii=False, indent=2)

    inverted_index_json = {
        token: sorted(list(doc_ids)) for token, doc_ids in inverted_index.items()
    }
    with open(config.INDEX_PATH, "w", encoding="utf-8") as output:
        json.dump(inverted_index_json, output, ensure_ascii=False, indent=2)

    LOGGER.info("Corpus JSON and inverted index saved.")


def save_database(document_corpus, inverted_index, embeddings_dict):
    """Write the corpus, inverted index, and embeddings to SQLite."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            doc_id TEXT PRIMARY KEY,
            season INTEGER,
            number INTEGER,
            title TEXT,
            description TEXT,
            preprocessed_combined BLOB
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inverted_index (
            token TEXT,
            doc_id TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id TEXT PRIMARY KEY,
            embedding BLOB
        )
        """
    )

    cur.execute("DELETE FROM episodes")
    cur.execute("DELETE FROM inverted_index")
    cur.execute("DELETE FROM embeddings")

    for doc_id, doc in document_corpus.items():
        preprocessed_document = preprocess_text(f"{doc['title']} {doc['description']}")
        cur.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                doc["season"],
                doc["number"],
                doc["title"],
                doc["description"],
                pickle.dumps(preprocessed_document),
            ),
        )

    for token, doc_ids in inverted_index.items():
        for doc_id in sorted(doc_ids):
            cur.execute("INSERT INTO inverted_index VALUES (?, ?)", (token, doc_id))

    for doc_id, embedding in embeddings_dict.items():
        cur.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            (doc_id, sqlite3.Binary(pickle.dumps(embedding))),
        )

    conn.commit()
    conn.close()
    LOGGER.info("SQLite database saved.")


def build_faiss_index(embeddings_dict):
    """Build and save a FAISS index for the corpus embeddings."""
    doc_ids = sorted(embeddings_dict.keys())
    embedding_matrix = np.vstack([embeddings_dict[doc_id] for doc_id in doc_ids]).astype("float32")

    faiss.normalize_L2(embedding_matrix)
    dimension = embedding_matrix.shape[1]
    os.environ["OMP_NUM_THREADS"] = "1"

    index = faiss.IndexHNSWFlat(dimension, config.FAISS_M)
    index.hnsw.efConstruction = config.FAISS_EF_CONSTRUCTION
    index.add(embedding_matrix)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    index_to_doc_id = {i: doc_id for i, doc_id in enumerate(doc_ids)}
    with open(config.FAISS_MAPPING_PATH, "w", encoding="utf-8") as mapping_file:
        json.dump(index_to_doc_id, mapping_file, ensure_ascii=False, indent=2)

    LOGGER.info("FAISS index and mapping saved.")


def main():
    logging.basicConfig(level=logging.INFO)
    df = load_episode_data()
    df = filter_seasons(df)
    document_corpus, inverted_index, embeddings_dict = build_corpus(df)
    document_corpus = sort_corpus(document_corpus)

    save_json_corpus(document_corpus, inverted_index)
    save_database(document_corpus, inverted_index, embeddings_dict)
    build_faiss_index(embeddings_dict)

    print(f"Corpus created: {len(document_corpus)} episodes saved.")


if __name__ == "__main__":
    main()
