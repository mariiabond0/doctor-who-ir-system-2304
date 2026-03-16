from rank_bm25 import BM25Okapi
from src.preprocessing import preprocess_text
import numpy as np

def build_bm25_corpus(document_corpus):
    """Return tokenized texts and corresponding doc IDs"""
    texts = []
    doc_ids = []
    for doc_id, doc in document_corpus.items():
        texts.append(preprocess_text(f"{doc['title']} {doc['description']}"))
        doc_ids.append(doc_id)
    return texts, doc_ids

def bm25_search(query: str, texts, doc_ids, top_n=5):
    """Return top_n document IDs ranked by BM25"""
    bm25 = BM25Okapi(texts)
    query_tokens = preprocess_text(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [doc_ids[i] for i in top_indices]