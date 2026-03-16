from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def encode_corpus(document_corpus):
    texts = [f"{doc['title']} {doc['description']}" for doc in document_corpus.values()]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

def semantic_search(query: str, document_corpus, corpus_embeddings, top_n=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_n)[0]
    results = [list(document_corpus.keys())[hit['corpus_id']] for hit in hits]
    return results