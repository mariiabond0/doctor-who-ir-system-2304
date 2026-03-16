from src.preprocessing import preprocess_text

def boolean_search(query: str, index: dict):
    """Simple Boolean search using inverted index"""
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []
    results = set(index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        results = results.union(index.get(token, []))
    return list(results)[:5]