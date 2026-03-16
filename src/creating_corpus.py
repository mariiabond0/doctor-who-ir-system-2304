import pandas as pd
from nltk.tokenize import word_tokenize
import string
import json
from collections import defaultdict
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "dw_data")

# Load CSV files
df_details = pd.read_csv(os.path.join(DATA_DIR, "all-detailsepisodes.csv"))
df_imdb = pd.read_csv(os.path.join(DATA_DIR, "imdb_details.csv"))

# Merge datasets
df = pd.merge(df_details, df_imdb, on="title", how="inner")
df.to_csv(os.path.join(DATA_DIR, "merged_dataset.csv"), index=False)

def preprocess(text):
    if pd.isna(text):
        return []
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in string.punctuation]
    return tokens

inverted_index = defaultdict(list)
document_corpus = {}

descriptions = df['description'].fillna("").tolist()
episode_keys = (df['season'].astype(str) + "x" + df['number'].astype(str)).tolist()
titles = df['title'].tolist()

for i, doc in enumerate(descriptions):
    doc_id = episode_keys[i]
    document_corpus[doc_id] = {
        "title": titles[i],
        "description": doc,
        "id": doc_id
    }
    tokens = preprocess(doc)
    for token in tokens:
        if doc_id not in inverted_index[token]:
            inverted_index[token].append(doc_id)

def sort_key(epid):
    season, number = epid.split('x')
    return int(season), int(number)

sorted_corpus = dict(sorted(document_corpus.items(), key=lambda x: sort_key(x[0])))
filtered_corpus = {k: v for k, v in sorted_corpus.items() if not k.startswith("11x")}

with open(os.path.join(DATA_DIR, "document_corpus_dw.json"), "w", encoding="utf-8") as f:
    json.dump(filtered_corpus, f, ensure_ascii=False, indent=2)

with open(os.path.join(DATA_DIR, "inverted_index.json"), "w", encoding="utf-8") as f:
    json.dump(dict(inverted_index), f, ensure_ascii=False, indent=2)

print(f"Corpus created: {len(filtered_corpus)} episodes saved, {len(document_corpus) - len(filtered_corpus)} removed.")