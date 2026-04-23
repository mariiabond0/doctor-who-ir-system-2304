# Doctor Who Information Retrieval System

This project implements an information retrieval system for the **Doctor Who** TV series. It supports multiple search methods, including **Boolean search**, **BM25**, **semantic search with Sentence Transformers**, and **FAISS-based nearest neighbor search**.

## Features

* **Corpus Creation**
  * Collects episode data from CSV files (`all-detailsepisodes.csv`, `imdb_details.csv`)
  * Preprocesses text with tokenization, optional stopword removal, and stemming
  * Builds a **document corpus**, **inverted index**, and **SQLite database**
* **Search Methods**
  * **Boolean Search** using an inverted index
  * **BM25 Search** using `rank_bm25`
  * **Semantic Search** using `SentenceTransformers` (`all-MiniLM-L6-v2`)
  * **FAISS Semantic Search** using a nearest neighbor index
* **Storage Options**
  * JSON files (`document_corpus_dw.json`, `inverted_index.json`)
  * SQLite database (`doctor_who.db`) with episodes, inverted index, and embeddings
  * FAISS index for fast semantic nearest neighbor retrieval
* **Evaluation**
  * Supports test queries with expected answers
  * Computes IR metrics: P@5, R@5, AP, and MRR
* **Deployment**
  * `uv`-based dependency management

## Project Structure

```
doctor-who-ir-project/
│
├─ src/
│   ├─ __init__.py
│   ├─ bm_25.py
│   ├─ boolean_search.py
│   ├─ creating_corpus.py
│   ├─ preprocessing.py
│   ├─ semantic_search.py
│
├─ dw_data/
│   ├─ all-detailsepisodes.csv
│   ├─ all-scripts.csv
│   ├─ doctor_who.db
│   ├─ document_corpus_dw.json
│   ├─ dwguide.csv
│   ├─ first_example_10_queries.json
│   ├─ imdb_details.csv
│   ├─ inverted_index.json
│   ├─ merged_dataset.csv
│   ├─ faiss_mapping.json
│   ├─ faiss.index
│   └─ search_results_summary.csv
│
├─ main.py
├─ README.md
├─ requirements.txt
├─ pyproject.toml
```

## Installation

### Using `uv` (Recommended)

Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### Using `pip` (Alternative)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Build the corpus and database:

```bash
uv run python src/creating_corpus.py
```

2. Run evaluation:

```bash
uv run python main.py
```

## Notes

* `config.py` centralizes paths and search settings.
* `src/semantic_search.py` provides semantic nearest-neighbor search from SQLite embeddings.

