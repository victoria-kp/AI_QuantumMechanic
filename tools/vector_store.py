"""ChromaDB vector store for semantic equation search.

Indexes each equation's search_text into a ChromaDB collection using
the built-in ONNX embedding model (all-MiniLM-L6-v2).  The collection
is rebuilt on each process start — fast for ~50 documents.
"""

import logging
from typing import List, Optional

import chromadb

from tools.catalog_loader import get_catalog

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "equations"
_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, initializing on first call."""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.Client()  # in-memory, rebuilt each run
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    catalog = get_catalog()
    if not catalog:
        logger.warning("Empty catalog — vector store has no documents")
        return _collection

    keys = []
    documents = []
    metadatas = []

    for key, entry in catalog.items():
        search_text = entry.get("search_text", "")
        if not search_text:
            search_text = f"{entry['name']} {entry['description']}"

        keys.append(key)
        documents.append(search_text)
        metadatas.append({
            "key": key,
            "name": entry["name"],
            "description": entry["description"],
            "tags": ",".join(entry.get("tags", [])),
        })

    _collection.add(ids=keys, documents=documents, metadatas=metadatas)
    logger.info("Indexed %d equations into ChromaDB", len(keys))

    return _collection


def search_equations(
    query: str,
    n_results: int = 5,
    tag: str = "",
) -> List[dict]:
    """Semantic search over the equation catalog.

    Args:
        query:     Natural-language search query.
        n_results: Maximum number of results to return.
        tag:       Optional tag to filter results (e.g. "potential").

    Returns:
        List of dicts with keys: key, name, description, tags, score.
        No expression strings are included.
    """
    collection = _get_collection()

    # Fetch extra results when tag-filtering so we can still
    # return n_results after post-filtering.
    fetch_n = n_results * 3 if tag else n_results

    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
    )

    hits = []
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    tag_lower = tag.lower() if tag else ""

    for id_, dist, meta in zip(ids, distances, metadatas):
        tags_list = meta["tags"].split(",") if meta["tags"] else []

        if tag_lower and tag_lower not in [t.lower() for t in tags_list]:
            continue

        hits.append({
            "key": id_,
            "name": meta["name"],
            "description": meta["description"],
            "tags": tags_list,
            "score": round(1 - dist, 4),  # cosine similarity
        })

        if len(hits) >= n_results:
            break

    return hits
