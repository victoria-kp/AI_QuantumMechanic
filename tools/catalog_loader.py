"""Loads equation catalog from JSON files in data/equations/.

Merges all category JSON files into a single Python dict that is a
drop-in replacement for the old hardcoded EQUATION_CATALOG.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "equations"
_catalog_cache: Optional[dict] = None


def get_catalog() -> dict:
    """Return the unified equation catalog, loading from disk on first call."""
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = _load_all()
    return _catalog_cache


def reload_catalog() -> dict:
    """Force-reload the catalog from disk."""
    global _catalog_cache
    _catalog_cache = _load_all()
    return _catalog_cache


def _load_all() -> dict:
    """Read every .json file in data/equations/ and merge into one dict."""
    catalog: dict = {}

    if not _DATA_DIR.is_dir():
        logger.warning("Equation data directory not found: %s", _DATA_DIR)
        return catalog

    for json_path in sorted(_DATA_DIR.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
            catalog.update(data)
            logger.info("Loaded %d equations from %s", len(data), json_path.name)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load %s: %s", json_path.name, exc)

    logger.info("Total catalog size: %d equations", len(catalog))
    return catalog
