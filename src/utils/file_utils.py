"""
Utility functions for file I/O operations.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import pickle

from src.models import Article, Newspaper

logger = logging.getLogger(__name__)


def ensure_dirs(dirs: List[str]) -> None:
    """
    Ensure that directories exist, create if they don't.

    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")


def save_articles_json(articles: List[Article], filepath: str) -> None:
    """
    Save articles to JSON file.

    Args:
        articles: List of Article objects
        filepath: Path to output JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = [article.to_dict() for article in articles]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(articles)} articles to {filepath}")


def load_articles_json(filepath: str) -> List[Article]:
    """
    Load articles from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of Article objects
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = [Article.from_dict(item) for item in data]

    logger.info(f"Loaded {len(articles)} articles from {filepath}")
    return articles


def save_newspaper_json(newspaper: Newspaper, filepath: str) -> None:
    """
    Save newspaper object to JSON file.

    Args:
        newspaper: Newspaper object
        filepath: Path to output JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(newspaper.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Saved newspaper '{newspaper.name}' to {filepath}")


def load_newspaper_json(filepath: str) -> Newspaper:
    """
    Load newspaper object from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Newspaper object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    newspaper = Newspaper.from_dict(data)

    logger.info(f"Loaded newspaper '{newspaper.name}' from {filepath}")
    return newspaper


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Path to output pickle file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    logger.info(f"Saved object to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    logger.info(f"Loaded object from {filepath}")
    return obj


def save_dict_json(data: Dict, filepath: str) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to output JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved data to {filepath}")


def load_dict_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"Loaded data from {filepath}")
    return data
