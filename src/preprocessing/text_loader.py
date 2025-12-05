"""
Module for loading and combining newspaper text files.
"""

import logging
from pathlib import Path
from typing import Tuple

from src.models import Newspaper

logger = logging.getLogger(__name__)


def load_il_piccolo() -> str:
    """
    Load and combine all 4 pages of Il Piccolo newspaper.

    Returns:
        str: Combined text from all pages with page markers
    """
    data_dir = Path(__file__).parent.parent.parent / 'data'
    combined_text = []

    for page_num in range(1, 5):
        file_path = data_dir / f'il_piccolo_19020909_pagina{page_num}.txt'

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        logger.info(f"Loading {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add page marker
            combined_text.append(f"\n\n=== PAGINA {page_num} ===\n\n")
            combined_text.append(content)

            logger.info(f"Loaded {len(content)} characters from page {page_num}")

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    full_text = ''.join(combined_text)
    logger.info(f"Total Il Piccolo text: {len(full_text)} characters")

    return full_text


def load_edinost() -> str:
    """
    Load Edinost newspaper complete transcription.

    Returns:
        str: Full text of Edinost newspaper
    """
    data_dir = Path(__file__).parent.parent.parent / 'data'
    file_path = data_dir / 'edinost_19020909_trascrizione_completa.txt'

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Edinost file not found: {file_path}")

    logger.info(f"Loading {file_path.name}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"Loaded {len(content)} characters from Edinost")
        return content

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise


def create_newspaper_objects() -> Tuple[Newspaper, Newspaper]:
    """
    Create Newspaper objects for both newspapers (without articles yet).

    Articles will be added later by the parser.

    Returns:
        Tuple[Newspaper, Newspaper]: Il Piccolo and Edinost newspaper objects
    """
    logger.info("Creating newspaper objects")

    piccolo = Newspaper(
        name='Il Piccolo',
        language='it',
        date='1902-09-09'
    )

    edinost = Newspaper(
        name='Edinost',
        language='sl',
        date='1902-09-09'
    )

    logger.info(f"Created newspaper objects: {piccolo.name}, {edinost.name}")

    return piccolo, edinost


def load_all_texts() -> Tuple[str, str, Newspaper, Newspaper]:
    """
    Convenience function to load all texts and create newspaper objects.

    Returns:
        Tuple containing:
        - Il Piccolo text (str)
        - Edinost text (str)
        - Il Piccolo newspaper object
        - Edinost newspaper object
    """
    piccolo_text = load_il_piccolo()
    edinost_text = load_edinost()
    piccolo_obj, edinost_obj = create_newspaper_objects()

    return piccolo_text, edinost_text, piccolo_obj, edinost_obj
