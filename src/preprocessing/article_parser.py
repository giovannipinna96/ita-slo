"""
Module for parsing newspaper texts into individual articles.
"""

import re
import logging
from typing import List

from src.models import Article

logger = logging.getLogger(__name__)

# Minimum article content length (characters)
MIN_ARTICLE_LENGTH = 100

# Regex patterns for Il Piccolo
PICCOLO_SEPARATOR_PATTERN = re.compile(r'^={40,}$', re.MULTILINE)
PICCOLO_PAGE_MARKER = re.compile(r'^=== PAGINA \d+ ===$', re.MULTILINE)

# Regex patterns for Edinost
# Match titles in all caps (at least 3 words or 10 characters)
EDINOST_TITLE_PATTERN = re.compile(r'^([A-ZČŠŽ][A-ZČŠŽ\s]{10,})$', re.MULTILINE)

# Words to exclude from being article titles (common in headers/footers)
EXCLUDED_TITLE_WORDS = {
    'TRIESTE', 'TRST', 'EDINOST', 'IL PICCOLO', 'DELLA SERA',
    'ANNO', 'NUMERO', 'REDAZIONE', 'AMMINISTRAZIONE'
}


def parse_il_piccolo(text: str) -> List[Article]:
    """
    Parse Il Piccolo newspaper text into individual articles.

    Articles are identified by titles delimited with '====' separators.

    Args:
        text: Combined text from all 4 pages

    Returns:
        List of Article objects
    """
    logger.info("Parsing Il Piccolo articles")

    articles = []
    current_page = 1

    # Split by page markers first
    page_sections = PICCOLO_PAGE_MARKER.split(text)

    for page_idx, page_text in enumerate(page_sections):
        if not page_text.strip():
            continue

        # Update page number based on marker found
        page_matches = PICCOLO_PAGE_MARKER.finditer(text)
        for i, match in enumerate(page_matches):
            if i == page_idx - 1:  # -1 because split creates one extra section at start
                page_num_str = match.group().replace('=', '').replace('PAGINA', '').strip()
                current_page = int(page_num_str)
                break

        # Find all separator patterns (====)
        separator_positions = [m.start() for m in PICCOLO_SEPARATOR_PATTERN.finditer(page_text)]

        if len(separator_positions) < 2:
            # No properly formatted articles in this section
            logger.debug(f"No delimited articles found in page section {page_idx}")
            continue

        # Process pairs of separators (before title, after title)
        i = 0
        while i < len(separator_positions) - 1:
            # Find title between two consecutive separators
            start_sep = separator_positions[i]
            end_sep = separator_positions[i + 1]

            # Extract title (between the two separators)
            title_start = page_text.find('\n', start_sep) + 1
            title_end = page_text.rfind('\n', title_start, end_sep)

            if title_start >= title_end:
                i += 1
                continue

            title = page_text[title_start:title_end].strip()

            # Skip if title is empty or too long (likely not a title)
            if not title or len(title) > 200 or len(title) < 3:
                i += 1
                continue

            # Find content (after end separator until next start separator or end)
            content_start = page_text.find('\n', end_sep) + 1

            # Find next article or end of page
            if i + 2 < len(separator_positions):
                content_end = separator_positions[i + 2]
            else:
                content_end = len(page_text)

            content = page_text[content_start:content_end].strip()

            # Remove any trailing separator
            content = PICCOLO_SEPARATOR_PATTERN.sub('', content).strip()

            # Validate article
            if len(content) < MIN_ARTICLE_LENGTH:
                logger.debug(f"Skipping short article: '{title[:50]}...' ({len(content)} chars)")
                i += 2  # Move to next pair
                continue

            # Check if it's an advertisement (simple heuristic)
            is_ad = _is_advertisement(title, content)
            if is_ad:
                logger.debug(f"Skipping advertisement: '{title[:50]}...'")
                i += 2
                continue

            # Create article
            article = Article(
                title=title,
                content=content,
                source='il_piccolo',
                language='it',
                page=current_page
            )

            articles.append(article)
            logger.debug(f"Extracted article: '{title[:50]}...' ({len(content)} chars, page {current_page})")

            i += 2  # Move to next pair of separators

    logger.info(f"Extracted {len(articles)} articles from Il Piccolo")
    return articles


def parse_edinost(text: str) -> List[Article]:
    """
    Parse Edinost newspaper text into individual articles.

    Articles are identified by titles in ALL CAPS on separate lines.

    Args:
        text: Complete Edinost newspaper text

    Returns:
        List of Article objects
    """
    logger.info("Parsing Edinost articles")

    articles = []

    # Find all potential titles (lines in all caps)
    title_matches = list(EDINOST_TITLE_PATTERN.finditer(text))

    if not title_matches:
        logger.warning("No articles found in Edinost (no titles matched)")
        return articles

    # Skip the first few matches (likely header/masthead)
    # Start from match that comes after reasonable offset (skip first 500 chars)
    valid_matches = [m for m in title_matches if m.start() > 500]

    for i, match in enumerate(valid_matches):
        title = match.group(1).strip()

        # Filter out excluded title words
        if any(excluded in title for excluded in EXCLUDED_TITLE_WORDS):
            logger.debug(f"Skipping excluded title: '{title}'")
            continue

        # Skip very short titles
        if len(title) < 10:
            continue

        # Get content (from end of title to next title or end)
        content_start = match.end()

        if i + 1 < len(valid_matches):
            content_end = valid_matches[i + 1].start()
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()

        # Validate article
        if len(content) < MIN_ARTICLE_LENGTH:
            logger.debug(f"Skipping short article: '{title[:50]}...' ({len(content)} chars)")
            continue

        # Check if it's an advertisement
        is_ad = _is_advertisement(title, content)
        if is_ad:
            logger.debug(f"Skipping advertisement: '{title[:50]}...'")
            continue

        # Detect section (if title contains section keywords)
        section = None
        if 'TRŽAŠKE VESTI' in title or 'TRŽAŠKE' in title:
            section = 'Notizie locali'
        elif 'PODLISTEK' in title:
            section = 'Feuilleton'
        elif 'POLITIČNI' in title:
            section = 'Politica'

        # Create article
        article = Article(
            title=title,
            content=content,
            source='edinost',
            language='sl',
            section=section
        )

        articles.append(article)
        logger.debug(f"Extracted article: '{title[:50]}...' ({len(content)} chars)")

    logger.info(f"Extracted {len(articles)} articles from Edinost")
    return articles


def _is_advertisement(title: str, content: str) -> bool:
    """
    Simple heuristic to detect advertisements.

    Args:
        title: Article title
        content: Article content

    Returns:
        True if likely an advertisement
    """
    # Check for advertisement keywords
    ad_keywords = [
        'cor.', 'f.', 'fiorini', 'corone',  # Currency
        'vendesi', 'affittasi', 'cercasi',  # Ads
        'comunicati', 'avviso', 'annuncio',  # Announcements
        'prezzo', 'lire', 'centesimi'  # Prices
    ]

    text_lower = (title + ' ' + content).lower()

    # Count ad keywords
    ad_keyword_count = sum(1 for keyword in ad_keywords if keyword in text_lower)

    # Only consider as advertisement if content is short AND has ad keywords
    # Long articles (>1000 chars) are unlikely to be pure advertisements
    if len(content) > 1000:
        return False
    
    # If multiple ad keywords found, likely an advertisement
    if ad_keyword_count >= 3:  # Raised threshold
        return True

    # Check for price patterns (number followed by currency)
    price_pattern = re.compile(r'\d+\s*(cor\.|f\.|fiorini|corone|lire|centesimi)', re.IGNORECASE)
    if len(price_pattern.findall(content)) >= 3:  # Raised threshold
        return True

    return False


def validate_article(article: Article) -> bool:
    """
    Validate that an article meets quality criteria.

    Args:
        article: Article to validate

    Returns:
        True if article is valid
    """
    # Check basic requirements (dataclass __post_init__ already validates these)
    if not article.title or not article.content:
        return False

    # No minimum length check - keep all articles

    # Check that content is not just whitespace
    if not article.content.strip():
        return False

    # Check reasonable title length
    if len(article.title) > 300:
        logger.warning(f"Article title too long: '{article.title[:50]}...' ({len(article.title)} chars)")
        return False

    return True


def parse_and_validate(text: str, source: str) -> List[Article]:
    """
    Parse text and return only validated articles.
    
    Auto-detects format:
    - If text contains '=== PAGINA' markers, uses old Il Piccolo parser
    - Otherwise uses generic parser for new format
    
    Args:
        text: Newspaper text
        source: Source identifier
        
    Returns:
        List of validated Article objects
    """
    # Auto-detect format
    if '=== PAGINA' in text:
        # Old format with page markers
        logger.info(f"Detected old format with page markers for {source}")
        if 'piccolo' in source.lower():
            articles = parse_il_piccolo(text)
        else:
            articles = parse_edinost(text)
    else:
        # New format with === separators
        logger.info(f"Detected new format for {source}")
        articles = parse_generic(text, source)
    
    # Validate articles
    valid_articles = [a for a in articles if validate_article(a)]
    
    logger.info(f"Validated {len(valid_articles)}/{len(articles)} articles from {source}")
    
    return valid_articles


# Regex patterns for generic format (new files)
GENERIC_ARTICLE_SEPARATOR = re.compile(r'^={3,}$', re.MULTILINE)
GENERIC_SUBSECTION_SEPARATOR = re.compile(r'^-{3,}$', re.MULTILINE)

# Header keywords to skip (newspaper masthead)
HEADER_KEYWORDS = {
    'EDINOST', 'IL PICCOLO', 'PICCOLO', 'GLASILO', 'UFFICI', 'TELEFONO',
    'ABBONAMENTO', 'NAROČNINA', 'OGLASI', 'INSERZIONI', 'IZDAJATELJ',
    'UREDNIŠTVO', 'REDAZIONE', 'AMMINISTRAZIONE', 'PAGINA', 'ANNUNCI',
    'PUBBLICITÀ', 'OGLASI / ANNUNCI'
}


def parse_generic(text: str, source: str) -> List[Article]:
    """
    Parse newspaper text using generic separator-based format.
    
    Simply splits by === separators and creates an article from each section.
    No filtering applied.
    
    Args:
        text: Newspaper text
        source: Source identifier (used for Article metadata)
        
    Returns:
        List of Article objects
    """
    logger.info(f"Parsing {source} articles using generic parser")
    
    # Detect language from source name
    language = 'sl' if 'edinost' in source.lower() else 'it'
    
    articles = []
    
    # Split by main article separators (===)
    sections = GENERIC_ARTICLE_SEPARATOR.split(text)
    
    # Process all sections
    for idx, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        
        # Extract title (first non-empty line)
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        if not lines:
            continue
        
        title = lines[0]
        
        # If title is too long, truncate it
        if len(title) > 200:
            title = title[:50] + "..."
            content = '\n'.join(lines)
        else:
            content = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        
        # If content is empty, use full section as content
        if not content:
            content = section
        
        # Create article
        try:
            article = Article(
                title=title,
                content=content,
                source=source,
                language=language
            )
            articles.append(article)
            logger.debug(f"Extracted article {idx}: '{title[:40]}...' ({len(content)} chars)")
        except ValueError as e:
            logger.debug(f"Skipping section {idx}: {e}")
    
    logger.info(f"Extracted {len(articles)} articles from {source} using generic parser")
    return articles


