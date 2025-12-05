"""
Module for cleaning and preprocessing article texts.
"""

import re
import logging
from typing import List
import spacy
from spacy.lang.it.stop_words import STOP_WORDS as IT_STOP_WORDS
from spacy.lang.sl.stop_words import STOP_WORDS as SL_STOP_WORDS

from src.models import Article

logger = logging.getLogger(__name__)

# Load spaCy models (lazy loading)
_nlp_it = None
_nlp_sl = None


def get_nlp(language: str):
    """
    Get spaCy NLP model for the given language (lazy loading).

    Args:
        language: 'it' or 'sl'

    Returns:
        spaCy NLP model
    """
    global _nlp_it, _nlp_sl

    if language == 'it':
        if _nlp_it is None:
            logger.info("Loading Italian spaCy model...")
            _nlp_it = spacy.load('it_core_news_lg')
        return _nlp_it
    elif language == 'sl':
        if _nlp_sl is None:
            logger.info("Loading Slovenian spaCy model...")
            _nlp_sl = spacy.load('sl_core_news_sm')
        return _nlp_sl
    else:
        raise ValueError(f"Unsupported language: {language}")


def clean_historical_text(text: str, language: str) -> str:
    """
    Clean and normalize historical text.

    Applies conservative normalization to preserve historical orthography
    while removing obvious artifacts.

    Args:
        text: Raw text
        language: 'it' or 'sl'

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove separator lines (=====)
    text = re.sub(r'={3,}', '', text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Remove isolated single characters (likely OCR errors)
    # But preserve single letters like "a", "e" in Italian
    # text = re.sub(r'\b[^aeiouAEIOU\s]\b', '', text)  # Too aggressive

    # Normalize multiple punctuation
    text = re.sub(r'\.{2,}', '...', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    # Remove page numbers (isolated numbers)
    text = re.sub(r'\b\d+\b\s*$', '', text, flags=re.MULTILINE)

    # Trim
    text = text.strip()

    return text


def remove_boilerplate(text: str) -> str:
    """
    Remove boilerplate text (headers, footers, repetitive content).

    Args:
        text: Text to clean

    Returns:
        Text without boilerplate
    """
    # Remove common headers/footers
    boilerplate_patterns = [
        r'Per dispaccio.*?Dal nostro inviato speciale',
        r'Redazione.*?Amministrazione',
        r'TRIESTE.*?\d{4}',
    ]

    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


def tokenize_and_lemmatize(text: str, language: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize and lemmatize text using spaCy.

    Args:
        text: Text to process
        language: 'it' or 'sl'
        remove_stopwords: Whether to remove stopwords

    Returns:
        List of lemmatized tokens
    """
    nlp = get_nlp(language)

    # Get stopwords for language
    if language == 'it':
        stopwords = IT_STOP_WORDS
    else:
        stopwords = SL_STOP_WORDS

    # Process text with spaCy
    doc = nlp(text)

    # Extract lemmas
    tokens = []
    for token in doc:
        # Skip punctuation, spaces, and numbers
        if token.is_punct or token.is_space or token.like_num:
            continue

        # Skip stopwords if requested
        if remove_stopwords and token.text.lower() in stopwords:
            continue

        # Skip very short tokens (likely artifacts)
        if len(token.text) < 2:
            continue

        # Add lemma (lowercase)
        lemma = token.lemma_.lower()
        if lemma and lemma != '-PRON-':  # spaCy uses -PRON- for pronouns
            tokens.append(lemma)

    return tokens


def clean_article(article: Article) -> None:
    """
    Clean a single article (modifies article in-place).

    Sets the cleaned_content attribute.

    Args:
        article: Article to clean
    """
    # Clean text
    cleaned = clean_historical_text(article.content, article.language)
    cleaned = remove_boilerplate(cleaned)

    # Tokenize and lemmatize
    tokens = tokenize_and_lemmatize(cleaned, article.language)

    # Store both cleaned text and tokenized version
    article.cleaned_content = ' '.join(tokens)

    logger.debug(f"Cleaned article '{article.title[:50]}...' ({len(tokens)} tokens)")


def clean_articles(articles: List[Article], language: str) -> None:
    """
    Clean a list of articles (modifies articles in-place).

    Args:
        articles: List of articles to clean
        language: Expected language ('it' or 'sl')
    """
    logger.info(f"Cleaning {len(articles)} {language} articles...")

    # Verify language consistency
    for article in articles:
        if article.language != language:
            logger.warning(
                f"Article language mismatch: expected {language}, got {article.language}"
            )

    # Clean each article
    for i, article in enumerate(articles):
        try:
            clean_article(article)

            if (i + 1) % 10 == 0:
                logger.info(f"Cleaned {i + 1}/{len(articles)} articles")

        except Exception as e:
            logger.error(f"Error cleaning article '{article.title}': {e}")
            # Set empty cleaned content on error
            article.cleaned_content = ""

    logger.info(f"Cleaning complete: {len(articles)} articles processed")


def get_vocabulary(articles: List[Article]) -> set:
    """
    Extract vocabulary (unique tokens) from cleaned articles.

    Args:
        articles: List of cleaned articles

    Returns:
        Set of unique tokens
    """
    vocabulary = set()

    for article in articles:
        if article.cleaned_content:
            tokens = article.cleaned_content.split()
            vocabulary.update(tokens)

    logger.info(f"Vocabulary size: {len(vocabulary)} unique tokens")
    return vocabulary


def get_token_statistics(articles: List[Article]) -> dict:
    """
    Compute token statistics for a collection of articles.

    Args:
        articles: List of cleaned articles

    Returns:
        Dictionary with statistics
    """
    total_tokens = 0
    vocabulary = set()
    article_lengths = []

    for article in articles:
        if article.cleaned_content:
            tokens = article.cleaned_content.split()
            total_tokens += len(tokens)
            vocabulary.update(tokens)
            article_lengths.append(len(tokens))

    stats = {
        'total_articles': len(articles),
        'total_tokens': total_tokens,
        'vocabulary_size': len(vocabulary),
        'avg_article_length': total_tokens / len(articles) if articles else 0,
        'min_article_length': min(article_lengths) if article_lengths else 0,
        'max_article_length': max(article_lengths) if article_lengths else 0
    }

    logger.info(f"Token statistics: {stats}")
    return stats
