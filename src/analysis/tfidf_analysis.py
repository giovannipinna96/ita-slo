"""
Module for TF-IDF analysis of articles.
"""

import logging
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import Article

logger = logging.getLogger(__name__)


class TFIDFAnalyzer:
    """TF-IDF analyzer for article collections."""

    def __init__(self, max_features: int = 100, min_df: int = 2, max_df: float = 0.8):
        """
        Initialize TF-IDF analyzer.

        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency (proportion)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None

    def fit_transform(self, articles: List[Article]) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform articles.

        Args:
            articles: List of articles with cleaned_content

        Returns:
            TF-IDF matrix (n_articles x n_features)
        """
        if not articles:
            raise ValueError("No articles provided")

        # Check that articles have been cleaned
        if not all(a.cleaned_content for a in articles):
            logger.warning("Not all articles have cleaned_content, some may be skipped")

        # Extract cleaned texts
        texts = [a.cleaned_content for a in articles if a.cleaned_content]

        if not texts:
            raise ValueError("No cleaned content found in articles")

        logger.info(f"Fitting TF-IDF on {len(texts)} articles...")

        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,  # Already lowercased in cleaning
            token_pattern=r'\b\w+\b'  # Word tokens
        )

        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        logger.info(f"Feature names: {len(self.feature_names)} terms")

        return self.tfidf_matrix

    def get_top_keywords_global(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top N keywords by average TF-IDF score across all documents.

        Args:
            n: Number of top keywords to return

        Returns:
            List of (word, avg_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")

        # Calculate average TF-IDF score for each term
        avg_scores = np.asarray(self.tfidf_matrix.mean(axis=0)).ravel()

        # Get indices of top N scores
        top_indices = avg_scores.argsort()[-n:][::-1]

        # Extract top keywords with scores
        top_keywords = [
            (self.feature_names[i], avg_scores[i])
            for i in top_indices
        ]

        logger.info(f"Top {n} global keywords extracted")
        return top_keywords

    def get_article_keywords(self, article_idx: int, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N keywords for a specific article.

        Args:
            article_idx: Index of article in the collection
            n: Number of top keywords to return

        Returns:
            List of (word, score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")

        if article_idx >= self.tfidf_matrix.shape[0]:
            raise ValueError(f"Article index {article_idx} out of range")

        # Get TF-IDF scores for this article
        article_scores = self.tfidf_matrix[article_idx].toarray().ravel()

        # Get indices of top N scores
        top_indices = article_scores.argsort()[-n:][::-1]

        # Extract keywords with non-zero scores
        keywords = [
            (self.feature_names[i], article_scores[i])
            for i in top_indices
            if article_scores[i] > 0
        ]

        return keywords

    def assign_keywords_to_articles(self, articles: List[Article], n: int = 10) -> None:
        """
        Assign top TF-IDF keywords to each article (modifies articles in-place).

        Args:
            articles: List of articles
            n: Number of keywords per article
        """
        if self.tfidf_matrix is None:
            raise ValueError("Must call fit_transform first")

        logger.info(f"Assigning top {n} keywords to each article...")

        # Filter articles that have cleaned content
        articles_with_content = [a for a in articles if a.cleaned_content]

        for idx, article in enumerate(articles_with_content):
            keywords = self.get_article_keywords(idx, n)
            article.tfidf_keywords = keywords

        logger.info(f"Keywords assigned to {len(articles_with_content)} articles")

    def analyze(self, articles: List[Article], n_global: int = 20, n_per_article: int = 10) -> dict:
        """
        Complete TF-IDF analysis pipeline.

        Args:
            articles: List of articles with cleaned_content
            n_global: Number of global top keywords
            n_per_article: Number of keywords per article

        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting TF-IDF analysis...")

        # Fit and transform
        self.fit_transform(articles)

        # Get global top keywords
        global_keywords = self.get_top_keywords_global(n_global)

        # Assign keywords to articles
        self.assign_keywords_to_articles(articles, n_per_article)

        results = {
            'global_keywords': global_keywords,
            'vocabulary_size': len(self.feature_names),
            'n_articles': self.tfidf_matrix.shape[0],
            'n_features': self.tfidf_matrix.shape[1]
        }

        logger.info("TF-IDF analysis complete")
        return results


def compare_tfidf(keywords_a: List[Tuple[str, float]],
                  keywords_b: List[Tuple[str, float]],
                  n: int = 20) -> dict:
    """
    Compare TF-IDF keywords between two newspapers.

    Args:
        keywords_a: Top keywords from newspaper A
        keywords_b: Top keywords from newspaper B
        n: Number of keywords to consider

    Returns:
        Dictionary with comparison metrics
    """
    # Extract top N words from each
    words_a = set([word for word, score in keywords_a[:n]])
    words_b = set([word for word, score in keywords_b[:n]])

    # Calculate overlap
    common_words = words_a & words_b
    unique_to_a = words_a - words_b
    unique_to_b = words_b - words_a

    comparison = {
        'common_words': list(common_words),
        'unique_to_a': list(unique_to_a),
        'unique_to_b': list(unique_to_b),
        'overlap_percentage': len(common_words) / n * 100 if n > 0 else 0
    }

    logger.info(f"TF-IDF comparison: {len(common_words)} common, "
                f"{len(unique_to_a)} unique to A, {len(unique_to_b)} unique to B")

    return comparison
