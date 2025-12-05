"""
Module for comparative analysis between two newspapers.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats

from src.models import Article

logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """Comparative analysis between two newspaper collections."""

    def __init__(self):
        pass

    def align_topics(self, topics_a: Dict[int, List[str]],
                     topics_b: Dict[int, List[str]]) -> Dict[int, Tuple[int, float]]:
        """
        Align topics between two newspapers using Hungarian algorithm.

        Args:
            topics_a: Dictionary mapping topic_id to keyword list (newspaper A)
            topics_b: Dictionary mapping topic_id to keyword list (newspaper B)

        Returns:
            Dictionary mapping topic_id_a to (topic_id_b, similarity_score)
        """
        logger.info(f"Aligning {len(topics_a)} topics from A with {len(topics_b)} topics from B...")

        # Create topic documents (keywords joined as strings)
        topics_a_ids = list(topics_a.keys())
        topics_b_ids = list(topics_b.keys())

        topics_a_docs = [' '.join(topics_a[tid]) for tid in topics_a_ids]
        topics_b_docs = [' '.join(topics_b[tid]) for tid in topics_b_ids]

        # Vectorize using TF-IDF
        all_docs = topics_a_docs + topics_b_docs
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_docs)

        # Split back into A and B matrices
        n_topics_a = len(topics_a_ids)
        matrix_a = tfidf_matrix[:n_topics_a]
        matrix_b = tfidf_matrix[n_topics_a:]

        # Calculate cosine similarity between all topic pairs
        similarity_matrix = cosine_similarity(matrix_a, matrix_b)

        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")

        # Use Hungarian algorithm to find optimal alignment (maximize similarity)
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Negative to maximize

        # Create alignment mapping
        alignment = {}
        for i, j in zip(row_ind, col_ind):
            topic_a_id = topics_a_ids[i]
            topic_b_id = topics_b_ids[j]
            similarity = similarity_matrix[i, j]

            alignment[topic_a_id] = (topic_b_id, float(similarity))

            logger.debug(f"Topic {topic_a_id} (A) aligned with Topic {topic_b_id} (B), "
                         f"similarity: {similarity:.3f}")

        logger.info(f"Topic alignment complete: {len(alignment)} pairs")
        return alignment

    def find_common_topics(self, alignment: Dict[int, Tuple[int, float]],
                           threshold: float = 0.6) -> List[Tuple[int, int, float]]:
        """
        Filter aligned topics to find those with sufficient similarity.

        Args:
            alignment: Topic alignment mapping from align_topics()
            threshold: Minimum similarity threshold

        Returns:
            List of (topic_a_id, topic_b_id, similarity) tuples
        """
        common_topics = []

        for topic_a_id, (topic_b_id, similarity) in alignment.items():
            if similarity >= threshold:
                common_topics.append((topic_a_id, topic_b_id, similarity))

        logger.info(f"Found {len(common_topics)} common topics (threshold={threshold})")
        return common_topics

    def compare_topic_coverage(self, articles_a: List[Article],
                                articles_b: List[Article],
                                common_topics: List[Tuple[int, int, float]]) -> Dict:
        """
        Compare topic coverage (number of articles) between newspapers.

        Args:
            articles_a: Articles from newspaper A
            articles_b: Articles from newspaper B
            common_topics: List of common topic pairs

        Returns:
            Dictionary with coverage comparison
        """
        logger.info("Comparing topic coverage...")

        coverage = []

        for topic_a_id, topic_b_id, similarity in common_topics:
            # Count articles in A where this topic is dominant
            count_a = 0
            for article in articles_a:
                if article.topics:
                    dominant = max(article.topics, key=lambda x: x[1])[0]
                    if dominant == topic_a_id:
                        count_a += 1

            # Count articles in B where this topic is dominant
            count_b = 0
            for article in articles_b:
                if article.topics:
                    dominant = max(article.topics, key=lambda x: x[1])[0]
                    if dominant == topic_b_id:
                        count_b += 1

            coverage.append({
                'topic_a_id': topic_a_id,
                'topic_b_id': topic_b_id,
                'similarity': similarity,
                'count_a': count_a,
                'count_b': count_b,
                'difference': count_a - count_b
            })

        logger.info(f"Coverage comparison complete for {len(coverage)} topics")
        return {'coverage': coverage}

    def compare_sentiment_on_common_topics(self, articles_a: List[Article],
                                            articles_b: List[Article],
                                            common_topics: List[Tuple[int, int, float]]) -> Dict:
        """
        Compare sentiment for common topics between newspapers.

        Args:
            articles_a: Articles from newspaper A
            articles_b: Articles from newspaper B
            common_topics: List of common topic pairs

        Returns:
            Dictionary with sentiment comparison
        """
        logger.info("Comparing sentiment on common topics...")

        comparisons = []

        # Sentiment label to numeric mapping
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}

        for topic_a_id, topic_b_id, similarity in common_topics:
            # Get articles for topic A
            sentiments_a = []
            for article in articles_a:
                if article.topics:
                    dominant = max(article.topics, key=lambda x: x[1])[0]
                    if dominant == topic_a_id and article.sentiment:
                        label = article.sentiment['label'].lower()
                        # Normalize label
                        if 'pos' in label:
                            sentiments_a.append(1)
                        elif 'neg' in label:
                            sentiments_a.append(-1)
                        else:
                            sentiments_a.append(0)

            # Get articles for topic B
            sentiments_b = []
            for article in articles_b:
                if article.topics:
                    dominant = max(article.topics, key=lambda x: x[1])[0]
                    if dominant == topic_b_id and article.sentiment:
                        label = article.sentiment['label'].lower()
                        if 'pos' in label:
                            sentiments_b.append(1)
                        elif 'neg' in label:
                            sentiments_b.append(-1)
                        else:
                            sentiments_b.append(0)

            if not sentiments_a or not sentiments_b:
                logger.debug(f"Insufficient sentiment data for topic pair ({topic_a_id}, {topic_b_id})")
                continue

            # Calculate statistics
            mean_a = np.mean(sentiments_a)
            mean_b = np.mean(sentiments_b)
            difference = mean_a - mean_b

            # Perform t-test if we have enough samples
            if len(sentiments_a) >= 2 and len(sentiments_b) >= 2:
                t_stat, p_value = stats.ttest_ind(sentiments_a, sentiments_b)
            else:
                t_stat, p_value = None, None

            comparisons.append({
                'topic_a_id': topic_a_id,
                'topic_b_id': topic_b_id,
                'topic_similarity': similarity,
                'mean_sentiment_a': float(mean_a),
                'mean_sentiment_b': float(mean_b),
                'sentiment_difference': float(difference),
                'n_articles_a': len(sentiments_a),
                'n_articles_b': len(sentiments_b),
                't_statistic': float(t_stat) if t_stat is not None else None,
                'p_value': float(p_value) if p_value is not None else None,
                'significant': p_value < 0.05 if p_value is not None else None
            })

        logger.info(f"Sentiment comparison complete for {len(comparisons)} topic pairs")
        return {'sentiment_comparison': comparisons}

    def identify_unique_topics(self, topics_a: Dict[int, List[str]],
                                topics_b: Dict[int, List[str]],
                                common_topics: List[Tuple[int, int, float]]) -> Dict:
        """
        Identify topics unique to each newspaper.

        Args:
            topics_a: Topics from newspaper A
            topics_b: Topics from newspaper B
            common_topics: List of common topic pairs

        Returns:
            Dictionary with unique topics for each newspaper
        """
        # Extract common topic IDs
        common_a_ids = set([t[0] for t in common_topics])
        common_b_ids = set([t[1] for t in common_topics])

        # Find unique topics
        unique_to_a = {}
        for topic_id, keywords in topics_a.items():
            if topic_id not in common_a_ids:
                unique_to_a[topic_id] = keywords

        unique_to_b = {}
        for topic_id, keywords in topics_b.items():
            if topic_id not in common_b_ids:
                unique_to_b[topic_id] = keywords

        logger.info(f"Unique topics: {len(unique_to_a)} in A, {len(unique_to_b)} in B")

        return {
            'unique_to_a': unique_to_a,
            'unique_to_b': unique_to_b
        }

    def compare_vocabularies(self, articles_a: List[Article],
                              articles_b: List[Article]) -> Dict:
        """
        Compare vocabularies between two newspapers.

        Args:
            articles_a: Articles from newspaper A
            articles_b: Articles from newspaper B

        Returns:
            Dictionary with vocabulary comparison
        """
        logger.info("Comparing vocabularies...")

        # Extract vocabularies
        vocab_a = set()
        for article in articles_a:
            if article.cleaned_content:
                vocab_a.update(article.cleaned_content.split())

        vocab_b = set()
        for article in articles_b:
            if article.cleaned_content:
                vocab_b.update(article.cleaned_content.split())

        # Calculate overlap
        common_words = vocab_a & vocab_b
        unique_to_a = vocab_a - vocab_b
        unique_to_b = vocab_b - vocab_a

        overlap_pct = len(common_words) / len(vocab_a | vocab_b) * 100 if (vocab_a | vocab_b) else 0

        comparison = {
            'vocab_size_a': len(vocab_a),
            'vocab_size_b': len(vocab_b),
            'common_words_count': len(common_words),
            'unique_to_a_count': len(unique_to_a),
            'unique_to_b_count': len(unique_to_b),
            'overlap_percentage': overlap_pct
        }

        logger.info(f"Vocabulary comparison: {len(vocab_a)} (A), {len(vocab_b)} (B), "
                    f"{len(common_words)} common ({overlap_pct:.1f}% overlap)")

        return comparison

    def extract_distinctive_terms(self, articles_a: List[Article],
                                   articles_b: List[Article],
                                   top_n: int = 50,
                                   min_word_length: int = 3) -> Dict:
        """
        Extract distinctive terms for each newspaper using TF-IDF.

        Args:
            articles_a: Articles from newspaper A
            articles_b: Articles from newspaper B
            top_n: Number of top terms to extract
            min_word_length: Minimum word length to include

        Returns:
            Dictionary with distinctive terms
        """
        logger.info(f"Extracting top {top_n} distinctive terms...")

        # Combine all texts for each newspaper
        text_a = ' '.join([a.cleaned_content for a in articles_a if a.cleaned_content])
        text_b = ' '.join([a.cleaned_content for a in articles_b if a.cleaned_content])

        # Vectorize with token pattern to filter short words
        vectorizer = TfidfVectorizer(
            max_features=top_n * 3,
            token_pattern=rf'\b[a-zA-ZàèéìòùÀÈÉÌÒÙčšžČŠŽ]{{{min_word_length},}}\b'
        )
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])

        feature_names = vectorizer.get_feature_names_out()

        # Get TF-IDF scores
        scores_a = tfidf_matrix[0].toarray().ravel()
        scores_b = tfidf_matrix[1].toarray().ravel()

        # Calculate distinctiveness: high in A, low in B (and vice versa)
        distinctiveness_a = scores_a - scores_b
        distinctiveness_b = scores_b - scores_a

        # Get top distinctive terms (only those with positive distinctiveness)
        top_a_indices = distinctiveness_a.argsort()[-top_n * 2:][::-1]
        top_b_indices = distinctiveness_b.argsort()[-top_n * 2:][::-1]

        # Build lists with deduplication
        seen_a = set()
        distinctive_a = []
        for i in top_a_indices:
            word = feature_names[i]
            if distinctiveness_a[i] > 0 and word not in seen_a:
                seen_a.add(word)
                distinctive_a.append((word, float(distinctiveness_a[i])))
                if len(distinctive_a) >= top_n:
                    break

        seen_b = set()
        distinctive_b = []
        for i in top_b_indices:
            word = feature_names[i]
            if distinctiveness_b[i] > 0 and word not in seen_b:
                seen_b.add(word)
                distinctive_b.append((word, float(distinctiveness_b[i])))
                if len(distinctive_b) >= top_n:
                    break

        logger.info(f"Extracted {len(distinctive_a)} distinctive terms for A, "
                    f"{len(distinctive_b)} for B")

        return {
            'distinctive_to_a': distinctive_a,
            'distinctive_to_b': distinctive_b
        }

    def full_comparison(self, articles_a: List[Article],
                        articles_b: List[Article],
                        topics_a: Dict[int, List[str]],
                        topics_b: Dict[int, List[str]],
                        similarity_threshold: float = 0.6) -> Dict:
        """
        Complete comparative analysis pipeline.

        Args:
            articles_a: Articles from newspaper A
            articles_b: Articles from newspaper B
            topics_a: Topics from newspaper A
            topics_b: Topics from newspaper B
            similarity_threshold: Threshold for common topics

        Returns:
            Dictionary with all comparison results
        """
        logger.info("Starting full comparative analysis...")

        # Align topics
        alignment = self.align_topics(topics_a, topics_b)

        # Find common topics
        common_topics = self.find_common_topics(alignment, threshold=similarity_threshold)

        # Coverage comparison
        coverage = self.compare_topic_coverage(articles_a, articles_b, common_topics)

        # Sentiment comparison
        sentiment = self.compare_sentiment_on_common_topics(articles_a, articles_b, common_topics)

        # Unique topics
        unique = self.identify_unique_topics(topics_a, topics_b, common_topics)

        # Vocabulary comparison
        vocab = self.compare_vocabularies(articles_a, articles_b)

        # Distinctive terms
        distinctive = self.extract_distinctive_terms(articles_a, articles_b)

        results = {
            'topic_alignment': alignment,
            'common_topics': common_topics,
            'coverage_comparison': coverage,
            'sentiment_comparison': sentiment,
            'unique_topics': unique,
            'vocabulary_comparison': vocab,
            'distinctive_terms': distinctive,
            'summary': {
                'n_topics_a': len(topics_a),
                'n_topics_b': len(topics_b),
                'n_common_topics': len(common_topics),
                'n_unique_to_a': len(unique['unique_to_a']),
                'n_unique_to_b': len(unique['unique_to_b'])
            }
        }

        logger.info("Full comparative analysis complete")
        logger.info(f"Summary: {len(topics_a)} topics in A, {len(topics_b)} topics in B, "
                    f"{len(common_topics)} common topics")

        return results
