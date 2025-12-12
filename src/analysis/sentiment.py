"""
Module for sentiment analysis using transformer models.

Uses language-specific models for better accuracy:
- Italian: MilaNLProc/feel-it-italian-sentiment (FEEL-IT model, binary: positive/negative)
- Slovenian: classla/xlm-r-parlasent (trained on parliamentary/formal texts)

Supports sentence-level analysis with voting for more accurate article-level sentiment.
"""

import logging
import re
from typing import List, Dict, Optional
from collections import Counter
from transformers import pipeline
from tqdm import tqdm
import numpy as np

from src.models import Article

logger = logging.getLogger(__name__)


# Model configurations for different languages
SENTIMENT_MODELS = {
    'it': {
        'model_name': 'neuraly/bert-base-italian-cased-sentiment',
        'description': 'Italian sentiment (3-class: positive/neutral/negative)',
        'labels': ['negative', 'neutral', 'positive'],
        'task': 'sentiment-analysis'
    },
    'sl': {
        'model_name': 'classla/xlm-r-parlasent',
        'description': 'Slovenian parliamentary/formal text sentiment',
        'labels': None,  # Regression model, returns score
        'task': 'text-classification',
        'is_regression': True
    },
    'multilingual': {
        'model_name': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'description': 'Multilingual sentiment (fallback)',
        'labels': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
        'task': 'sentiment-analysis'
    }
}


class SentimentAnalyzer:
    """Sentiment analysis using language-specific models with sentence-level voting."""

    def __init__(self, language: str = 'multilingual', use_sentences: bool = True):
        """
        Initialize sentiment analyzer for a specific language.

        Args:
            language: Language code ('it' for Italian, 'sl' for Slovenian, 
                     'multilingual' for fallback)
            use_sentences: If True, analyze at sentence level and aggregate.
                          If False, analyze entire text (truncated to 512 chars)
        """
        self.language = language
        self.use_sentences = use_sentences
        self.model_config = SENTIMENT_MODELS.get(language, SENTIMENT_MODELS['multilingual'])
        self.model_name = self.model_config['model_name']
        self.model = None
        self.is_regression = self.model_config.get('is_regression', False)

    def load_model(self):
        """Load sentiment analysis model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading sentiment model for {self.language}: {self.model_name}")
            logger.info(f"Model description: {self.model_config['description']}")
            logger.info(f"Sentence-level analysis: {self.use_sentences}")
            try:
                if self.is_regression:
                    # For regression models like parlasent
                    self.model = pipeline(
                        "text-classification",
                        model=self.model_name,
                        tokenizer=self.model_name,
                        truncation=True,
                        max_length=512
                    )
                else:
                    self.model = pipeline(
                        "sentiment-analysis",
                        model=self.model_name,
                        top_k=None,  # Return all labels with scores
                        truncation=True,
                        max_length=512
                    )
                logger.info("Sentiment model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading sentiment model: {e}")
                raise

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Pattern for sentence splitting (handles multiple punctuation marks)
        # Works for Italian, Slovenian and other European languages
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-ZÀÈÉÌÒÙČŠŽ])'
        
        # Split by pattern
        sentences = re.split(sentence_pattern, text)
        
        # Filter out very short sentences (less than 10 chars)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 10]
        
        # If no sentences found, return the whole text as one sentence
        if not sentences:
            return [text] if len(text.strip()) >= 10 else []
        
        return sentences

    def _normalize_label(self, label_raw: str, score: float = None) -> str:
        """
        Normalize sentiment label to standard format (positive/neutral/negative).
        """
        label_lower = label_raw.lower()
        
        # For FEEL-IT Italian model
        if 'positive' in label_lower or 'positivo' in label_lower:
            return 'positive'
        elif 'negative' in label_lower or 'negativo' in label_lower:
            return 'negative'
        elif 'neutral' in label_lower or 'neutro' in label_lower:
            return 'neutral'
        
        # For nlptown model (star ratings)
        if '1 star' in label_lower or '2 star' in label_lower:
            return 'negative'
        elif '3 star' in label_lower:
            return 'neutral'
        elif '4 star' in label_lower or '5 star' in label_lower:
            return 'positive'
        
        # For regression models
        if score is not None:
            if score < -0.2:
                return 'negative'
            elif score > 0.2:
                return 'positive'
            else:
                return 'neutral'
        
        return 'neutral'

    def _process_regression_result(self, result: List[Dict]) -> Dict:
        """
        Process result from regression model (like parlasent).
        
        Parlasent returns scores from 0 to 5:
        - 0: Negative, 1: Mixed Negative
        - 2: Neutral Negative, 3: Neutral Positive
        - 4: Mixed Positive, 5: Positive
        
        We map to 3 categories: 0-1=negative, 2-3=neutral, 4-5=positive
        """
        if isinstance(result, list) and len(result) > 0:
            item = result[0] if isinstance(result[0], dict) else result
            
            if isinstance(item, dict):
                raw_score = item.get('score', 2.5)
                label_raw = item.get('label', 'LABEL_0')
                
                # Clip to valid range and convert to 3 categories
                clipped_score = np.clip(raw_score, 0, 5)
                category_index = int(np.round(clipped_score) // 2)
                
                three_category_mapper = {
                    0: 'negative',
                    1: 'neutral',
                    2: 'positive'
                }
                
                sentiment_label = three_category_mapper.get(category_index, 'neutral')
                confidence = 1.0 - abs(clipped_score - (category_index * 2 + 0.5)) / 2.5
                
                return {
                    'label': sentiment_label,
                    'score': float(confidence),
                    'raw_label': label_raw,
                    'raw_score': float(raw_score)
                }
        
        return {'label': 'neutral', 'score': 0.5, 'raw_label': 'unknown'}

    def analyze_single_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text segment.
        
        Args:
            text: Text to analyze (should be <= 512 chars)
            
        Returns:
            Dictionary with label and score
        """
        if self.model is None:
            self.load_model()

        if len(text) > 512:
            text = text[:512]

        try:
            result = self.model(text)
            
            if self.is_regression:
                return self._process_regression_result(result)
            
            if isinstance(result, list) and len(result) > 0:
                scores_list = result[0] if isinstance(result[0], list) else result
                best_sentiment = max(scores_list, key=lambda x: x['score'])
                label_raw = best_sentiment['label']
                sentiment_label = self._normalize_label(label_raw)

                return {
                    'label': sentiment_label,
                    'score': best_sentiment['score'],
                    'raw_label': label_raw
                }
            
            return {'label': 'neutral', 'score': 0.0}

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'neutral', 'score': 0.0, 'error': str(e)}

    def aggregate_sentence_sentiments(self, sentence_results: List[Dict]) -> Dict:
        """
        Aggregate sentence-level sentiments into article-level sentiment.
        
        Uses weighted voting based on confidence scores.
        
        Args:
            sentence_results: List of sentiment results for each sentence
            
        Returns:
            Aggregated sentiment dictionary
        """
        if not sentence_results:
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentence_count': 0,
                'vote_distribution': {}
            }
        
        # Count labels with weighted scores
        label_weights = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        label_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for result in sentence_results:
            label = result.get('label', 'neutral')
            score = result.get('score', 0.5)
            
            if label in label_weights:
                label_weights[label] += score
                label_counts[label] += 1
        
        # Find winning label (by weighted sum)
        winning_label = max(label_weights, key=label_weights.get)
        
        # Calculate confidence as percentage of votes for winning label
        total_sentences = len(sentence_results)
        vote_percentage = label_counts[winning_label] / total_sentences if total_sentences > 0 else 0
        
        # Calculate average confidence for winning label
        avg_confidence = (label_weights[winning_label] / label_counts[winning_label] 
                         if label_counts[winning_label] > 0 else 0.5)
        
        # Final score combines vote percentage and average confidence
        final_score = (vote_percentage + avg_confidence) / 2
        
        return {
            'label': winning_label,
            'score': float(final_score),
            'sentence_count': total_sentences,
            'vote_distribution': {
                'positive': label_counts['positive'],
                'neutral': label_counts['neutral'],
                'negative': label_counts['negative']
            },
            'vote_percentages': {
                'positive': label_counts['positive'] / total_sentences * 100 if total_sentences > 0 else 0,
                'neutral': label_counts['neutral'] / total_sentences * 100 if total_sentences > 0 else 0,
                'negative': label_counts['negative'] / total_sentences * 100 if total_sentences > 0 else 0
            },
            'weighted_scores': label_weights
        }

    def analyze_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Analyze sentiment of a text.
        
        If use_sentences=True, splits into sentences and aggregates.
        Otherwise, analyzes the first max_length characters.

        Args:
            text: Text to analyze
            max_length: Maximum text length for single-pass analysis

        Returns:
            Dictionary with label, score, and detailed results
        """
        if self.model is None:
            self.load_model()

        if not self.use_sentences:
            # Legacy mode: analyze truncated text
            return self.analyze_single_text(text[:max_length])
        
        # Sentence-level analysis with voting
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentence_count': 0
            }
        
        # Analyze each sentence
        sentence_results = []
        for sentence in sentences:
            result = self.analyze_single_text(sentence)
            sentence_results.append({
                'sentence': sentence,  # Full sentence, no truncation
                **result
            })
        
        # Aggregate results
        aggregated = self.aggregate_sentence_sentiments(sentence_results)
        aggregated['sentence_details'] = sentence_results
        
        return aggregated

    def analyze_article(self, article: Article) -> Dict:
        """
        Analyze sentiment of an article (modifies article in-place).

        Args:
            article: Article to analyze

        Returns:
            Sentiment dictionary with sentence-level details
        """
        # Use original content for sentence splitting (not lemmatized)
        text = article.content if article.content else article.cleaned_content

        if not text or len(text.strip()) < 10:
            logger.warning(f"Article '{article.title}' has insufficient text for sentiment analysis")
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentence_count': 0
            }

        sentiment = self.analyze_text(text)
        article.sentiment = sentiment

        return sentiment

    def analyze_articles(self, articles: List[Article]) -> None:
        """
        Analyze sentiment for all articles (modifies articles in-place).

        Args:
            articles: List of articles
        """
        if self.model is None:
            self.load_model()

        mode = "sentence-level" if self.use_sentences else "document-level"
        logger.info(f"Analyzing sentiment for {len(articles)} articles using {self.model_name} ({mode})...")

        for article in tqdm(articles, desc=f"Sentiment ({self.language})"):
            self.analyze_article(article)

        logger.info("Sentiment analysis complete")

    def compute_aggregated_sentiment(self, articles: List[Article]) -> Dict:
        """
        Compute aggregated sentiment statistics for a collection of articles.

        Args:
            articles: List of articles with sentiment results

        Returns:
            Dictionary with aggregated statistics
        """
        label_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        scores = []
        total_sentences = 0
        valid_articles = 0

        for article in articles:
            if article.sentiment and 'label' in article.sentiment:
                label = article.sentiment['label'].lower()

                if 'pos' in label:
                    label_counts['positive'] += 1
                elif 'neg' in label:
                    label_counts['negative'] += 1
                else:
                    label_counts['neutral'] += 1

                scores.append(article.sentiment.get('score', 0.5))
                total_sentences += article.sentiment.get('sentence_count', 1)
                valid_articles += 1

        if valid_articles > 0:
            label_percentages = {
                label: (count / valid_articles) * 100
                for label, count in label_counts.items()
            }
            avg_confidence = sum(scores) / len(scores) if scores else 0.0
        else:
            label_percentages = label_counts
            avg_confidence = 0.0

        aggregated = {
            'counts': label_counts,
            'percentages': label_percentages,
            'avg_confidence': avg_confidence,
            'total_articles': valid_articles,
            'total_sentences_analyzed': total_sentences,
            'model_used': self.model_name,
            'analysis_mode': 'sentence-level' if self.use_sentences else 'document-level'
        }

        logger.info(f"Aggregated sentiment: {label_percentages}")
        logger.info(f"Total sentences analyzed: {total_sentences}")
        return aggregated

    def analyze_newspaper(self, articles: List[Article]) -> Dict:
        """
        Complete sentiment analysis pipeline for a newspaper.

        Args:
            articles: List of articles

        Returns:
            Aggregated sentiment statistics
        """
        mode = "sentence-level" if self.use_sentences else "document-level"
        logger.info(f"Starting newspaper sentiment analysis ({mode}) with {self.model_name}...")

        self.analyze_articles(articles)
        aggregated = self.compute_aggregated_sentiment(articles)

        logger.info("Newspaper sentiment analysis complete")
        return aggregated


def compare_sentiments(sentiment_a: Dict, sentiment_b: Dict) -> Dict:
    """
    Compare sentiment distributions between two newspapers.

    Args:
        sentiment_a: Aggregated sentiment for newspaper A
        sentiment_b: Aggregated sentiment for newspaper B

    Returns:
        Dictionary with comparison metrics
    """
    percentages_a = sentiment_a.get('percentages', {})
    percentages_b = sentiment_b.get('percentages', {})

    differences = {}
    for label in ['positive', 'neutral', 'negative']:
        pct_a = percentages_a.get(label, 0)
        pct_b = percentages_b.get(label, 0)
        differences[label] = pct_a - pct_b

    comparison = {
        'newspaper_a': percentages_a,
        'newspaper_b': percentages_b,
        'differences': differences,
        'abs_difference': sum(abs(d) for d in differences.values()),
        'models_used': {
            'newspaper_a': sentiment_a.get('model_used', 'unknown'),
            'newspaper_b': sentiment_b.get('model_used', 'unknown')
        },
        'sentences_analyzed': {
            'newspaper_a': sentiment_a.get('total_sentences_analyzed', 0),
            'newspaper_b': sentiment_b.get('total_sentences_analyzed', 0)
        }
    }

    logger.info(f"Sentiment comparison: differences = {differences}")
    return comparison
