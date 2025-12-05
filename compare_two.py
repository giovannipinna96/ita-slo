#!/usr/bin/env python
"""
Quick script to compare two text files with sentiment and topic analysis.

Usage:
    uv run python compare_two.py file1.txt file2.txt --language it
    uv run python compare_two.py file1.txt file2.txt --language sl
    
    # With BERTopic (recommended for better topic discovery):
    uv run python compare_two.py file1.txt file2.txt --language it --use-bertopic
"""

import argparse
import logging
from pathlib import Path

from src.preprocessing.article_parser import parse_and_validate
from src.preprocessing.text_cleaner import clean_articles
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.topic_modeling import TopicModeler
from src.analysis.tfidf_analysis import TFIDFAnalyzer
from src.visualization.sentiment_plots import plot_all_articles_sentiment, plot_sentiment_comparison
from src.utils.file_utils import ensure_dirs, save_dict_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_file(file_path: str, name: str, language: str = 'it', use_bertopic: bool = False):
    """Analyze a single text file.
    
    Args:
        file_path: Path to the text file
        name: Display name for the file
        language: Language code ('it' or 'sl')
        use_bertopic: If True, also run BERTopic analysis (slower but more accurate)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {name}")
    logger.info(f"{'='*60}")
    
    # Load text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Parse articles
    articles = parse_and_validate(text, name.lower().replace(' ', '_'))
    logger.info(f"Found {len(articles)} articles/paragraphs")
    
    if not articles:
        logger.warning("No articles found!")
        return None, {}
    
    # Clean
    clean_articles(articles, language)
    
    # Sentiment analysis (sentence-level)
    logger.info("Running sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer(language=language, use_sentences=True)
    sentiment = sentiment_analyzer.analyze_newspaper(articles)
    
    # TF-IDF
    logger.info("Running TF-IDF...")
    tfidf = TFIDFAnalyzer(max_features=50, min_df=1, max_df=0.9)
    tfidf_results = tfidf.analyze(articles, n_global=20, n_per_article=5)
    
    # Topic modeling (LDA)
    logger.info("Running LDA topic modeling...")
    topic_modeler = TopicModeler()
    num_topics = min(3, max(2, len(articles) // 2))
    lda_results = topic_modeler.analyze_lda(articles, num_topics=num_topics)
    
    results = {
        'name': name,
        'n_articles': len(articles),
        'sentiment': sentiment,
        'tfidf': tfidf_results,
        'topics_lda': lda_results
    }
    
    # BERTopic analysis (optional, but recommended for Italian)
    if use_bertopic:
        logger.info("Running BERTopic topic modeling...")
        bertopic_modeler = TopicModeler()
        # Use same number of topics as LDA for consistency
        bertopic_results = bertopic_modeler.analyze_bertopic(articles, language=language, num_topics=num_topics)
        results['topics_bertopic'] = bertopic_results
        logger.info(f"BERTopic found {bertopic_results['num_topics']} topics")
    
    return articles, results


def extract_extreme_sentences(articles, name: str, top_n: int = 10):
    """
    Extract the most positive and most negative sentences from all articles.
    
    Args:
        articles: List of analyzed articles with sentence-level sentiment
        name: Name of the source
        top_n: Number of sentences to extract for each category
        
    Returns:
        Dictionary with top positive and negative sentences
    """
    all_sentences = []
    
    # Collect all sentences with their sentiment
    for article in articles:
        if not article.sentiment or 'sentence_details' not in article.sentiment:
            continue
        
        for sent_info in article.sentiment['sentence_details']:
            all_sentences.append({
                'sentence': sent_info.get('sentence', ''),
                'label': sent_info.get('label', 'neutral'),
                'score': sent_info.get('score', 0.5),
                'raw_score': sent_info.get('raw_score', None),
                'article_title': article.title[:50] if article.title else 'Untitled'
            })
    
    if not all_sentences:
        logger.warning(f"No sentence-level data found for {name}")
        return {'source': name, 'top_positive': [], 'top_negative': []}
    
    # Separate by label
    positive_sentences = [s for s in all_sentences if s['label'] == 'positive']
    negative_sentences = [s for s in all_sentences if s['label'] == 'negative']
    neutral_sentences = [s for s in all_sentences if s['label'] == 'neutral']
    
    # Sort by score (descending for positive, descending for negative too as higher score = more confident)
    positive_sentences.sort(key=lambda x: x['score'], reverse=True)
    negative_sentences.sort(key=lambda x: x['score'], reverse=True)
    
    # Take top N
    top_positive = positive_sentences[:top_n]
    top_negative = negative_sentences[:top_n]
    
    result = {
        'source': name,
        'total_sentences': len(all_sentences),
        'positive_count': len(positive_sentences),
        'negative_count': len(negative_sentences),
        'neutral_count': len(neutral_sentences),
        'top_positive': [
            {
                'rank': i + 1,
                'sentence': s['sentence'],
                'confidence': round(s['score'], 4),
                'article': s['article_title']
            }
            for i, s in enumerate(top_positive)
        ],
        'top_negative': [
            {
                'rank': i + 1,
                'sentence': s['sentence'],
                'confidence': round(s['score'], 4),
                'article': s['article_title']
            }
            for i, s in enumerate(top_negative)
        ]
    }
    
    logger.info(f"Extracted top {len(top_positive)} positive and {len(top_negative)} negative sentences for {name}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Compare two text files')
    parser.add_argument('file1', help='Path to first text file')
    parser.add_argument('file2', help='Path to second text file')
    parser.add_argument('--language', '-l', default='it', choices=['it', 'sl'],
                       help='Language (it=Italian, sl=Slovenian)')
    parser.add_argument('--name1', default=None, help='Name for first file')
    parser.add_argument('--name2', default=None, help='Name for second file')
    parser.add_argument('--output', '-o', default='outputs/comparison',
                       help='Output directory')
    parser.add_argument('--use-bertopic', '-b', action='store_true',
                       help='Enable BERTopic analysis (slower but more accurate topic discovery)')
    
    args = parser.parse_args()
    
    # Default names from filenames
    name1 = args.name1 or Path(args.file1).stem
    name2 = args.name2 or Path(args.file2).stem
    
    # Setup
    ensure_dirs([args.output])
    
    logger.info("="*70)
    logger.info("TWO-FILE COMPARISON")
    logger.info(f"File 1: {args.file1} -> {name1}")
    logger.info(f"File 2: {args.file2} -> {name2}")
    logger.info(f"Language: {args.language}")
    logger.info(f"BERTopic: {'enabled' if args.use_bertopic else 'disabled'}")
    logger.info("="*70)
    
    # Analyze both files
    articles1, results1 = analyze_file(args.file1, name1, args.language, args.use_bertopic)
    articles2, results2 = analyze_file(args.file2, name2, args.language, args.use_bertopic)
    
    if not articles1 or not articles2:
        logger.error("Failed to analyze one or both files")
        return
    
    # Save results
    output_dir = Path(args.output)
    save_dict_json(results1, output_dir / f"{name1.lower().replace(' ', '_')}_results.json")
    save_dict_json(results2, output_dir / f"{name2.lower().replace(' ', '_')}_results.json")
    
    # Extract top positive/negative sentences
    logger.info("\n" + "="*60)
    logger.info("Extracting extreme sentences...")
    logger.info("="*60)
    
    extreme1 = extract_extreme_sentences(articles1, name1, top_n=10)
    extreme2 = extract_extreme_sentences(articles2, name2, top_n=10)
    
    # Save extreme sentences to JSON
    save_dict_json(extreme1, output_dir / f"{name1.lower().replace(' ', '_')}_extreme_sentences.json")
    save_dict_json(extreme2, output_dir / f"{name2.lower().replace(' ', '_')}_extreme_sentences.json")
    
    # Create visualizations
    logger.info("\n" + "="*60)
    logger.info("Creating visualizations...")
    logger.info("="*60)
    
    # Per-article sentiment plots
    plot_all_articles_sentiment(articles1, name1, str(output_dir))
    plot_all_articles_sentiment(articles2, name2, str(output_dir))
    
    # Comparison plot
    plot_sentiment_comparison(
        results1['sentiment'], results2['sentiment'],
        name1, name2,
        str(output_dir / 'sentiment_comparison.png')
    )
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    logger.info(f"\n{name1}:")
    logger.info(f"  Articles: {results1['n_articles']}")
    logger.info(f"  Sentences analyzed: {results1['sentiment'].get('total_sentences_analyzed', 'N/A')}")
    logger.info(f"  Sentiment: {results1['sentiment']['percentages']}")
    logger.info(f"  Top keywords: {results1['tfidf']['global_keywords'][:5]}")
    logger.info(f"  LDA topics: {results1['topics_lda']['num_topics']}")
    if 'topics_bertopic' in results1:
        logger.info(f"  BERTopic topics: {results1['topics_bertopic']['num_topics']}")
        # Show BERTopic topic keywords
        for topic_id, topic_info in list(results1['topics_bertopic']['topics'].items())[:3]:
            logger.info(f"    Topic {topic_id}: {topic_info['keywords'][:5]}")
    
    logger.info(f"\n{name2}:")
    logger.info(f"  Articles: {results2['n_articles']}")
    logger.info(f"  Sentences analyzed: {results2['sentiment'].get('total_sentences_analyzed', 'N/A')}")
    logger.info(f"  Sentiment: {results2['sentiment']['percentages']}")
    logger.info(f"  Top keywords: {results2['tfidf']['global_keywords'][:5]}")
    logger.info(f"  LDA topics: {results2['topics_lda']['num_topics']}")
    if 'topics_bertopic' in results2:
        logger.info(f"  BERTopic topics: {results2['topics_bertopic']['num_topics']}")
        # Show BERTopic topic keywords
        for topic_id, topic_info in list(results2['topics_bertopic']['topics'].items())[:3]:
            logger.info(f"    Topic {topic_id}: {topic_info['keywords'][:5]}")
    
    logger.info(f"\nResults saved to: {output_dir.absolute()}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
