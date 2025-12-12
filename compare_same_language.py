"""
Script for comparing articles in the same language.

Allows comparison of:
- Different pages of the same newspaper
- Different sections of a single newspaper
- Any two sets of Italian or Slovenian articles

Usage:
    uv run python compare_same_language.py --language it --source1 page1 --source2 page2
    uv run python compare_same_language.py --language it --mode pages  # Compare all pages
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

# Preprocessing
from src.preprocessing.article_parser import parse_and_validate
from src.preprocessing.text_cleaner import clean_articles, get_token_statistics

# Analysis
from src.analysis.tfidf_analysis import TFIDFAnalyzer
from src.analysis.topic_modeling import TopicModeler
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.comparative import ComparativeAnalyzer

# Visualization
from src.visualization.topic_plots import create_all_topic_plots
from src.visualization.sentiment_plots import create_all_sentiment_plots
from src.visualization.comparison_plots import create_all_comparison_plots

# Utilities
from src.utils.file_utils import ensure_dirs, save_dict_json, save_articles_json
from src.models import Article, Newspaper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_italian_page(page_num: int) -> Tuple[str, List[Article]]:
    """
    Load a specific page of Il Piccolo.
    
    Args:
        page_num: Page number (1-4)
        
    Returns:
        Tuple of (raw text, parsed articles)
    """
    page_path = Path(f"data/il_piccolo_19020909_pagina{page_num}.txt")
    
    if not page_path.exists():
        raise FileNotFoundError(f"Page {page_num} not found at {page_path}")
    
    with open(page_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    articles = parse_and_validate(text, f'il_piccolo_page{page_num}', language='it')
    
    return text, articles


def analyze_article_set(
    articles: List[Article], 
    name: str, 
    language: str = 'it',
    output_prefix: str = ''
) -> dict:
    """
    Perform full analysis on a set of articles.
    
    Args:
        articles: List of articles to analyze
        name: Name for this set (e.g., "Page 1", "Section A")
        language: Language code ('it' or 'sl')
        output_prefix: Prefix for output files
        
    Returns:
        Dictionary with all analysis results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {name}")
    logger.info(f"{'='*60}")
    
    results = {'name': name, 'language': language}
    
    # Clean articles
    logger.info(f"Cleaning {len(articles)} articles...")
    clean_articles(articles, language)
    results['token_stats'] = get_token_statistics(articles)
    
    # TF-IDF Analysis
    logger.info("Running TF-IDF analysis...")
    tfidf_analyzer = TFIDFAnalyzer(max_features=100, min_df=1, max_df=0.9)
    results['tfidf'] = tfidf_analyzer.analyze(articles, n_global=20, n_per_article=10)
    
    # Topic Modeling (LDA)
    logger.info("Running topic modeling (LDA)...")
    topic_modeler = TopicModeler()
    num_topics = min(5, max(2, len(articles) // 3))  # Adaptive topic count
    results['lda'] = topic_modeler.analyze_lda(articles, num_topics=num_topics)
    
    # Sentiment Analysis
    logger.info(f"Running sentiment analysis (language: {language})...")
    sentiment_analyzer = SentimentAnalyzer(language=language)
    results['sentiment'] = sentiment_analyzer.analyze_newspaper(articles)
    
    # Save individual results
    if output_prefix:
        save_dict_json(results['tfidf'], f'outputs/results/{output_prefix}_tfidf.json')
        save_dict_json(results['lda'], f'outputs/results/{output_prefix}_lda.json')
        save_dict_json(results['sentiment'], f'outputs/results/{output_prefix}_sentiment.json')
        save_articles_json(articles, f'outputs/parsed/{output_prefix}_articles.json')
    
    results['articles'] = articles
    results['n_articles'] = len(articles)
    
    return results


def compare_article_sets(
    results_a: dict,
    results_b: dict,
    output_dir: str = 'outputs/results'
) -> dict:
    """
    Compare two sets of analyzed articles.
    
    Args:
        results_a: Analysis results for set A
        results_b: Analysis results for set B
        output_dir: Directory for output files
        
    Returns:
        Comparison results dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing: {results_a['name']} vs {results_b['name']}")
    logger.info(f"{'='*60}")
    
    comparative = ComparativeAnalyzer()
    
    comparison = comparative.full_comparison(
        articles_a=results_a['articles'],
        articles_b=results_b['articles'],
        topics_a=results_a['lda']['topics'],
        topics_b=results_b['lda']['topics'],
        similarity_threshold=0.3  # Lower threshold for same-language comparison
    )
    
    # Add sentiment comparison
    from src.analysis.sentiment import compare_sentiments
    comparison['sentiment_comparison'] = compare_sentiments(
        results_a['sentiment'],
        results_b['sentiment']
    )
    
    # Summary
    comparison['sets'] = {
        'a': {'name': results_a['name'], 'n_articles': results_a['n_articles']},
        'b': {'name': results_b['name'], 'n_articles': results_b['n_articles']}
    }
    
    return comparison


def compare_all_pages(language: str = 'it'):
    """
    Compare all pages of Il Piccolo newspaper.
    
    Args:
        language: Language code
    """
    ensure_dirs([
        'outputs/parsed',
        'outputs/results',
        'outputs/visualizations',
        'outputs/results/page_comparisons'
    ])
    
    logger.info("="*80)
    logger.info("SAME-LANGUAGE COMPARISON: Il Piccolo Pages")
    logger.info("="*80)
    
    # Load and analyze all pages
    page_results = {}
    for page_num in range(1, 5):
        try:
            logger.info(f"\n--- Loading Page {page_num} ---")
            text, articles = load_italian_page(page_num)
            
            page_results[page_num] = analyze_article_set(
                articles=articles,
                name=f"Il Piccolo - Pagina {page_num}",
                language=language,
                output_prefix=f"il_piccolo_page{page_num}"
            )
        except FileNotFoundError as e:
            logger.warning(f"Could not load page {page_num}: {e}")
    
    # Compare all page pairs
    logger.info("\n" + "="*80)
    logger.info("PAIRWISE PAGE COMPARISONS")
    logger.info("="*80)
    
    all_comparisons = {}
    pages = list(page_results.keys())
    
    for i, page_a in enumerate(pages):
        for page_b in pages[i+1:]:
            comparison_key = f"page{page_a}_vs_page{page_b}"
            logger.info(f"\nComparing Page {page_a} vs Page {page_b}...")
            
            comparison = compare_article_sets(
                results_a=page_results[page_a],
                results_b=page_results[page_b]
            )
            
            all_comparisons[comparison_key] = comparison
            save_dict_json(
                comparison, 
                f'outputs/results/page_comparisons/{comparison_key}.json'
            )
    
    # Overall summary
    summary = {
        'total_pages': len(page_results),
        'pages': {
            f"page_{num}": {
                'name': res['name'],
                'n_articles': res['n_articles'],
                'vocabulary_size': res['token_stats']['vocabulary_size'],
                'sentiment_distribution': res['sentiment']['percentages'],
                'top_keywords': res['tfidf']['global_keywords'][:10]
            }
            for num, res in page_results.items()
        },
        'comparisons': {
            key: {
                'common_topics': len(comp.get('common_topics', [])),
                'vocabulary_overlap': comp.get('vocabulary_comparison', {}).get('overlap_percentage', 0),
                'sentiment_diff': comp.get('sentiment_comparison', {}).get('differences', {})
            }
            for key, comp in all_comparisons.items()
        }
    }
    
    save_dict_json(summary, 'outputs/results/page_comparison_summary.json')
    
    # Create visualizations
    logger.info("\n" + "="*80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*80)
    
    ensure_dirs(['outputs/visualizations/page_comparisons'])
    
    # Create topic plots for each page
    for page_num, results in page_results.items():
        create_all_topic_plots(
            articles=results['articles'],
            topics=results['lda']['topics'],
            newspaper_name=f'Il Piccolo Page {page_num}',
            output_dir='outputs/visualizations/page_comparisons'
        )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    for page_num, results in page_results.items():
        logger.info(f"\nPage {page_num}:")
        logger.info(f"  Articles: {results['n_articles']}")
        logger.info(f"  Vocabulary: {results['token_stats']['vocabulary_size']} words")
        logger.info(f"  Sentiment: {results['sentiment']['percentages']}")
        logger.info(f"  Top keywords: {results['tfidf']['global_keywords'][:5]}")
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON COMPLETE!")
    logger.info(f"Results saved to: outputs/results/page_comparisons/")
    logger.info(f"Summary saved to: outputs/results/page_comparison_summary.json")
    logger.info("="*80)
    
    return summary, page_results, all_comparisons


def compare_two_sources(
    source1_path: str,
    source2_path: str,
    name1: str,
    name2: str,
    language: str = 'it'
):
    """
    Compare two custom text sources.
    
    Args:
        source1_path: Path to first text file
        source2_path: Path to second text file
        name1: Name for first source
        name2: Name for second source
        language: Language code
    """
    ensure_dirs([
        'outputs/parsed',
        'outputs/results',
        'outputs/visualizations'
    ])
    
    logger.info("="*80)
    logger.info(f"COMPARING: {name1} vs {name2}")
    logger.info("="*80)
    
    # Load texts
    with open(source1_path, 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(source2_path, 'r', encoding='utf-8') as f:
        text2 = f.read()
    
    # Parse articles
    articles1 = parse_and_validate(text1, name1.lower().replace(' ', '_'), language=language)
    articles2 = parse_and_validate(text2, name2.lower().replace(' ', '_'), language=language)
    
    # Analyze both
    results1 = analyze_article_set(
        articles=articles1,
        name=name1,
        language=language,
        output_prefix=name1.lower().replace(' ', '_')
    )
    
    results2 = analyze_article_set(
        articles=articles2,
        name=name2,
        language=language,
        output_prefix=name2.lower().replace(' ', '_')
    )
    
    # Compare
    comparison = compare_article_sets(results1, results2)
    
    # Save comparison
    output_name = f"{name1}_vs_{name2}".lower().replace(' ', '_')
    save_dict_json(comparison, f'outputs/results/{output_name}_comparison.json')
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{name1}:")
    logger.info(f"  Articles: {results1['n_articles']}")
    logger.info(f"  Sentiment: {results1['sentiment']['percentages']}")
    
    logger.info(f"\n{name2}:")
    logger.info(f"  Articles: {results2['n_articles']}")
    logger.info(f"  Sentiment: {results2['sentiment']['percentages']}")
    
    logger.info(f"\nComparison:")
    logger.info(f"  Common topics: {len(comparison.get('common_topics', []))}")
    logger.info(f"  Vocabulary overlap: {comparison.get('vocabulary_comparison', {}).get('overlap_percentage', 0):.1f}%")
    
    sent_diff = comparison.get('sentiment_comparison', {}).get('differences', {})
    logger.info(f"  Sentiment differences: {sent_diff}")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Compare articles in the same language'
    )
    parser.add_argument(
        '--language', '-l',
        choices=['it', 'sl'],
        default='it',
        help='Language of the articles (default: it)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['pages', 'custom'],
        default='pages',
        help='Comparison mode: pages (compare Il Piccolo pages) or custom'
    )
    parser.add_argument(
        '--source1',
        help='Path to first text file (for custom mode)'
    )
    parser.add_argument(
        '--source2',
        help='Path to second text file (for custom mode)'
    )
    parser.add_argument(
        '--name1',
        default='Source 1',
        help='Name for first source'
    )
    parser.add_argument(
        '--name2',
        default='Source 2',
        help='Name for second source'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'pages':
        compare_all_pages(language=args.language)
    elif args.mode == 'custom':
        if not args.source1 or not args.source2:
            parser.error("Custom mode requires --source1 and --source2")
        compare_two_sources(
            source1_path=args.source1,
            source2_path=args.source2,
            name1=args.name1,
            name2=args.name2,
            language=args.language
        )


if __name__ == "__main__":
    main()
