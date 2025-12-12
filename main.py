"""
Main script for historical newspaper NLP analysis pipeline.

Analyzes and compares two historical newspapers from Trieste (Sept 9, 1902):
- Il Piccolo (Italian)
- Edinost (Slovenian)

Performs: topic analysis (LDA + BERTopic), TF-IDF, clustering, sentiment analysis,
and comparative analysis between the two newspapers.
"""

import logging
import json
from pathlib import Path

# Preprocessing
from src.preprocessing.text_loader import load_all_texts
from src.preprocessing.article_parser import parse_and_validate
from src.preprocessing.text_cleaner import clean_articles, get_token_statistics

# Analysis
from src.analysis.tfidf_analysis import TFIDFAnalyzer
from src.analysis.topic_modeling import TopicModeler
from src.analysis.clustering import Clusterer
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.comparative import ComparativeAnalyzer

# Visualization
from src.visualization.topic_plots import create_all_topic_plots
from src.visualization.tfidf_plots import create_all_tfidf_plots
from src.visualization.sentiment_plots import create_all_sentiment_plots
from src.visualization.comparison_plots import create_all_comparison_plots

# Utilities
from src.utils.file_utils import (
    ensure_dirs,
    save_articles_json,
    save_newspaper_json,
    save_dict_json,
    save_pickle
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("HISTORICAL NEWSPAPER NLP ANALYSIS PIPELINE")
    logger.info("Il Piccolo (IT) vs Edinost (SL) - September 9, 1902")
    logger.info("=" * 80)

    # Setup directories
    ensure_dirs([
        'outputs/parsed',
        'outputs/models',
        'outputs/results',
        'outputs/visualizations'
    ])

    # ========== PHASE 1: PREPROCESSING ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: PREPROCESSING")
    logger.info("=" * 80)

    # Load texts
    logger.info("\n--- Loading texts ---")
    piccolo_text, edinost_text, piccolo_obj, edinost_obj = load_all_texts()

    # Parse articles
    logger.info("\n--- Parsing articles ---")
    piccolo_articles = parse_and_validate(piccolo_text, 'il_piccolo', language='it')
    edinost_articles = parse_and_validate(edinost_text, 'edinost', language='sl')

    logger.info(f"Extracted {len(piccolo_articles)} articles from Il Piccolo")
    logger.info(f"Extracted {len(edinost_articles)} articles from Edinost")

    # Add articles to newspaper objects
    for article in piccolo_articles:
        piccolo_obj.add_article(article)
    for article in edinost_articles:
        edinost_obj.add_article(article)

    # Clean articles
    logger.info("\n--- Cleaning articles ---")
    clean_articles(piccolo_articles, 'it')
    clean_articles(edinost_articles, 'sl')

    # Token statistics
    logger.info("\n--- Token statistics ---")
    piccolo_stats = get_token_statistics(piccolo_articles)
    edinost_stats = get_token_statistics(edinost_articles)

    logger.info(f"Il Piccolo: {piccolo_stats}")
    logger.info(f"Edinost: {edinost_stats}")

    # Save parsed articles
    logger.info("\n--- Saving parsed articles ---")
    save_articles_json(piccolo_articles, 'outputs/parsed/il_piccolo_articles.json')
    save_articles_json(edinost_articles, 'outputs/parsed/edinost_articles.json')

    # ========== PHASE 2: SEPARATE NLP ANALYSIS ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: SEPARATE NLP ANALYSIS")
    logger.info("=" * 80)

    # ----- TF-IDF Analysis -----
    logger.info("\n--- TF-IDF Analysis ---")

    logger.info("Il Piccolo TF-IDF...")
    piccolo_tfidf = TFIDFAnalyzer(max_features=100, min_df=2, max_df=0.8)
    piccolo_tfidf_results = piccolo_tfidf.analyze(piccolo_articles, n_global=20, n_per_article=10)
    save_dict_json(piccolo_tfidf_results, 'outputs/results/il_piccolo_tfidf.json')
    save_pickle(piccolo_tfidf.vectorizer, 'outputs/models/il_piccolo_tfidf_vectorizer.pkl')

    logger.info("Edinost TF-IDF...")
    edinost_tfidf = TFIDFAnalyzer(max_features=100, min_df=2, max_df=0.8)
    edinost_tfidf_results = edinost_tfidf.analyze(edinost_articles, n_global=20, n_per_article=10)
    save_dict_json(edinost_tfidf_results, 'outputs/results/edinost_tfidf.json')
    save_pickle(edinost_tfidf.vectorizer, 'outputs/models/edinost_tfidf_vectorizer.pkl')

    logger.info(f"Il Piccolo top keywords: {piccolo_tfidf_results['global_keywords'][:5]}")
    logger.info(f"Edinost top keywords: {edinost_tfidf_results['global_keywords'][:5]}")

    # ----- Topic Modeling -----
    logger.info("\n--- Topic Modeling (LDA) ---")

    logger.info("Il Piccolo LDA...")
    piccolo_topic_modeler = TopicModeler()
    piccolo_lda_results = piccolo_topic_modeler.analyze_lda(piccolo_articles, num_topics=8)
    save_dict_json(piccolo_lda_results, 'outputs/results/il_piccolo_lda_topics.json')
    piccolo_topic_modeler.lda_model.save('outputs/models/il_piccolo_lda.model')

    logger.info("Edinost LDA...")
    edinost_topic_modeler = TopicModeler()
    edinost_lda_results = edinost_topic_modeler.analyze_lda(edinost_articles, num_topics=6)
    save_dict_json(edinost_lda_results, 'outputs/results/edinost_lda_topics.json')
    edinost_topic_modeler.lda_model.save('outputs/models/edinost_lda.model')

    logger.info(f"Il Piccolo LDA: {piccolo_lda_results['num_topics']} topics, "
                f"coherence: {piccolo_lda_results['coherence_score']:.3f}")
    logger.info(f"Edinost LDA: {edinost_lda_results['num_topics']} topics, "
                f"coherence: {edinost_lda_results['coherence_score']:.3f}")

    # Note: BERTopic skipped for now (can be slow on CPU)
    # Uncomment to enable:
    # logger.info("\n--- Topic Modeling (BERTopic) ---")
    # piccolo_bertopic_results = piccolo_topic_modeler.analyze_bertopic(piccolo_articles, language='it')
    # edinost_bertopic_results = edinost_topic_modeler.analyze_bertopic(edinost_articles, language='sl')

    # ----- Clustering -----
    logger.info("\n--- Clustering ---")

    logger.info("Il Piccolo clustering...")
    piccolo_clusterer = Clusterer()
    piccolo_cluster_results = piccolo_clusterer.cluster(piccolo_articles, n_clusters=5, method='kmeans')
    save_dict_json(piccolo_cluster_results, 'outputs/results/il_piccolo_clusters.json')

    logger.info("Edinost clustering...")
    edinost_clusterer = Clusterer()
    edinost_cluster_results = edinost_clusterer.cluster(edinost_articles, n_clusters=4, method='kmeans')
    save_dict_json(edinost_cluster_results, 'outputs/results/edinost_clusters.json')

    logger.info(f"Il Piccolo: {piccolo_cluster_results['n_clusters']} clusters")
    logger.info(f"Edinost: {edinost_cluster_results['n_clusters']} clusters")

    # ----- Sentiment Analysis -----
    logger.info("\n--- Sentiment Analysis ---")

    # Use language-specific sentiment models for better accuracy
    # Italian: paride92/feel-ic-ita-sentiment (trained on Italian newspapers)
    # Slovenian: classla/xlm-r-parlasent (trained on parliamentary/formal texts)
    
    logger.info("Il Piccolo sentiment (using Italian newspaper model)...")
    piccolo_sentiment_analyzer = SentimentAnalyzer(language='it')
    piccolo_sentiment = piccolo_sentiment_analyzer.analyze_newspaper(piccolo_articles)
    save_dict_json(piccolo_sentiment, 'outputs/results/il_piccolo_sentiment.json')

    logger.info("Edinost sentiment (using Slovenian formal text model)...")
    edinost_sentiment_analyzer = SentimentAnalyzer(language='sl')
    edinost_sentiment = edinost_sentiment_analyzer.analyze_newspaper(edinost_articles)
    save_dict_json(edinost_sentiment, 'outputs/results/edinost_sentiment.json')

    logger.info(f"Il Piccolo sentiment: {piccolo_sentiment['percentages']}")
    logger.info(f"Edinost sentiment: {edinost_sentiment['percentages']}")

    # Save updated articles (with all analysis results)
    save_articles_json(piccolo_articles, 'outputs/parsed/il_piccolo_articles_analyzed.json')
    save_articles_json(edinost_articles, 'outputs/parsed/edinost_articles_analyzed.json')

    # ========== PHASE 3: COMPARATIVE ANALYSIS ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: COMPARATIVE ANALYSIS")
    logger.info("=" * 80)

    comparative = ComparativeAnalyzer()

    # Full comparison
    logger.info("\n--- Running full comparative analysis ---")
    comparison_results = comparative.full_comparison(
        articles_a=piccolo_articles,
        articles_b=edinost_articles,
        topics_a=piccolo_lda_results['topics'],
        topics_b=edinost_lda_results['topics'],
        similarity_threshold=0.4  # Lowered threshold for cross-lingual comparison
    )

    # Save comparison results
    save_dict_json(comparison_results, 'outputs/results/comparative_analysis.json')

    # Log summary
    logger.info("\n--- Comparative Analysis Summary ---")
    logger.info(f"Common topics: {comparison_results['summary']['n_common_topics']}")
    logger.info(f"Unique to Il Piccolo: {comparison_results['summary']['n_unique_to_a']}")
    logger.info(f"Unique to Edinost: {comparison_results['summary']['n_unique_to_b']}")
    logger.info(f"Vocabulary overlap: {comparison_results['vocabulary_comparison']['overlap_percentage']:.1f}%")

    # Log common topics
    logger.info("\nCommon topics found:")
    for topic_a_id, topic_b_id, similarity in comparison_results['common_topics']:
        keywords_a = piccolo_lda_results['topics'][topic_a_id][:5]
        keywords_b = edinost_lda_results['topics'][topic_b_id][:5]
        logger.info(f"  Topic {topic_a_id} (IT): {keywords_a}")
        logger.info(f"  Topic {topic_b_id} (SL): {keywords_b}")
        logger.info(f"  Similarity: {similarity:.3f}\n")

    # ========== PHASE 4: FINALIZATION ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: FINALIZATION")
    logger.info("=" * 80)

    # Update newspaper objects with analysis results
    piccolo_obj.overall_sentiment = piccolo_sentiment
    piccolo_obj.vocabulary_size = piccolo_stats['vocabulary_size']

    edinost_obj.overall_sentiment = edinost_sentiment
    edinost_obj.vocabulary_size = edinost_stats['vocabulary_size']

    # Save complete newspaper objects
    save_newspaper_json(piccolo_obj, 'outputs/results/il_piccolo_complete.json')
    save_newspaper_json(edinost_obj, 'outputs/results/edinost_complete.json')

    # Create summary report
    summary = {
        'preprocessing': {
            'il_piccolo_articles': len(piccolo_articles),
            'edinost_articles': len(edinost_articles),
            'il_piccolo_stats': piccolo_stats,
            'edinost_stats': edinost_stats
        },
        'tfidf': {
            'il_piccolo': piccolo_tfidf_results,
            'edinost': edinost_tfidf_results
        },
        'topic_modeling': {
            'il_piccolo': piccolo_lda_results,
            'edinost': edinost_lda_results
        },
        'clustering': {
            'il_piccolo': piccolo_cluster_results,
            'edinost': edinost_cluster_results
        },
        'sentiment': {
            'il_piccolo': piccolo_sentiment,
            'edinost': edinost_sentiment
        },
        'comparative': comparison_results['summary']
    }

    save_dict_json(summary, 'outputs/results/analysis_summary.json')

    # ========== PHASE 5: VISUALIZATION ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: VISUALIZATION")
    logger.info("=" * 80)

    # Create visualization directories
    ensure_dirs([
        'outputs/visualizations',
        'outputs/visualizations/topic_wordclouds'
    ])

    # Topic visualizations for both newspapers
    logger.info("\n--- Creating topic visualizations ---")
    logger.info("Il Piccolo topic plots...")
    create_all_topic_plots(
        articles=piccolo_articles,
        topics=piccolo_lda_results['topics'],
        newspaper_name='Il Piccolo',
        output_dir='outputs/visualizations'
    )

    logger.info("Edinost topic plots...")
    create_all_topic_plots(
        articles=edinost_articles,
        topics=edinost_lda_results['topics'],
        newspaper_name='Edinost',
        output_dir='outputs/visualizations'
    )

    # TF-IDF visualizations
    logger.info("\n--- Creating TF-IDF visualizations ---")
    create_all_tfidf_plots(
        keywords_a=piccolo_tfidf_results['global_keywords'],
        keywords_b=edinost_tfidf_results['global_keywords'],
        newspaper_a='Il Piccolo',
        newspaper_b='Edinost',
        output_dir='outputs/visualizations'
    )

    # Sentiment visualizations
    logger.info("\n--- Creating sentiment visualizations ---")
    create_all_sentiment_plots(
        sentiment_a=piccolo_sentiment,
        sentiment_b=edinost_sentiment,
        newspaper_a='Il Piccolo',
        newspaper_b='Edinost',
        output_dir='outputs/visualizations',
        articles_a=piccolo_articles,
        articles_b=edinost_articles
    )

    # Comparison visualizations
    logger.info("\n--- Creating comparison visualizations ---")
    create_all_comparison_plots(
        comparative_results=comparison_results,
        topics_a=piccolo_lda_results['topics'],
        topics_b=edinost_lda_results['topics'],
        summary_data=summary,
        newspaper_a='Il Piccolo',
        newspaper_b='Edinost',
        output_dir='outputs/visualizations'
    )

    logger.info("\nAll visualizations created successfully!")
    logger.info(f"Visualizations saved to: {Path('outputs/visualizations').absolute()}")

    # ========== COMPLETE ==========
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total articles analyzed: {len(piccolo_articles) + len(edinost_articles)}")
    logger.info(f"Results saved to: {Path('outputs').absolute()}")
    logger.info("\nKey findings:")
    logger.info(f"  - {len(piccolo_articles)} Il Piccolo articles ({piccolo_stats['vocabulary_size']} unique words)")
    logger.info(f"  - {len(edinost_articles)} Edinost articles ({edinost_stats['vocabulary_size']} unique words)")
    logger.info(f"  - {comparison_results['summary']['n_common_topics']} common topics identified")
    logger.info(f"  - {comparison_results['vocabulary_comparison']['overlap_percentage']:.1f}% vocabulary overlap")
    logger.info("\nOutput locations:")
    logger.info("  - Analysis results: outputs/results/")
    logger.info("  - Visualizations: outputs/visualizations/")
    logger.info("  - Parsed articles: outputs/parsed/")


if __name__ == "__main__":
    main()
