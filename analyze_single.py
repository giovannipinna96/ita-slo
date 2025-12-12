#!/usr/bin/env python
"""
Analyze a single text file with sentiment and topic analysis.

Usage:
    uv run python analyze_single.py data/piccolo_19020217.txt --language it
    uv run python analyze_single.py data/edinost_19020217.txt --language sl
    
    # With BERTopic (recommended for better topic discovery):
    uv run python analyze_single.py data/piccolo_19020217.txt --language it --use-bertopic
"""

import argparse
import logging
from pathlib import Path

from src.preprocessing.article_parser import parse_and_validate
from src.preprocessing.text_cleaner import clean_articles, get_token_statistics
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.topic_modeling import TopicModeler
from src.analysis.tfidf_analysis import TFIDFAnalyzer
from src.visualization.sentiment_plots import plot_all_articles_sentiment
from src.utils.file_utils import ensure_dirs, save_dict_json

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_file(file_path: str, name: str, language: str = 'it', 
                 use_bertopic: bool = False, output_dir: str = 'outputs/single'):
    """
    Analyze a single text file with full NLP pipeline.
    
    Args:
        file_path: Path to the text file
        name: Display name for the file
        language: Language code ('it' or 'sl')
        use_bertopic: If True, also run BERTopic analysis
        output_dir: Output directory for results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"SINGLE FILE ANALYSIS")
    logger.info(f"File: {file_path}")
    logger.info(f"Name: {name}")
    logger.info(f"Language: {language}")
    logger.info(f"BERTopic: {'enabled' if use_bertopic else 'disabled'}")
    logger.info(f"{'='*70}")
    
    # Setup output directory
    ensure_dirs([output_dir])
    output_path = Path(output_dir)
    
    # Load text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Parse articles
    logger.info("\n--- Parsing articles ---")
    articles = parse_and_validate(text, name.lower().replace(' ', '_'), language=language)
    logger.info(f"Found {len(articles)} articles/paragraphs")
    
    if not articles:
        logger.warning("No articles found!")
        return None
    
    # Clean articles
    logger.info("\n--- Cleaning text ---")
    clean_articles(articles, language)
    token_stats = get_token_statistics(articles)
    logger.info(f"Token statistics: {token_stats}")
    
    # Results dictionary (global)
    results = {
        'name': name,
        'file': str(file_path),
        'language': language,
        'n_articles': len(articles),
        'token_stats': token_stats
    }
    
    # Sentiment Analysis (global)
    logger.info("\n--- Sentiment Analysis ---")
    sentiment_analyzer = SentimentAnalyzer(language=language, use_sentences=True)
    sentiment = sentiment_analyzer.analyze_newspaper(articles)
    results['sentiment'] = sentiment
    
    logger.info(f"Sentiment distribution: {sentiment['percentages']}")
    logger.info(f"Total sentences analyzed: {sentiment.get('total_sentences_analyzed', 'N/A')}")
    
    # TF-IDF Analysis (global)
    logger.info("\n--- TF-IDF Analysis ---")
    tfidf = TFIDFAnalyzer(max_features=50, min_df=1, max_df=0.9)
    tfidf_results = tfidf.analyze(articles, n_global=20, n_per_article=5)
    results['tfidf'] = tfidf_results
    
    logger.info(f"Top keywords: {tfidf_results['global_keywords'][:10]}")
    
    # Topic Modeling (LDA) - global
    logger.info("\n--- Topic Modeling (LDA) ---")
    topic_modeler = TopicModeler()
    num_topics = min(5, max(2, len(articles) // 3))
    lda_results = topic_modeler.analyze_lda(articles, num_topics=num_topics)
    results['topics_lda'] = lda_results
    
    logger.info(f"LDA found {lda_results['num_topics']} topics")
    logger.info(f"Coherence score: {lda_results['coherence_score']:.4f}")
    
    for topic_id, keywords in lda_results['topics'].items():
        logger.info(f"  Topic {topic_id}: {keywords[:5]}")
    
    # BERTopic (optional) - global
    bertopic_results = None
    if use_bertopic:
        logger.info("\n--- Topic Modeling (BERTopic) ---")
        bertopic_modeler = TopicModeler()
        bertopic_results = bertopic_modeler.analyze_bertopic(
            articles, language=language, num_topics=num_topics
        )
        results['topics_bertopic'] = bertopic_results
        
        logger.info(f"BERTopic found {bertopic_results['num_topics']} topics")
        for topic_id, keywords in list(bertopic_results['topics'].items())[:5]:
            logger.info(f"  Topic {topic_id}: {keywords[:5]}")
    
    # Extract extreme sentences (global)
    logger.info("\n--- Extracting Extreme Sentences ---")
    extreme_sentences = extract_extreme_sentences(articles, name, top_n=10)
    results['extreme_sentences'] = extreme_sentences
    
    # === SAVE GLOBAL RESULTS ===
    logger.info("\n--- Saving Global Results ---")
    safe_name = name.lower().replace(' ', '_').replace('/', '_')
    save_dict_json(results, output_path / f"global_results.json")
    save_dict_json(extreme_sentences, output_path / f"global_extreme_sentences.json")
    
    # Create global visualizations
    plot_all_articles_sentiment(articles, name, str(output_path))
    
    # === SAVE PER-ARTICLE RESULTS ===
    logger.info("\n--- Saving Per-Article Results ---")
    articles_dir = output_path / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, article in enumerate(articles, start=1):
        # Create numbered subfolder (01, 02, 03, ...)
        article_folder = articles_dir / f"{idx:02d}"
        article_folder.mkdir(parents=True, exist_ok=True)
        
        # Save original article text
        with open(article_folder / "original_text.txt", 'w', encoding='utf-8') as f:
            f.write(f"TITOLO: {article.title}\n")
            f.write("=" * 50 + "\n\n")
            f.write(article.content)
        
        # Prepare article-level results
        article_results = {
            'article_number': idx,
            'title': article.title,
            'content_length': len(article.content),
            'language': article.language,
            'source': article.source
        }
        
        # Sentiment for this article
        if article.sentiment:
            article_results['sentiment'] = article.sentiment
        
        # Topics for this article (LDA)
        if article.topics:
            article_results['topics_lda'] = [
                {'topic_id': t[0], 'probability': float(t[1])} 
                for t in article.topics
            ]
            # Add topic keywords
            if lda_results and 'topics' in lda_results:
                article_results['topic_keywords'] = {}
                for topic_id, prob in article.topics:
                    if str(topic_id) in lda_results['topics']:
                        article_results['topic_keywords'][topic_id] = lda_results['topics'][str(topic_id)][:10]
        
        # TF-IDF keywords for this article
        if article.tfidf_keywords:
            article_results['tfidf_keywords'] = article.tfidf_keywords
        
        # BERTopic for this article (if available)
        if bertopic_results and 'article_topics' in bertopic_results:
            article_idx = idx - 1  # idx starts at 1
            if article_idx < len(bertopic_results['article_topics']):
                topic_info = bertopic_results['article_topics'][article_idx]
                topic_id = topic_info.get('topic', -1)
                
                article_results['topics_bertopic'] = {
                    'topic_id': topic_id,
                    'topic_name': topic_info.get('name', f'Topic {topic_id}')
                }
                
                # Add BERTopic keywords for this topic
                # Try both string and int keys
                if 'topics' in bertopic_results:
                    topics_dict = bertopic_results['topics']
                    keywords = None
                    
                    # Try string key first
                    if str(topic_id) in topics_dict:
                        keywords = topics_dict[str(topic_id)]
                    # Try int key
                    elif topic_id in topics_dict:
                        keywords = topics_dict[topic_id]
                    
                    if keywords:
                        # Keywords might be a list or a dict with 'keywords' key
                        if isinstance(keywords, dict) and 'keywords' in keywords:
                            article_results['topics_bertopic']['keywords'] = keywords['keywords'][:10]
                        elif isinstance(keywords, list):
                            article_results['topics_bertopic']['keywords'] = keywords[:10]
        
        # Save article results JSON
        save_dict_json(article_results, article_folder / "analysis.json")
        
        # === CREATE ARTICLE VISUALIZATIONS ===
        create_article_visualizations(article, article_folder, lda_results, bertopic_results, idx)
        
        logger.debug(f"Saved article {idx}: {article.title[:40]}...")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"  File: {file_path}")
    logger.info(f"  Articles: {len(articles)}")
    logger.info(f"  Vocabulary: {token_stats['vocabulary_size']} unique words")
    logger.info(f"  Sentiment: {sentiment['percentages']}")
    logger.info(f"  Top keywords: {tfidf_results['global_keywords'][:5]}")
    logger.info(f"  Topics (LDA): {lda_results['num_topics']}")
    if use_bertopic:
        logger.info(f"  Topics (BERT): {bertopic_results['num_topics']}")
    logger.info(f"\nResults saved to: {output_path.absolute()}")
    logger.info(f"  - Global analysis: global_results.json")
    logger.info(f"  - Per-article: articles/01/, articles/02/, ...")
    logger.info("="*70)
    
    return results


def create_article_visualizations(article, output_folder, lda_results, bertopic_results, idx):
    """
    Create visualizations for a single article.
    
    Generates:
    - sentiment_pie.png: Pie chart of sentiment distribution
    - topic_distribution_lda.png: Bar chart of LDA topic probabilities
    - topic_distribution_bertopic.png: Bar chart of BERTopic (if available)
    - wordcloud.png: Word cloud of article content
    """
    from pathlib import Path
    output_folder = Path(output_folder)
    
    # 1. SENTIMENT PIE CHART
    if article.sentiment and 'distribution' in article.sentiment:
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            dist = article.sentiment['distribution']
            labels = list(dist.keys())
            sizes = list(dist.values())
            colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
            chart_colors = [colors.get(l, '#95a5a6') for l in labels]
            
            # Filter out zero values
            non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, chart_colors) if s > 0]
            if non_zero:
                labels, sizes, chart_colors = zip(*non_zero)
                ax.pie(sizes, labels=labels, colors=chart_colors, autopct='%1.1f%%', startangle=90)
                ax.set_title(f"Sentiment - Art. {idx}", fontsize=12)
                plt.tight_layout()
                plt.savefig(output_folder / "sentiment_pie.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Could not create sentiment pie for article {idx}: {e}")
    
    # 2. LDA TOPIC DISTRIBUTION BAR CHART
    if article.topics and len(article.topics) > 0:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            topic_ids = [f"Topic {t[0]}" for t in article.topics]
            probs = [float(t[1]) for t in article.topics]
            
            bars = ax.barh(topic_ids, probs, color='steelblue')
            ax.set_xlabel('Probability')
            ax.set_title(f"LDA Topics - Art. {idx}", fontsize=12)
            ax.set_xlim(0, 1)
            
            # Add probability values on bars
            for bar, prob in zip(bars, probs):
                ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.2f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_folder / "topic_distribution_lda.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Could not create LDA topic chart for article {idx}: {e}")
    
    # 3. BERTOPIC TOPIC DISTRIBUTION (if available)
    if bertopic_results and 'article_topics' in bertopic_results:
        try:
            # Get BERTopic assignment for this article (idx-1 because enumerate starts at 1)
            article_idx = idx - 1
            if article_idx < len(bertopic_results.get('article_topics', [])):
                topic_info = bertopic_results['article_topics'][article_idx]
                topic_id = topic_info.get('topic', -1)
                
                # Get topic keywords
                topic_keywords = []
                if 'topics' in bertopic_results and str(topic_id) in bertopic_results['topics']:
                    topic_keywords = bertopic_results['topics'][str(topic_id)][:8]
                
                # Create a simple info plot for BERTopic
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.text(0.5, 0.7, f"BERTopic Assignment", fontsize=14, fontweight='bold',
                       ha='center', transform=ax.transAxes)
                ax.text(0.5, 0.5, f"Topic {topic_id}", fontsize=20, color='darkorange',
                       ha='center', transform=ax.transAxes)
                if topic_keywords:
                    keywords_str = ", ".join(topic_keywords[:5])
                    ax.text(0.5, 0.25, f"Keywords: {keywords_str}", fontsize=10,
                           ha='center', transform=ax.transAxes, style='italic')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(output_folder / "topic_distribution_bertopic.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            logger.debug(f"Could not create BERTopic chart for article {idx}: {e}")
    
    # 3. WORD CLOUD
    if WORDCLOUD_AVAILABLE:
        try:
            text = article.cleaned_content if article.cleaned_content else article.content
            if text and len(text.strip()) > 10:
                wc = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"Word Cloud - Art. {idx}", fontsize=12)
                plt.tight_layout()
                plt.savefig(output_folder / "wordcloud.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            logger.debug(f"Could not create word cloud for article {idx}: {e}")


def extract_extreme_sentences(articles, name: str, top_n: int = 10):
    """Extract the most positive and most negative sentences."""
    all_sentences = []
    
    for article in articles:
        if not article.sentiment or 'sentence_details' not in article.sentiment:
            continue
        
        for sent_info in article.sentiment['sentence_details']:
            all_sentences.append({
                'sentence': sent_info.get('sentence', ''),
                'label': sent_info.get('label', 'neutral'),
                'score': sent_info.get('score', 0.5),
                'article_title': article.title[:50] if article.title else 'Untitled'
            })
    
    if not all_sentences:
        return {'source': name, 'top_positive': [], 'top_negative': []}
    
    positive = sorted([s for s in all_sentences if s['label'] == 'positive'],
                     key=lambda x: x['score'], reverse=True)
    negative = sorted([s for s in all_sentences if s['label'] == 'negative'],
                     key=lambda x: x['score'], reverse=True)
    
    return {
        'source': name,
        'total_sentences': len(all_sentences),
        'positive_count': len(positive),
        'negative_count': len(negative),
        'neutral_count': len([s for s in all_sentences if s['label'] == 'neutral']),
        'top_positive': [
            {'rank': i+1, 'sentence': s['sentence'], 'confidence': round(s['score'], 4), 
             'article': s['article_title']}
            for i, s in enumerate(positive[:top_n])
        ],
        'top_negative': [
            {'rank': i+1, 'sentence': s['sentence'], 'confidence': round(s['score'], 4),
             'article': s['article_title']}
            for i, s in enumerate(negative[:top_n])
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a single text file with sentiment and topic analysis'
    )
    parser.add_argument('file', help='Path to the text file to analyze')
    parser.add_argument('--language', '-l', default='it', choices=['it', 'sl'],
                       help='Language (it=Italian, sl=Slovenian)')
    parser.add_argument('--name', '-n', default=None, 
                       help='Display name for the file (default: filename)')
    parser.add_argument('--output', '-o', default='outputs/single',
                       help='Output directory')
    parser.add_argument('--use-bertopic', '-b', action='store_true',
                       help='Enable BERTopic analysis (slower but more accurate)')
    
    args = parser.parse_args()
    
    # Default name from filename
    name = args.name or Path(args.file).stem
    
    # Create output directory with file name
    file_basename = Path(args.file).stem
    output_dir = Path(args.output) / file_basename
    
    analyze_file(
        file_path=args.file,
        name=name,
        language=args.language,
        use_bertopic=args.use_bertopic,
        output_dir=str(output_dir)
    )


if __name__ == "__main__":
    main()
