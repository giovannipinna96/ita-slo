"""
Test script for validating article parsing.
"""

import logging
from src.preprocessing.text_loader import load_all_texts
from src.preprocessing.article_parser import parse_and_validate
from src.utils.file_utils import save_articles_json, ensure_dirs

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== TESTING ARTICLE PARSING ===")

    # Ensure output directories exist
    ensure_dirs(['outputs/parsed'])

    # Load texts
    logger.info("Loading texts...")
    piccolo_text, edinost_text, piccolo_obj, edinost_obj = load_all_texts()

    # Parse Il Piccolo
    logger.info("\n--- Parsing Il Piccolo ---")
    piccolo_articles = parse_and_validate(piccolo_text, 'il_piccolo')
    logger.info(f"Total articles extracted: {len(piccolo_articles)}")

    # Parse Edinost
    logger.info("\n--- Parsing Edinost ---")
    edinost_articles = parse_and_validate(edinost_text, 'edinost')
    logger.info(f"Total articles extracted: {len(edinost_articles)}")

    # Statistics
    logger.info("\n=== STATISTICS ===")
    logger.info(f"Il Piccolo: {len(piccolo_articles)} articles")
    if piccolo_articles:
        avg_length = sum(len(a.content) for a in piccolo_articles) / len(piccolo_articles)
        logger.info(f"  Average length: {avg_length:.0f} characters")
        logger.info(f"  Min length: {min(len(a.content) for a in piccolo_articles)} characters")
        logger.info(f"  Max length: {max(len(a.content) for a in piccolo_articles)} characters")

    logger.info(f"\nEdinost: {len(edinost_articles)} articles")
    if edinost_articles:
        avg_length = sum(len(a.content) for a in edinost_articles) / len(edinost_articles)
        logger.info(f"  Average length: {avg_length:.0f} characters")
        logger.info(f"  Min length: {min(len(a.content) for a in edinost_articles)} characters")
        logger.info(f"  Max length: {max(len(a.content) for a in edinost_articles)} characters")

    # Show sample articles
    logger.info("\n=== SAMPLE ARTICLES (Il Piccolo) ===")
    for i, article in enumerate(piccolo_articles[:3]):
        logger.info(f"\nArticle {i+1}:")
        logger.info(f"  Title: {article.title}")
        logger.info(f"  Page: {article.page}")
        logger.info(f"  Length: {len(article.content)} chars")
        logger.info(f"  Preview: {article.content[:150]}...")

    logger.info("\n=== SAMPLE ARTICLES (Edinost) ===")
    for i, article in enumerate(edinost_articles[:3]):
        logger.info(f"\nArticle {i+1}:")
        logger.info(f"  Title: {article.title}")
        logger.info(f"  Section: {article.section}")
        logger.info(f"  Length: {len(article.content)} chars")
        logger.info(f"  Preview: {article.content[:150]}...")

    # Save parsed articles
    logger.info("\n--- Saving parsed articles ---")
    save_articles_json(piccolo_articles, 'outputs/parsed/il_piccolo_articles.json')
    save_articles_json(edinost_articles, 'outputs/parsed/edinost_articles.json')

    logger.info("\n=== PARSING TEST COMPLETE ===")
    logger.info(f"Total articles: {len(piccolo_articles) + len(edinost_articles)}")


if __name__ == "__main__":
    main()
