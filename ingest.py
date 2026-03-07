import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion import run_ingestion
from src.embeddings import EmbeddingIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  NUST Bank AI Assistant – Data Ingestion Pipeline")
    print("=" * 60 + "\n")


def print_stats(index: EmbeddingIndex) -> None:
    stats = index.get_stats()
    print("\n── Index Statistics ──────────────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print("──────────────────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NUST Bank KB Ingestion Pipeline")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if index already exists")
    parser.add_argument("--stats",   action="store_true",
                        help="Print index statistics and exit")
    args = parser.parse_args()

    print_banner()
    cfg   = load_config()
    index = EmbeddingIndex(cfg)

    # ── Stats only ──
    if args.stats:
        print_stats(index)
        return

    # ── Skip if already built (unless --rebuild) ──
    if index.collection_exists() and not args.rebuild:
        logger.info("Index already exists. Use --rebuild to force rebuild.")
        print_stats(index)
        print(" Index is ready. Run: streamlit run app.py\n")
        return

    # ── Step 1: Ingest data ──
    logger.info("Step 1/2 – Parsing and preprocessing source documents...")
    t0   = time.time()
    docs = run_ingestion(cfg)
    t1   = time.time()
    logger.info("Ingestion complete: %d documents in %.1fs", len(docs), t1 - t0)

    if not docs:
        logger.error("No documents found! Check data/raw/ for source files.")
        sys.exit(1)

    # Print breakdown by category
    from collections import Counter
    cats = Counter(d["category"] for d in docs)
    print("\n── Document breakdown by product ────────────────────────")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}  {cat}")
    print(f"  {'─'*30}")
    print(f"  {len(docs):>4}  TOTAL")
    print("─────────────────────────────────────────────────────────\n")

    # ── Step 2: Build vector index ──
    logger.info("Step 2/2 – Building Qdrant vector index...")
    t2 = time.time()
    index.build(docs)
    t3 = time.time()
    logger.info("Index built in %.1fs", t3 - t2)

    print_stats(index)
    print(f"All done in {t3 - t0:.1f}s\n")
    print("   Launch the app:  streamlit run app.py\n")


if __name__ == "__main__":
    main()
