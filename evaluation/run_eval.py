import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import EmbeddingIndex


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=0, help="Max examples (0=all)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg["data"]
    kb_path = Path(data_cfg["processed_dir"]) / data_cfg["knowledge_base_file"]
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    docs = kb.get("documents", [])
    if args.limit:
        docs = docs[: args.limit]

    index = EmbeddingIndex(cfg)
    top_k = cfg["retrieval"]["top_k"]

    hits_at_1 = 0
    hits_at_k = 0
    n = 0

    for doc in docs:
        q = (doc.get("question") or "").strip()
        if not q:
            continue
        gold_a = (doc.get("answer") or "").strip()
        n += 1
        results = index.search(q, top_k=top_k)
        if not results:
            continue
        top = results[0]
        if top.get("answer", "").strip() == gold_a or top.get("question", "").strip() == q:
            hits_at_1 += 1
        if any(
            r.get("answer", "").strip() == gold_a or r.get("question", "").strip() == q
            for r in results
        ):
            hits_at_k += 1

    print("Retrieval evaluation (held-out style: query = stored question)")
    print(f"Examples evaluated: {n}")
    if n:
        print(f"Top-1 match rate: {hits_at_1 / n:.3f}")
        print(f"Top-{top_k} match rate: {hits_at_k / n:.3f}")
    else:
        print("No examples with questions found.")


if __name__ == "__main__":
    main()
