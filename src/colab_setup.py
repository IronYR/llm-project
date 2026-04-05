"""
Helpers for Google Colab notebooks (ingest / knowledge base check).

Clone + ``pip install`` is done in the first notebook cell (see ``notebooks/README.md``)
because the repo must exist before ``src`` can be imported.

Usage after clone:

    from src.colab_setup import ensure_knowledge_base
    ensure_knowledge_base(ROOT)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def is_colab() -> bool:
    return "google.colab" in sys.modules


def _kb_paths(config: dict, root: Path) -> tuple[Path, Path]:
    data_cfg = config["data"]
    processed = root / data_cfg["processed_dir"]
    kb = processed / data_cfg["knowledge_base_file"]
    raw = root / data_cfg["raw_dir"]
    return kb, raw


def ensure_knowledge_base(root: Path, upload_in_colab: bool = True) -> None:
    """
    Ensure ``knowledge_base.json`` exists (run ``ingest.py`` if needed).

    On Colab, optionally prompt for file upload of the two bank source files into ``data/raw/``.
    Locally, expects ``data/raw`` sources or an existing processed KB; otherwise run ``python ingest.py`` first.
    """
    import yaml

    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    kb_path, raw_dir = _kb_paths(cfg, root)
    if kb_path.is_file():
        print("Knowledge base found:", kb_path)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)

    if is_colab() and upload_in_colab:
        from google.colab import files  # type: ignore[import-untyped]

        print(
            "Upload the course dataset files:\n"
            "  - NUST Bank-Product-Knowledge.xlsx\n"
            "  - funds_transfer_app_features_faq.json\n"
        )
        uploaded = files.upload()
        for name, data in uploaded.items():
            (raw_dir / name).write_bytes(data)
            print("Wrote", raw_dir / name)
    elif not any(raw_dir.glob("*.xlsx")) and not any(raw_dir.glob("*.json")):
        raise FileNotFoundError(
            f"Missing {kb_path} and no .xlsx/.json under {raw_dir}. "
            "Run `python ingest.py` locally after adding sources, or open this notebook on Colab to upload files."
        )

    subprocess.run([sys.executable, str(root / "ingest.py")], cwd=str(root), check=True)
    if not kb_path.is_file():
        raise FileNotFoundError(
            f"Ingest did not produce {kb_path}. Place sources in {raw_dir} and run python ingest.py"
        )
    print("Ingest complete:", kb_path)
