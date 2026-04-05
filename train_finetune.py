"""
CLI entry point for LoRA fine-tuning. Interactive workflow: see `notebooks/02_lora_finetuning.ipynb`.
"""

import argparse
import logging
import sys
from pathlib import Path

from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).parent))

from src.finetune_lib import (
    attach_lora,
    build_dataset_rows,
    build_sft_dataset,
    build_sft_training_args,
    load_base_model_for_training,
    load_config,
    load_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ft = cfg.get("finetune", {})
    data_cfg = cfg.get("data", {})
    model_id = ft.get("base_model_id", "Qwen/Qwen2.5-3B-Instruct")
    out_dir = Path(ft.get("output_dir", "./models/lora_nust_bank"))
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = build_dataset_rows(data_cfg["processed_dir"], data_cfg["knowledge_base_file"])
    if not raw_rows:
        logger.error("No training rows. Run ingest.py first.")
        sys.exit(1)
    logger.info("Training examples: %d", len(raw_rows))

    tokenizer = load_tokenizer(model_id)
    ds = build_sft_dataset(tokenizer, raw_rows)

    model = load_base_model_for_training(model_id, ft)
    model = attach_lora(model, ft)

    sft_cfg = build_sft_training_args(ft, out_dir)

    try:
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=ds,
            tokenizer=tokenizer,
        )
    logger.info(
        "Starting trainer.train() — tqdm stays at 0%% until the first step finishes; "
        "on CPU with a 3B model that can take many minutes per step."
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved LoRA adapter to %s", out_dir)


if __name__ == "__main__":
    main()
