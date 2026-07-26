from pathlib import Path

import torch

from src.config import load_yaml
from src.data.loader import load_local_dataset
from src.model.downloader import load_local_model
from src.model.lora import apply_lora
from src.training.dataloader import create_dataloaders
from src.training.preprocess import prepare_batch
from src.utils import setup_logger

logger = setup_logger("forward-pass")


def main():

    model_cfg = load_yaml("model.yaml")
    train_cfg = load_yaml("train.yaml")

    logger.info("Loading tokenized dataset...")

    dataset = load_local_dataset(Path("data/tokenized"))

    train_loader, _ = create_dataloaders(
        dataset,
        batch_size=1,
    )

    logger.info("Loading model...")

    tokenizer, model = load_local_model(
        Path(model_cfg["model"]["local_path"])
    )

    model = apply_lora(
        model,
        train_cfg["lora"],
    )

    model.train()

    logger.info("Preparing batch...")

    batch = next(iter(train_loader))
    batch = prepare_batch(batch)

    logger.info("Running forward pass...")

    outputs = model(**batch)

    logger.info("Loss : %.4f", outputs.loss.item())


if __name__ == "__main__":
    main()