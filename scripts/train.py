from pathlib import Path
import time

import torch

from src.config import load_yaml
from src.data.loader import load_local_dataset
from src.model.downloader import load_local_model
from src.model.lora import apply_lora, print_trainable_parameters
from src.training.dataloader import create_dataloaders
from src.training.trainer import Trainer
from src.utils import setup_logger

logger = setup_logger("train")


def main():

    start_total = time.perf_counter()

    # ---------------- Config ---------------- #

    logger.info("Loading configuration...")

    model_cfg = load_yaml("model.yaml")
    train_cfg = load_yaml("train.yaml")

    # ---------------- Dataset ---------------- #

    logger.info("Loading tokenized dataset...")

    dataset = load_local_dataset(Path("data/tokenized"))

    train_loader, validation_loader = create_dataloaders(
        dataset,
        batch_size=train_cfg["training"]["batch_size"],
    )

    logger.info(
        "Train batches: %d | Validation batches: %d",
        len(train_loader),
        len(validation_loader),
    )

    # ---------------- Model ---------------- #

    logger.info("Loading local model...")

    tokenizer, model = load_local_model(
        Path(model_cfg["model"]["local_path"])
    )

    # ---------------- LoRA ---------------- #

    logger.info("Applying LoRA adapters...")

    model = apply_lora(model, train_cfg["lora"])

    print_trainable_parameters(model)

    # ---------------- Trainer ---------------- #

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=train_cfg,
    )

    # ---------------- Train ---------------- #

    trainer.train()

    # ---------------- Done ---------------- #

    total_time = time.perf_counter() - start_total

    logger.info("Total pipeline time : %.2f sec", total_time)


if __name__ == "__main__":
    main()
