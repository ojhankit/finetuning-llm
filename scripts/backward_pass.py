from pathlib import Path
import os
import time

import torch
from torch.optim import AdamW

from src.config import load_yaml
from src.data.loader import load_local_dataset
from src.model.downloader import load_local_model
from src.model.lora import apply_lora, print_trainable_parameters
from src.training.dataloader import create_dataloaders
from src.training.preprocess import prepare_batch
from src.utils import setup_logger

logger = setup_logger("backward-pass")


def log_time(stage: str, start: float):
    elapsed = time.perf_counter() - start
    logger.info("%s completed in %.2f sec", stage, elapsed)


def main():

    # ---------------- Device Setup ---------------- #

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU : %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        CPU_THREADS = 4
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(CPU_THREADS)
        logger.info("CUDA not available — using CPU with %d threads", CPU_THREADS)

    # ---------------------------------------------- #

    model_cfg = load_yaml("model.yaml")
    train_cfg = load_yaml("train.yaml")

    # ---------------- Dataset ---------------- #

    start = time.perf_counter()

    logger.info("Loading tokenized dataset...")

    dataset = load_local_dataset(Path("data/tokenized"))

    train_loader, _ = create_dataloaders(
        dataset,
        batch_size=train_cfg["training"]["batch_size"],
    )

    log_time("Dataset loading", start)

    # ---------------- Model ---------------- #

    start = time.perf_counter()

    logger.info("Loading local model...")

    _, model = load_local_model(
        Path(model_cfg["model"]["local_path"])
    )

    log_time("Model loading", start)

    # ---------------- LoRA ---------------- #

    start = time.perf_counter()

    logger.info("Applying LoRA...")

    model = apply_lora(
        model,
        train_cfg["lora"],
    )

    log_time("LoRA setup", start)

    print_trainable_parameters(model)

    # ---------------- Optimizer ---------------- #

    trainable_params = filter(
        lambda p: p.requires_grad,
        model.parameters(),
    )

    optimizer = AdamW(
        trainable_params,
        lr=float(train_cfg["training"]["learning_rate"]),
        weight_decay=float(train_cfg["training"]["weight_decay"]),
    )

    logger.info("Optimizer initialized.")

    model.to(device)
    model.train()

    # ---------------- Batch ---------------- #

    batch = next(iter(train_loader))
    batch = prepare_batch(batch)
    batch = {key: value.to(device) for key, value in batch.items()}

    logger.info("Batch Keys : %s", list(batch.keys()))

    # ---------------- Forward ---------------- #

    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()

    outputs = model(**batch)

    loss = outputs.loss

    log_time("Forward pass", start)

    logger.info("Loss : %.6f", loss.item())

    # ---------------- Backward ---------------- #

    logger.info("Starting backward pass...")

    start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()

    loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()

    log_time("Backward pass", start)

    # ---------------- Optimizer Step ---------------- #

    logger.info("Running optimizer step...")

    start = time.perf_counter()

    optimizer.step()
    optimizer.zero_grad()

    log_time("Optimizer step", start)

    logger.info("Backward pass test completed successfully.")


if __name__ == "__main__":
    main()