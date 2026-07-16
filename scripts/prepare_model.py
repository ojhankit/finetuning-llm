from pathlib import Path

from src.config import load_yaml
from src.model.downloader import load_local_model
from src.model.lora import apply_lora, print_trainable_parameters
from src.utils import setup_logger

logger = setup_logger("prepare-model")


def main():

    logger.info("Loading configuration...")

    model_cfg = load_yaml("model.yaml")
    train_cfg = load_yaml("train.yaml")

    logger.info("Loading local model...")

    tokenizer, model = load_local_model(
        Path(model_cfg["model"]["local_path"])
    )

    logger.info("Applying LoRA...")

    model = apply_lora(
        model,
        train_cfg["lora"],
    )

    print_trainable_parameters(model)

    logger.info("Model is ready for fine-tuning.")


if __name__ == "__main__":
    main()