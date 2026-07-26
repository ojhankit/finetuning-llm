from pathlib import Path

from src.config import load_yaml
from src.data.loader import load_local_dataset
from src.training.dataloader import create_dataloaders
from src.utils import setup_logger

logger = setup_logger("inspect")


TOKENIZED_DATASET = Path("data/tokenized")


def main():

    train_cfg = load_yaml("train.yaml")

    logger.info("Loading tokenized dataset...")

    dataset = load_local_dataset(TOKENIZED_DATASET)

    train_loader, validation_loader = create_dataloaders(
        dataset,
        batch_size=train_cfg["training"]["batch_size"],
    )

    logger.info("Fetching one batch...")

    from src.training.preprocess import prepare_batch

    batch = next(iter(train_loader))
    batch = prepare_batch(batch)

    logger.info("=" * 60)

    for key, value in batch.items():

        logger.info("%s", key)
        logger.info("Shape : %s", tuple(value.shape))
        logger.info("Dtype : %s", value.dtype)

    logger.info("=" * 60)

    logger.info("Batch inspection completed.")


if __name__ == "__main__":
    main()