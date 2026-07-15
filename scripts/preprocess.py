from pathlib import Path

from src.data.loader import load_local_dataset, save_dataset
from src.data.preprocess import preprocess_dataset
from src.utils import setup_logger

logger = setup_logger("preprocess")


RAW_DATASET = Path("data/raw/train")
PROCESSED_DATASET = Path("data/processed/train")


def main():

    logger.info("Loading raw dataset...")

    dataset = load_local_dataset(RAW_DATASET)

    logger.info("Preprocessing dataset...")

    processed = preprocess_dataset(dataset)

    logger.info("Processed dataset:")

    logger.info(processed)

    logger.info("First formatted sample:\n%s", processed[0]["text"])

    save_dataset(processed, PROCESSED_DATASET)

    logger.info("Saved processed dataset to %s", PROCESSED_DATASET)


if __name__ == "__main__":
    main()