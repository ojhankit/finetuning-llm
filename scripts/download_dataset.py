from pathlib import Path

from src.data.loader import download_dataset, save_dataset
from src.utils import setup_logger

logger = setup_logger("download")

DATASET_NAME = "yahma/alpaca-cleaned"
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data/raw")
SUBSET_SIZE = 1000


def main():

    logger.info("start dataset download")

    dataset = download_dataset(
        dataset_name = DATASET_NAME,
        cache_dir = CACHE_DIR,
        subset_size = SUBSET_SIZE,
    )

    logger.info("dataset summary: \n%s", dataset)
    logger.info("First Example:\n%s", dataset["train"][0])
    save_dataset(dataset, OUTPUT_DIR)

    logger.info("dataset saved to %s", OUTPUT_DIR)

    logger.info("done.")

if __name__ == "__main__":
    main()