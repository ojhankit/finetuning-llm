from pathlib import Path
from datasets import load_dataset, Dataset, load_from_disk

from src.utils import setup_logger

logger = setup_logger("dataset")

def download_dataset(
    dataset_name: str,
    cache_dir: Path,
    subset_size: int | None = None,
) -> Dataset:
    """
    Download a dataset from Hugging Face and optionally keep only
    a subset of the training split.
    """

    logger.info("Downloading dataset: %s", dataset_name)

    dataset = load_dataset(
        dataset_name,
        cache_dir=str(cache_dir),
    )["train"]

    logger.info("Dataset downloaded successfully.")

    if subset_size is not None:
        logger.info("Selecting first %d training samples.", subset_size)

        dataset["train"] = dataset.select(
            range(min(subset_size, len(dataset)))
        )

    logger.info("Training samples: %d", len(dataset["train"]))

    return dataset


def save_dataset(dataset: Dataset, path: Path) -> None:
    """
    Save dataset locally.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving dataset to %s", path)

    dataset.save_to_disk(str(path))

    logger.info("Dataset saved successfully.")


def load_local_dataset(path: Path) -> Dataset:
    """
    Load dataset from disk.
    """

    logger.info("Loading dataset from %s", path)

    return load_from_disk(str(path))