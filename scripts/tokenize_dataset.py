from pathlib import Path

from src.config import load_yaml
from src.data.loader import load_local_dataset, save_dataset
from src.data.tokenize import DatasetTokenizer, tokenize_dataset
from src.utils import setup_logger

logger = setup_logger("tokenize")


RAW_DATASET = Path("data/processed/train")
OUTPUT_DATASET = Path("data/tokenized")


def main():

    logger.info("Loading configuration...")

    model_cfg = load_yaml("model.yaml")
    dataset_cfg = load_yaml("dataset.yaml")

    logger.info("Loading processed dataset...")

    dataset = load_local_dataset(RAW_DATASET)

    logger.info("Creating train/validation split...")

    dataset = dataset.train_test_split(
        test_size=dataset_cfg["dataset"]["validation_split"],
        seed=dataset_cfg["dataset"]["random_seed"],
    )

    dataset["validation"] = dataset.pop("test")

    tokenizer = DatasetTokenizer(
        model_path=model_cfg["model"]["local_path"],
        max_length=model_cfg["tokenizer"]["max_length"],
        padding=model_cfg["tokenizer"]["padding"],
        truncation=model_cfg["tokenizer"]["truncation"],
    )

    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
    )

    logger.info(tokenized)

    logger.info("Train sample:")

    logger.info(tokenized["train"][0])

    save_dataset(
        tokenized,
        OUTPUT_DATASET,
    )

    logger.info("Saved tokenized dataset to %s", OUTPUT_DATASET)


if __name__ == "__main__":
    main()