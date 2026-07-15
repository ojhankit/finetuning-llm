import json
from pathlib import Path

from src.data.loader import load_local_dataset
from src.utils import setup_logger

logger = setup_logger("explore")


DATASET_PATH = Path("data/raw/train")
REPORT_PATH = Path("outputs/metrics/dataset_report.json")


def compute_statistics(dataset):

    train = dataset

    total_samples = len(train)

    empty_inputs = sum(
        1
        for example in train
        if not example["input"].strip()
    )

    instruction_lengths = [
        len(example["instruction"])
        for example in train
    ]

    input_lengths = [
        len(example["input"])
        for example in train
    ]

    output_lengths = [
        len(example["output"])
        for example in train
    ]

    report = {
        "total_samples": total_samples,
        "features": train.column_names,
        "empty_inputs": empty_inputs,
        "non_empty_inputs": total_samples - empty_inputs,
        "instruction": {
            "average": round(sum(instruction_lengths) / total_samples, 2),
            "minimum": min(instruction_lengths),
            "maximum": max(instruction_lengths),
        },
        "input": {
            "average": round(sum(input_lengths) / total_samples, 2),
            "minimum": min(input_lengths),
            "maximum": max(input_lengths),
        },
        "output": {
            "average": round(sum(output_lengths) / total_samples, 2),
            "minimum": min(output_lengths),
            "maximum": max(output_lengths),
        },
    }

    return report


def main():

    logger.info("Loading dataset...")

    dataset = load_local_dataset(DATASET_PATH)

    logger.info("Computing statistics...")

    report = compute_statistics(dataset)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("Dataset Report")
    logger.info("=" * 60)

    for key, value in report.items():
        logger.info("%s : %s", key, value)

    logger.info("=" * 60)

    logger.info("Report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    main()