from datasets import Dataset

from src.utils import setup_logger

logger = setup_logger("preprocess")

def format_prompt(example: dict) -> dict:

    instruction_ = example["instruction"].strip()
    input_ = example["input"].strip()
    output_ = example["output"].strip()

    if input_:
        text = (
            f"### Instruction:\n"
            f"{instruction_}\n\n"
            f"### Input:\n"
            f"{input_}\n\n"
            f"### Response:\n"
            f"{output_}\n\n"
        )
    else:
        text = (
            f"### Instruction:\n"
            f"{instruction_}\n\n"
            f"### Response:\n"
            f"{output_}"
        )

    return {"text": text}

def preprocess_dataset(dataset: Dataset) -> Dataset:
    """
    Format every example into a training prompt.
    """

    logger.info("Formatting prompts...")

    processed_dataset = dataset.map(
        format_prompt,
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
    )

    logger.info("Formatting complete.")

    return processed_dataset