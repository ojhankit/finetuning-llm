from datasets import DatasetDict
from transformers import AutoTokenizer

from src.utils import setup_logger

logger = setup_logger("tokenizer")

class DatasetTokenizer:

    def __init__(self, model_path: str, max_length: int, padding: str, truncation: bool):
        logger.info("loading tokenizer from %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, examples):

        return self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )

def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: DatasetTokenizer,
) -> DatasetDict:

    logger.info("Tokenizing dataset...")

    tokenized = dataset.map(
        tokenizer,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    logger.info("Tokenization complete.")

    return tokenized