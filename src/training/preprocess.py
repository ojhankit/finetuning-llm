from src.utils import setup_logger

logger = setup_logger("training-preprocess")


def prepare_batch(batch):
    """
    Prepare a batch for causal language modeling.

    Labels are a copy of input_ids.
    """

    batch["labels"] = batch["input_ids"].clone()

    return batch