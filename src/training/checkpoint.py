from pathlib import Path

from src.utils import setup_logger

logger = setup_logger("checkpoint")


def save_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    epoch: int,
):
    """
    Save LoRA adapter checkpoint.
    """

    checkpoint_dir = output_dir / f"epoch_{epoch}"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Saving checkpoint to %s",
        checkpoint_dir,
    )

    model.save_pretrained(checkpoint_dir)

    tokenizer.save_pretrained(checkpoint_dir)

    logger.info("Checkpoint saved successfully.")