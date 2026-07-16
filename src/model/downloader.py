from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import setup_logger

logger = setup_logger("model")


def download_model(
    model_name: str,
    local_path: Path,
    cache_dir: Path,
):
    """
    Download model and tokenizer from Hugging Face.

    If already present locally, load from disk instead.
    """

    local_path = Path(local_path)
    cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and any(local_path.iterdir()):

        logger.info("Local model found.")

        logger.info("Loading tokenizer from %s", local_path)

        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True,
        )

        logger.info("Tokenizer loaded.")

        logger.info("Loading model from %s", local_path)

        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
        )

        logger.info("Model loaded successfully.")

        return tokenizer, model

    logger.info("No local model found.")
    logger.info("Downloading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
    )

    logger.info("Tokenizer downloaded.")

    logger.info("Downloading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
    )

    logger.info("Model downloaded.")

    local_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving tokenizer to %s", local_path)

    tokenizer.save_pretrained(local_path)

    logger.info("Tokenizer saved.")

    logger.info("Saving model to %s", local_path)

    model.save_pretrained(local_path)

    logger.info("Model saved successfully.")

    return tokenizer, model


def load_local_model(local_path: Path):
    """
    Load tokenizer and model from local directory.
    """

    logger.info("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
    )

    logger.info("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
    )

    logger.info("Local model loaded successfully.")

    return tokenizer, model