from pathlib import Path
import time

from src.config import load_yaml
from src.model.downloader import download_model
from src.utils import setup_logger

logger = setup_logger("download-model")


def main():

    start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Starting model download pipeline...")
    logger.info("=" * 60)

    config = load_yaml("model.yaml")

    model_name = config["model"]["name"]
    local_path = Path(config["model"]["local_path"])
    cache_dir = Path(config["cache"]["huggingface"])

    logger.info("Model Name : %s", model_name)
    logger.info("Local Path : %s", local_path)
    logger.info("Cache Path : %s", cache_dir)

    tokenizer, model = download_model(
        model_name=model_name,
        local_path=local_path,
        cache_dir=cache_dir,
    )

    logger.info("-" * 60)
    logger.info("Model Information")
    logger.info("-" * 60)

    logger.info("Model Type       : %s", model.config.model_type)
    logger.info("Hidden Size      : %d", model.config.hidden_size)
    logger.info("Layers           : %d", model.config.num_hidden_layers)
    logger.info("Attention Heads  : %d", model.config.num_attention_heads)
    logger.info("Vocabulary Size  : %d", tokenizer.vocab_size)

    elapsed = time.perf_counter() - start

    logger.info("-" * 60)
    logger.info("Completed in %.2f seconds", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()