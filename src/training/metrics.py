import math

import torch
from tqdm import tqdm

from src.training.preprocess import prepare_batch
from src.utils import setup_logger

logger = setup_logger("metrics")


@torch.inference_mode()
def compute_perplexity(
    model,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Compute average loss and perplexity over a dataloader.

    Perplexity = exp(average cross-entropy loss).
    Lower is better — a random model on a 50k vocab has PPL ~50,000.
    """

    model.eval()

    total_loss = 0.0
    total_steps = 0

    progress_bar = tqdm(
        dataloader,
        desc="Evaluating perplexity",
        leave=False,
    )

    for batch in progress_bar:

        batch = prepare_batch(batch)
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)

        loss = outputs.loss.item()
        total_loss += loss
        total_steps += 1

        progress_bar.set_postfix(loss=f"{loss:.4f}")

    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)

    logger.info("Average Loss : %.4f", avg_loss)
    logger.info("Perplexity   : %.4f", perplexity)

    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
    }


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Measures n-gram overlap between generated and reference responses.
    Requires the `evaluate` library (already in pyproject.toml).
    """

    try:
        import evaluate
        rouge = evaluate.load("rouge")
    except ImportError as e:
        raise RuntimeError(
            "ROUGE evaluation requires: uv add rouge-score absl-py nltk"
        ) from e

    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    logger.info("ROUGE-1 : %.4f", results["rouge1"])
    logger.info("ROUGE-2 : %.4f", results["rouge2"])
    logger.info("ROUGE-L : %.4f", results["rougeL"])

    return {
        "rouge1": round(results["rouge1"], 4),
        "rouge2": round(results["rouge2"], 4),
        "rougeL": round(results["rougeL"], 4),
    }
