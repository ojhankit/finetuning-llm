from pathlib import Path
import torch

from src.config import load_yaml
from src.data.loader import load_local_dataset
from src.inference.generate import load_finetuned_model, generate_response
from src.training.dataloader import create_dataloaders
from src.training.metrics import compute_perplexity, compute_rouge
from src.utils import setup_logger

logger = setup_logger("evaluate")


def main():

    # ---------------- Config ---------------- #

    model_cfg = load_yaml("model.yaml")
    train_cfg = load_yaml("train.yaml")

    # ---------------- Device ---------------- #

    device_str = train_cfg["training"].get("device", "cpu")
    device = torch.device(device_str)

    logger.info("Running evaluation on: %s", device)

    # ---------------- Paths ---------------- #

    base_model_path = Path(model_cfg["model"]["local_path"])

    checkpoint_dir = Path(train_cfg["training"]["output_dir"])
    checkpoints = sorted(checkpoint_dir.glob("epoch_*"))

    if not checkpoints:
        logger.error(
            "No checkpoints found in %s. Run training first.",
            checkpoint_dir,
        )
        return

    adapter_path = checkpoints[-1]

    # ---------------- Dataset ---------------- #

    logger.info("Loading tokenized dataset...")

    dataset = load_local_dataset(Path("data/tokenized"))

    _, validation_loader = create_dataloaders(
        dataset,
        batch_size=train_cfg["training"]["batch_size"],
    )

    # ---------------- Load Model ---------------- #

    tokenizer, model = load_finetuned_model(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        device=device,
    )

    # ---------------- Perplexity ---------------- #

    logger.info("=" * 60)
    logger.info("Evaluating Perplexity on Validation Set")
    logger.info("=" * 60)

    ppl_results = compute_perplexity(
        model=model,
        dataloader=validation_loader,
        device=device,
    )
    
    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
