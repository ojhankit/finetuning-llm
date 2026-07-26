from tqdm import tqdm
import torch

from src.training.preprocess import prepare_batch
from src.utils import setup_logger

logger = setup_logger("engine")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    gradient_accumulation_steps: int,
):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    optimizer_steps = 0

    optimizer.zero_grad()

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="Training",
        leave=False,
    )

    for step, batch in progress_bar:

        batch = prepare_batch(batch)

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        running_loss += loss.item()

        (loss / gradient_accumulation_steps).backward()

        should_step = (
            (step + 1) % gradient_accumulation_steps == 0
            or (step + 1) == len(dataloader)
        )

        if should_step:

            optimizer.step()
            optimizer.zero_grad()

            optimizer_steps += 1

        average_loss = running_loss / (step + 1)

        current_lr = optimizer.param_groups[0]["lr"]

        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}",
            lr=f"{current_lr:.2e}",
        )

    epoch_loss = running_loss / len(dataloader)

    logger.info(
        "Training Loss : %.4f | Optimizer Steps : %d",
        epoch_loss,
        optimizer_steps,
    )

    return epoch_loss


@torch.inference_mode()
def validate(
    model,
    dataloader,
    device,
):
    """
    Evaluate model on validation dataset.
    """

    model.eval()

    running_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc="Validation",
        leave=False,
    )

    for step, batch in enumerate(progress_bar):

        batch = prepare_batch(batch)

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        running_loss += loss.item()

        average_loss = running_loss / (step + 1)

        progress_bar.set_postfix(
            loss=f"{average_loss:.4f}"
        )

    validation_loss = running_loss / len(dataloader)

    logger.info(
        "Validation Loss : %.4f",
        validation_loss,
    )

    return validation_loss