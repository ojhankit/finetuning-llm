from pathlib import Path
import time

import torch
from torch.optim import AdamW

from src.training.engine import train_one_epoch, validate
from src.training.checkpoint import save_checkpoint
from src.utils import setup_logger

logger = setup_logger("trainer")


class Trainer:
    """
    Handles the complete training pipeline.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        validation_loader,
        config,
    ):

        self.model = model
        self.tokenizer = tokenizer

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.config = config

        self.device = torch.device(
            config["training"]["device"]
        )

        self.model.to(self.device)

        trainable_parameters = filter(
            lambda p: p.requires_grad,
            self.model.parameters(),
        )

        self.optimizer = AdamW(
            trainable_parameters,
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

        self.output_dir = Path(
            config["training"]["output_dir"]
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_validation_loss = float("inf")

    def train(self):

        total_epochs = self.config["training"]["num_epochs"]

        logger.info("=" * 70)
        logger.info("Starting Fine-Tuning")
        logger.info("=" * 70)

        logger.info("Device             : %s", self.device)
        logger.info("Epochs             : %d", total_epochs)
        logger.info(
            "Batch Size         : %d",
            self.config["training"]["batch_size"],
        )
        logger.info(
            "Gradient Accum     : %d",
            self.config["training"]["gradient_accumulation_steps"],
        )
        logger.info(
            "Learning Rate      : %s",
            self.config["training"]["learning_rate"],
        )

        logger.info("=" * 70)

        for epoch in range(1, total_epochs + 1):

            logger.info("")
            logger.info(
                "Epoch %d/%d",
                epoch,
                total_epochs,
            )

            epoch_start = time.perf_counter()

            train_loss = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                gradient_accumulation_steps=self.config["training"][
                    "gradient_accumulation_steps"
                ],
            )

            validation_loss = validate(
                model=self.model,
                dataloader=self.validation_loader,
                device=self.device,
            )

            epoch_time = time.perf_counter() - epoch_start

            logger.info("-" * 70)
            logger.info("Training Loss   : %.4f", train_loss)
            logger.info("Validation Loss : %.4f", validation_loss)
            logger.info("Epoch Time      : %.2f sec", epoch_time)

            if validation_loss < self.best_validation_loss:

                logger.info(
                    "Validation improved %.4f → %.4f",
                    self.best_validation_loss,
                    validation_loss,
                )

                self.best_validation_loss = validation_loss

                save_checkpoint(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    output_dir=self.output_dir,
                    epoch=epoch,
                )

            else:

                logger.info(
                    "Validation did not improve."
                )

        logger.info("=" * 70)
        logger.info("Training completed.")
        logger.info(
            "Best Validation Loss : %.4f",
            self.best_validation_loss,
        )
        logger.info("=" * 70)