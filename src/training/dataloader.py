from torch.utils.data import DataLoader
from transformers import default_data_collator

from src.utils import setup_logger

logger = setup_logger("dataloader")

def create_dataloaders(dataset, batch_size: int):
    logger.info("Creating train DataLoader.")

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    logger.info("Creating validation DataLoader.")

    validation_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    logger.info("DataLoaders created successfully.")

    return train_loader, validation_loader