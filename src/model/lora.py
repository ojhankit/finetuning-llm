from peft import LoraConfig, TaskType, get_peft_model

from src.utils import setup_logger

logger = setup_logger("lora")


def apply_lora(model, config: dict):
    """
    Attach LoRA adapters to a pretrained model.
    """

    logger.info("Configuring LoRA...")

    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        bias=config["bias"],
        task_type=TaskType[config["task_type"]],
        target_modules=config["target_modules"],
    )

    logger.info("Applying LoRA adapters...")

    model = get_peft_model(model, lora_config)

    logger.info("LoRA adapters attached successfully.")

    return model


def print_trainable_parameters(model):
    """
    Print trainable vs total parameters.
    """

    trainable_params = 0
    total_params = 0

    for parameter in model.parameters():

        total_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()

    percentage = 100 * trainable_params / total_params

    logger.info("=" * 60)
    logger.info("Model Parameter Summary")
    logger.info("=" * 60)

    logger.info("Trainable Parameters : %s", f"{trainable_params:,}")
    logger.info("Total Parameters     : %s", f"{total_params:,}")
    logger.info("Trainable %%          : %.4f", percentage)

    logger.info("=" * 60)