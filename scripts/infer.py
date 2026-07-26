from pathlib import Path
import torch

from src.config import load_yaml
from src.inference.generate import load_finetuned_model, generate_response
from src.utils import setup_logger

logger = setup_logger("infer")


PROMPTS = [
    # General Knowledge
    {
        "instruction": "Explain the difference between machine learning and deep learning.",
        "input": "",
    },

    # Summarization
    {
        "instruction": "Summarize the following paragraph in two sentences.",
        "input": (
            "Artificial Intelligence has rapidly transformed industries by "
            "automating repetitive tasks, improving decision-making, and "
            "enabling new products and services. However, ethical concerns "
            "such as bias, privacy, and transparency remain important challenges."
        ),
    },

    # List Generation
    {
        "instruction": "List five benefits of learning Python for beginners.",
        "input": "",
    },

    # Explanation
    {
        "instruction": "Explain recursion as if you are teaching a high school student.",
        "input": "",
    },

    # Comparison
    {
        "instruction": "Compare Python and C++ in terms of performance, ease of use, and applications.",
        "input": "",
    },

    # Step-by-Step Guide
    {
        "instruction": "Explain how to install Python on Windows step by step.",
        "input": "",
    },

    # Email Writing
    {
        "instruction": "Write a professional email requesting an internship opportunity.",
        "input": "",
    },

    # Creative Writing
    {
        "instruction": "Write a short story about a robot who wants to become a teacher.",
        "input": "",
    },

    # Translation
    {
        "instruction": "Translate the following sentence into Hindi.",
        "input": "Learning never stops if you stay curious.",
    },

    # Advice
    {
        "instruction": "Give five practical tips for preparing for technical interviews.",
        "input": "",
    },

    # Code Generation
    {
        "instruction": "Write a Python function to check whether a number is prime.",
        "input": "",
    },

    # Constraint Following
    {
        "instruction": "Explain what Docker is in exactly three sentences.",
        "input": "",
    },
]

def main():

    # ---------------- Config ---------------- #

    model_cfg = load_yaml("model.yaml")
    train_cfg = load_yaml("train.yaml")

    # ---------------- Device ---------------- #

    device_str = train_cfg["training"].get("device", "cpu")
    device = torch.device(device_str)

    logger.info("Running inference on: %s", device)

    # ---------------- Paths ---------------- #

    base_model_path = Path(model_cfg["model"]["local_path"])

    # Use the best checkpoint saved during training
    checkpoint_dir = Path(train_cfg["training"]["output_dir"])
    checkpoints = sorted(checkpoint_dir.glob("epoch_*"))

    if not checkpoints:
        logger.error(
            "No checkpoints found in %s. Run training first.",
            checkpoint_dir,
        )
        return

    adapter_path = checkpoints[-1]  # last = best saved epoch

    logger.info("Base model : %s", base_model_path)
    logger.info("Adapter    : %s", adapter_path)

    # ---------------- Load Model ---------------- #

    tokenizer, model = load_finetuned_model(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        device=device,
    )

    # ---------------- Generate ---------------- #

    for i, prompt in enumerate(PROMPTS, 1):

        logger.info("=" * 70)
        logger.info("Prompt %d/%d", i, len(PROMPTS))
        logger.info("Instruction : %s", prompt["instruction"])

        if prompt["input"]:
            logger.info("Input       : %s", prompt["input"])

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            instruction=prompt["instruction"],
            input=prompt["input"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            device=device,
        )

        logger.info("Response    :\n%s", response)

    logger.info("=" * 70)
    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
