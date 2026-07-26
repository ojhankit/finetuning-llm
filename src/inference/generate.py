import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils import setup_logger

logger = setup_logger("generate")


def load_finetuned_model(
    base_model_path: Path,
    adapter_path: Path,
    device: torch.device,
):
    """
    Load the base model and attach the saved LoRA adapter.
    """

    logger.info("Loading tokenizer from %s", base_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        local_files_only=True,
    )

    logger.info("Loading base model from %s", base_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    logger.info("Attaching LoRA adapter from %s", adapter_path)

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        local_files_only=True,
    )

    model.to(device)
    model.eval()

    logger.info("Fine-tuned model ready.")

    return tokenizer, model


def build_prompt(instruction: str, input: str = "") -> str:
    """
    Build the Alpaca-style prompt used during training.
    """

    if input.strip():
        return (
            f"### Instruction:\n"
            f"{instruction.strip()}\n\n"
            f"### Input:\n"
            f"{input.strip()}\n\n"
            f"### Response:\n"
        )

    return (
        f"### Instruction:\n"
        f"{instruction.strip()}\n\n"
        f"### Response:\n"
    )


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    instruction: str,
    input: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    Generate a response from the fine-tuned model given an instruction.
    """

    prompt = build_prompt(instruction, input)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens (strip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()
