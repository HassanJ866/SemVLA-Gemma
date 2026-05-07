"""
Merge LoRA adapter (checkpoint-1500) into the Gemma4 base model and save the
full merged weights to ckpts/brain_phase1/final.

Run once on the HPC server before training the action head:
    python scripts/merge_brain.py
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor


def merge(adapter_path: str, output_path: str) -> None:
    from unsloth import FastVisionModel

    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model from adapter config at {adapter_path} ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(adapter_path),
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )

    print("Merging LoRA weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Also save the processor (needed by SemVLAPolicy at inference time)
    try:
        processor = AutoProcessor.from_pretrained(str(adapter_path))
        processor.save_pretrained(str(output_path))
        print("Processor saved.")
    except Exception as e:
        print(f"Warning: could not save processor ({e}). Copy it manually if needed.")

    print("Done. Merged model is at:", output_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="ckpts/checkpoint-1500",
        help="Path to the LoRA adapter directory (default: ckpts/checkpoint-1500)",
    )
    parser.add_argument(
        "--output",
        default="ckpts/brain_phase1/final",
        help="Where to save the merged model (default: ckpts/brain_phase1/final)",
    )
    args = parser.parse_args()
    merge(args.adapter, args.output)
