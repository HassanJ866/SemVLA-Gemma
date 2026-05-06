"""
Phase 1: Brain fine-tuning on LIBERO-Spatial augmented data.

Usage:
    python -m models.brain.train --config-name=phase1_libero

The trainer:
  1. Loads Gemma 4 (multimodal) with LoRA from a Hydra config.
  2. Streams the 3-task JSONL from data/splits/libero_train.jsonl.
  3. Applies chat-template + image tokenisation.
  4. Trains with causal-LM loss masked to output tokens only.
  5. Evaluates on data/splits/libero_train_val.jsonl every eval_steps.
  6. Saves checkpoints and logs to wandb.
"""

import json
import logging
import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from models.brain.prompts import format_training_sample

log = logging.getLogger(__name__)


# ── dataset ────────────────────────────────────────────────────────────────────

class ThreeTaskDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, processor, max_new_tokens: int = 256):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sample = format_training_sample(rec)
        messages = sample["messages"]
        target_text = sample["target"]

        # load image if present in any message
        image = None
        img_rel = rec.get("image")
        if img_rel:
            img_path = self.image_root / img_rel
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")

        # build full conversation text using apply_chat_template
        # append the target as the assistant response
        full_messages = messages + [{"role": "assistant", "content": target_text}]

        return {
            "messages": full_messages,
            "image": image,
            "target_text": target_text,
            "task_type": rec["task_type"],
        }


def collate_fn(batch, processor, device, max_length: int = 1024):
    """Tokenise a batch; mask everything except output tokens in labels."""
    texts = []
    images = []
    for item in batch:
        text = processor.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        images.append(item["image"])

    # Some samples may not have an image; use a blank placeholder so the
    # processor can handle a uniform batch.
    has_image = [img is not None for img in images]
    placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
    images_or_placeholder = [img if img is not None else placeholder for img in images]

    encoding = processor(
        text=texts,
        images=images_or_placeholder,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    # build labels: -100 everywhere except target tokens
    labels = encoding["input_ids"].clone()
    for i, item in enumerate(batch):
        # find where the assistant turn starts by searching for the target text
        target_ids = processor.tokenizer(
            item["target_text"], add_special_tokens=False
        )["input_ids"]
        full_ids = encoding["input_ids"][i].tolist()
        # find the last occurrence of the target sequence
        tgt_len = len(target_ids)
        start_pos = -1
        for j in range(len(full_ids) - tgt_len, -1, -1):
            if full_ids[j:j + tgt_len] == target_ids:
                start_pos = j
                break
        if start_pos >= 0:
            labels[i, :start_pos] = -100
        else:
            # fallback: mask first 80% as prompt
            cutoff = int(0.8 * labels.shape[1])
            labels[i, :cutoff] = -100

    # mask padding
    labels[encoding["attention_mask"] == 0] = -100

    return {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "pixel_values": encoding.get("pixel_values", None),
        "labels": labels.to(device),
    }


# ── training loop ──────────────────────────────────────────────────────────────

def evaluate(model, val_loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()
            n += 1
    model.train()
    return {"val_loss": total_loss / max(n, 1)}


@hydra.main(config_path="../../configs/brain", config_name="phase1_libero", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.get("wandb_project"):
        wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── model ──────────────────────────────────────────────────────────────
    log.info(f"Loading processor and model: {cfg.model_id}")
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            target_modules=cfg.lora.target_modules,
            lora_dropout=cfg.lora.dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if cfg.get("compile", False):
        model = torch.compile(model)

    # ── data ───────────────────────────────────────────────────────────────
    train_ds = ThreeTaskDataset(cfg.train_jsonl, cfg.image_root, processor)
    val_ds   = ThreeTaskDataset(cfg.val_jsonl,   cfg.image_root, processor)

    _collate = lambda b: collate_fn(b, processor, device, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.per_device_batch_size,
                              shuffle=True,  collate_fn=_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.per_device_batch_size,
                              shuffle=False, collate_fn=_collate, num_workers=0)

    # ── optimiser & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=0.01,
    )
    total_steps = cfg.max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
    )

    # ── training ───────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    accum_loss = 0.0
    grad_accum = cfg.get("grad_accum_steps", 1)
    optimizer.zero_grad()

    while global_step < total_steps:
        for batch in train_loader:
            if global_step >= total_steps:
                break

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (global_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % cfg.log_steps == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(f"step={global_step} loss={accum_loss:.4f} lr={lr:.2e}")
                if cfg.get("wandb_project"):
                    wandb.log({"train/loss": accum_loss, "train/lr": lr,
                               "step": global_step})
                accum_loss = 0.0

            if global_step % cfg.eval_steps == 0:
                metrics = evaluate(model, val_loader, device)
                log.info(f"[eval] step={global_step} {metrics}")
                if cfg.get("wandb_project"):
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()} |
                              {"step": global_step})

            if global_step % cfg.save_steps == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt_dir))
                processor.save_pretrained(str(ckpt_dir))
                log.info(f"Checkpoint saved: {ckpt_dir}")

    # final save
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    log.info(f"Final checkpoint: {final_dir}")
    if cfg.get("wandb_project"):
        wandb.finish()


if __name__ == "__main__":
    main()
