"""
Phase 1: Brain fine-tuning on LIBERO-Spatial augmented data.

Usage:
    python -m models.brain.train --config-name=phase1_libero

The trainer:
  1. Loads Gemma 4 E4B multimodal with Unsloth FastVisionModel + QLoRA.
  2. Streams the 3-task JSONL from data/splits/libero_train.jsonl.
  3. Applies chat-template + image tokenisation.
  4. Trains with causal-LM loss masked to output tokens only.
  5. Evaluates on data/splits/libero_train_val.jsonl every eval_steps.
  6. Logs to wandb + local CSV (logs/brain_train_metrics.csv).

Logged metrics:
  train/loss          — mean cross-entropy over output tokens (every log_steps)
  train/loss_grounding, train/loss_parsing, train/loss_task_synthesis
                      — per-task-type loss breakdown (every log_steps)
  train/grad_norm     — gradient norm before clipping
  train/lr            — current learning rate
  train/gpu_mem_gb    — peak GPU memory allocated
  eval/val_loss       — aggregate val loss (every eval_steps)
  eval/val_loss_grounding, eval/val_loss_parsing, eval/val_loss_task_synthesis
"""

import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

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

        image = None
        img_rel = rec.get("image")
        if img_rel:
            img_path = self.image_root / img_rel
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")

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

    placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
    images_filled = [img if img is not None else placeholder for img in images]

    encoding = processor(
        text=texts,
        images=images_filled,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    labels = encoding["input_ids"].clone()
    for i, item in enumerate(batch):
        target_ids = processor.tokenizer(
            item["target_text"], add_special_tokens=False
        )["input_ids"]
        full_ids = encoding["input_ids"][i].tolist()
        tgt_len = len(target_ids)
        start_pos = -1
        for j in range(len(full_ids) - tgt_len, -1, -1):
            if full_ids[j:j + tgt_len] == target_ids:
                start_pos = j
                break
        if start_pos >= 0:
            labels[i, :start_pos] = -100
        else:
            cutoff = int(0.8 * labels.shape[1])
            labels[i, :cutoff] = -100

    labels[encoding["attention_mask"] == 0] = -100

    return {
        "input_ids":      encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device),
        "pixel_values":   encoding.get("pixel_values", None),
        "labels":         labels.to(device),
        "task_types":     [item["task_type"] for item in batch],
    }


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    task_loss: dict[str, list[float]] = defaultdict(list)
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=pixel_values,
                labels=batch["labels"],
            )
            loss_val = outputs.loss.item()
            total_loss += loss_val
            n += 1
            for tt in batch["task_types"]:
                task_loss[tt].append(loss_val)

    model.train()

    metrics = {"val_loss": total_loss / max(n, 1)}
    for tt, vals in task_loss.items():
        metrics[f"val_loss_{tt}"] = sum(vals) / len(vals)
    return metrics


# ── CSV logger ─────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = None

    def log(self, row: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


# ── training loop ──────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs/brain", config_name="phase1_libero", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output_dir, exist_ok=True)

    use_wandb = bool(cfg.get("wandb_project"))
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    csv_log = CSVLogger("logs/brain_train_metrics.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── model: Unsloth FastVisionModel ─────────────────────────────────────
    log.info(f"Loading: {cfg.model_id}")
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_length,
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=torch.bfloat16,
    )

    if cfg.use_lora:
        model = FastVisionModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            # fine-tune language layers only; vision tower stays frozen
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
        )
        model.print_trainable_parameters()

    if cfg.get("compile", False):
        model = torch.compile(model)

    # ── data ───────────────────────────────────────────────────────────────
    train_ds = ThreeTaskDataset(cfg.train_jsonl, cfg.image_root, processor)
    val_ds   = ThreeTaskDataset(cfg.val_jsonl,   cfg.image_root, processor)
    log.info(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")

    _collate = lambda b: collate_fn(b, processor, device, cfg.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.per_device_batch_size,
                              shuffle=True,  collate_fn=_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.per_device_batch_size,
                              shuffle=False, collate_fn=_collate, num_workers=0)

    # ── optimiser & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2), weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    # ── training ───────────────────────────────────────────────────────────
    model.train()
    global_step = 0
    grad_accum  = cfg.get("grad_accum_steps", 1)
    optimizer.zero_grad()

    accum: dict[str, float] = defaultdict(float)
    accum_counts: dict[str, int] = defaultdict(int)

    while global_step < cfg.max_steps:
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=pixel_values,
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            raw_loss = outputs.loss.item()
            accum["loss"] += raw_loss
            accum_counts["loss"] += 1
            for tt in batch["task_types"]:
                accum[f"loss_{tt}"] += raw_loss
                accum_counts[f"loss_{tt}"] += 1

            if (global_step + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = 0.0

            global_step += 1

            # ── log ────────────────────────────────────────────────────────
            if global_step % cfg.log_steps == 0:
                lr = scheduler.get_last_lr()[0]
                gpu_mem = (torch.cuda.memory_allocated(device) / 1e9
                           if device.type == "cuda" else 0.0)

                row = {"step": global_step, "lr": lr,
                       "grad_norm": grad_norm, "gpu_mem_gb": gpu_mem}
                for k, total in accum.items():
                    cnt = accum_counts[k]
                    row[k] = total / max(cnt, 1)

                log.info(
                    f"step={global_step}  loss={row['loss']:.4f}  "
                    f"grounding={row.get('loss_grounding', 0):.4f}  "
                    f"parsing={row.get('loss_parsing', 0):.4f}  "
                    f"synthesis={row.get('loss_task_synthesis', 0):.4f}  "
                    f"lr={lr:.2e}  grad_norm={grad_norm:.3f}  gpu={gpu_mem:.1f}GB"
                )

                if use_wandb:
                    import wandb
                    wandb.log(
                        {f"train/{k}": v for k, v in row.items() if k != "step"},
                        step=global_step,
                    )

                csv_log.log({"phase": "train", **row})
                accum.clear()
                accum_counts.clear()

            # ── eval ───────────────────────────────────────────────────────
            if global_step % cfg.eval_steps == 0:
                metrics = evaluate(model, val_loader, device)
                log.info(f"[eval] step={global_step}  " +
                         "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

                if use_wandb:
                    import wandb
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()},
                              step=global_step)

                csv_log.log({"phase": "eval", "step": global_step, **metrics})

            # ── checkpoint ─────────────────────────────────────────────────
            if global_step % cfg.save_steps == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                model.save_pretrained(str(ckpt_dir))
                processor.save_pretrained(str(ckpt_dir))
                log.info(f"Checkpoint saved: {ckpt_dir}")

    # ── final save ─────────────────────────────────────────────────────────
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))
    log.info(f"Final checkpoint: {final_dir}")

    csv_log.close()
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
