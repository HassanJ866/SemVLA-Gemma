"""
Phase 1: Brain fine-tuning on LIBERO-Spatial augmented data.

Usage:
    python -m models.brain.train --config-name=phase1_libero

Resume from latest checkpoint automatically if one exists in output_dir.
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
    def __init__(self, jsonl_path: str, image_root: str):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.image_root = Path(image_root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sample = format_training_sample(rec)
        image = None
        img_rel = rec.get("image")
        if img_rel:
            img_path = self.image_root / img_rel
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
        return {
            "messages":    sample["messages"],
            "image":       image,
            "target_text": sample["target"],
            "task_type":   rec["task_type"],
        }


def collate_fn(batch, processor, device, max_length: int = 1024):
    """
    Process each sample individually using processor.apply_chat_template with
    tokenize=True so that image tokens and pixel_values are created together
    in the same call — required by Gemma4 to avoid token/feature mismatch.
    Then manually pad to the same length.
    """
    placeholder = Image.new("RGB", (224, 224), color=(128, 128, 128))
    pad_id = processor.tokenizer.pad_token_id or 0

    all_encodings = []
    for item in batch:
        img = item["image"] if item["image"] is not None else placeholder
        full_messages = item["messages"] + [
            {"role": "assistant", "content": item["target_text"]}
        ]
        enc = processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images=[img],
            truncation=True,
            max_length=max_length,
        )
        all_encodings.append(enc)

    max_len = max(e["input_ids"].shape[1] for e in all_encodings)
    input_ids_list, attn_mask_list, labels_list, pixel_values_list = [], [], [], []

    for enc, item in zip(all_encodings, batch):
        ids  = enc["input_ids"][0]
        mask = enc["attention_mask"][0]
        pad_len = max_len - ids.shape[0]
        ids  = torch.cat([ids,  ids.new_full((pad_len,), pad_id)])
        mask = torch.cat([mask, mask.new_zeros(pad_len)])

        lbl = ids.clone()
        target_ids = processor.tokenizer(
            item["target_text"], add_special_tokens=False
        )["input_ids"]
        tgt_len   = len(target_ids)
        full_list = ids.tolist()
        start_pos = -1
        for j in range(len(full_list) - tgt_len, -1, -1):
            if full_list[j:j + tgt_len] == target_ids:
                start_pos = j
                break
        if start_pos >= 0:
            lbl[:start_pos] = -100
        else:
            lbl[:int(0.8 * len(lbl))] = -100
        lbl[mask == 0] = -100

        input_ids_list.append(ids)
        attn_mask_list.append(mask)
        labels_list.append(lbl)
        if "pixel_values" in enc:
            pixel_values_list.append(enc["pixel_values"])

    result = {
        "input_ids":      torch.stack(input_ids_list).to(device),
        "attention_mask": torch.stack(attn_mask_list).to(device),
        "labels":         torch.stack(labels_list).to(device),
        "task_types":     [item["task_type"] for item in batch],
    }
    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0).to(device)
    return result


# ── checkpoint helpers ─────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> tuple[Path | None, int]:
    """Return (ckpt_path, step) of the latest checkpoint, or (None, 0)."""
    out = Path(output_dir)
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    step = int(latest.name.split("-")[1])
    return latest, step


def save_checkpoint(model, processor, optimizer, scheduler, step: int,
                    output_dir: str, keep_last: int = 3) -> None:
    out = Path(output_dir)
    ckpt_dir = out / f"checkpoint-{step}"
    model.save_pretrained(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))
    torch.save(optimizer.state_dict(),  ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(),  ckpt_dir / "scheduler.pt")
    torch.save({"global_step": step},   ckpt_dir / "train_state.pt")
    log.info(f"Checkpoint saved: {ckpt_dir}")

    # keep only the last N checkpoints to save disk
    all_ckpts = sorted(out.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
    for old in all_ckpts[:-keep_last]:
        import shutil
        shutil.rmtree(old)
        log.info(f"Removed old checkpoint: {old}")


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, device) -> dict:
    model.eval()
    total_loss = 0.0
    task_loss: dict[str, list[float]] = defaultdict(list)
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            task_types = batch.pop("task_types")
            outputs = model(**batch)
            loss_val = outputs.loss.item()
            total_loss += loss_val
            n += 1
            for tt in task_types:
                task_loss[tt].append(loss_val)
    model.train()
    metrics = {"val_loss": total_loss / max(n, 1)}
    for tt, vals in task_loss.items():
        metrics[f"val_loss_{tt}"] = sum(vals) / len(vals)
    return metrics


# ── CSV logger ─────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path: str, resume: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume and self.path.exists() else "w"
        self._file   = open(self.path, mode, newline="")
        self._writer = None
        self._resume = resume and mode == "a"

    def log(self, row: dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            if not self._resume:
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

    # ── resume detection ───────────────────────────────────────────────────
    resume_ckpt, start_step = find_latest_checkpoint(cfg.output_dir)
    if resume_ckpt:
        log.info(f"Resuming from checkpoint: {resume_ckpt} (step {start_step})")
    else:
        log.info("No checkpoint found — starting from scratch.")

    use_wandb = bool(cfg.get("wandb_project"))
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                   resume="allow",
                   config=OmegaConf.to_container(cfg, resolve=True))

    csv_log = CSVLogger("logs/brain_train_metrics.csv", resume=bool(resume_ckpt))
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── model ──────────────────────────────────────────────────────────────
    from unsloth import FastVisionModel

    # load from checkpoint if resuming, otherwise from base model
    model_source = str(resume_ckpt) if resume_ckpt else cfg.model_id
    log.info(f"Loading model from: {model_source}")

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_source,
        max_seq_length=cfg.max_length,
        load_in_4bit=cfg.get("load_in_4bit", True),
        dtype=None,
    )

    if cfg.use_lora and not resume_ckpt:
        # only apply LoRA config on fresh start; resume loads adapters from ckpt
        model = FastVisionModel.get_peft_model(
            model,
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            bias="none",
            random_state=42,
        )
    model.print_trainable_parameters()

    # ── data ───────────────────────────────────────────────────────────────
    train_ds = ThreeTaskDataset(cfg.train_jsonl, cfg.image_root)
    val_ds   = ThreeTaskDataset(cfg.val_jsonl,   cfg.image_root)
    log.info(f"Train samples: {len(train_ds):,}  Val samples: {len(val_ds):,}")

    _collate     = lambda b: collate_fn(b, processor, device, cfg.max_length)
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
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    # restore optimizer + scheduler state if resuming
    if resume_ckpt:
        opt_path = resume_ckpt / "optimizer.pt"
        sch_path = resume_ckpt / "scheduler.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            log.info("Optimizer state restored.")
        if sch_path.exists():
            scheduler.load_state_dict(torch.load(sch_path, map_location=device))
            log.info("Scheduler state restored.")

    # ── training ───────────────────────────────────────────────────────────
    model.train()
    global_step = start_step
    grad_accum  = cfg.get("grad_accum_steps", 1)
    optimizer.zero_grad()

    accum: dict[str, float]      = defaultdict(float)
    accum_counts: dict[str, int] = defaultdict(int)

    # how many batches to skip at the start of the first epoch when resuming
    batches_to_skip = start_step % len(train_loader) if start_step > 0 else 0

    while global_step < cfg.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= cfg.max_steps:
                break

            # skip already-completed batches in the resumed epoch
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            task_types = batch.pop("task_types")
            outputs    = model(**batch)
            loss       = outputs.loss / grad_accum
            loss.backward()

            raw_loss = outputs.loss.item()
            accum["loss"] += raw_loss
            accum_counts["loss"] += 1
            for tt in task_types:
                accum[f"loss_{tt}"] += raw_loss
                accum_counts[f"loss_{tt}"] += 1

            if (global_step + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = 0.0

            global_step += 1

            if global_step % cfg.log_steps == 0:
                lr      = scheduler.get_last_lr()[0]
                gpu_mem = (torch.cuda.memory_allocated(device) / 1e9
                           if device.type == "cuda" else 0.0)
                row = {"step": global_step, "lr": lr,
                       "grad_norm": grad_norm, "gpu_mem_gb": gpu_mem}
                for k, total in accum.items():
                    row[k] = total / max(accum_counts[k], 1)
                log.info(
                    f"step={global_step}  loss={row['loss']:.4f}  "
                    f"grounding={row.get('loss_grounding', 0):.4f}  "
                    f"parsing={row.get('loss_parsing', 0):.4f}  "
                    f"synthesis={row.get('loss_task_synthesis', 0):.4f}  "
                    f"lr={lr:.2e}  grad_norm={grad_norm:.3f}  gpu={gpu_mem:.1f}GB"
                )
                if use_wandb:
                    import wandb
                    wandb.log({f"train/{k}": v for k, v in row.items()
                               if k != "step"}, step=global_step)
                csv_log.log({"phase": "train", **row})
                accum.clear()
                accum_counts.clear()

            if global_step % cfg.eval_steps == 0:
                metrics = evaluate(model, val_loader, device)
                log.info(f"[eval] step={global_step}  " +
                         "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if use_wandb:
                    import wandb
                    wandb.log({f"eval/{k}": v for k, v in metrics.items()},
                              step=global_step)
                csv_log.log({"phase": "eval", "step": global_step, **metrics})

            if global_step % cfg.save_steps == 0:
                save_checkpoint(model, processor, optimizer, scheduler,
                                global_step, cfg.output_dir)

        batches_to_skip = 0  # only skip on first epoch pass after resume

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
