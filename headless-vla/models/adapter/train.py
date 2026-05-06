"""
Adapter training: conditional flow matching on LIBERO-Spatial motion primitives.

Prerequisite: compute action normalisation stats first:
    python -c "
    from models.middleware.normalize import compute_and_save_stats
    compute_and_save_stats('data/splits/adapter_franka.jsonl',
                           'ckpts/franka_7dof/action_stats.json')
    "

Usage:
    python -m models.adapter.train --config-name=franka_7dof
"""

import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from models.adapter.architecture import SemanticActionAdapter
from models.adapter.flow_matching import flow_matching_loss
from models.middleware.graph_encoder import GRAPH_FEAT_DIM
from models.middleware.normalize import ActionNormalizer, compute_and_save_stats

log = logging.getLogger(__name__)


# ── dataset ────────────────────────────────────────────────────────────────────

class AdapterDataset(Dataset):
    def __init__(self, jsonl_path: str, normalizer: ActionNormalizer,
                 action_dim: int, chunk_size: int, graph_feat_dim: int,
                 state_dim: int):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.normalizer   = normalizer
        self.action_dim   = action_dim
        self.chunk_size   = chunk_size
        self.graph_feat_dim = graph_feat_dim
        self.state_dim    = state_dim

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        task_emb = torch.tensor(rec["task_emb"], dtype=torch.float32)  # [task_embed_dim]

        graph_feats = np.array(rec["graph_feats"], dtype=np.float32)
        if len(graph_feats) < self.graph_feat_dim:
            graph_feats = np.pad(graph_feats, (0, self.graph_feat_dim - len(graph_feats)))
        graph_feats = torch.tensor(graph_feats[:self.graph_feat_dim],
                                   dtype=torch.float32).unsqueeze(0)  # [1, G]

        proprio = np.array(rec["proprio"], dtype=np.float32)
        if len(proprio) < self.state_dim:
            proprio = np.pad(proprio, (0, self.state_dim - len(proprio)))
        state = torch.tensor(proprio[:self.state_dim], dtype=torch.float32)  # [state_dim]

        chunk_raw = np.array(rec["action_chunk"], dtype=np.float32)  # [T, A_raw]
        T_raw = chunk_raw.shape[0]
        # pad / truncate to chunk_size
        if T_raw < self.chunk_size:
            pad = np.zeros((self.chunk_size - T_raw, chunk_raw.shape[1]), dtype=np.float32)
            chunk_raw = np.concatenate([chunk_raw, pad], axis=0)
        chunk_raw = chunk_raw[:self.chunk_size, :self.action_dim]  # [T, A]

        chunk_norm = self.normalizer.normalize_chunk(chunk_raw)
        action_chunk = torch.tensor(chunk_norm, dtype=torch.float32)  # [T, A]

        return {
            "task_emb":     task_emb,
            "graph_feats":  graph_feats,
            "state":        state,
            "action_chunk": action_chunk,
        }


def collate(batch):
    return {
        "task_emb":     torch.stack([b["task_emb"]     for b in batch]),   # [B, E]
        "graph_feats":  torch.stack([b["graph_feats"]  for b in batch]),   # [B, 1, G]
        "state":        torch.stack([b["state"]        for b in batch]),   # [B, S]
        "action_chunk": torch.stack([b["action_chunk"] for b in batch]),   # [B, T, A]
    }


# ── training loop ──────────────────────────────────────────────────────────────

def evaluate_loss(adapter, val_loader, device) -> float:
    adapter.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            loss = flow_matching_loss(
                adapter,
                batch["task_emb"].to(device),
                batch["graph_feats"].to(device),
                batch["state"].to(device),
                batch["action_chunk"].to(device),
            )
            total += loss.item(); n += 1
    adapter.train()
    return total / max(n, 1)


@hydra.main(config_path="../../configs/adapter", config_name="franka_7dof", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── action normalisation stats (mandatory) ─────────────────────────────
    stats_path = Path(cfg.output_dir) / "action_stats.json"
    if not stats_path.exists():
        log.info("Computing action normalisation stats …")
        compute_and_save_stats(cfg.train_jsonl, str(stats_path), cfg.action_dim)

    normalizer = ActionNormalizer(str(stats_path))

    # ── wandb ──────────────────────────────────────────────────────────────
    if cfg.get("wandb_project"):
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── model ──────────────────────────────────────────────────────────────
    adapter = SemanticActionAdapter(
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size,
        hidden=cfg.hidden,
        n_heads=cfg.n_heads,
        n_blocks=cfg.n_blocks,
        graph_feat_dim=cfg.graph_feat_dim,
        state_dim=cfg.state_dim,
        task_embed_dim=cfg.get("task_embed_dim", 384),
    ).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    log.info(f"Adapter parameters: {n_params:,}")

    # ── data ───────────────────────────────────────────────────────────────
    train_ds = AdapterDataset(cfg.train_jsonl, normalizer, cfg.action_dim,
                               cfg.chunk_size, cfg.graph_feat_dim, cfg.state_dim)
    val_ds   = AdapterDataset(cfg.val_jsonl,   normalizer, cfg.action_dim,
                               cfg.chunk_size, cfg.graph_feat_dim, cfg.state_dim)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=0)

    # ── optimiser ──────────────────────────────────────────────────────────
    from transformers import get_cosine_schedule_with_warmup
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg.peak_lr,
                                   betas=(cfg.beta1, cfg.beta2), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps
    )

    # ── train ──────────────────────────────────────────────────────────────
    adapter.train()
    global_step = 0
    accum_loss = 0.0

    while global_step < cfg.max_steps:
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            loss = flow_matching_loss(
                adapter,
                batch["task_emb"].to(device),
                batch["graph_feats"].to(device),
                batch["state"].to(device),
                batch["action_chunk"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            accum_loss += loss.item()
            global_step += 1

            if global_step % cfg.log_steps == 0:
                lr = scheduler.get_last_lr()[0]
                log.info(f"step={global_step} loss={accum_loss/cfg.log_steps:.5f} lr={lr:.2e}")
                if cfg.get("wandb_project"):
                    import wandb
                    wandb.log({"train/loss": accum_loss / cfg.log_steps,
                               "train/lr": lr, "step": global_step})
                accum_loss = 0.0

            if global_step % cfg.eval_steps == 0:
                val_loss = evaluate_loss(adapter, val_loader, device)
                log.info(f"[eval] step={global_step} val_loss={val_loss:.5f}")
                if cfg.get("wandb_project"):
                    import wandb
                    wandb.log({"eval/loss": val_loss, "step": global_step})

            if global_step % cfg.save_steps == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                adapter.save(str(ckpt_dir))
                # copy stats into checkpoint
                import shutil
                shutil.copy(stats_path, ckpt_dir / "action_stats.json")
                log.info(f"Checkpoint saved: {ckpt_dir}")

    # final save
    final_dir = Path(cfg.output_dir) / "final"
    adapter.save(str(final_dir))
    import shutil
    shutil.copy(stats_path, final_dir / "action_stats.json")
    log.info(f"Final checkpoint: {final_dir}")
    if cfg.get("wandb_project"):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
