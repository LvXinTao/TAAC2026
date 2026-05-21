"""PCVRHyFormer trainer (binary-classification, AUC-monitored).

Uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
Supports EMA and SWA for weight averaging.
"""

import os
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class EMA:
    """Exponential Moving Average for model dense parameters."""

    def __init__(self, model: nn.Module, decay: float, device: str) -> None:
        self.decay = decay
        self.model = model
        self.shadow: dict = {}
        for p in model.get_dense_params():
            self.shadow[p.data_ptr()] = p.data.clone().to(device)

    @torch.no_grad()
    def step(self) -> None:
        """Update shadow copies: shadow = decay * shadow + (1 - decay) * param."""
        for p in self.model.get_dense_params():
            ptr = p.data_ptr()
            if ptr in self.shadow:
                self.shadow[ptr].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to_model(self) -> dict:
        """Swap EMA weights into model, return old values for restore."""
        old = {}
        for p in self.model.get_dense_params():
            ptr = p.data_ptr()
            if ptr in self.shadow:
                old[ptr] = p.data.clone()
                p.data.copy_(self.shadow[ptr])
        return old

    @torch.no_grad()
    def restore(self, old: dict) -> None:
        """Restore model params from apply_to_model return value."""
        for p in self.model.get_dense_params():
            ptr = p.data_ptr()
            if ptr in old:
                p.data.copy_(old[ptr])


class SWA:
    """Stochastic Weight Averaging for model dense parameters.

    Collects snapshots of dense parameters at fixed intervals (swa_freq steps)
    starting after swa_start steps, maintaining a running average.

    Usage:
        swa = SWA(model, swa_start=5000, swa_freq=500, device='cuda')
        # After optimizer.step():
        swa.step(global_step)
        # Before saving best checkpoint:
        old = swa.apply_to_model()
        torch.save(model.state_dict(), path)
        swa.restore(old)
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start: int,
        swa_freq: int,
        device: str,
    ) -> None:
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.model = model
        self.device = device
        self.swa_state: dict = {}   # data_ptr -> averaged tensor
        self.n_averaged: int = 0    # number of snapshots averaged so far

        for p in model.get_dense_params():
            self.swa_state[p.data_ptr()] = p.data.clone().to(device)

    def should_update(self, global_step: int) -> bool:
        """Check if we should collect a snapshot at this step."""
        return global_step > self.swa_start and (
            global_step - self.swa_start
        ) % self.swa_freq == 0

    @torch.no_grad()
    def step(self, global_step: int) -> None:
        """Collect snapshot and update running average if due."""
        if not self.should_update(global_step):
            return

        for p in self.model.get_dense_params():
            ptr = p.data_ptr()
            if ptr in self.swa_state:
                old_avg = self.swa_state[ptr]
                n = self.n_averaged + 1
                # Running average: new_avg = (old_avg * n_old + param) / n
                self.swa_state[ptr].copy_(
                    old_avg * self.n_averaged + p.data.to(self.device),
                )
                self.swa_state[ptr].div_(n)
        self.n_averaged += 1

    @torch.no_grad()
    def apply_to_model(self) -> dict:
        """Swap SWA weights into model, return old values for restore."""
        old = {}
        for p in self.model.get_dense_params():
            ptr = p.data_ptr()
            if ptr in self.swa_state:
                old[ptr] = p.data.clone()
                p.data.copy_(self.swa_state[ptr])
        return old

    @torch.no_grad()
    def restore(self, old: dict) -> None:
        """Restore model params from apply_to_model return value."""
        for p in self.model.get_dense_params():
            ptr = p.data_ptr()
            if ptr in old:
                p.data.copy_(old[ptr])


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        ema_decay: float = 0.0,
        swa_start: int = 0,
        swa_freq: int = 0,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        self.schema_path: Optional[str] = schema_path
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr}, wd={weight_decay})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.warmup_steps: int = warmup_steps
        self.total_step: int = 0  # tracked for SWA step scheduling
        self.warmup_scheduler = None
        if warmup_steps > 0:
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.dense_optimizer,
                lr_lambda=lambda step: min(1.0, step / warmup_steps),
            )

        # EMA
        self.ema_decay = ema_decay
        self.ema: Optional[EMA] = None
        if ema_decay > 0:
            self.ema = EMA(model, decay=ema_decay, device=device)
            logging.info(f"EMA enabled with decay={ema_decay}")

        # SWA
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa: Optional[SWA] = None
        if swa_freq > 0:
            self.swa = SWA(model, swa_start=swa_start, swa_freq=swa_freq, device=device)
            logging.info(f"SWA enabled: start={swa_start}, freq={swa_freq}")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}, "
                     f"warmup_steps={warmup_steps}, weight_decay={weight_decay}")

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name."""
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``."""
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir."""
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories."""
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device``."""
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically."""
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Swap to weight-averaged weights before saving (EMA or SWA).
        # Priority: EMA > SWA (only one should be active at a time).
        swap_old = None
        if self.ema is not None:
            swap_old = self.ema.apply_to_model()
        elif self.swa is not None:
            swap_old = self.swa.apply_to_model()

        try:
            best_dir = os.path.join(
                self.save_dir,
                self._build_step_dir_name(total_step, is_best=True),
            )
            self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")
            self._remove_old_best_dirs()

            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })

            if self.early_stopping.best_score != old_best and os.path.exists(
                self.early_stopping.checkpoint_path
            ):
                self._save_step_checkpoint(
                    total_step, is_best=True, skip_model_file=True)
        finally:
            if swap_old is not None:
                if self.ema is not None:
                    self.ema.restore(swap_old)
                elif self.swa is not None:
                    self.swa.restore(swap_old)

    def train(self) -> None:
        """Main training loop."""
        print("Start training (PCVRHyFormer)")
        self.model.train()
        self.total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                self.total_step += 1
                loss_sum += loss

                if self.warmup_scheduler is not None and self.total_step <= self.warmup_steps:
                    self.warmup_scheduler.step()

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, self.total_step)
                    self.writer.add_scalar('LR/train', self.dense_optimizer.param_groups[0]['lr'], self.total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                if self.eval_every_n_steps > 0 and self.total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {self.total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {self.total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, self.total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, self.total_step)

                    self._handle_validation_result(self.total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {self.total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, self.total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, self.total_step)

            self._handle_validation_result(self.total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.ema is not None:
            self.ema.step()
        if self.swa is not None:
            self.swa.step(self.total_step)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation and return ``(AUC, logloss)``."""
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
