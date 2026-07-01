from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def select_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor | None:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if np.any(counts == 0):
        return None
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_grad_scaler(device: torch.device, enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    except TypeError:
        return torch.cuda.amp.autocast()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, int]:
    model.train()
    scaler = make_grad_scaler(device, enabled=use_amp and device.type == "cuda")
    total_loss = 0.0
    total_steps = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach().cpu())
        total_steps += 1

    if total_steps == 0:
        raise ValueError("Training loader produced no batches")
    return total_loss / total_steps, total_steps


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_labels: list[int] = []
    all_probs: list[list[float]] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total += batch_size
        correct += int((pred == labels).sum().detach().cpu())
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    if total == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}

    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }
    auc = compute_auc(all_labels, all_probs)
    if auc is not None:
        metrics["auc"] = auc
    return metrics


def compute_auc(labels: list[int], probs: list[list[float]]) -> float | None:
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    prob_array = np.asarray(probs)
    label_array = np.asarray(labels)
    try:
        if prob_array.shape[1] == 2:
            return float(roc_auc_score(label_array, prob_array[:, 1]))
        return float(roc_auc_score(label_array, prob_array, multi_class="ovr"))
    except ValueError:
        return None


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    metrics: dict[str, float],
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": metrics,
        "class_names": class_names,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output_path)


def printable_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
