from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import build_dataloaders, parse_target_shape
from model import Net
from training import (
    class_weights,
    evaluate,
    printable_metrics,
    save_checkpoint,
    select_device,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LAM/BHD/Control classifier on one local site.")
    parser.add_argument("--data-dir", default="auto", help="Dataset root. Use 'auto' for Rhino/local defaults.")
    parser.add_argument("--output-dir", default="outputs/single_site")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-shape", default="64,64,64", help="'none' or D,H,W, for example 64,64,64")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    target_shape = parse_target_shape(args.target_shape)
    device = select_device(args.device)

    train_loader, val_loader, test_loader, summary = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        target_shape=target_shape,
    )
    print(f"Data dir: {summary.data_dir}")
    if summary.labels_csv:
        print(f"Labels CSV: {summary.labels_csv}")
    print(f"Classes: {summary.class_names} counts={summary.class_counts}")
    print(f"Split: train={summary.train_count}, val={summary.val_count}, test={summary.test_count}")
    print(f"Device: {device}")

    train_labels = list(train_loader.dataset.labels)
    model = Net(num_classes=summary.num_classes).to(device)
    weights = class_weights(train_labels, summary.num_classes, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_accuracy = -1.0
    output_dir = Path(args.output_dir)
    for epoch in range(1, args.epochs + 1):
        train_loss, steps = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        print(
            f"epoch={epoch}/{args.epochs}, steps={steps}, train_loss={train_loss:.4f}, "
            f"val_{printable_metrics(val_metrics)}"
        )
        accuracy = val_metrics.get("accuracy", float("nan"))
        if accuracy == accuracy and accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                output_dir / "best_model.pt",
                model,
                val_metrics,
                summary.class_names,
                extra={"epoch": epoch, "target_shape": target_shape},
            )

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    save_checkpoint(
        output_dir / "final_model.pt",
        model,
        test_metrics,
        summary.class_names,
        extra={"target_shape": target_shape},
    )
    print(f"test_{printable_metrics(test_metrics)}")
    print(f"Saved checkpoints under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
