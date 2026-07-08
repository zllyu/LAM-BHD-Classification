from __future__ import annotations

import argparse
import re

import torch

from data import build_dataloaders, parse_target_shape
from model import Net
from training import class_weights, evaluate, printable_metrics, select_device, train_one_epoch


try:
    from nvflare.apis.fl_constant import FLMetaKey

    NUM_STEPS_KEY = FLMetaKey.NUM_STEPS_CURRENT_ROUND
except Exception:
    NUM_STEPS_KEY = "NUM_STEPS_CURRENT_ROUND"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVFlare Client API trainer for LAM/BHD/Control classification.")
    parser.add_argument("--data-dir", default="auto", help="Dataset root. Use 'auto' on Rhino.")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per federated round.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-shape", default="64,64,64")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-classes", type=int, default=3, help="0 means infer from local labels.")
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    return parser.parse_args()


def infer_partition_index(site_name: str, num_partitions: int) -> int:
    match = re.search(r"(\d+)$", site_name or "")
    if not match:
        return 0
    return (int(match.group(1)) - 1) % num_partitions


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    target_shape = parse_target_shape(args.target_shape)
    device = select_device(args.device)

    import nvflare.client as flare
    from nvflare.client.tracking import SummaryWriter

    flare.init()
    sys_info = flare.system_info()
    site_name = sys_info.get("site_name", "unknown")

    partition_index = args.partition_index
    if args.num_partitions < 1:
        raise ValueError("--num-partitions must be >= 1")
    if args.num_partitions == 1:
        partition_index = 0
    elif partition_index < 0:
        partition_index = infer_partition_index(site_name, args.num_partitions)

    train_loader, val_loader, _, summary = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        target_shape=target_shape,
        partition_index=partition_index,
        num_partitions=args.num_partitions,
    )
    num_classes = args.num_classes or summary.num_classes
    if num_classes != summary.num_classes:
        raise ValueError(
            f"Configured num_classes={num_classes}, but data contains {summary.num_classes}: {summary.class_names}"
        )

    model = Net(num_classes=num_classes)
    weights = class_weights(list(train_loader.dataset.labels), num_classes, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    writer = SummaryWriter()
    print(f"site={site_name}, data_dir={summary.data_dir}, classes={summary.class_names}")
    if summary.labels_csv:
        print(f"site={site_name}, labels_csv={summary.labels_csv}")
    print(
        f"site={site_name}, split train={summary.train_count}, val={summary.val_count}, "
        f"partition={partition_index}/{args.num_partitions}"
    )

    while flare.is_running():
        input_model = flare.receive()
        current_round = getattr(input_model, "current_round", 0) or 0
        print(f"site={site_name}, round={current_round}, received global model")

        if input_model.params:
            model.load_state_dict(input_model.params, strict=True)
        model.to(device)

        before_metrics = evaluate(model, val_loader, loss_fn, device)
        print(f"site={site_name}, round={current_round}, before_{printable_metrics(before_metrics)}")

        if flare.is_evaluate():
            flare.send(flare.FLModel(metrics=before_metrics))
            continue

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = 0
        last_loss = 0.0
        for local_epoch in range(1, args.epochs + 1):
            last_loss, steps = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            total_steps += steps
            global_step = current_round * max(1, args.epochs) + local_epoch
            writer.add_scalar("train_loss", last_loss, global_step)
            print(
                f"site={site_name}, round={current_round}, local_epoch={local_epoch}/{args.epochs}, "
                f"loss={last_loss:.4f}"
            )

        metrics = evaluate(model, val_loader, loss_fn, device)
        metrics["train_loss"] = float(last_loss)
        writer.add_scalar("accuracy", metrics["accuracy"], current_round)
        print(f"site={site_name}, round={current_round}, after_{printable_metrics(metrics)}")

        output_model = flare.FLModel(
            params=state_dict_to_cpu(model),
            metrics=metrics,
            meta={NUM_STEPS_KEY: total_steps},
        )
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        flare.send(output_model)


if __name__ == "__main__":
    main()
