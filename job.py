from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_CUSTOM_DIR = PROJECT_ROOT / "app" / "custom"
sys.path.insert(0, str(APP_CUSTOM_DIR))

from model import Net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or export local NVFlare FedAvg for LAM/BHD/Control classification.")
    parser.add_argument("--job-name", default="LAM_BHD_fedavg")
    parser.add_argument("--n-clients", "--n_clients", type=int, default=2)
    parser.add_argument("--num-rounds", "--num_rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per federated round.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=2)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-shape", default="64,64,64")
    parser.add_argument("--data-dir", default="auto")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--gpu-config", default="", help="Comma-separated GPU ids for SimEnv, for example 0,1.")
    parser.add_argument("--launch-external-process", action="store_true")
    parser.add_argument("--enable-log-streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--partition-sites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export", action="store_true", help="Export the NVFlare job instead of running simulation.")
    parser.add_argument("--export-dir", default="/tmp/nvflare_jobs/lam_bhd")
    return parser.parse_args()


def build_train_args(args: argparse.Namespace) -> str:
    parts = [
        "--data-dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--target-shape",
        args.target_shape,
        "--num-classes",
        str(args.num_classes),
        "--device",
        args.device,
    ]
    if args.partition_sites:
        parts.extend(["--num-partitions", str(args.n_clients), "--partition-index", "-1"])
    return " ".join(shlex.quote(part) for part in parts)


def build_recipe(args: argparse.Namespace):
    from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
    from nvflare.recipe import add_experiment_tracking

    recipe = FedAvgRecipe(
        name=args.job_name,
        min_clients=args.n_clients,
        num_rounds=args.num_rounds,
        initial_model=Net(num_classes=args.num_classes),
        train_script=str(APP_CUSTOM_DIR / "fl_client.py"),
        train_args=build_train_args(args),
        launch_external_process=args.launch_external_process,
    )

    add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.enable_log_streaming and hasattr(recipe, "enable_log_streaming"):
        recipe.enable_log_streaming()
    return recipe


def build_sim_env(args: argparse.Namespace):
    from nvflare.recipe import SimEnv

    site_names = [f"site-{idx + 1}" for idx in range(args.n_clients)]
    env_kwargs = {
        "clients": site_names,
        "num_threads": args.num_threads or args.n_clients,
    }
    if args.gpu_config:
        env_kwargs["gpu_config"] = args.gpu_config
    return SimEnv(**env_kwargs)


def main() -> None:
    args = parse_args()
    recipe = build_recipe(args)
    env = build_sim_env(args)

    if args.export:
        recipe.export(args.export_dir, env=env)
        print(f"Exported NVFlare job to: {args.export_dir}")
        return

    run = recipe.execute(env=env)
    print()
    print("Job status:", run.get_status())
    print("Result path:", run.get_result())


if __name__ == "__main__":
    main()
