from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_LOCAL_DATA_DIR = Path("data/LAM_BHD_synthetic_nrrd_dataset")
RHINO_DATASETS_DIR = Path("/input/datasets")
DEFAULT_CLASS_ORDER = ("LAM", "BHD", "Control")
LABELS_CSV_NAME = "labels.csv"
VOLUME_EXTENSIONS = (".nrrd", ".nhdr", ".npy")


@dataclass(frozen=True)
class DatasetSummary:
    data_dir: Path
    class_names: list[str]
    class_counts: list[int]
    train_count: int
    val_count: int
    test_count: int
    labels_csv: Path | None = None

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def parse_target_shape(value: str | Sequence[int] | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "none", "native", "off", "false"}:
            return None
        parts = cleaned.replace("x", ",").replace(" ", ",").split(",")
        shape = [int(part) for part in parts if part]
    else:
        shape = [int(part) for part in value]
    if len(shape) != 3 or any(dim <= 0 for dim in shape):
        raise ValueError("target_shape must be three positive integers, for example 64,64,64")
    return tuple(shape)


def resolve_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    if data_dir and str(data_dir).lower() != "auto":
        return Path(data_dir).expanduser().resolve()

    env_dir = os.environ.get("LAM_BHD_DATA_DIR")
    if env_dir and env_dir.lower() != "auto":
        return Path(env_dir).expanduser().resolve()

    rhino_dir = find_rhino_data_dir()
    if rhino_dir is not None:
        return rhino_dir

    return DEFAULT_LOCAL_DATA_DIR.expanduser().resolve()


def find_rhino_data_dir(root: Path = RHINO_DATASETS_DIR) -> Path | None:
    if not root.exists():
        return None

    candidates: list[Path] = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        candidates.append(dataset_dir / "file_data")
        candidates.append(dataset_dir)
    candidates.append(root)

    for candidate in candidates:
        if candidate.exists() and has_dataset_records(candidate):
            return candidate.resolve()
    return None


def has_dataset_records(path: Path) -> bool:
    return (path / LABELS_CSV_NAME).is_file() or has_class_subdirs(path)


def has_class_subdirs(path: Path) -> bool:
    return any(child.is_dir() and list_volume_files(child) for child in path.iterdir())


def list_volume_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and _is_supported_volume(child):
            files.append(child)
    return files


def _is_supported_volume(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(VOLUME_EXTENSIONS)


def discover_dataset(data_dir: Path) -> tuple[list[Path], list[int], list[str], list[int], Path | None]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    labels_csv = data_dir / LABELS_CSV_NAME
    if labels_csv.is_file():
        files, labels, class_names, class_counts = discover_dataset_from_csv(data_dir, labels_csv)
        return files, labels, class_names, class_counts, labels_csv

    files, labels, class_names, class_counts = discover_dataset_from_class_dirs(data_dir)
    return files, labels, class_names, class_counts, None


def discover_dataset_from_csv(
    data_dir: Path,
    labels_csv: Path,
) -> tuple[list[Path], list[int], list[str], list[int]]:
    rows: list[tuple[Path, str]] = []
    with labels_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"filepath", "GT"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{labels_csv} is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            label = (row.get("GT") or "").strip()
            raw_filepath = (row.get("filepath") or "").strip()
            if not label or not raw_filepath:
                raise ValueError(f"{labels_csv}:{row_number} must have non-empty filepath and GT")

            file_path = Path(raw_filepath)
            if not file_path.is_absolute():
                file_path = data_dir / file_path
            file_path = file_path.resolve()
            if not _is_supported_volume(file_path):
                raise ValueError(f"Unsupported volume extension in {labels_csv}:{row_number}: {file_path}")
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing file referenced by {labels_csv}:{row_number}: {file_path}")
            rows.append((file_path, label))

    if not rows:
        raise ValueError(f"No dataset rows found in {labels_csv}")

    labels_in_csv = [label for _, label in rows]
    class_names = ordered_class_names(labels_in_csv, include_default_order=True)
    label_to_index = {label: index for index, label in enumerate(class_names)}
    files = [file_path for file_path, _ in rows]
    labels = [label_to_index[label] for _, label in rows]
    class_counts = [labels.count(index) for index in range(len(class_names))]
    return files, labels, class_names, class_counts


def discover_dataset_from_class_dirs(data_dir: Path) -> tuple[list[Path], list[int], list[str], list[int]]:
    class_dirs_by_name = {p.name: p for p in data_dir.iterdir() if p.is_dir() and list_volume_files(p)}
    if len(class_dirs_by_name) < 2:
        raise ValueError(
            f"Expected at least two class subdirectories with {VOLUME_EXTENSIONS} files under {data_dir}"
        )

    class_names = ordered_class_names(class_dirs_by_name.keys(), include_default_order=False)
    files: list[Path] = []
    labels: list[int] = []
    class_counts: list[int] = []
    for label, class_name in enumerate(class_names):
        class_files = list_volume_files(class_dirs_by_name[class_name])
        class_counts.append(len(class_files))
        files.extend(class_files)
        labels.extend([label] * len(class_files))

    return files, labels, class_names, class_counts


def ordered_class_names(labels: Iterable[str], include_default_order: bool) -> list[str]:
    observed = set(labels)
    ordered = [name for name in DEFAULT_CLASS_ORDER if name in observed or include_default_order]
    ordered.extend(sorted(observed.difference(DEFAULT_CLASS_ORDER)))
    if len(ordered) < 2:
        raise ValueError(f"Expected at least two classes, found: {ordered}")
    return ordered


def partition_records(
    files: Sequence[Path],
    labels: Sequence[int],
    num_partitions: int = 1,
    partition_index: int = 0,
    seed: int = 42,
) -> tuple[list[Path], list[int]]:
    if num_partitions <= 1:
        return list(files), list(labels)
    if partition_index < 0 or partition_index >= num_partitions:
        raise ValueError("partition_index must be between 0 and num_partitions - 1")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    labels_np = np.asarray(labels)
    for label in sorted(set(labels)):
        indices = np.where(labels_np == label)[0]
        rng.shuffle(indices)
        selected.extend(indices[partition_index::num_partitions].tolist())

    selected.sort()
    if not selected:
        raise ValueError("Partition produced no samples; reduce num_partitions or check the dataset")
    return [files[i] for i in selected], [labels[i] for i in selected]


def stratified_split(
    files: Sequence[Path],
    labels: Sequence[int],
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be non-negative and sum to less than 1")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    labels_np = np.asarray(labels)

    for label in sorted(set(labels)):
        indices = np.where(labels_np == label)[0]
        rng.shuffle(indices)
        n = len(indices)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, n - 2))
            n_val = max(0, min(n_val, n - n_test - 1))

        test_idx.extend(indices[:n_test].tolist())
        val_idx.extend(indices[n_test : n_test + n_val].tolist())
        train_idx.extend(indices[n_test + n_val :].tolist())

    for bucket in (train_idx, val_idx, test_idx):
        bucket.sort()

    def pick(indices: Iterable[int]) -> tuple[list[Path], list[int]]:
        idx = list(indices)
        return [files[i] for i in idx], [labels[i] for i in idx]

    train_files, train_labels = pick(train_idx)
    val_files, val_labels = pick(val_idx)
    test_files, test_labels = pick(test_idx)
    if not train_files:
        raise ValueError("Training split is empty; provide more data or reduce validation/test fractions")
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


class VolumeClassificationDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        labels: Sequence[int],
        target_shape: tuple[int, int, int] | None = (64, 64, 64),
        augment: bool = False,
    ):
        self.files = list(files)
        self.labels = list(labels)
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        volume = load_volume(self.files[index], self.target_shape)
        if self.augment:
            volume = random_flip_3d(volume)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return volume, label


def load_volume(path: Path, target_shape: tuple[int, int, int] | None = (64, 64, 64)) -> torch.Tensor:
    name = path.name.lower()
    if name.endswith(".npy"):
        array = np.load(path)
    else:
        try:
            import nrrd
        except ImportError as exc:
            raise ImportError("Install pynrrd to read .nrrd/.nhdr volumes: pip install pynrrd") from exc
        array, _ = nrrd.read(str(path))

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, copy=False)
    if array.ndim == 4 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume in {path}, got shape {array.shape}")

    tensor = torch.from_numpy(normalize_intensity(array)).unsqueeze(0)
    if target_shape is not None and tuple(tensor.shape[1:]) != tuple(target_shape):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.contiguous()


def normalize_intensity(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        raise ValueError("Cannot normalize an empty volume")
    low, high = np.percentile(array, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    array = np.clip(array, low, high)
    return ((array - low) / (high - low)).astype(np.float32)


def random_flip_3d(volume: torch.Tensor) -> torch.Tensor:
    for axis in (1, 2, 3):
        if torch.rand(()) < 0.5:
            volume = torch.flip(volume, dims=(axis,))
    return volume


def build_dataloaders(
    data_dir: str | os.PathLike[str] | None = None,
    batch_size: int = 2,
    num_workers: int = 0,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 42,
    target_shape: tuple[int, int, int] | None = (64, 64, 64),
    partition_index: int = 0,
    num_partitions: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, DatasetSummary]:
    resolved_dir = resolve_data_dir(data_dir)
    files, labels, class_names, class_counts, labels_csv = discover_dataset(resolved_dir)
    files, labels = partition_records(files, labels, num_partitions, partition_index, seed)
    train_x, train_y, val_x, val_y, test_x, test_y = stratified_split(
        files, labels, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    train_ds = VolumeClassificationDataset(train_x, train_y, target_shape=target_shape, augment=True)
    val_ds = VolumeClassificationDataset(val_x, val_y, target_shape=target_shape, augment=False)
    test_ds = VolumeClassificationDataset(test_x, test_y, target_shape=target_shape, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    summary = DatasetSummary(
        data_dir=resolved_dir,
        class_names=class_names,
        class_counts=class_counts,
        train_count=len(train_ds),
        val_count=len(val_ds),
        test_count=len(test_ds),
        labels_csv=labels_csv,
    )
    return train_loader, val_loader, test_loader, summary
