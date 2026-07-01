#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = ("LAM", "BHD", "Control")
DEFAULT_OUT_DIR = Path("data/LAM_BHD_synthetic_nrrd_dataset")


@dataclass(frozen=True)
class CystSummary:
    count: int
    mean_radius_vox: float
    max_radius_vox: float


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").replace(" ", ",").split(",")
    dims = tuple(int(part) for part in parts if part)
    if len(dims) != 3 or any(dim < 32 for dim in dims):
        raise argparse.ArgumentTypeError("shape must be three integers >= 32, for example 96,128,128")
    return dims


def lung_masks(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_size, y_size, x_size = shape
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, z_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, y_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, x_size, dtype=np.float32),
        indexing="ij",
    )

    body = (x / 0.90) ** 2 + ((y + 0.02) / 0.78) ** 2 + (z / 1.05) ** 2 < 1.0
    left_score = ((x + 0.34) / 0.34) ** 2 + ((y + 0.02) / 0.53) ** 2 + ((z + 0.03) / 0.93) ** 2
    right_score = ((x - 0.34) / 0.34) ** 2 + ((y + 0.02) / 0.53) ** 2 + ((z + 0.03) / 0.93) ** 2
    left = left_score < 1.0
    right = right_score < 1.0
    lungs = left | right

    mediastinum = (np.abs(x) < 0.16) & (np.abs(y) < 0.32) & (np.abs(z) < 0.72)
    lungs &= ~mediastinum
    body |= lungs

    edge_score = np.minimum(left_score, right_score)
    return body, lungs, z, y, x, edge_score


def make_base_volume(
    rng: np.random.Generator,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body, lungs, z, y, x, edge_score = lung_masks(shape)
    volume = np.full(shape, -1024.0, dtype=np.float32)

    body_noise = rng.normal(0.0, 24.0, size=shape).astype(np.float32)
    lung_noise = rng.normal(0.0, 36.0, size=shape).astype(np.float32)
    cranio_caudal_gradient = 45.0 * z

    volume[body] = 25.0 + body_noise[body]
    volume[lungs] = -835.0 + lung_noise[lungs] + cranio_caudal_gradient[lungs]

    add_vessel_like_markings(volume, lungs, x, y, z, rng)
    add_chest_wall_rind(volume, body, lungs)
    return volume, lungs, z, y, x, edge_score


def add_chest_wall_rind(volume: np.ndarray, body: np.ndarray, lungs: np.ndarray) -> None:
    rind = body & ~lungs
    volume[rind] = np.maximum(volume[rind], -120.0)


def add_vessel_like_markings(
    volume: np.ndarray,
    lungs: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
) -> None:
    for side, x_center in (("left", -0.34), ("right", 0.34)):
        _ = side
        for _index in range(18):
            z0 = rng.uniform(-0.65, 0.70)
            y0 = rng.uniform(-0.25, 0.25)
            angle = rng.uniform(-0.8, 0.8)
            slope = rng.uniform(-0.15, 0.15)
            radius = rng.uniform(0.012, 0.027)
            x_line = x_center + slope * z + 0.05 * np.sin(2.5 * z + angle)
            y_line = y0 + 0.16 * (z - z0) + 0.03 * np.cos(3.0 * z + angle)
            dist2 = ((x - x_line) ** 2 + (y - y_line) ** 2) / (radius**2)
            vessel = lungs & (np.abs(z - z0) < rng.uniform(0.22, 0.42)) & (dist2 < 1.0)
            if vessel.any():
                volume[vessel] = np.maximum(volume[vessel], rng.uniform(-650.0, -500.0))


def sample_lung_center(
    rng: np.random.Generator,
    lungs: np.ndarray,
    z: np.ndarray,
    edge_score: np.ndarray,
    mode: str,
) -> tuple[int, int, int]:
    candidates = np.argwhere(lungs)
    if mode == "lam":
        weights = np.ones(len(candidates), dtype=np.float32)
    elif mode == "bhd":
        z_values = z[lungs]
        edge_values = edge_score[lungs]
        weights = np.clip((z_values + 0.30) / 1.30, 0.0, 1.0) ** 2
        weights *= np.clip((edge_values - 0.42) / 0.58, 0.0, 1.0) ** 1.5
        weights += 1e-4
    else:
        raise ValueError(f"Unknown center sampling mode: {mode}")

    weights = weights / weights.sum()
    index = rng.choice(len(candidates), p=weights)
    return tuple(int(v) for v in candidates[index])


def add_cyst(
    volume: np.ndarray,
    lungs: np.ndarray,
    center: tuple[int, int, int],
    radii: tuple[float, float, float],
    wall_thickness: float,
    rng: np.random.Generator,
) -> float:
    zc, yc, xc = center
    rz, ry, rx = radii
    z_min = max(0, int(math.floor(zc - rz - wall_thickness - 2)))
    z_max = min(volume.shape[0], int(math.ceil(zc + rz + wall_thickness + 3)))
    y_min = max(0, int(math.floor(yc - ry - wall_thickness - 2)))
    y_max = min(volume.shape[1], int(math.ceil(yc + ry + wall_thickness + 3)))
    x_min = max(0, int(math.floor(xc - rx - wall_thickness - 2)))
    x_max = min(volume.shape[2], int(math.ceil(xc + rx + wall_thickness + 3)))

    zz, yy, xx = np.meshgrid(
        np.arange(z_min, z_max, dtype=np.float32),
        np.arange(y_min, y_max, dtype=np.float32),
        np.arange(x_min, x_max, dtype=np.float32),
        indexing="ij",
    )
    ellipsoid = ((zz - zc) / rz) ** 2 + ((yy - yc) / ry) ** 2 + ((xx - xc) / rx) ** 2
    local_lung = lungs[z_min:z_max, y_min:y_max, x_min:x_max]
    cavity = local_lung & (ellipsoid <= 1.0)
    wall = local_lung & (ellipsoid > 1.0) & (ellipsoid <= (1.0 + wall_thickness / max(radii)) ** 2)

    if cavity.any():
        volume[z_min:z_max, y_min:y_max, x_min:x_max][cavity] = rng.normal(-990.0, 10.0, size=int(cavity.sum()))
    if wall.any():
        wall_values = rng.normal(-610.0, 45.0, size=int(wall.sum()))
        local = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        local[wall] = np.maximum(local[wall], wall_values)

    return float((rz * ry * rx) ** (1.0 / 3.0))


def add_lam_pattern(
    volume: np.ndarray,
    lungs: np.ndarray,
    z: np.ndarray,
    edge_score: np.ndarray,
    rng: np.random.Generator,
) -> CystSummary:
    cyst_count = int(rng.integers(95, 165))
    radii: list[float] = []
    for _ in range(cyst_count):
        center = sample_lung_center(rng, lungs, z, edge_score, "lam")
        radius = float(rng.uniform(2.0, 6.2))
        jitter = rng.uniform(0.82, 1.18, size=3)
        radii.append(add_cyst(volume, lungs, center, tuple(radius * jitter), rng.uniform(0.55, 1.20), rng))
    return summarize_cysts(radii)


def add_bhd_pattern(
    volume: np.ndarray,
    lungs: np.ndarray,
    z: np.ndarray,
    edge_score: np.ndarray,
    rng: np.random.Generator,
) -> CystSummary:
    cyst_count = int(rng.integers(22, 56))
    radii: list[float] = []
    for _ in range(cyst_count):
        center = sample_lung_center(rng, lungs, z, edge_score, "bhd")
        base = float(rng.uniform(3.5, 11.5))
        elongation = rng.uniform(1.1, 2.8)
        flattening = rng.uniform(0.60, 1.05)
        axis_choice = rng.integers(0, 3)
        local_radii = np.array([base * flattening, base, base], dtype=np.float32)
        local_radii[axis_choice] *= elongation
        local_radii *= rng.uniform(0.85, 1.25, size=3)
        radii.append(add_cyst(volume, lungs, center, tuple(float(v) for v in local_radii), rng.uniform(0.45, 1.0), rng))
    return summarize_cysts(radii)


def summarize_cysts(radii: list[float]) -> CystSummary:
    if not radii:
        return CystSummary(count=0, mean_radius_vox=0.0, max_radius_vox=0.0)
    values = np.asarray(radii, dtype=np.float32)
    return CystSummary(count=len(radii), mean_radius_vox=float(values.mean()), max_radius_vox=float(values.max()))


def generate_sample(
    class_name: str,
    sample_index: int,
    shape: tuple[int, int, int],
    seed: int,
) -> tuple[np.ndarray, CystSummary]:
    rng = np.random.default_rng(seed)
    volume, lungs, z, _y, _x, edge_score = make_base_volume(rng, shape)

    if class_name == "LAM":
        summary = add_lam_pattern(volume, lungs, z, edge_score, rng)
    elif class_name == "BHD":
        summary = add_bhd_pattern(volume, lungs, z, edge_score, rng)
    elif class_name == "Control":
        summary = CystSummary(count=0, mean_radius_vox=0.0, max_radius_vox=0.0)
    else:
        raise ValueError(f"Unsupported class: {class_name}")

    rng = np.random.default_rng(seed + 10_000 + sample_index)
    volume += rng.normal(0.0, 12.0, size=shape).astype(np.float32)
    volume = np.clip(volume, -1024.0, 300.0)
    return volume.astype(np.int16), summary


def write_nrrd(path: Path, volume: np.ndarray, spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if volume.dtype != np.int16:
        raise TypeError("Synthetic CT volumes are expected to be int16")

    header = "\n".join(
        [
            "NRRD0004",
            "# Synthetic LAM/BHD/Control CT-like volume for software testing only.",
            "type: short",
            "dimension: 3",
            "space: left-posterior-superior",
            f"sizes: {volume.shape[0]} {volume.shape[1]} {volume.shape[2]}",
            "space directions: "
            f"({spacing[0]},0,0) (0,{spacing[1]},0) (0,0,{spacing[2]})",
            "kinds: domain domain domain",
            "endian: little",
            "encoding: gzip",
            "space origin: (0,0,0)",
            "",
            "",
        ]
    ).encode("ascii")

    with path.open("wb") as file:
        file.write(header)
        with gzip.GzipFile(fileobj=file, mode="wb", compresslevel=6) as gz:
            gz.write(np.ascontiguousarray(volume).tobytes(order="C"))


def write_dataset_readme(out_dir: Path, samples_per_class: int, shape: tuple[int, int, int], seed: int) -> None:
    readme = f"""# Synthetic LAM/BHD/Control NRRD Dataset

This dataset is algorithmically generated and is intended for pipeline smoke tests,
software integration, and model-development dry runs. It is not clinical data and
must not be used to claim diagnostic performance.

Classes:

- LAM: diffuse, numerous, mostly round thin-walled synthetic cysts.
- BHD: fewer, larger, basilar/subpleural, more elliptical synthetic cysts.
- Control: CT-like lung volumes without synthetic cysts.

Generation settings:

- samples_per_class: {samples_per_class}
- shape_zyx: {shape}
- seed: {seed}
- intensity: approximate CT HU in int16
- file_format: NRRD0004 with gzip-compressed inline data

Metadata for every volume is stored in `metadata.csv`.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def generate_dataset(
    out_dir: Path,
    samples_per_class: int,
    shape: tuple[int, int, int],
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.csv"
    spacing = (1.25, 1.0, 1.0)

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file",
                "class",
                "sample_index",
                "seed",
                "shape_zyx",
                "spacing_zyx_mm",
                "synthetic_cyst_count",
                "synthetic_mean_radius_vox",
                "synthetic_max_radius_vox",
            ],
        )
        writer.writeheader()
        for class_offset, class_name in enumerate(CLASS_NAMES):
            class_dir = out_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(1, samples_per_class + 1):
                sample_seed = seed + class_offset * 100_000 + index
                volume, cysts = generate_sample(class_name, index, shape, sample_seed)
                file_name = f"{class_name}_{index:03d}.nrrd"
                out_path = class_dir / file_name
                write_nrrd(out_path, volume, spacing)
                writer.writerow(
                    {
                        "file": str(out_path.relative_to(out_dir)),
                        "class": class_name,
                        "sample_index": index,
                        "seed": sample_seed,
                        "shape_zyx": "x".join(str(v) for v in shape),
                        "spacing_zyx_mm": "x".join(str(v) for v in spacing),
                        "synthetic_cyst_count": cysts.count,
                        "synthetic_mean_radius_vox": f"{cysts.mean_radius_vox:.3f}",
                        "synthetic_max_radius_vox": f"{cysts.max_radius_vox:.3f}",
                    }
                )

    write_dataset_readme(out_dir, samples_per_class, shape, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic LAM/BHD/Control CT-like NRRD volumes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--shape", type=parse_shape, default=parse_shape("96,128,128"))
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    if args.samples_per_class <= 0:
        raise SystemExit("--samples-per-class must be positive")

    generate_dataset(
        out_dir=args.out_dir.expanduser().resolve(),
        samples_per_class=args.samples_per_class,
        shape=args.shape,
        seed=args.seed,
    )
    print(f"Wrote synthetic dataset to {args.out_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
