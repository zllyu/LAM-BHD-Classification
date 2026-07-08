# LAM/BHD/Control 3D Classification

This project trains a 3-class 3D medical volume classifier and supports:

- Local single-site training.
- Local NVFlare federated learning simulation.
- Rhino Health / Rhino FCP NVFlare deployment.

Target classes:

```text
0 = LAM
1 = BHD
2 = Control  # non-LAM and non-BHD
```

## Environment

Run commands from the repository root. To create the conda environment from the checked-in config:

```bash
cd <repo-root>
conda env create -f environment.yml
conda activate lam-bhd-nvflare
```

Validated dependency baseline:

```text
Python 3.12.10
NVFlare 2.7.0
PyTorch 2.9.1 + CUDA 12.6 wheels
numpy 2.2.6
pynrrd 1.1.3
scikit-learn 1.9.0
tensorboard 2.21.0
```

Rhino target: NVFlare v2.7.

## Data Type And Format

Input data type: 3D medical volume files.

Supported file formats:

```text
.nrrd
.nhdr
.npy
```

Recommended local dataset layout:

```text
data/LAM_BHD_synthetic_nrrd_dataset/
  labels.csv
  LAM/*.nrrd
  BHD/*.nrrd
  Control/*.nrrd
```

`labels.csv` is preferred when present. Required columns:

```text
filepath,GT
```

Current CSV format:

```text
StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID,filepath,GT
```

Class order is fixed in `app/custom/data.py`:

```python
DEFAULT_CLASS_ORDER = ("LAM", "BHD", "Control")
```

Data path resolution order:

1. CLI `--data-dir`, unless `auto`.
2. Environment variable `LAM_BHD_DATA_DIR`, unless `auto`.
3. Rhino `/input/datasets/<dataset_uid>/file_data`.
4. Rhino `/input/datasets/<dataset_uid>`.
5. `data/LAM_BHD_synthetic_nrrd_dataset` relative to the current working directory.

If your dataset lives elsewhere, pass it explicitly:

```bash
python LAM_BHD_classification.py --data-dir /path/to/LAM_BHD_synthetic_nrrd_dataset
```

or set:

```bash
export LAM_BHD_DATA_DIR=/path/to/LAM_BHD_synthetic_nrrd_dataset
```

Preprocessing:

- Load volume as `float32`.
- Clip/normalize intensity by 0.5/99.5 percentiles.
- Add channel dimension: `(1, D, H, W)`.
- Resize to `--target-shape`, default `64,64,64`; use `none` for native shape.
- Random 3D flips during training.

Main data code:

```text
app/custom/data.py
```

## Model

Model file:

```text
app/custom/model.py
```

Model type: small 3D CNN.

Architecture:

```text
Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d
Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d
Conv3d -> BatchNorm3d -> ReLU
AdaptiveAvgPool3d(1)
Linear(num_classes=3)
```

Training:

```text
Loss: CrossEntropyLoss with class weights
Optimizer: AdamW
Metrics: loss, accuracy, multiclass one-vs-rest AUC
```

Training utilities:

```text
app/custom/training.py
```

## Local Single-Site Training

Entry point:

```text
LAM_BHD_classification.py
```

Implementation:

```text
app/custom/train_single_site.py
```

Default run, assuming data is under `data/LAM_BHD_synthetic_nrrd_dataset`:

```bash
python LAM_BHD_classification.py
```

Typical run:

```bash
python LAM_BHD_classification.py \
  --data-dir data/LAM_BHD_synthetic_nrrd_dataset \
  --output-dir outputs/single_site \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --target-shape 64,64,64 \
  --device auto
```

CPU smoke test:

```bash
python LAM_BHD_classification.py \
  --data-dir data/LAM_BHD_synthetic_nrrd_dataset \
  --output-dir outputs/smoke_single \
  --epochs 1 \
  --batch-size 8 \
  --target-shape 8,8,8 \
  --device cpu
```

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | `auto` | Dataset root or automatic resolution. |
| `--output-dir` | `outputs/single_site` | Checkpoint output directory. |
| `--epochs` | `3` | Number of epochs. |
| `--batch-size` | `2` | Batch size. |
| `--num-workers` | `0` | DataLoader workers. |
| `--lr` | `1e-4` | Learning rate. |
| `--seed` | `42` | Split/training seed. |
| `--target-shape` | `64,64,64` | Resize shape; `none` means native shape. |
| `--device` | `auto` | `auto`, `cpu`, `cuda:0`, etc. |

Inputs:

```text
Dataset root with labels.csv and volume files.
```

Outputs:

```text
<output-dir>/best_model.pt
<output-dir>/final_model.pt
```

Checkpoint contents:

```text
model_state_dict
metrics
class_names
epoch / target_shape metadata
```

Modify local outputs in:

```text
app/custom/train_single_site.py
app/custom/training.py::save_checkpoint
```

## Local FL Simulation

Entry point:

```text
LAM_BHD_job.py
```

Implementation:

```text
job.py
app/custom/fl_client.py
```

Default run, assuming data is under `data/LAM_BHD_synthetic_nrrd_dataset`:

```bash
python LAM_BHD_job.py
```

Typical run:

```bash
python LAM_BHD_job.py \
  --n-clients 2 \
  --num-rounds 5 \
  --epochs 2 \
  --batch-size 4 \
  --lr 1e-4 \
  --target-shape 64,64,64 \
  --data-dir data/LAM_BHD_synthetic_nrrd_dataset \
  --num-classes 3 \
  --device auto
```

CPU smoke test:

```bash
python LAM_BHD_job.py \
  --n-clients 1 \
  --num-rounds 1 \
  --epochs 1 \
  --batch-size 16 \
  --target-shape 8,8,8 \
  --data-dir data/LAM_BHD_synthetic_nrrd_dataset \
  --device cpu
```

FL parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--job-name` | `LAM_BHD_fedavg` | NVFlare job/simulation folder name. |
| `--n-clients` | `1` | Number of simulated clients. |
| `--num-rounds` | `2` | Federated aggregation rounds. |
| `--epochs` | `1` | Local epochs per client per FL round. |
| `--batch-size` | `2` | Client batch size. |
| `--num-workers` | `0` | DataLoader workers per client. |
| `--lr` | `1e-4` | Client learning rate. |
| `--seed` | `42` | Split/training seed. |
| `--target-shape` | `64,64,64` | Resize shape. |
| `--data-dir` | `auto` | Dataset root or automatic resolution. |
| `--num-classes` | `3` | Model output classes. |
| `--device` | `auto` | Client training device. |
| `--num-threads` | `0` | Simulator threads; `0` means number of clients. |
| `--gpu-config` | empty | NVFlare simulator GPU config. |
| `--partition-sites` | enabled | Stratified split of one dataset across clients when `--n-clients > 1`; no-op for one client. |
| `--no-partition-sites` | disabled | Every client uses the full dataset. |
| `--export` | disabled | Export NVFlare job instead of running. |
| `--export-dir` | `/tmp/nvflare_jobs/lam_bhd` | Export destination. |

Inputs:

```text
Dataset root from --data-dir, LAM_BHD_DATA_DIR, Rhino input mount, or local data/ fallback.
Initial model Net(num_classes=3).
```

Outputs:

```text
/tmp/nvflare/simulation/<job-name>/
```

The run prints:

```text
Result path: /tmp/nvflare/simulation/LAM_BHD_fedavg
```

Useful output files:

```text
server/log.txt
server/log_fl.txt
site-1/log.txt
site-2/log.txt
server/simulate_job/tb_events/
```

TensorBoard:

```bash
tensorboard --logdir=/tmp/nvflare/simulation/LAM_BHD_fedavg/server/simulate_job/tb_events
```

Export without running:

```bash
python LAM_BHD_job.py --export --export-dir /tmp/nvflare_jobs/lam_bhd
```

## Rhino Deployment

Rhino code object payload:

```text
app/
meta.json
Dockerfile
requirements.txt
```

Build/push image:

```bash
AWS_PROFILE=rhino ./docker-push.sh <workgroup-ecr-repository> <image-tag>
```

`<workgroup-ecr-repository>` is the Workgroup ECR repository shown in Rhino Settings / Containers & Artifacts. For `dashboard.rhinohealth.com`, the script defaults to Rhino's documented AWS production registry:

```text
865551847959.dkr.ecr.us-east-1.amazonaws.com
```

Example:

```bash
AWS_PROFILE=rhino ./docker-push.sh <workgroup-ecr-repository> v1.0
```

If Rhino shows a different registry for your workgroup, pass it explicitly:

```bash
AWS_PROFILE=rhino ./docker-push.sh \
  --image-registry <registry-host> \
  <workgroup-ecr-repository> \
  v1.0
```

Use the printed container image URI in Rhino.

Rhino SDK code object settings:

```text
code_type = CodeTypes.NVIDIA_FLARE_V2_7
config = {"container_image_uri": "<URI from docker-push.sh>"}
```

If the SDK does not expose `NVIDIA_FLARE_V2_7`, use the Rhino UI NVFlare v2.7 option or upgrade the Rhino SDK.

Rhino client config:

```text
app/config/config_fed_client.json
```

Current client command inside the Rhino container:

```text
/workspace/app/custom/fl_client.py --data-dir auto --epochs 1 --batch-size 2 --num-workers 0 --target-shape 64,64,64 --num-classes 3
```

Rhino server config:

```text
app/config/config_fed_server.json
```

Current server settings:

```text
num_clients = 1
num_rounds = 2
global_model_file_name = /output/model_parameters.pt
model = model.Net(num_classes=3)
```

Rhino metadata:

```text
meta.json
```

Current metadata:

```text
min_clients = 1
deploy_map = app -> @ALL
```

The checked-in Rhino config is valid for one client and is intended to be the default image config. Leave `meta.json` `min_clients = 1` so the same code object can run single-client jobs.

To run N clients from the same image, do not rebuild just to change the client count. On the Rhino Run Model Training page:

1. Select the N participating site datasets under Training Datasets.
2. Copy the full contents of `app/config/config_fed_server.json` into Federated Server Config Override.
3. Change only `workflows[0].args.num_clients` to N.

The selected training clients/datasets must match `num_clients`. If fewer clients are selected, FedAvg can wait until timeout. If more clients are selected, the server only requires `num_clients` client results for aggregation.

For SDK runs, set `ModelTrainInput.config_fed_server` to the same full JSON string. Rhino documents this override in the NVFlare run guide: https://docs.rhinohealth.com/hc/en-us/articles/12522228144669-Running-NVFlare-Code

Rhino inputs:

```text
One Rhino dataset per site.
Each dataset should expose labels.csv and volume files under:
/input/datasets/<dataset_uid>/
or
/input/datasets/<dataset_uid>/file_data/
```

Rhino outputs:

```text
/output/model_parameters.pt
TensorBoard analytics events when enabled by Rhino/NVFlare runtime.
```

## Where To Modify Things

| Change | File |
| --- | --- |
| Local dataset default | `app/custom/data.py` |
| Class order / label mapping | `app/custom/data.py` |
| CSV column assumptions | `app/custom/data.py` |
| Model architecture | `app/custom/model.py` |
| Local single-site parameters | `app/custom/train_single_site.py` |
| Local single-site outputs | `app/custom/train_single_site.py`, `app/custom/training.py` |
| Local FL simulation parameters | `job.py` |
| FL client local training behavior | `app/custom/fl_client.py` |
| Rhino client training args | `app/config/config_fed_client.json` |
| Rhino rounds/client count/model output path | `app/config/config_fed_server.json` |
| Rhino min clients/deploy map | `meta.json` |
| Conda environment | `environment.yml` |
| Python dependencies | `requirements.txt` |
| Container image build | `Dockerfile` |
| Docker build exclusions | `.dockerignore` |
| Push target registry/image tag | `docker-push.sh` command arguments |

## Quick Checks

Compile check:

```bash
python -m py_compile LAM_BHD_classification.py LAM_BHD_job.py LAM_BHD_fedavg.py job.py app/custom/*.py
```

Check data discovery:

```bash
python - <<'CHECK_DATA'
import sys
sys.path.insert(0, 'app/custom')
from data import build_dataloaders
_, _, _, summary = build_dataloaders(data_dir='data/LAM_BHD_synthetic_nrrd_dataset', target_shape=(8, 8, 8), batch_size=4)
print(summary)
CHECK_DATA
```

Expected class names:

```text
['LAM', 'BHD', 'Control']
```
