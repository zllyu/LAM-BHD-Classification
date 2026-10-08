#!/usr/bin/env python
# coding: utf-8

# # 胸部 CT 肺脏分割（TotalSegmentator 批处理）
# 
# 本 notebook 会：
# 
# 1. 读取一张表格（CSV 或 Excel），其中 `filepath` 一栏是胸部 CT **nrrd** 文件的绝对路径；
# 2. 对每个病例用 [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) 做肺脏分割（默认合并 5 个肺叶为整肺二值掩膜）；
# 3. 把每个 mask 保存为 nrrd（与原图坐标对齐），并把 **原路径 + mask 绝对路径 + 处理状态** 写入一张结果表（CSV / Excel）。
# 
# > **注意**：TotalSegmentator 的输入只支持 NIfTI 文件或 DICOM 文件夹，不直接读 nrrd。本 notebook 内部用 SimpleITK 把 nrrd 转成临时 nifti 再分割，最后把 mask 重采样回原始网格保存。
# >
# > TotalSegmentator 不是医疗器械，仅供科研与技术用途，不用于临床诊断。

# ## 1. 环境准备
# 
# 首次运行请安装依赖；已装好可跳过本单元格。第一次调用分割时还会自动下载模型权重（需要联网）。

# In[1]:


# 首次运行取消注释安装
# %pip install TotalSegmentator SimpleITK nibabel numpy pandas openpyxl


# In[2]:


import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from totalsegmentator.python_api import totalsegmentator

print("依赖导入成功")


# ## 2. 配置参数
# 
# 只需要改这一格。

# In[3]:


# ------- 输入 -------
INPUT_TABLE = "/media/jk636/d/rhino/rhino_data/dummy_schema_filled.csv"   # 你的 csv 或 xlsx，含 filepath 列
FILEPATH_COLUMN = "absolute_path"                         # 存放 nrrd 绝对路径的列名

# ------- 输出 -------
OUTPUT_DIR = "/home/kaggieteam/rhino_data/stefan_lam_bhd_project/Totalsegmentator_mask/seperate_lobe"        # mask 保存目录（不存在会自动创建）
RESULTS_TABLE = "/home/kaggieteam/rhino_data/stefan_lam_bhd_project/Totalsegmentator_mask/mask_index_seperate_lobe.csv"  # 结果表；后缀 .csv -> CSV，.xlsx -> Excel
MASK_FORMAT = "nrrd"                                  # mask 输出格式："nrrd" 或 "nii.gz"

# ------- 分割选项 -------
DEVICE = "gpu"        # "gpu" / "cpu" / "mps"（无 GPU 时改成 "cpu"，建议同时把 FAST 设为 True）
FAST = False          # True: 3mm 快速模型，速度快、精度略低；纯 CPU 建议开启
KEEP_LOBES = True    # False: 输出整肺二值掩膜(0/1)；True: 保留 5 个肺叶的多标签

OVERWRITE = False     # False: 已存在的 mask 跳过，便于断点续跑


# In[4]:


# 五个肺叶的类别名（TotalSegmentator "total" 任务）
LUNG_LOBES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
Path(RESULTS_TABLE).parent.mkdir(parents=True, exist_ok=True)
assert MASK_FORMAT in ("nrrd", "nii.gz"), "MASK_FORMAT 只能是 'nrrd' 或 'nii.gz'"
print("输出目录:", OUTPUT_DIR)


# ## 3. 读取输入表格
# 
# 自动识别 CSV / Excel，并检查 `filepath` 列与文件是否存在。

# In[5]:


def load_table(path):
    ext = Path(path).suffix.lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"不支持的表格格式: {ext}")

df_in = load_table(INPUT_TABLE)
assert FILEPATH_COLUMN in df_in.columns, (
    f"表里没有 '{FILEPATH_COLUMN}' 列，实际列为: {list(df_in.columns)}"
)

paths = df_in[FILEPATH_COLUMN].astype(str).tolist()
missing = [p for p in paths if not os.path.exists(p)]
print(f"共 {len(paths)} 个病例；缺失文件 {len(missing)} 个")
if missing:
    print("以下路径不存在（将被跳过）:")
    for p in missing[:10]:
        print("  ", p)
    if len(missing) > 10:
        print(f"  ... 以及另外 {len(missing) - 10} 个")


# ## 4. 分割核心函数
# 
# 流程：读 nrrd → 写临时 nifti → TotalSegmentator（只跑 5 个肺叶）→ 合并/保留标签 → 重采样回原始网格 → 保存 mask。

# In[6]:


def segment_one(nrrd_path, mask_path, device=DEVICE, fast=FAST, keep_lobes=KEEP_LOBES):
    """对单个 nrrd 做肺脏分割，mask 保存到 mask_path。"""
    original = sitk.ReadImage(nrrd_path)  # 保留原始几何信息用于最终对齐

    with tempfile.TemporaryDirectory() as tmp:
        tmp_in = os.path.join(tmp, "input.nii.gz")
        tmp_out = os.path.join(tmp, "seg.nii.gz")
        sitk.WriteImage(original, tmp_in)

        # 只分割肺叶，ml=True 输出单个多标签文件
        totalsegmentator(
            input=tmp_in,
            output=tmp_out,
            task="total",
            roi_subset=LUNG_LOBES,
            ml=True,
            fast=fast,
            device=device,
            quiet=True,
        )

        seg = sitk.ReadImage(tmp_out)  # 使用 nifti header 中的真实几何
        if not keep_lobes:
            # 只分割了肺叶，>0 即为整肺
            seg = sitk.BinaryThreshold(seg, 1, 100000, 1, 0)

        # 重采样回原始 nrrd 网格，保证与原图完全对齐（最近邻）
        mask = sitk.Resample(
            seg, original, sitk.Transform(),
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
        sitk.WriteImage(mask, mask_path, useCompression=True)

    return mask_path


def make_mask_path(nrrd_path, out_dir, fmt, idx):
    stem = Path(nrrd_path).stem  # 去掉 .nrrd
    name = f"{stem}_lung_mask.{fmt}"
    candidate = os.path.join(out_dir, name)
    if os.path.exists(candidate):  # 防止不同目录下同名文件冲突
        candidate = os.path.join(out_dir, f"{stem}_{idx}_lung_mask.{fmt}")
    return candidate

print("函数已定义")


# ## 5. 批量处理
# 
# 逐个处理；出错的病例会被记录但不中断整体流程。

# In[7]:


records = []

for i, nrrd_path in enumerate(paths):
    row = {"filepath": nrrd_path, "mask_filepath": "", "status": "", "message": ""}

    if not os.path.exists(nrrd_path):
        row["status"] = "skipped"
        row["message"] = "input file not found"
        records.append(row)
        print(f"[{i+1}/{len(paths)}] 跳过（文件不存在）: {nrrd_path}")
        continue

    mask_path = make_mask_path(nrrd_path, OUTPUT_DIR, MASK_FORMAT, i)

    if os.path.exists(mask_path) and not OVERWRITE:
        row["mask_filepath"] = os.path.abspath(mask_path)
        row["status"] = "exists"
        row["message"] = "mask already exists, skipped"
        records.append(row)
        print(f"[{i+1}/{len(paths)}] 已存在，跳过: {mask_path}")
        continue

    try:
        print(f"[{i+1}/{len(paths)}] 分割中: {os.path.basename(nrrd_path)} ...")
        segment_one(nrrd_path, mask_path)
        row["mask_filepath"] = os.path.abspath(mask_path)
        row["status"] = "success"
        print(f"    -> 完成: {mask_path}")
    except Exception as e:
        row["status"] = "failed"
        row["message"] = repr(e)
        print(f"    !! 失败: {e}")

    records.append(row)

print("\n全部处理结束。")


# ## 6. 保存结果表并汇总

# In[8]:


df_out = pd.DataFrame(records, columns=["filepath", "mask_filepath", "status", "message"])

ext = Path(RESULTS_TABLE).suffix.lower()
if ext in (".xlsx", ".xls"):
    df_out.to_excel(RESULTS_TABLE, index=False)
else:
    df_out.to_csv(RESULTS_TABLE, index=False)

print("结果表已保存:", os.path.abspath(RESULTS_TABLE))
print("\n状态统计:")
print(df_out["status"].value_counts().to_string())
df_out.head(10)


# ## 7.（可选）快速检查一个结果
# 
# 随便看一张成功的 mask，确认体素数不为 0、几何与原图一致。

# In[9]:


ok = df_out[df_out["status"] == "success"]
if len(ok) == 0:
    print("没有成功的病例可供检查。")
else:
    r = ok.iloc[0]
    ct = sitk.ReadImage(r["filepath"])
    mk = sitk.ReadImage(r["mask_filepath"])
    arr = sitk.GetArrayFromImage(mk)
    print("原图:", r["filepath"])
    print("Mask:", r["mask_filepath"])
    print("尺寸一致:", ct.GetSize() == mk.GetSize())
    print("Spacing 一致:", np.allclose(ct.GetSpacing(), mk.GetSpacing()))
    print("非零体素数:", int((arr > 0).sum()), "/", arr.size)
    print("标签取值:", np.unique(arr).tolist())

