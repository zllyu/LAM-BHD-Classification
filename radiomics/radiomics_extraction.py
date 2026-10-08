#!/usr/bin/env python
# coding: utf-8

# # 影像组学特征批量提取（PyRadiomics）
# 
# 读取 Excel（`filepath` = CT 影像的 nrrd，`mask_final` = 对应 mask 的 nrrd），逐行提取影像组学特征，汇总成一张表并保存为 CSV。
# 
# ## ⚠️ 先看安装（很重要）
# 
# PyRadiomics 官方包在**较新的 Python 上装不上或 import 会报错**（Python 3.10+ 装不上，3.9 能装但导入失败）。最稳的做法是**专门建一个 Python 3.9 的 conda 环境**，并把 numpy 降一档：
# 
# ```bash
# conda create -n radiomics python=3.9 -y
# conda activate radiomics
# pip install pyradiomics
# pip install "numpy<1.24"          # 避免用到已弃用的 numpy 接口导致 import 报错
# pip install jupyter pandas openpyxl SimpleITK
# ```
# 
# 然后**从这个环境里启动 Jupyter**（`jupyter notebook`）再打开本文件。
# 
# 如果你必须用更新的 Python（3.10–3.13），可以考虑社区维护的分支（如 `pyradiomics-cuda`，支持到 3.13），但 API 略有差异，建议优先用上面的 3.9 环境——它是参考实现，结果最可复现。

# In[1]:


import os
import logging
import pandas as pd
from radiomics import featureextractor

# ==== 配置 ====
EXCEL_PATH  = "/media/jk636/d/rhino/rhino_data/stefan_lam_bhd_project/file_path_summary.xlsx"          # 你的 Excel 路径
IMAGE_COL   = "filepath"                 # CT 影像列
MASK_COL    = "mask_final"               # mask 列
LABEL       = 1                          # mask 里 ROI 的标签值（前景通常是 1）
OUTPUT_CSV  = "radiomics_features.csv"   # 输出特征表

# 关掉 pyradiomics 的冗长日志（想看细节可改成 logging.INFO）
logging.getLogger("radiomics").setLevel(logging.ERROR)


# ## 配置提取器
# 
# `binWidth`、重采样、要不要小波/LoG 滤波等都在这里调。默认只用 Original 图像 + 全部特征类，先跑通再按需加。

# In[2]:


settings = {
    "binWidth": 25,                 # 灰度离散化的 bin 宽度，CT 常用 25
    "resampledPixelSpacing": None,  # 如需各向同性重采样，改成 [1, 1, 1]
    "interpolator": "sitkBSpline",
    "geometryTolerance": 1e-4,      # 放宽一点，容忍 CT 与 mask 的微小几何差异
    "correctMask": True,            # mask 与影像几何轻微不一致时自动修正对齐
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableAllFeatures()               # 启用所有特征类（firstorder / shape / glcm ...）
extractor.enableImageTypeByName("Original")  # 只用原始图像

# 想要更多特征可解开下面两行（特征数和耗时都会明显增加）：
# extractor.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0]})
# extractor.enableImageTypeByName("Wavelet")

print("启用的图像类型：", extractor.enabledImagetypes)


# ## 读取 Excel 并核对路径

# In[3]:


df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.str.strip()

for c in (IMAGE_COL, MASK_COL):
    if c not in df.columns:
        raise ValueError(f"Excel 里找不到列 {c!r}，实际列名：{list(df.columns)}")

# 提取前先确认文件都在（缺文件的会在提取时报错，这里提前提示一下）
missing = []
for idx, row in df.iterrows():
    for c in (IMAGE_COL, MASK_COL):
        p = str(row[c]).strip()
        if not os.path.isfile(p):
            missing.append((idx, c, p))
if missing:
    print(f"⚠️ 有 {len(missing)} 个文件路径不存在，这些行会提取失败：")
    for idx, c, p in missing[:20]:
        print(f"   行{idx} [{c}] {p}")
else:
    print(f"文件都在，共 {len(df)} 对待提取")


# ## 逐行提取
# 
# 用 `try/except` 包住每一行，某一对出错不会中断整体；失败的会单独记录下来。

# In[ ]:


records = []
errors  = []

for idx, row in df.iterrows():
    image_path = str(row[IMAGE_COL]).strip()
    mask_path  = str(row[MASK_COL]).strip()
    try:
        result = extractor.execute(image_path, mask_path, label=LABEL)
        rec = {IMAGE_COL: image_path, MASK_COL: mask_path}
        for k, v in result.items():
            if k.startswith("diagnostics_"):   # 跳过诊断信息，只保留特征
                continue
            rec[k] = float(v)
        records.append(rec)
        print(f"[{idx+1}/{len(df)}] OK   {os.path.basename(image_path)}")
    except Exception as e:
        errors.append({"index": idx, IMAGE_COL: image_path,
                       MASK_COL: mask_path, "error": str(e)})
        print(f"[{idx+1}/{len(df)}] 失败 {os.path.basename(image_path)}  ->  {e}")

print(f"\n完成：成功 {len(records)}，失败 {len(errors)}")


# ## 保存结果

# In[ ]:


features_df = pd.DataFrame(records)
features_df.to_csv(OUTPUT_CSV, index=False)

n_feat = features_df.shape[1] - 2 if len(features_df) else 0
print(f"已保存 {OUTPUT_CSV}：{len(features_df)} 行，{n_feat} 个特征")
features_df.head()


# In[ ]:


# 失败清单（如果有）
if errors:
    err_df = pd.DataFrame(errors)
    err_df.to_csv("radiomics_errors.csv", index=False)
    print(f"有 {len(errors)} 个样本失败，已存到 radiomics_errors.csv")
    display(err_df)
else:
    print("没有失败的样本 ✅")


# In[ ]:




