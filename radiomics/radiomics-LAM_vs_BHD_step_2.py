#!/usr/bin/env python
# coding: utf-8

# # 影像组学二分类建模：BHD vs LAM
# 
# ## 研究问题
# 
# 本 notebook 专门用于 **BHD 与 LAM 的鉴别诊断**。与之前的
# **Control vs (BHD + LAM)** 疾病检测任务不同，这里会：
# 
# - **剔除 Control**，仅保留 BHD 与 LAM；
# - 固定编码为 **BHD = 0（阴性类）**、**LAM = 1（阳性类）**；
# - 所有特征选择、模型筛选、超参数搜索和阈值选择都只在 BHD/LAM 训练数据中完成；
# - 独立测试集只用于最终评估，不参与阈值或模型选择。
# 
# ## 本 notebook 的内容
# 
# 数据与标签检查 → （可选）三组两两可分性背景诊断 → **仅保留 BHD/LAM** →
# 分层训练/测试划分 → Pipeline 化预处理与特征选择 → 20+ 算法广筛 →
# 超参搜索与集成 → **训练集 OOF 概率确定阈值** → 独立测试集验证
# （ROC/PR/校准/决策曲线、DeLong 与 bootstrap 置信区间）→
# 置换检验、学习曲线、特征重要性与单变量分析。
# 

# ## 1. 环境与配置

# In[1]:


import warnings, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sklearn
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_validate,
                                     cross_val_predict, cross_val_score,
                                     RandomizedSearchCV, learning_curve,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve,
                             log_loss, brier_score_loss, confusion_matrix,
                             classification_report, make_scorer)
from scipy import stats
import joblib

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
print("scikit-learn", sklearn.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)


# In[2]:


CFG = dict(
    data_file    = "./radiomics_features_upper_middle.csv",
    id_col       = "ID",
    gt_col       = "GT",
    exclude_cols = ["filepath", "mask_final"],
    gt_levels    = ["control", "bhd", "lam"],
    gt_display   = ["Control", "BHD", "LAM"],

    seed         = 20260723,
    test_size    = 0.30,
    fs_method    = "lasso",
    fs_C         = 0.10,
    fs_min_features = 5,             # L1 若把特征选空，至少回退保留这么多个
    imbalance    = "class_weight",   # "class_weight" | "smote" | "none"

    cv_folds       = 5,
    screen_repeats = 3,
    tune_repeats   = 10,
    top_k          = 5,
    n_iter         = 40,
    tune_metric    = "roc_auc",      # 二分类主指标

    exclude_models = [],
    n_boot         = 1000,
    n_perm         = 200,            # 置换检验次数（0 = 跳过）
    do_learning_curve = True,
    do_nested_cv   = False,
    n_jobs         = -1,
    out_dir        = "results_LAM_vs_BHD",
)

# ---------------- 本 notebook 固定任务：BHD vs LAM ----------------
# 显式指定正负类，避免依赖 gt_levels / gt_display 的排列顺序。
BINARY = dict(
    keep_classes = ["BHD", "LAM"],
    negative     = "BHD",            # 编码为 0
    positive     = "LAM",            # 编码为 1；predict_proba[:, 1] = P(LAM)
)

OUT  = Path(CFG["out_dir"]); OUT.mkdir(parents=True, exist_ok=True)
SEED = CFG["seed"]
np.random.seed(SEED)
LV3  = CFG["gt_display"]
PAL3 = {"Control": "#8C8C8C", "BHD": "#4E79A7", "LAM": "#E15759"}
CFG


# ## 2. 载入数据

# In[3]:


def load_data(cfg):
    f = Path(cfg["data_file"])
    if f.exists():
        df = pd.read_csv(f)
        print(f"已载入 {f.resolve()}  ->  {df.shape[0]} 行 x {df.shape[1]} 列")
    else:
        print(f"⚠ 未找到 {f} —— 生成【模拟数据】演示。"
              "注意：模拟数据信号是人为设定的，任何指标都不代表你的真实数据。")
        rng = np.random.default_rng(cfg["seed"])
        n, p = [80, 45, 25], 102
        X = rng.normal(size=(sum(n), p))
        # 刻意模拟"Control 好分、BHD 与 LAM 难分"的真实情形
        X[:, :12] += np.repeat([0.0, 1.0, 1.0], n)[:, None]   # 病 vs 非病：信号强
        X[:, 12:18] += np.repeat([0.0, 0.0, 0.45], n)[:, None]  # BHD vs LAM：信号弱
        df = pd.DataFrame(X, columns=[f"feature_{i:03d}" for i in range(1, p + 1)])
        df.insert(0, cfg["id_col"], [f"case_{i:03d}" for i in range(1, sum(n) + 1)])
        df.insert(1, cfg["gt_col"], np.repeat(cfg["gt_levels"], n))
        df.insert(2, "filepath", "dummy/path.nii.gz")
        df.insert(3, "mask_final", 1)
    return df


df = load_data(CFG)

lab_map = {a.lower(): b for a, b in zip(CFG["gt_levels"], CFG["gt_display"])}
y_raw   = df[CFG["gt_col"]].astype(str).str.strip().str.lower()
unknown = sorted(set(y_raw) - set(lab_map))
if unknown:
    raise ValueError(f"GT 列中有未登记的取值 {unknown}")
lab3 = y_raw.map(lab_map).to_numpy()
y3   = pd.Series(lab3).map({c: i for i, c in enumerate(LV3)}).to_numpy()

drop_lower = {c.lower() for c in CFG["exclude_cols"]} | {CFG["id_col"].lower(), CFG["gt_col"].lower()}
X_all = df.drop(columns=[c for c in df.columns if c.lower() in drop_lower])
X_all = X_all.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
print(f"\n特征 {X_all.shape[1]} 个 | 三分类分布：",
      {c: int((y3 == i).sum()) for i, c in enumerate(LV3)})


# ## 3. 背景诊断（可选）：三类两两可分性
# 
# 这里仍保留 Control vs BHD、Control vs LAM、BHD vs LAM 的重复分层 CV，
# **仅作为背景 sanity check，不再用于决定是否合并类别**。
# 
# 本 notebook 后续的正式建模任务固定为 **BHD vs LAM**；从第 4 节开始，
# Control 会被完全剔除。
# 

# In[4]:


# LassoSelector 需要能被 joblib 正常序列化，因此写入独立模块再导入
# （在 notebook 的 __main__ 中直接定义的类无法被 pickle 保存）
import sys, importlib
from sklearn.linear_model import LogisticRegression

_UTILS = '''
# 影像组学建模辅助模块（由 notebook 自动生成）
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif


class LassoSelector(BaseEstimator, TransformerMixin):
    # L1 逻辑回归特征选择，带空集保护。
    # 当两类差异很小时，L1 惩罚可能把所有系数压到 0，选出空特征集，
    # 后续分类器会直接失败（表现为 CV 分数全是 nan）。此时回退到
    # 单变量 F 检验补足到 min_features 个，保证流水线始终可用。

    def __init__(self, C=0.1, min_features=5, class_weight="balanced", random_state=None):
        self.C = C
        self.min_features = min_features
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        solver = "liblinear" if len(np.unique(y)) == 2 else "saga"
        est = LogisticRegression(penalty="l1", solver=solver, C=self.C, max_iter=5000,
                                 class_weight=self.class_weight,
                                 random_state=self.random_state)
        est.fit(X, y)
        imp = np.abs(np.atleast_2d(est.coef_)).sum(axis=0)
        support = imp > 1e-8
        self.lasso_n_ = int(support.sum())
        need = min(self.min_features, X.shape[1])
        if support.sum() < need:
            F = np.nan_to_num(f_classif(X, y)[0], nan=0.0)
            for j in np.argsort(-F):
                if support.sum() >= need:
                    break
                support[j] = True
        self.support_ = support
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self):
        return self.support_
'''

Path("radiomics_utils.py").write_text(_UTILS, encoding="utf-8")
if str(Path.cwd()) not in sys.path:          # 确保当前目录可被导入
    sys.path.insert(0, str(Path.cwd()))
if "radiomics_utils" in sys.modules:
    importlib.reload(sys.modules["radiomics_utils"])
from radiomics_utils import LassoSelector
print("已生成 radiomics_utils.py —— 模型可正常保存与加载")


def base_pipe(clf=None, seed=SEED, cfg=CFG):
    '''统一的预处理 + 特征选择 + 分类器流水线（防泄露的核心）。'''
    sel = "passthrough"
    if cfg["fs_method"] == "lasso":
        sel = LassoSelector(C=cfg["fs_C"], min_features=cfg["fs_min_features"],
                            random_state=seed)
    if clf is None:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    steps = [("impute", SimpleImputer(strategy="median")),
             ("var", VarianceThreshold(1e-12)),
             ("scale", StandardScaler()),
             ("select", sel),
             ("clf", clf)]
    if cfg["imbalance"] == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            steps.insert(3, ("smote", SMOTE(random_state=seed, k_neighbors=3)))
            return ImbPipeline(steps)
        except ImportError:
            print("未安装 imbalanced-learn，回退到 class_weight")
    return Pipeline(steps)


# In[5]:


cv_diag = RepeatedStratifiedKFold(n_splits=CFG["cv_folds"], n_repeats=5, random_state=SEED)
pairs, rows = [(0, 1), (0, 2), (1, 2)], []
for a, b in pairs:
    m = np.isin(y3, [a, b])
    Xp, yp = X_all[m], (y3[m] == b).astype(int)
    auc = cross_val_score(base_pipe(), Xp, yp, cv=cv_diag, scoring="roc_auc",
                          n_jobs=CFG["n_jobs"])
    bac = cross_val_score(base_pipe(), Xp, yp, cv=cv_diag, scoring="balanced_accuracy",
                          n_jobs=CFG["n_jobs"])
    rows.append({"Pair": f"{LV3[a]} vs {LV3[b]}", "n": int(m.sum()),
                 "AUC": auc.mean(), "AUC_sd": auc.std(),
                 "BalAcc": bac.mean(), "BalAcc_sd": bac.std()})
    print(f"  {rows[-1]['Pair']:<22s} n={m.sum():3d}  "
          f"AUC={auc.mean():.3f}±{auc.std():.3f}  BalAcc={bac.mean():.3f}")

pair_tab = pd.DataFrame(rows).round(4)
pair_tab.to_csv(OUT / "diagnosis_pairwise_separability.csv", index=False)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
cols = ["#59A14F" if v >= .8 else "#F28E2B" if v >= .7 else "#E15759" for v in pair_tab.AUC]
ax.barh(pair_tab.Pair, pair_tab.AUC, xerr=pair_tab.AUC_sd, color=cols, alpha=.9,
        error_kw=dict(lw=1, ecolor="#444"))
ax.axvline(.5, ls="--", c="grey", lw=1); ax.set_xlim(.4, 1.0)
ax.set_xlabel("Cross-validated AUC (L2 logistic regression)")
ax.set_title("Pairwise separability — where does the 3-class model fail?", fontsize=10.5)
plt.tight_layout(); plt.savefig(OUT / "fig0_pairwise_separability.png", dpi=200, bbox_inches="tight")
plt.show()

target_pair = f"{BINARY['negative']} vs {BINARY['positive']}"
target_row = pair_tab.loc[pair_tab["Pair"] == target_pair]
if len(target_row):
    r = target_row.iloc[0]
    print(f"\n目标任务 {target_pair}: AUC={r['AUC']:.3f}±{r['AUC_sd']:.3f}, "
          f"BalAcc={r['BalAcc']:.3f}±{r['BalAcc_sd']:.3f}")
else:
    print(f"\n⚠ 未在两两诊断表中找到目标组合：{target_pair}")

print("→ 上述三组比较只作背景参考；下面正式模型只使用 BHD 与 LAM 样本。")


# ## 4. 构建 BHD vs LAM 二分类标签
# 
# 本节执行本 notebook 最关键的任务转换：
# 
# - **Control 全部剔除**；
# - 仅保留 **BHD** 与 **LAM**；
# - 固定 **BHD = 0**、**LAM = 1**；
# - 因而后续 `predict_proba(... )[:, 1]` 始终表示 **P(LAM)**。
# 
# 显式固定正负类比依赖类别出现顺序更安全，尤其是在更换数据文件或调整
# `gt_levels` 顺序时。
# 

# In[6]:


keep_classes = list(BINARY["keep_classes"])
NEG = BINARY["negative"]
POS = BINARY["positive"]

if NEG == POS:
    raise ValueError("BINARY['negative'] 与 BINARY['positive'] 不能相同")
if set(keep_classes) != {NEG, POS}:
    raise ValueError("BINARY['keep_classes'] 必须恰好包含 negative 与 positive 两类")

# 只保留 BHD / LAM，Control（以及任何不在 keep_classes 中的类别）不进入正式建模
keep = np.isin(lab3, keep_classes)
lab_bin = lab3[keep]
X_bin   = X_all.loc[keep].reset_index(drop=True)
id_bin  = df.loc[keep, CFG["id_col"]].to_numpy()
lab3_kept = lab3[keep]

present = set(pd.unique(lab_bin))
missing = set(keep_classes) - present
if missing:
    raise ValueError(f"数据中缺少目标类别：{sorted(missing)}")

y_bin = (lab_bin == POS).astype(int)
CLS   = [NEG, POS]
PAL   = [PAL3[NEG], PAL3[POS]]

excluded = pd.Series(lab3[~keep]).value_counts().to_dict()
print(f"已剔除非目标类别：{excluded if excluded else '无'}")
print(f"二分类任务：{NEG} (0)  vs  {POS} (1)")
print(f"样本量：{len(y_bin)}  |  {NEG}={int((y_bin==0).sum())}, "
      f"{POS}={int((y_bin==1).sum())}  |  LAM阳性率={y_bin.mean():.3f}")
print("\n正式建模样本构成：")
print(pd.Series(lab_bin, name="Label").value_counts().reindex(CLS))


# ## 5. 分层随机划分

# In[7]:


idx = np.arange(len(y_bin))
idx_tr, idx_te, y_tr, y_te = train_test_split(
    idx, y_bin, test_size=CFG["test_size"], stratify=y_bin, random_state=SEED)
X_tr = X_bin.iloc[idx_tr].reset_index(drop=True)
X_te = X_bin.iloc[idx_te].reset_index(drop=True)

split_tab = pd.DataFrame({
    "Train": [int((y_tr == 0).sum()), int((y_tr == 1).sum())],
    "Test":  [int((y_te == 0).sum()), int((y_te == 1).sum())]}, index=CLS)
print(f"训练 {len(y_tr)} 例 / 测试 {len(y_te)} 例\n"); print(split_tab)
print(f"\n阳性率  训练 {y_tr.mean():.3f} | 测试 {y_te.mean():.3f}")
if min(split_tab.Test) < 15:
    print("\n⚠ 测试集某一类 < 15 例：AUC 的置信区间会很宽，"
          "请以重复交叉验证为主要证据，并务必报告 95% CI。")

pd.DataFrame({"ID": id_bin, "OriginalLabel": lab3_kept, "BinaryLabel": lab_bin,
              "Set": np.where(np.isin(idx, idx_tr), "train", "test")}
             ).to_csv(OUT / "data_split.csv", index=False)


# ## 6. 模型库
# 
# 与三分类版本相同的 20+ 种算法。所有模型都被同一条 Pipeline 包裹，
# 预处理与特征选择在每一折内独立拟合。

# In[8]:


from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              AdaBoostClassifier, BaggingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

CW = "balanced" if CFG["imbalance"] == "class_weight" else None


def cal(est):
    """为没有 predict_proba 的分类器补上概率校准。"""
    return CalibratedClassifierCV(est, cv=3, method="sigmoid")


MODELS = {
    "LogReg-L2":        LogisticRegression(C=1.0, max_iter=5000, class_weight=CW, random_state=SEED),
    "LogReg-L1":        LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                           max_iter=5000, class_weight=CW, random_state=SEED),
    "LogReg-Elastic":   LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=.5,
                                           C=1.0, max_iter=5000, class_weight=CW, random_state=SEED),
    "RidgeClassifier":  cal(RidgeClassifier(class_weight=CW, random_state=SEED)),
    "SGD-LogLoss":      SGDClassifier(loss="log_loss", max_iter=5000, class_weight=CW, random_state=SEED),
    "LDA-shrinkage":    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "QDA":              QuadraticDiscriminantAnalysis(reg_param=.1),
    "GaussianNB":       GaussianNB(),
    "KNN":              KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "SVC-Linear":       SVC(kernel="linear", probability=True, class_weight=CW, random_state=SEED),
    "SVC-RBF":          SVC(kernel="rbf", probability=True, class_weight=CW, random_state=SEED),
    "SVC-Poly":         SVC(kernel="poly", degree=3, probability=True, class_weight=CW, random_state=SEED),
    "LinearSVC":        cal(LinearSVC(class_weight=CW, random_state=SEED, max_iter=10000)),
    "DecisionTree":     DecisionTreeClassifier(class_weight=CW, random_state=SEED),
    "RandomForest":     RandomForestClassifier(n_estimators=500, class_weight=CW,
                                               random_state=SEED, n_jobs=1),
    "ExtraTrees":       ExtraTreesClassifier(n_estimators=500, class_weight=CW,
                                             random_state=SEED, n_jobs=1),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    "HistGradientBoost":HistGradientBoostingClassifier(class_weight=CW, random_state=SEED),
    "AdaBoost":         AdaBoostClassifier(n_estimators=200, random_state=SEED),
    "Bagging-Tree":     BaggingClassifier(DecisionTreeClassifier(class_weight=CW, random_state=SEED),
                                          n_estimators=200, random_state=SEED, n_jobs=1),
    "MLP":              MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=3000,
                                      early_stopping=True, random_state=SEED),
    "GaussianProcess":  GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=SEED),
}
try:
    from xgboost import XGBClassifier
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    MODELS["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=.05, max_depth=3,
                                      subsample=.8, colsample_bytree=.8, reg_lambda=1.,
                                      scale_pos_weight=spw if CW else 1.,
                                      eval_metric="logloss", random_state=SEED, n_jobs=1)
except ImportError:
    print("未安装 xgboost（可选）")
try:
    from lightgbm import LGBMClassifier
    MODELS["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=15,
                                        class_weight=CW, random_state=SEED, n_jobs=1, verbose=-1)
except ImportError:
    print("未安装 lightgbm（可选）")

MODELS = {k: v for k, v in MODELS.items() if k not in CFG["exclude_models"]}
NO_CW = {"LDA-shrinkage", "QDA", "GaussianNB", "KNN", "GradientBoosting",
         "AdaBoost", "MLP", "GaussianProcess"}
print(f"共 {len(MODELS)} 个候选模型")


# ## 7. 二分类评估指标与阈值策略
# 
# 二分类的指标体系比三分类清晰，但有两个容易出错的地方：
# 
# **其一，AUC 与阈值指标要分开看。** AUC 和 AP 是阈值无关的排序能力；
# Sensitivity/Specificity/PPV/NPV 则取决于你选哪个阈值。只报 AUC 不报阈值指标，
# 临床上无法使用；只报 0.5 阈值下的准确率，则在类别不平衡时严重误导。
# 
# **其二，阈值必须在训练集上确定。** 常见错误是在测试集上扫一遍阈值挑最优的报告出来——
# 这等于用测试集调参，结果必然乐观。本 notebook 用训练集的 out-of-fold 预测按
# **Youden 指数**确定阈值，然后原封不动地应用到测试集。
# 
# **不平衡下 PR 曲线比 ROC 更敏感**：ROC 的横轴是 FPR，阴性类很大时 FPR 变化不明显，
# 曲线容易显得漂亮；PR 曲线以阳性率为基线，更能暴露 PPV 的真实水平。

# In[9]:


def binary_metrics(y_true, y_prob, thr=0.5):
    """二分类全套指标；阈值相关指标使用给定 thr。"""
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv  = tp / (tp + fp) if tp + fp else np.nan
    npv  = tn / (tn + fn) if tn + fn else np.nan
    m = {
        "AUC":         roc_auc_score(y_true, y_prob),
        "AP":          average_precision_score(y_true, y_prob),
        "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv,
        "F1":          f1_score(y_true, y_pred, zero_division=0),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred),
        "Accuracy":    accuracy_score(y_true, y_pred),
        "Youden":      (sens + spec - 1) if np.isfinite(sens) and np.isfinite(spec) else np.nan,
        "MCC":         matthews_corrcoef(y_true, y_pred),
        "Kappa":       cohen_kappa_score(y_true, y_pred),
        "Brier":       brier_score_loss(y_true, y_prob),
        "LogLoss":     log_loss(y_true, y_prob, labels=[0, 1]),
        "Threshold":   thr,
    }
    return m


def pick_threshold(y_true, y_prob, rule="youden", target=0.90):
    """在训练集 out-of-fold 概率上确定阈值。"""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    thr = np.clip(thr, 0, 1)
    if rule == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    # roc_curve 的 thresholds 是降序：索引越大阈值越低、敏感度越高、特异度越低
    if rule == "sens":                       # 保证敏感度 >= target（筛查场景）
        ok = np.where(tpr >= target)[0]      # 取第一个达标点 -> 阈值最高、特异度最好
        return float(np.clip(thr[ok[0]], .001, .999)) if len(ok) else 0.5
    if rule == "spec":                       # 保证特异度 >= target（确诊场景）
        ok = np.where(1 - fpr >= target)[0]  # 取最后一个达标点 -> 敏感度最好
        return float(np.clip(thr[ok[-1]], .001, .999)) if len(ok) else 0.5
    if rule == "f1":
        p, r, t = precision_recall_curve(y_true, y_prob)
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5
    if rule == "prevalence":
        return float(np.mean(y_true))
    return 0.5


# ---- DeLong 法的 AUC 方差（论文里报告 "AUC (95% CI)" 的标准做法）----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N); out[J] = T
    return out


def delong_auc_ci(y_true, y_prob, alpha=0.95):
    pos = y_prob[y_true == 1]; neg = y_prob[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan, np.nan
    tz = _midrank(np.concatenate([pos, neg]))
    tx, ty = _midrank(pos), _midrank(neg)
    auc = (tz[:m].sum() / m - (m + 1) / 2) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    var = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    se  = np.sqrt(max(var, 0))
    z   = stats.norm.ppf(1 - (1 - alpha) / 2)
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se))


SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision",
           "balanced_accuracy": "balanced_accuracy", "f1": "f1",
           "mcc": make_scorer(matthews_corrcoef), "neg_log_loss": "neg_log_loss"}
print("主指标：", CFG["tune_metric"])


# ## 8. 阶段 A —— 全模型广筛

# In[10]:


cv_screen = RepeatedStratifiedKFold(n_splits=CFG["cv_folds"],
                                    n_repeats=CFG["screen_repeats"], random_state=SEED)
rows, cv_raw = [], {}
t0 = time.time()
for name, est in MODELS.items():
    t1 = time.time()
    try:
        res = cross_validate(base_pipe(est), X_tr, y_tr, cv=cv_screen, scoring=SCORING,
                             n_jobs=CFG["n_jobs"], error_score=np.nan)
    except Exception as e:
        print(f"  [跳过] {name}: {type(e).__name__} {e}"); continue
    cv_raw[name] = res
    row = {"Model": name, "ClassWeighted": name not in NO_CW}
    for k in SCORING:
        row[k] = np.nanmean(res["test_" + k]); row[k + "_sd"] = np.nanstd(res["test_" + k])
    row["fit_s"] = res["fit_time"].mean()
    rows.append(row)
    print(f"  {name:<18s} AUC={row['roc_auc']:.3f}±{row['roc_auc_sd']:.3f}  "
          f"AP={row['average_precision']:.3f}  BalAcc={row['balanced_accuracy']:.3f}"
          f"  ({time.time()-t1:.1f}s)")

screen = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
screen.to_csv(OUT / "results_cv_screening.csv", index=False)
print(f"\n总耗时 {time.time()-t0:.1f}s")
screen[["Model", "ClassWeighted", "roc_auc", "roc_auc_sd", "average_precision",
        "balanced_accuracy", "f1", "mcc", "fit_s"]].round(4)


# In[11]:


order = screen["Model"].tolist()[::-1]
data  = [cv_raw[m]["test_roc_auc"] for m in order]
fig, ax = plt.subplots(figsize=(8, .34 * len(order) + 1.6))
bp = ax.boxplot(data, vert=False, patch_artist=True, widths=.62,
                medianprops=dict(color="black", lw=1.4),
                flierprops=dict(marker=".", ms=3, alpha=.5))
for patch, m in zip(bp["boxes"], order):
    patch.set_facecolor("#4E79A7" if m in screen["Model"].head(CFG["top_k"]).values else "#C8CDD2")
    patch.set_alpha(.85); patch.set_edgecolor("#33393F")
ax.axvline(.5, ls="--", c="crimson", lw=1.2)
ax.text(.5, len(order) + .6, " Random guess", color="crimson", fontsize=8, va="top")
ax.set_yticklabels(order, fontsize=8.5)
ax.set_xlabel(f"ROC AUC (repeated stratified CV) — {NEG} vs {POS}")
ax.set_title(f"Model screening — {CFG['cv_folds']}-fold x {CFG['screen_repeats']} repeats", fontsize=11)
plt.tight_layout(); plt.savefig(OUT / "fig1_model_screening.png", dpi=200, bbox_inches="tight")

plt.savefig(OUT / "fig1_model_screening.png", dpi=200, bbox_inches="tight")
plt.show()

top_models = screen["Model"].head(CFG["top_k"]).tolist()
print("进入超参搜索：", top_models)


# ## 9. 阶段 B —— 超参搜索与集成

# In[12]:


def grid_for(name):
    g = {}
    if CFG["fs_method"] == "lasso":
        g["select__C"] = [0.02, 0.05, 0.1, 0.3, 1.0]
    g.update({
        "LogReg-L2":        {"clf__C": np.logspace(-3, 3, 13)},
        "LogReg-L1":        {"clf__C": np.logspace(-3, 2, 11)},
        "LogReg-Elastic":   {"clf__C": np.logspace(-3, 2, 9), "clf__l1_ratio": [.1, .3, .5, .7, .9]},
        "RidgeClassifier":  {"clf__estimator__alpha": np.logspace(-3, 3, 13)},
        "SGD-LogLoss":      {"clf__alpha": np.logspace(-5, 0, 11),
                             "clf__penalty": ["l2", "l1", "elasticnet"]},
        "LDA-shrinkage":    {"clf__shrinkage": ["auto", .05, .1, .2, .4, .6]},
        "QDA":              {"clf__reg_param": np.linspace(0, .9, 10)},
        "KNN":              {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
                             "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
        "SVC-Linear":       {"clf__C": np.logspace(-3, 2, 11)},
        "SVC-RBF":          {"clf__C": np.logspace(-2, 3, 11), "clf__gamma": np.logspace(-4, 0, 9)},
        "SVC-Poly":         {"clf__C": np.logspace(-2, 2, 9), "clf__degree": [2, 3],
                             "clf__gamma": np.logspace(-4, 0, 7)},
        "LinearSVC":        {"clf__estimator__C": np.logspace(-3, 2, 11)},
        "DecisionTree":     {"clf__max_depth": [2, 3, 4, 6, None],
                             "clf__min_samples_leaf": [1, 3, 5, 10]},
        "RandomForest":     {"clf__max_features": ["sqrt", "log2", .3, .5],
                             "clf__min_samples_leaf": [1, 2, 4], "clf__max_depth": [None, 4, 8]},
        "ExtraTrees":       {"clf__max_features": ["sqrt", "log2", .3, .5],
                             "clf__min_samples_leaf": [1, 2, 4], "clf__max_depth": [None, 4, 8]},
        "GradientBoosting": {"clf__n_estimators": [200, 400], "clf__learning_rate": [.03, .1],
                             "clf__max_depth": [2, 3], "clf__subsample": [.8, 1.]},
        "HistGradientBoost":{"clf__learning_rate": [.03, .1], "clf__max_leaf_nodes": [7, 15, 31],
                             "clf__l2_regularization": [0, .1, 1.]},
        "AdaBoost":         {"clf__n_estimators": [100, 200, 400], "clf__learning_rate": [.1, .5, 1.]},
        "Bagging-Tree":     {"clf__max_features": [.3, .6, 1.], "clf__max_samples": [.6, .8, 1.]},
        "MLP":              {"clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                             "clf__alpha": np.logspace(-4, 1, 8)},
        "XGBoost":          {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [.03, .1],
                             "clf__subsample": [.7, .9], "clf__reg_lambda": [1, 5, 10]},
        "LightGBM":         {"clf__num_leaves": [7, 15, 31], "clf__learning_rate": [.03, .1],
                             "clf__min_child_samples": [5, 10, 20]},
    }.get(name, {}))
    return g


cv_tune = RepeatedStratifiedKFold(n_splits=CFG["cv_folds"],
                                  n_repeats=CFG["tune_repeats"], random_state=SEED + 1)
tuned, tune_rows = {}, []
for name in top_models:
    grid, pipe, t1 = grid_for(name), base_pipe(MODELS[name]), time.time()
    if grid:
        srch = RandomizedSearchCV(pipe, grid, n_iter=CFG["n_iter"], cv=cv_tune,
                                  scoring=CFG["tune_metric"], n_jobs=CFG["n_jobs"],
                                  random_state=SEED, refit=True, error_score=np.nan)
        srch.fit(X_tr, y_tr)
        best, score, params = srch.best_estimator_, srch.best_score_, srch.best_params_
    else:
        sc = cross_validate(pipe, X_tr, y_tr, cv=cv_tune, scoring=CFG["tune_metric"],
                            n_jobs=CFG["n_jobs"], error_score=np.nan)
        best = pipe.fit(X_tr, y_tr); score = np.nanmean(sc["test_score"]); params = {}
    tuned[name] = best
    nf = best.named_steps["select"].get_support().sum() if CFG["fs_method"] == "lasso" else X_tr.shape[1]
    tune_rows.append({"Model": name, "CV_AUC": round(score, 4), "n_features_kept": int(nf),
                      "best_params": json.dumps({k: str(v) for k, v in params.items()})})
    print(f"{name:<18s} CV AUC = {score:.4f}   保留特征 {nf}   ({time.time()-t1:.1f}s)")

tune_tab = pd.DataFrame(tune_rows).sort_values("CV_AUC", ascending=False).reset_index(drop=True)

# ---- 集成 ----
base3 = tune_tab["Model"].head(3).tolist()
ests = [(n, tuned[n]) for n in base3]
for nm, ens in {"Voting-Soft": VotingClassifier(ests, voting="soft", n_jobs=1),
                "Stacking-LR": StackingClassifier(
                    ests, final_estimator=LogisticRegression(max_iter=5000, class_weight=CW,
                                                             random_state=SEED),
                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1)}.items():
    sc = cross_validate(ens, X_tr, y_tr, scoring="roc_auc", n_jobs=CFG["n_jobs"],
                        cv=StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=SEED),
                        error_score=np.nan)
    s = np.nanmean(sc["test_score"]); ens.fit(X_tr, y_tr); tuned[nm] = ens
    tune_tab.loc[len(tune_tab)] = {"Model": nm, "CV_AUC": round(s, 4),
                                   "n_features_kept": np.nan, "best_params": "-"}
    print(f"{nm:<18s} CV AUC = {s:.4f}")

tune_tab = tune_tab.sort_values("CV_AUC", ascending=False).reset_index(drop=True)
tune_tab.to_csv(OUT / "results_cv_tuned.csv", index=False)
tune_tab[["Model", "CV_AUC", "n_features_kept"]]


# ## 10. 在训练集上确定分类阈值
# 
# 用 `cross_val_predict` 得到训练集的 out-of-fold 概率（每个样本的预测都来自没见过它的模型），
# 在此基础上按不同规则选阈值。**测试集不参与阈值选择**。
# 
# 下表列出几种规则的结果，可根据临床定位取舍：筛查排除更看重敏感度，
# 确诊更看重特异度，Youden 是两者的折中。

# In[13]:


BEST_CV = tune_tab.loc[0, "Model"]
print(f"CV 最优模型：{BEST_CV}")

oof = cross_val_predict(tuned[BEST_CV], X_tr, y_tr, method="predict_proba",
                        cv=StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=SEED),
                        n_jobs=CFG["n_jobs"])[:, 1]
print(f"训练集 out-of-fold AUC = {roc_auc_score(y_tr, oof):.4f}")

thr_rows = []
for rule, note in [("youden", "Youden 指数最大（默认）"), ("f1", "F1 最大"),
                   ("sens", "敏感度 ≥ 0.90"), ("spec", "特异度 ≥ 0.90"),
                   ("prevalence", "阳性率作阈值"), ("default", "固定 0.5")]:
    t = 0.5 if rule == "default" else pick_threshold(y_tr, oof, rule)
    m = binary_metrics(y_tr, oof, t)
    thr_rows.append({"Rule": rule, "说明": note, "Threshold": round(t, 4),
                     **{k: round(m[k], 4) for k in
                        ["Sensitivity", "Specificity", "PPV", "NPV", "BalancedAcc", "F1", "Youden"]}})
thr_tab = pd.DataFrame(thr_rows)
thr_tab.to_csv(OUT / "threshold_selection.csv", index=False)
print("\n各阈值规则在训练集 out-of-fold 上的表现：")
print(thr_tab.to_string(index=False))

# 阈值必须与模型配套：为每个候选模型分别在其 out-of-fold 概率上取 Youden 阈值
cv_oof = StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=SEED)
THR_BY_MODEL = {}
print("\n各模型各自的 out-of-fold 阈值：")
for name, mdl in tuned.items():
    p_oof = cross_val_predict(mdl, X_tr, y_tr, method="predict_proba",
                              cv=cv_oof, n_jobs=CFG["n_jobs"])[:, 1]
    THR_BY_MODEL[name] = pick_threshold(y_tr, p_oof, "youden")
    print(f"  {name:<18s} thr={THR_BY_MODEL[name]:.4f}   "
          f"OOF AUC={roc_auc_score(y_tr, p_oof):.4f}")

THR = THR_BY_MODEL[BEST_CV]
print(f"\n若最终选用 {BEST_CV}，阈值 = {THR:.4f}；"
      "测试集上每个模型都会使用它自己的阈值。")


# ## 11. 外部验证 —— 独立测试集
# 
# 测试集在此之前从未参与拟合、模型选择、超参搜索或阈值确定。

# In[14]:


test_prob, test_metrics = {}, {}
for name, mdl in tuned.items():
    p = mdl.predict_proba(X_te)[:, 1]
    test_prob[name] = p
    test_metrics[name] = binary_metrics(y_te, p, THR_BY_MODEL[name])  # 各用各的阈值

test_tab = pd.DataFrame(test_metrics).T.sort_values("AUC", ascending=False).round(4)
test_tab.index.name = "Model"
test_tab.to_csv(OUT / "results_test_external.csv")
print("测试集结果（每个模型使用其自身在训练集 out-of-fold 上确定的阈值）")
test_tab[["AUC", "AP", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV",
          "BalancedAcc", "F1", "MCC", "Brier"]]


# In[15]:


BEST = test_tab.index[0]
THR  = THR_BY_MODEL[BEST]                      # 与最终模型配套的阈值
prob = test_prob[BEST]; pred = (prob >= THR).astype(int)
auc, lo, hi = delong_auc_ci(y_te, prob)
print(f"最佳模型：{BEST}")
print(f"测试集 AUC = {auc:.3f} (95% CI {lo:.3f}–{hi:.3f}, DeLong)\n")
print(classification_report(y_te, pred, target_names=CLS, digits=3, zero_division=0))

cm = confusion_matrix(y_te, pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"混淆矩阵：TN={tn}  FP={fp}  FN={fn}  TP={tp}")

fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.imshow(cm / cm.sum(axis=1, keepdims=True),
          cmap=LinearSegmentedColormap.from_list("bl", ["#FFFFFF", "#4E79A7"]), vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/cm[i].sum()*100:.0f}%)", ha="center", va="center",
                fontsize=11, color="white" if cm[i, j] / cm[i].sum() > .55 else "#222")
ax.set_xticks([0, 1], CLS); ax.set_yticks([0, 1], CLS)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
ax.set_title(f"{BEST} — test set (thr={THR:.2f})", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "fig2_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()


# ## 12. ROC / PR / 校准 / 决策曲线
# 
# **决策曲线分析（DCA）** 回答的是 ROC 回答不了的问题：在临床上真正使用这个模型，
# 相比"全部当阳性处理"或"全部当阴性处理"，净获益是多少。
# 只有当模型曲线在合理的阈值概率区间内高于两条参考线，才说明它有临床价值。

# In[16]:


fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# --- ROC ---
ax = axes[0, 0]
for nm in test_tab.index[:3]:
    f, t, _ = roc_curve(y_te, test_prob[nm])
    a, l_, h_ = delong_auc_ci(y_te, test_prob[nm])
    ax.plot(f, t, lw=2, label=f"{nm}  {a:.3f} ({l_:.2f}–{h_:.2f})")
ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC — test set")
ax.legend(fontsize=8, loc="lower right")

# --- PR ---
ax = axes[0, 1]
for nm in test_tab.index[:3]:
    p_, r_, _ = precision_recall_curve(y_te, test_prob[nm])
    ax.plot(r_, p_, lw=2, label=f"{nm}  AP={average_precision_score(y_te, test_prob[nm]):.3f}")
ax.axhline(y_te.mean(), ls=":", c="grey", lw=1.2)
ax.text(.02, y_te.mean() + .02, f"prevalence={y_te.mean():.2f}", fontsize=8, color="grey")
ax.set(xlabel="Recall (Sensitivity)", ylabel="Precision (PPV)", title="Precision–Recall")
ax.legend(fontsize=8, loc="lower left")

# --- 校准 ---
ax = axes[1, 0]
ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="Perfect")
nb = min(5, max(3, int(min((y_te == 0).sum(), (y_te == 1).sum()) / 3)))
try:
    pt, pp = calibration_curve(y_te, prob, n_bins=nb, strategy="quantile")
    ax.plot(pp, pt, "o-", color="#4E79A7", lw=2, ms=6, label=BEST)
except Exception as e:
    print("校准曲线跳过：", e)
ax.set(xlabel="Mean predicted probability", ylabel="Observed frequency",
       title=f"Calibration (Brier={test_tab.loc[BEST,'Brier']:.3f})")
ax.legend(fontsize=9)

# --- 决策曲线分析 ---
ax = axes[1, 1]
pt_grid = np.linspace(0.01, 0.80, 120)
N, prev = len(y_te), y_te.mean()
nb_model = [( (prob >= t).astype(int) @ y_te ) / N
            - (((prob >= t).astype(int) @ (1 - y_te)) / N) * (t / (1 - t)) for t in pt_grid]
nb_all   = prev - (1 - prev) * (pt_grid / (1 - pt_grid))
ax.plot(pt_grid, nb_model, lw=2.2, color="#4E79A7", label=BEST)
ax.plot(pt_grid, nb_all, lw=1.4, color="#8C8C8C", ls="--", label="Treat all")
ax.axhline(0, lw=1.4, color="black", ls=":", label="Treat none")
ax.set_ylim(min(-0.03, prev * -0.15), prev * 1.15)
ax.set(xlabel="Threshold probability", ylabel="Net benefit",
       title="Decision curve analysis")
ax.legend(fontsize=9)

fig.suptitle(f"{NEG} vs {POS} — test set (n={len(y_te)})", fontsize=12)
plt.tight_layout(); plt.savefig(OUT / "fig3_roc_pr_calib_dca.png", dpi=200, bbox_inches="tight")
plt.show()


# ## 13. Bootstrap 置信区间与模型间比较

# In[17]:


rng = np.random.default_rng(SEED)
by_cls = {c: np.where(y_te == c)[0] for c in [0, 1]}
boots, diffs = [], []
second = test_tab.index[1] if len(test_tab) > 1 else BEST
for _ in range(CFG["n_boot"]):
    bi = np.concatenate([rng.choice(v, len(v), replace=True) for v in by_cls.values()])
    if len(np.unique(y_te[bi])) < 2:
        continue
    boots.append(binary_metrics(y_te[bi], prob[bi], THR))
    diffs.append(roc_auc_score(y_te[bi], prob[bi]) -
                 roc_auc_score(y_te[bi], test_prob[second][bi]))
bt = pd.DataFrame(boots)
ci = pd.DataFrame({"Point": pd.Series(test_metrics[BEST]),
                   "Lo": bt.quantile(.025), "Hi": bt.quantile(.975)})
ci["95% CI"] = ci.apply(lambda r: f"{r.Point:.3f} ({r.Lo:.3f}–{r.Hi:.3f})", axis=1)
ci.to_csv(OUT / f"results_test_bootCI_{BEST}.csv")
print(f"{BEST} — bootstrap 95% CI (B={len(bt)})")
print(ci[["95% CI"]].to_string())

if second != BEST:
    d = np.array(diffs); p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    print(f"\n{BEST} vs {second} 的 AUC 差值 = {d.mean():.4f} "
          f"(95% CI {np.quantile(d,.025):.4f}–{np.quantile(d,.975):.4f}), "
          f"bootstrap p = {max(p, 1/len(d)):.4f}")


# ## 14. 置换检验：性能是真的还是偶然？
# 
# 把标签随机打乱若干次，重跑整条建模流程，得到"零假设下的 AUC 分布"。
# 若真实 AUC 落在这个分布的极端尾部，说明模型确实抓到了信号。
# 
# **样本量小的时候这一步比任何指标都重要**：n=100 出一个 AUC 0.68 看着还行，
# 但如果打乱标签也常常能得到 0.65，那这个结果就没有意义。

# In[18]:


BEST_CV


# In[19]:


BEST


# In[20]:


if CFG["n_perm"] and CFG["n_perm"] > 0:
    t0 = time.time()
    base_est = base_pipe(MODELS[BEST]) if BEST in MODELS else tuned[BEST]
    score, perm_scores, pval = permutation_test_score(
        base_est, X_tr, y_tr, scoring="roc_auc",
        cv=StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=SEED),
        n_permutations=CFG["n_perm"], n_jobs=CFG["n_jobs"], random_state=SEED)
    print(f"真实 CV AUC = {score:.4f}")
    print(f"置换分布：均值 {perm_scores.mean():.4f}，95 分位 {np.quantile(perm_scores,.95):.4f}")
    print(f"置换检验 p = {pval:.4f}   ({time.time()-t0:.0f}s, {CFG['n_perm']} 次)")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(perm_scores, bins=25, color="#C8CDD2", edgecolor="#7A8288")
    ax.axvline(score, color="#E15759", lw=2.4, label=f"Observed AUC = {score:.3f}")
    ax.axvline(.5, color="grey", ls="--", lw=1)
    ax.set(xlabel="CV AUC under permuted labels", ylabel="Count",
           title=f"Permutation test — p = {pval:.4f}")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "fig4_permutation_test.png", dpi=200, bbox_inches="tight")
    plt.show()
    if pval > 0.05:
        print("\n⚠ p > 0.05：无法排除当前性能来自偶然。"
              "此时不应报告模型有效，应回到特征质量与样本量的问题上。")
else:
    print("已跳过置换检验（CFG['n_perm'] = 0）")


# ## 15. 学习曲线：是样本不够，还是特征无效？
# 
# 这张图直接回答"性能不佳该往哪个方向投入"：
# 
# - **验证曲线仍在上升** → 增加样本量有明确收益，值得继续收集病例。
# - **训练与验证都低且贴近** → 欠拟合，特征本身信息不足，加样本无济于事，
#   应回到特征提取（ICC 筛选、谐波化、分割一致性）或换更强的表征。
# - **训练高、验证低、间距大** → 过拟合，应加强正则、减少特征、简化模型。

# In[21]:


if CFG["do_learning_curve"]:
    est = base_pipe(MODELS[BEST]) if BEST in MODELS else tuned[BEST]
    sizes, tr_s, va_s = learning_curve(
        est, X_tr, y_tr, cv=StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=SEED),
        scoring="roc_auc", train_sizes=np.linspace(.35, 1.0, 7),
        n_jobs=CFG["n_jobs"], random_state=SEED)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for s, lab, c in [(tr_s, "Training", "#4E79A7"), (va_s, "Cross-validation", "#E15759")]:
        m_, sd_ = s.mean(1), s.std(1)
        ax.plot(sizes, m_, "o-", color=c, lw=2, label=lab)
        ax.fill_between(sizes, m_ - sd_, m_ + sd_, color=c, alpha=.15)
    ax.axhline(.5, ls="--", c="grey", lw=1)
    ax.set(xlabel="Training set size", ylabel="ROC AUC",
           title=f"Learning curve — {BEST}")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "fig5_learning_curve.png", dpi=200, bbox_inches="tight")
    plt.show()

    gap = tr_s.mean(1)[-1] - va_s.mean(1)[-1]
    slope = va_s.mean(1)[-1] - va_s.mean(1)[-3]
    print(f"末端 训练-验证 间距 = {gap:.3f}；验证曲线近端斜率 = {slope:+.3f}")
    if gap > .15 and va_s.mean(1)[-1] < .75:
        print("→ 过拟合特征：建议加强正则化、进一步减少特征数。")
    elif slope > .02:
        print("→ 验证曲线仍在上升：扩大样本量预计能带来实质提升。")
    else:
        print("→ 曲线已平台且水平不高：瓶颈在特征信息量，"
              "加样本收益有限，应回到特征提取与质控环节。")


# ## 16. 特征重要性与单变量分析

# In[22]:


pi = permutation_importance(tuned[BEST], X_te, y_te, scoring="roc_auc",
                            n_repeats=30, random_state=SEED, n_jobs=CFG["n_jobs"])
imp = pd.DataFrame({"feature": X_te.columns, "importance": pi.importances_mean,
                    "sd": pi.importances_std}).sort_values("importance", ascending=False)
imp.to_csv(OUT / f"feature_importance_{BEST}.csv", index=False)

# 单变量：每个特征自身的 AUC 与 Mann–Whitney p（BH 校正）
uni = []
for c in X_bin.columns:
    v = X_bin[c].to_numpy(); ok = np.isfinite(v)
    if ok.sum() < 10 or len(np.unique(v[ok])) < 2:
        continue
    a = roc_auc_score(y_bin[ok], v[ok])
    p = stats.mannwhitneyu(v[ok][y_bin[ok] == 1], v[ok][y_bin[ok] == 0]).pvalue
    uni.append({"feature": c, "AUC": max(a, 1 - a), "p": p})
uni = pd.DataFrame(uni).sort_values("AUC", ascending=False)
m = len(uni)
uni["p_BH"] = np.minimum.accumulate(
    (uni.sort_values("p").p.to_numpy() * m / np.arange(1, m + 1))[::-1])[::-1] \
    if m else []
uni = uni.sort_values("AUC", ascending=False)
uni.to_csv(OUT / "univariate_analysis.csv", index=False)
print(f"单变量 AUC ≥ 0.70 的特征：{(uni.AUC >= .70).sum()} / {m}")
print(f"BH 校正后 p < 0.05 的特征：{(uni.p_BH < .05).sum()} / {m}\n")

fig, axes = plt.subplots(1, 2, figsize=(12, .3 * 15 + 1.8))
t1 = imp.head(15).iloc[::-1]
axes[0].barh(t1.feature, t1.importance, xerr=t1.sd, color="#4E79A7", alpha=.9,
             error_kw=dict(lw=.8, ecolor="#555"))
axes[0].axvline(0, c="grey", lw=1)
axes[0].set(xlabel="Drop in AUC when permuted", title=f"Permutation importance — {BEST}")
t2 = uni.head(15).iloc[::-1]
axes[1].barh(t2.feature, t2.AUC, color="#59A14F", alpha=.9)
axes[1].axvline(.5, c="grey", ls="--", lw=1); axes[1].set_xlim(.45, 1.0)
axes[1].set(xlabel="Univariate AUC", title="Single-feature discrimination")
for a in axes:
    a.tick_params(axis="y", labelsize=7.5)
plt.tight_layout(); plt.savefig(OUT / "fig6_importance.png", dpi=200, bbox_inches="tight")
plt.show()


# ## 17. 嵌套交叉验证（可选）

# In[23]:


if CFG["do_nested_cv"]:
    NEST = BEST_CV if BEST_CV in MODELS else base3[0]
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    rows = []
    for k, (itr, ite) in enumerate(outer.split(X_bin, y_bin), 1):
        srch = RandomizedSearchCV(base_pipe(MODELS[NEST]),
                                  grid_for(NEST) or {"select__C": [.05, .1, .3]},
                                  n_iter=15, cv=inner, scoring="roc_auc",
                                  n_jobs=CFG["n_jobs"], random_state=SEED, error_score=np.nan)
        srch.fit(X_bin.iloc[itr], y_bin[itr])
        p_ = srch.best_estimator_.predict_proba(X_bin.iloc[ite])[:, 1]
        oof_thr = pick_threshold(y_bin[itr],
                                 cross_val_predict(srch.best_estimator_, X_bin.iloc[itr],
                                                   y_bin[itr], cv=inner,
                                                   method="predict_proba")[:, 1])
        m_ = binary_metrics(y_bin[ite], p_, oof_thr); m_["Fold"] = k
        rows.append(m_); print(f"  fold {k}: AUC={m_['AUC']:.3f}  BalAcc={m_['BalancedAcc']:.3f}")
    nested = pd.DataFrame(rows).set_index("Fold")
    nested.to_csv(OUT / "results_nested_cv.csv")
    print(f"\n嵌套 CV（{NEST}）均值 ± 标准差：")
    print(pd.DataFrame({"mean": nested.mean(), "sd": nested.std()}).round(4))
else:
    print("已跳过嵌套 CV（把 CFG['do_nested_cv'] 设为 True 可启用）")


# ## 18. 保存结果

# In[24]:


pd.DataFrame({
    "ID": id_bin[idx_te],
    "OriginalLabel": lab3_kept[idx_te],
    "True": [CLS[i] for i in y_te],
    "Prob_LAM": prob,
    "Pred": [CLS[i] for i in pred],
    "Threshold": THR,
}).to_csv(OUT / f"test_predictions_{BEST}.csv", index=False)

try:
    joblib.dump({
        "model": tuned[BEST],
        "name": BEST,
        "threshold": THR,
        "classes": CLS,
        "negative": NEG,
        "positive": POS,
        "probability_definition": f"P({POS})",
        "config": CFG,
        "binary": BINARY,
    }, OUT / f"best_model_{BEST}.joblib")
    print(f"模型已保存：0={NEG}, 1={POS}, predict_proba[:,1]=P({POS})")
    print("加载时需先 import radiomics_utils（与 notebook 同目录）。")
except Exception as e:
    print("模型序列化失败：", e, "\n预测结果 CSV 已正常保存，不影响结果报告。")

print("已保存到", OUT.resolve())
for f in sorted(OUT.iterdir()):
    print("  ", f.name)


# ## 19. 结果解读与后续方向
# 
# **本 notebook 的结论边界是 BHD vs LAM 鉴别诊断。**  
# 论文中建议报告：训练集重复 CV 的 AUC（均值±标准差）、独立测试集 AUC 与
# DeLong 95% CI、训练集 OOF 确定的阈值、该阈值下的
# Sensitivity/Specificity/PPV/NPV、校准（Brier/校准曲线）、决策曲线、
# bootstrap 置信区间与置换检验 p 值。
# 
# 由于这里固定 **LAM 为阳性类（1）**：
# 
# - Sensitivity = LAM 的检出率；
# - Specificity = BHD 的正确排除率；
# - PPV = 被模型判为 LAM 的病例中实际为 LAM 的比例；
# - NPV = 被模型判为 BHD 的病例中实际为 BHD 的比例；
# - `Prob_LAM` 与 `predict_proba[:, 1]` 都表示 **P(LAM)**。
# 
# **如果 BHD vs LAM 表现仍然不理想，优先按以下顺序排查：**
# 
# 1. 看置换检验：若 p > 0.05，当前性能不能可靠地区分于偶然。
# 2. 看学习曲线：验证曲线仍上升 → 增加样本可能有收益；训练高而验证低 → 过拟合；
#    两者都低且接近 → 当前特征对 BHD/LAM 的鉴别信息不足。
# 3. 做特征可重复性筛选：用重复/双人分割计算 ICC，剔除不稳定特征。
# 4. 检查重采样、灰度离散化、归一化等 PyRadiomics 提取参数是否完全一致。
# 5. 多中心/多设备数据考虑 ComBat 等批次效应校正。
# 6. 若多种算法结果接近，优先改善表征而不是继续堆模型：可加入囊肿分布、
#    形态、数量、体积占比等更具 BHD/LAM 鉴别意义的定量特征，以及合适的临床变量。
# 7. 小样本高维场景下，应依赖严格的 Pipeline 内特征选择、正则化、重复 CV，
#    并避免用测试集挑模型或阈值。
# 
