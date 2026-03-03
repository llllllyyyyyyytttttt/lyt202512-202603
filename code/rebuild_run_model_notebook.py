import json
import shutil
import textwrap
from pathlib import Path


def md(text: str) -> dict:
    s = textwrap.dedent(text).strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": s.splitlines(True),
    }


def code(text: str) -> dict:
    s = textwrap.dedent(text).strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": s.splitlines(True),
    }


cells = []

cells.append(
    md(
        """
        # 动态信用风险建模完整实验（重建版）

        本 Notebook 目标：
        1. 实现“静态 Logit + GAS 时变参数 + B-spline 变系数”完整流程；
        2. 代码注释与结果解读使用中文；
        3. 输出论文可复用的指标、图表、自动结论文本；
        4. 确保中文显示稳定（JSON 使用 ensure_ascii=True）。

        核心输出对象：
        - `model_df`, `result_df`, `quarter_eval_df`, `region_eval_df`, `monitor_df`, `bs_effect_df`
        - `analysis_summary_df`, `analysis_text_lines`
        - `fig_auc_trend`, `fig_region_bar`, `fig_gas_state`
        """
    )
)

cells.append(
    code(
        r"""
# ===============================
# 0) 依赖检查与自动安装（含 matplotlib）
# ===============================
import importlib
import subprocess
import sys


def ensure_pkg(pip_name: str, import_name: str | None = None) -> None:
    mod = import_name or pip_name
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"[INFO] 安装缺失依赖: {pip_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


for pkg, mod in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
]:
    ensure_pkg(pkg, mod)

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)
EPS = 1e-9

print("依赖加载完成。")
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 1) 通用函数：激活函数、评估指标、文件查找
# ===============================
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def ks_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(np.abs(tpr - fpr)))


def psi_score(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return float("nan")
    cuts[0], cuts[-1] = -np.inf, np.inf
    exp_hist = np.histogram(expected, bins=cuts)[0].astype(float)
    act_hist = np.histogram(actual, bins=cuts)[0].astype(float)
    exp_pct = np.clip(exp_hist / exp_hist.sum(), 1e-6, None)
    act_pct = np.clip(act_hist / act_hist.sum(), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_level(psi: float) -> str:
    if pd.isna(psi):
        return "NA"
    if psi < 0.10:
        return "稳定"
    if psi <= 0.25:
        return "轻微漂移"
    return "显著漂移"


def quarter_label_to_date(text: str):
    text = str(text)
    m = re.match(r"(\\d{4})年\\s*第?([一二三四1234])季度", text)
    if not m:
        return pd.NaT
    year = int(m.group(1))
    q_map = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}
    q = q_map[m.group(2)]
    md_map = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return pd.Timestamp(f"{year}-{md_map[q]}")

print("通用函数定义完成。")
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 2) 数据读取函数
# ===============================
def load_borrower_data():
    path = BORROWER_DATA_PATH
    df = pd.read_csv(path)

    x_cols = [c for c in df.columns if c.startswith("x_")]
    meta_cols = [c for c in df.columns if c not in x_cols]

    cfg = {
        "id_col": meta_cols[0],
        "target_col": "y" if "y" in meta_cols else meta_cols[1],
        "time_col": meta_cols[2],
        "region_col": meta_cols[3],
        "province_col": meta_cols[4],
        "x_cols": x_cols,
    }

    df = df.rename(
        columns={
            cfg["id_col"]: "user_id",
            cfg["target_col"]: "y",
            cfg["time_col"]: "time",
            cfg["region_col"]: "region",
            cfg["province_col"]: "province",
        }
    )
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df, cfg


def load_govern_macro_data():
    path = GOVERN_DATA_PATH
    df = pd.read_csv(path, encoding="utf-8")
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.rename(columns={tcol: "time"})

    raw_macro_cols = [c for c in df.columns if c not in ["time", "province", "region"]]
    rename = {c: f"gov_macro_{i:02d}" for i, c in enumerate(raw_macro_cols, 1)}
    df = df.rename(columns=rename)
    macro_cols = list(rename.values())
    return df[["time", "province", "region"] + macro_cols], macro_cols


def load_national_macro_optional():
    try:
        path = NATIONAL_DATA_PATH
        nat = pd.read_csv(path, encoding="gbk", skiprows=2)
        tcol = nat.columns[0]
        nat[tcol] = nat[tcol].map(quarter_label_to_date)
        nat = nat.rename(columns={tcol: "time"})

        keep = []
        for c in nat.columns:
            if c == "time":
                continue
            s = pd.to_numeric(nat[c], errors="coerce")
            if s.notna().sum() >= 8:
                nat[c] = s
                keep.append(c)

        rename = {c: f"nat_macro_{i:02d}" for i, c in enumerate(keep, 1)}
        nat = nat.rename(columns=rename)
        nat_cols = list(rename.values())
        return nat[["time"] + nat_cols], nat_cols
    except Exception:
        return pd.DataFrame(columns=["time"]), []

print("数据读取函数定义完成。")
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 3) 自定义模型：GAS + B-spline
# 性能优化：
# - 预构建时间索引缓存，减少优化迭代中的重复过滤
# - 前向递推使用预分配数组
# ===============================
class GASDynamicLogit:
    def __init__(self, l2=5e-3, maxiter=420):
        self.l2 = l2
        self.maxiter = maxiter

    @staticmethod
    def _split(theta, p, m):
        beta = theta[:p]
        gamma = theta[p : p + m]
        omega = theta[p + m]
        alpha = theta[p + m + 1]
        phi = theta[p + m + 2]
        return beta, gamma, omega, alpha, phi

    @staticmethod
    def _build_index_cache(t_idx, T):
        return [np.where(t_idx == t)[0] for t in range(T)]

    def _forward(self, X, y_for_state, z_by_t, theta, idx_cache):
        p = X.shape[1]
        m = z_by_t.shape[1]
        beta, gamma, omega, alpha, phi = self._split(theta, p, m)
        T = z_by_t.shape[0]

        pred = np.zeros(X.shape[0], dtype=float)
        states = np.full(T, np.nan, dtype=float)
        scores = np.full(T, np.nan, dtype=float)

        # 长期均值初始化
        f_prev = (omega + z_by_t.mean(axis=0).dot(gamma)) / max(1.0 - phi, 1e-3)

        for t in range(T):
            idx_t = idx_cache[t]
            if t == 0:
                f_t = f_prev
                score_t = 0.0
            else:
                idx_prev = idx_cache[t - 1]
                p_prev = sigmoid(X[idx_prev].dot(beta) + f_prev)
                score_t = np.mean(y_for_state[idx_prev] - p_prev)
                fisher_t = np.mean(p_prev * (1.0 - p_prev)) + EPS
                f_t = omega + z_by_t[t].dot(gamma) + alpha * (score_t / np.sqrt(fisher_t)) + phi * f_prev

            if idx_t.size > 0:
                pred[idx_t] = sigmoid(X[idx_t].dot(beta) + f_t)
            states[t] = f_t
            scores[t] = score_t
            f_prev = f_t

        return pred, states, scores

    def _loss(self, theta, X, y, z_by_t):
        pred, _, _ = self._forward(X, y, z_by_t, theta, self._fit_idx_cache)
        nll = -np.sum(y * np.log(pred + EPS) + (1.0 - y) * np.log(1.0 - pred + EPS))

        p = X.shape[1]
        m = z_by_t.shape[1]
        beta, gamma, _, _, _ = self._split(theta, p, m)
        penalty = self.l2 * (np.sum(beta**2) + np.sum(gamma**2))
        return float(nll + penalty)

    def fit(self, X, y, t_idx, z_by_t):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        t_idx = np.asarray(t_idx, dtype=int)
        z_by_t = np.asarray(z_by_t, dtype=float)

        p = X.shape[1]
        m = z_by_t.shape[1]
        T = z_by_t.shape[0]
        self._fit_idx_cache = self._build_index_cache(t_idx, T)

        # 初始化：静态 Logit 系数 + 默认状态参数
        static = LogisticRegression(max_iter=300, solver="lbfgs")
        static.fit(X, y)
        beta0 = static.coef_.reshape(-1)
        ybar = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        omega0 = float(np.log(ybar / (1 - ybar)))
        theta0 = np.concatenate([beta0, np.zeros(m), np.array([omega0, 0.0, 0.2])])

        bounds = [(-4, 4)] * (p + m) + [(-6, 6), (-1, 1), (-0.95, 0.95)]
        res = minimize(
            self._loss,
            theta0,
            args=(X, y, z_by_t),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter},
        )

        self.theta_ = res.x
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        self.p_ = p
        self.m_ = m
        return self

    def predict_proba(self, X, y_for_state, t_idx, z_by_t):
        X = np.asarray(X, dtype=float)
        y_for_state = np.asarray(y_for_state, dtype=float)
        t_idx = np.asarray(t_idx, dtype=int)
        z_by_t = np.asarray(z_by_t, dtype=float)
        idx_cache = self._build_index_cache(t_idx, z_by_t.shape[0])
        pred, _, _ = self._forward(X, y_for_state, z_by_t, self.theta_, idx_cache)
        return pred

    def predict_proba_with_details(self, X, y_for_state, t_idx, z_by_t, time_labels):
        X = np.asarray(X, dtype=float)
        y_for_state = np.asarray(y_for_state, dtype=float)
        t_idx = np.asarray(t_idx, dtype=int)
        z_by_t = np.asarray(z_by_t, dtype=float)
        idx_cache = self._build_index_cache(t_idx, z_by_t.shape[0])
        pred, states, scores = self._forward(X, y_for_state, z_by_t, self.theta_, idx_cache)
        detail = pd.DataFrame({"time": pd.to_datetime(time_labels), "gas_state": states, "gas_score": scores})
        return pred, detail


class BSplineVaryingLogit:
    def __init__(self, n_knots=7, degree=3, lam=1.5, ridge=5e-3, maxiter=260):
        self.n_knots = n_knots
        self.degree = degree
        self.lam = lam
        self.ridge = ridge
        self.maxiter = maxiter

    def _build_design(self, X, t_num, key_feature, fit=False):
        t_num = np.asarray(t_num).reshape(-1, 1)
        key_feature = np.asarray(key_feature)

        if fit:
            self.spline_ = SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                include_bias=True,
                extrapolation="linear",
            )
            B = self.spline_.fit_transform(t_num)
        else:
            B = self.spline_.transform(t_num)

        K = B.shape[1]
        Xd = np.hstack([X, B, key_feature.reshape(-1, 1) * B])
        return Xd, K

    def _loss(self, w, Xd, y, p, K):
        pred = sigmoid(Xd.dot(w))
        nll = -np.sum(y * np.log(pred + EPS) + (1.0 - y) * np.log(1.0 - pred + EPS))

        beta = w[:p]
        theta0 = w[p : p + K]
        theta1 = w[p + K : p + 2 * K]
        if K >= 3:
            D2 = np.diff(np.eye(K), n=2, axis=0)
            smooth = np.sum((D2 @ theta0) ** 2) + np.sum((D2 @ theta1) ** 2)
        else:
            smooth = np.sum(theta0**2) + np.sum(theta1**2)
        return float(nll + self.ridge * np.sum(beta**2) + self.lam * smooth)

    def fit(self, X, y, t_num, key_feature):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xd, K = self._build_design(X, t_num, key_feature, fit=True)
        p = X.shape[1]
        w0 = np.zeros(p + 2 * K)

        res = minimize(
            self._loss,
            w0,
            args=(Xd, y, p, K),
            method="L-BFGS-B",
            options={"maxiter": self.maxiter},
        )

        self.w_ = res.x
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        self.p_ = p
        self.K_ = K
        return self

    def predict_proba(self, X, t_num, key_feature):
        X = np.asarray(X, dtype=float)
        Xd, _ = self._build_design(X, t_num, key_feature, fit=False)
        return sigmoid(Xd.dot(self.w_))

    def get_time_effects(self, t_num_grid):
        B = self.spline_.transform(np.asarray(t_num_grid).reshape(-1, 1))
        p = self.p_
        K = self.K_
        theta0 = self.w_[p : p + K]
        theta1 = self.w_[p + K : p + 2 * K]
        return B.dot(theta0), B.dot(theta1)

print("模型类定义完成。")
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 4) 数据加载与合并
# ===============================
borrower_df, borrower_cfg = load_borrower_data()
govern_df, govern_cols = load_govern_macro_data()
national_df, national_cols = load_national_macro_optional()

model_df = borrower_df.merge(govern_df, on=["time", "province"], how="left")
if len(national_cols) > 0:
    model_df = model_df.merge(national_df, on="time", how="left")

if "region" not in model_df.columns:
    if "region_x" in model_df.columns and "region_y" in model_df.columns:
        model_df["region"] = model_df["region_x"].fillna(model_df["region_y"])
    elif "region_x" in model_df.columns:
        model_df["region"] = model_df["region_x"]
    elif "region_y" in model_df.columns:
        model_df["region"] = model_df["region_y"]
for c in ["region_x", "region_y"]:
    if c in model_df.columns:
        model_df = model_df.drop(columns=c)

x_cols = borrower_cfg["x_cols"]
macro_cols = [c for c in model_df.columns if c.startswith("gov_macro_") or c.startswith("nat_macro_")]
candidate_cols = x_cols + macro_cols
for c in candidate_cols:
    model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

miss_ratio = model_df[candidate_cols].isna().mean()
keep_cols = [c for c in candidate_cols if miss_ratio[c] <= 0.98]
var_series = model_df[keep_cols].var(numeric_only=True)
keep_cols = [c for c in keep_cols if var_series.get(c, 0.0) > 1e-8]

print("借款样本:", borrower_df.shape)
print("合并后样本:", model_df.shape)
print("候选特征:", len(candidate_cols), "保留特征:", len(keep_cols))
print("时间范围:", model_df["time"].min(), "->", model_df["time"].max(), "季度数:", model_df["time"].nunique())
print("地区数:", model_df["region"].nunique(), "省份数:", model_df["province"].nunique())
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 5) 时间切分 + 特征预处理
# ===============================
all_times = np.sort(model_df["time"].dropna().unique())
train_times = all_times[:-6]
valid_times = all_times[-6:-3]
test_times = all_times[-3:]

train_mask = model_df["time"].isin(train_times)
valid_mask = model_df["time"].isin(valid_times)
test_mask = model_df["time"].isin(test_times)
train_mask_arr = train_mask.to_numpy()
valid_mask_arr = valid_mask.to_numpy()
test_mask_arr = test_mask.to_numpy()

corr_score = {}
for c in [k for k in keep_cols if k.startswith("x_")]:
    s = model_df.loc[train_mask, c]
    if s.notna().sum() < 200 or s.std(skipna=True) < 1e-12:
        corr_score[c] = 0.0
        continue
    cor = np.corrcoef(s.fillna(s.median()), model_df.loc[train_mask, "y"])[0, 1]
    corr_score[c] = abs(cor) if np.isfinite(cor) else 0.0

borrower_features = sorted(corr_score, key=corr_score.get, reverse=True)[:20]
macro_features = [c for c in keep_cols if c.startswith("gov_macro_")][:4]
selected_features = borrower_features + macro_features

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(model_df.loc[train_mask, selected_features]))
X_valid = scaler.transform(imputer.transform(model_df.loc[valid_mask, selected_features]))
X_test = scaler.transform(imputer.transform(model_df.loc[test_mask, selected_features]))
X_all = scaler.transform(imputer.transform(model_df[selected_features]))

y_train = model_df.loc[train_mask, "y"].to_numpy()
y_valid = model_df.loc[valid_mask, "y"].to_numpy()
y_test = model_df.loc[test_mask, "y"].to_numpy()
y_all = model_df["y"].to_numpy()

print("入模特征数:", len(selected_features))
print("借款行为Top5:", borrower_features[:5])
print("宏观特征:", macro_features)
print("样本切分 train/valid/test:", train_mask.sum(), valid_mask.sum(), test_mask.sum())
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 6) 时间索引与 GAS 状态变量构建
# ===============================
time_order = np.sort(model_df["time"].dropna().unique())
time_to_idx = {t: i for i, t in enumerate(time_order)}
t_idx_all = model_df["time"].map(time_to_idx).to_numpy()
train_t_idx = np.array([time_to_idx[t] for t in train_times], dtype=int)

state_cols = [c for c in macro_features if c in selected_features]
if len(state_cols) == 0:
    state_cols = selected_features[:3]

z_df = model_df.groupby("time")[state_cols].mean().sort_index()
z_df = z_df.ffill().bfill().fillna(z_df.median())
z_mu = z_df.iloc[train_t_idx].mean()
z_sd = z_df.iloc[train_t_idx].std().replace(0, 1.0)
z_std_df = (z_df - z_mu) / z_sd
z_by_t_all = z_std_df.to_numpy()

print("状态变量:", state_cols)
print("状态矩阵:", z_by_t_all.shape)
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 7) 训练模型与全样本预测
# ===============================
baseline = LogisticRegression(max_iter=400, solver="lbfgs")
baseline.fit(X_train, y_train)

p_base = np.zeros(len(model_df), dtype=float)
p_base[train_mask_arr] = baseline.predict_proba(X_train)[:, 1]
p_base[valid_mask_arr] = baseline.predict_proba(X_valid)[:, 1]
p_base[test_mask_arr] = baseline.predict_proba(X_test)[:, 1]

gas_model = GASDynamicLogit(l2=5e-3, maxiter=420)
gas_model.fit(X_train, y_train, t_idx_all[train_mask_arr], z_by_t_all[train_t_idx])
p_gas, gas_detail_df = gas_model.predict_proba_with_details(
    X_all, y_all, t_idx_all, z_by_t_all, time_order
)

key_feature = borrower_features[0] if len(borrower_features) > 0 else selected_features[0]
key_idx = selected_features.index(key_feature)
t_num_all = np.array([time_to_idx[t] for t in model_df["time"]], dtype=int)

bs_model = BSplineVaryingLogit(n_knots=7, degree=3, lam=1.5, ridge=5e-3, maxiter=260)
bs_model.fit(X_train, y_train, t_num_all[train_mask_arr], X_train[:, key_idx])
p_bs = bs_model.predict_proba(X_all, t_num_all, X_all[:, key_idx])

print("GAS 收敛:", gas_model.success_, gas_model.message_)
print("B-spline 收敛:", bs_model.success_, bs_model.message_)
print("时变关键特征:", key_feature)
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 8) 分阶段集成（稳定期 / 波动期）
# ===============================
if z_by_t_all.shape[0] > 1:
    vol_series = np.r_[0.0, np.mean(np.abs(np.diff(z_by_t_all, axis=0)), axis=1)]
else:
    vol_series = np.zeros(z_by_t_all.shape[0])

vol_df = pd.DataFrame({"time": time_order, "volatility": vol_series})
vol_threshold = float(vol_df.loc[vol_df["time"].isin(train_times), "volatility"].quantile(0.70))
vol_df["regime"] = np.where(vol_df["volatility"] >= vol_threshold, "volatile", "stable")

regime_map = dict(zip(vol_df["time"], vol_df["regime"]))
row_regime = model_df["time"].map(regime_map).to_numpy()

w_grid = np.linspace(0.0, 1.0, 21)
best_w = {"stable": 0.5, "volatile": 0.5}
for regime in ["stable", "volatile"]:
    mask = valid_mask_arr & (row_regime == regime)
    if mask.sum() < 50 or np.unique(model_df.loc[mask, "y"]).size < 2:
        continue
    best_auc = -1.0
    for w in w_grid:
        p_try = w * p_gas[mask] + (1.0 - w) * p_bs[mask]
        auc = safe_auc(model_df.loc[mask, "y"], p_try)
        if auc > best_auc:
            best_auc = auc
            best_w[regime] = float(w)

row_w = np.array([best_w[r] for r in row_regime], dtype=float)
p_ens = row_w * p_gas + (1.0 - row_w) * p_bs

print("波动阈值:", round(vol_threshold, 6))
print("最优权重（GAS占比）:", best_w)
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 9) 核心评估：AUC / KS / PSI
# ===============================
records = []
split_items = [("train", train_mask_arr), ("valid", valid_mask_arr), ("test", test_mask_arr)]
model_items = [("baseline", p_base), ("gas", p_gas), ("bspline", p_bs), ("ensemble", p_ens)]

for model_name, pred in model_items:
    train_ref = pred[train_mask_arr]
    for split_name, split_mask in split_items:
        y_true = model_df.loc[split_mask, "y"].to_numpy()
        y_hat = pred[split_mask]
        psi = 0.0 if split_name == "train" else psi_score(train_ref, y_hat)
        records.append(
            {
                "model": model_name,
                "split": split_name,
                "n": int(split_mask.sum()),
                "default_rate": float(y_true.mean()),
                "auc": safe_auc(y_true, y_hat),
                "ks": ks_score(y_true, y_hat),
                "psi_vs_train": psi,
                "psi_level": psi_level(psi),
            }
        )

result_df = pd.DataFrame(records).sort_values(["model", "split"]).reset_index(drop=True)
result_df
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 10) 季度监控与 GAS 状态联动
# ===============================
q_rows = []
for t, g in model_df.groupby("time"):
    y_t = g["y"].to_numpy()
    if np.unique(y_t).size < 2:
        continue
    idx = g.index.to_numpy()
    q_rows.append(
        {
            "time": t,
            "n": int(len(g)),
            "default_rate": float(y_t.mean()),
            "auc_base": safe_auc(y_t, p_base[idx]),
            "auc_gas": safe_auc(y_t, p_gas[idx]),
            "auc_bs": safe_auc(y_t, p_bs[idx]),
            "auc_ens": safe_auc(y_t, p_ens[idx]),
            "ks_ens": ks_score(y_t, p_ens[idx]),
        }
    )

quarter_eval_df = pd.DataFrame(q_rows).sort_values("time").reset_index(drop=True)
monitor_df = (
    quarter_eval_df.merge(gas_detail_df, on="time", how="left")
    .merge(vol_df, on="time", how="left")
)
monitor_df.tail(10)
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 11) 区域拆分（测试集）
# ===============================
region_rows = []
test_df = model_df.loc[test_mask_arr].copy()
for region_name, g in test_df.groupby("region"):
    y_r = g["y"].to_numpy()
    if np.unique(y_r).size < 2:
        continue
    idx = g.index.to_numpy()
    region_rows.append(
        {
            "region": region_name,
            "n": int(len(g)),
            "default_rate": float(y_r.mean()),
            "auc_base": safe_auc(y_r, p_base[idx]),
            "auc_gas": safe_auc(y_r, p_gas[idx]),
            "auc_bs": safe_auc(y_r, p_bs[idx]),
            "auc_ens": safe_auc(y_r, p_ens[idx]),
            "ks_ens": ks_score(y_r, p_ens[idx]),
        }
    )

region_eval_df = pd.DataFrame(region_rows).sort_values("auc_ens", ascending=False).reset_index(drop=True)
region_eval_df
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 12) B-spline 时变参数曲线
# ===============================
unique_t = np.arange(len(time_order))
intercept_curve, key_slope_curve = bs_model.get_time_effects(unique_t)

bs_effect_df = pd.DataFrame(
    {
        "time": pd.to_datetime(time_order),
        "bs_intercept_effect": intercept_curve,
        "bs_key_slope_effect": key_slope_curve,
    }
)
bs_effect_df.tail(10)
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 13) 图表输出（3张核心图）
# ===============================
fig_auc_trend, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(quarter_eval_df["time"], quarter_eval_df["auc_base"], marker="o", label="baseline")
ax1.plot(quarter_eval_df["time"], quarter_eval_df["auc_gas"], marker="o", label="gas")
ax1.plot(quarter_eval_df["time"], quarter_eval_df["auc_bs"], marker="o", label="bspline")
ax1.plot(quarter_eval_df["time"], quarter_eval_df["auc_ens"], marker="o", label="ensemble")
ax1.set_title("季度AUC趋势对比")
ax1.set_xlabel("季度")
ax1.set_ylabel("AUC")
ax1.legend()
ax1.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

fig_region_bar, ax2 = plt.subplots(figsize=(8, 4))
plot_region = region_eval_df.sort_values("auc_ens", ascending=True)
ax2.barh(plot_region["region"], plot_region["auc_ens"])
ax2.set_title("测试集分区域AUC（集成模型）")
ax2.set_xlabel("AUC")
ax2.set_ylabel("区域")
ax2.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

fig_gas_state, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(monitor_df["time"], monitor_df["gas_state"], marker="o", color="#1f77b4", label="gas_state")
ax3.set_title("GAS时变状态轨迹")
ax3.set_xlabel("季度")
ax3.set_ylabel("状态值")
ax3.grid(alpha=0.3)
ax3.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    code(
        """
# ===============================
# 14) 自动中文结果分析
# 输出：
# - analysis_summary_df
# - analysis_text_lines
# ===============================
analysis_text_lines = []

test_view = result_df[result_df["split"] == "test"].sort_values("auc", ascending=False).reset_index(drop=True)
best_model = str(test_view.loc[0, "model"])
best_auc = float(test_view.loc[0, "auc"])
best_ks = float(test_view.loc[0, "ks"])
analysis_text_lines.append(f"测试集表现最优模型为 {best_model}，AUC={best_auc:.4f}，KS={best_ks:.4f}。")

for _, r in test_view.iterrows():
    analysis_text_lines.append(
        f"模型 {r['model']}：测试AUC={r['auc']:.4f}，KS={r['ks']:.4f}，PSI={r['psi_vs_train']:.4f}（{r['psi_level']}）。"
    )

q_best = quarter_eval_df.loc[quarter_eval_df["auc_ens"].idxmax()]
q_worst = quarter_eval_df.loc[quarter_eval_df["auc_ens"].idxmin()]
analysis_text_lines.append(
    f"季度维度上，集成模型最佳季度为 {q_best['time'].date()}（AUC={q_best['auc_ens']:.4f}），"
    f"较弱季度为 {q_worst['time'].date()}（AUC={q_worst['auc_ens']:.4f}）。"
)

if len(region_eval_df) > 0:
    top_r = region_eval_df.iloc[0]
    low_r = region_eval_df.iloc[-1]
    analysis_text_lines.append(
        f"区域维度上，表现最好区域为 {top_r['region']}（AUC={top_r['auc_ens']:.4f}），"
        f"较弱区域为 {low_r['region']}（AUC={low_r['auc_ens']:.4f}）。"
    )

analysis_text_lines.append("综合来看，动态模型与分阶段集成能更好适应宏观波动下的风险识别需求。")

analysis_summary_df = pd.DataFrame(
    {
        "item": [
            "best_model_test",
            "best_auc_test",
            "best_ks_test",
            "best_quarter_auc_ens",
            "worst_quarter_auc_ens",
        ],
        "value": [
            best_model,
            best_auc,
            best_ks,
            float(q_best["auc_ens"]),
            float(q_worst["auc_ens"]),
        ],
    }
)

print("【自动分析结论】")
for line in analysis_text_lines:
    print("-", line)

print("\\n【摘要表】")
analysis_summary_df
"""
    )
)

cells.append(
    md(
        """
        ## 参数调优与复现建议

        1. GAS 调优优先级：`l2` → `alpha/phi` 边界 → 状态变量组合；
        2. B-spline 调优优先级：`n_knots` → `lam` → 时变关键特征选择；
        3. 若追求更高性能，可将特征筛选与超参搜索拆到离线脚本并缓存中间结果；
        4. 建议将模型类迁移到 `model.py`，推理流程迁移到 `try.py`，形成可复用工程结构。
        """
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path("code/run_model.ipynb")
backup = Path("code/run_model.corrupt_backup.ipynb")

if target.exists():
    shutil.copy2(target, backup)
    print(f"[backup] {backup}")

# 关键：ensure_ascii=True，防止终端编码异常导致中文变问号
target.write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")
print(f"[written] {target} cells={len(cells)}")
