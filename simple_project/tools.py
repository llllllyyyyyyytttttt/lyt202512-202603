from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ==============================
# 固定路径配置（直接使用项目路径，不再 find_file）
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

BORROWER_DATA_PATH = DATA_DIR / "data1_\u94f6\u8054" / "model_sample_with_time_region.csv"
GOVERN_DATA_PATH = DATA_DIR / "china_globaldata" / "govern_data.csv"
NATIONAL_DATA_PATH = DATA_DIR / "china_globaldata" / "national_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "simple_project" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# 基础指标函数
# ==============================
def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUC 安全计算：若标签单一，返回 NaN。"""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def ks_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """KS 指标计算。"""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(np.abs(tpr - fpr)))


def psi_score(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI 计算（用 expected 分箱）。"""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return float("nan")
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=cuts)[0].astype(float)
    a = np.histogram(actual, bins=cuts)[0].astype(float)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ==============================
# 时间处理函数
# ==============================
def quarter_text_to_date(text: str) -> pd.Timestamp:
    """
    把“2025年第三季度”转换为季度末日期。
    """
    text = str(text)
    m = re.match(r"(\d{4})\s*年\s*第?([一二三四1234])季度", text)
    if not m:
        return pd.NaT
    year = int(m.group(1))
    q_map = {"一": 1, "二": 2, "三": 3, "四": 4, "1": 1, "2": 2, "3": 3, "4": 4}
    q = q_map[m.group(2)]
    md_map = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    return pd.Timestamp(f"{year}-{md_map[q]}")


# ==============================
# 数据加载函数
# ==============================
def load_borrower_sample(path: Path = BORROWER_DATA_PATH) -> pd.DataFrame:
    """
    读取借款样本并标准化字段：
    - user_id, y, time, region, province
    - x_ 开头特征原样保留
    """
    df = pd.read_csv(path)
    x_cols = [c for c in df.columns if c.startswith("x_")]
    meta_cols = [c for c in df.columns if c not in x_cols]

    if len(meta_cols) < 2:
        raise ValueError("借款数据缺少基础字段，至少需要 ID 与目标变量。")

    rename_map: dict[str, str] = {
        meta_cols[0]: "user_id",
        ("y" if "y" in meta_cols else meta_cols[1]): "y",
    }
    if len(meta_cols) > 2:
        rename_map[meta_cols[2]] = "time"
    if len(meta_cols) > 3:
        rename_map[meta_cols[3]] = "region"
    if len(meta_cols) > 4:
        rename_map[meta_cols[4]] = "province"

    df = df.rename(columns=rename_map)

    # 若缺少字段，补空列，保证后续流程稳定
    for c in ["user_id", "y", "time", "region", "province"]:
        if c not in df.columns:
            df[c] = np.nan

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df


def load_govern_macro(path: Path = GOVERN_DATA_PATH) -> pd.DataFrame:
    """
    读取省级宏观数据并重命名为 gov_macro_XX。
    """
    g = pd.read_csv(path, encoding="utf-8")
    tcol = g.columns[0]
    g[tcol] = pd.to_datetime(g[tcol], errors="coerce")
    g = g.rename(columns={tcol: "time"})

    if "province" not in g.columns:
        raise ValueError("省级宏观数据缺少 province 列。")
    if "region" not in g.columns:
        g["region"] = np.nan

    metric_cols = [c for c in g.columns if c not in ["time", "province", "region"]]
    rename_map = {c: f"gov_macro_{i:02d}" for i, c in enumerate(metric_cols, 1)}
    g = g.rename(columns=rename_map)
    keep_cols = ["time", "province", "region"] + list(rename_map.values())
    return g[keep_cols]


def load_national_macro(path: Path = NATIONAL_DATA_PATH) -> pd.DataFrame:
    """
    读取全国宏观数据并重命名为 nat_macro_XX。
    文件前两行是说明，实际表头从第3行开始。
    """
    n = pd.read_csv(path, encoding="gbk", skiprows=2)
    tcol = n.columns[0]
    n[tcol] = n[tcol].map(quarter_text_to_date)
    n = n.rename(columns={tcol: "time"})

    numeric_cols: list[str] = []
    for c in n.columns:
        if c == "time":
            continue
        s = pd.to_numeric(n[c], errors="coerce")
        if s.notna().sum() >= 8:
            n[c] = s
            numeric_cols.append(c)

    rename_map = {c: f"nat_macro_{i:02d}" for i, c in enumerate(numeric_cols, 1)}
    n = n.rename(columns=rename_map)
    keep_cols = ["time"] + list(rename_map.values())
    return n[keep_cols]


def merge_macro_with_fallback(
    borrower_df: pd.DataFrame,
    govern_df: pd.DataFrame,
    national_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    宏观数据合并逻辑：
    1. 先按 time + province 合并省级宏观；
    2. 再按 time 合并全国宏观；
    3. 对“没有地区/省份”或“省级宏观缺失”的样本，用全国宏观回填。
    """
    df = borrower_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    merged = df.merge(govern_df, on=["time", "province"], how="left", suffixes=("", "_gov"))
    merged = merged.merge(national_df, on="time", how="left")

    gov_cols = sorted([c for c in merged.columns if c.startswith("gov_macro_")])
    nat_cols = sorted([c for c in merged.columns if c.startswith("nat_macro_")])
    pair_cnt = min(len(gov_cols), len(nat_cols))

    # 没有地区信息时，必须使用全国宏观
    no_region_mask = merged["region"].isna() | merged["province"].isna()
    merged["macro_source"] = "province_macro"
    merged.loc[no_region_mask, "macro_source"] = "national_macro"

    # 使用全国宏观做回填
    for i in range(pair_cnt):
        gcol = gov_cols[i]
        ncol = nat_cols[i]
        merged.loc[no_region_mask, gcol] = merged.loc[no_region_mask, ncol]
        merged[gcol] = merged[gcol].fillna(merged[ncol])

    return merged


def split_train_test_by_time(df: pd.DataFrame, test_periods: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间切分 train/test（默认最后3个季度做测试）。
    """
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    times = np.sort(data["time"].dropna().unique())
    if len(times) <= test_periods:
        raise ValueError("时间点数量不足，无法按当前 test_periods 切分。")

    train_times = times[:-test_periods]
    test_times = times[-test_periods:]

    train_df = data[data["time"].isin(train_times)].copy()
    test_df = data[data["time"].isin(test_times)].copy()
    return train_df, test_df


def evaluate_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    target_col: str = "y",
) -> dict[str, Any]:
    """
    计算测试集评估指标，并附上 PSI（test vs train）。
    """
    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    return {
        "auc_test": safe_auc(y_test, test_pred),
        "ks_test": ks_score(y_test, test_pred),
        "psi_test_vs_train": psi_score(train_pred, test_pred),
        "default_rate_train": float(np.nanmean(y_train)),
        "default_rate_test": float(np.nanmean(y_test)),
    }

