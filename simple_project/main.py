from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simple_project.model import DynamicCreditModel
from simple_project.tools import (
    BORROWER_DATA_PATH,
    GOVERN_DATA_PATH,
    NATIONAL_DATA_PATH,
    OUTPUT_DIR,
    evaluate_predictions,
    load_borrower_sample,
    load_govern_macro,
    load_national_macro,
    merge_macro_with_fallback,
    split_train_test_by_time,
)


def run_pipeline() -> None:
    """
    简化版主流程：
    1) 读取借款样本 + 省级宏观 + 全国宏观；
    2) 宏观合并并处理“无地区样本回退全国宏观”；
    3) 按时间切分 train/test；
    4) model 直接输入 train_df/test_df 训练与预测；
    5) 输出预测结果、指标与元信息。
    """
    print("========== 路径检查 ==========")
    print("借款样本:", BORROWER_DATA_PATH)
    print("省级宏观:", GOVERN_DATA_PATH)
    print("全国宏观:", NATIONAL_DATA_PATH)

    borrower_df = load_borrower_sample(BORROWER_DATA_PATH)
    govern_df = load_govern_macro(GOVERN_DATA_PATH)
    national_df = load_national_macro(NATIONAL_DATA_PATH)

    model_df = merge_macro_with_fallback(borrower_df, govern_df, national_df)

    # 缺失地区回退统计
    fallback_cnt = int((model_df["macro_source"] == "national_macro").sum())
    print("无地区/省份或省级宏观缺失 -> 使用全国宏观样本数:", fallback_cnt)

    train_df, test_df = split_train_test_by_time(model_df, test_periods=3)
    print("train 样本数:", len(train_df), "test 样本数:", len(test_df))

    model = DynamicCreditModel(random_state=42)
    result = model.fit_predict(train_df=train_df, test_df=test_df, target_col="y", time_col="time")

    train_pred_df: pd.DataFrame = result["train_pred_df"]  # type: ignore[assignment]
    test_pred_df: pd.DataFrame = result["test_pred_df"]  # type: ignore[assignment]
    meta: dict[str, str] = result["meta"]  # type: ignore[assignment]

    # 如果测试集有 y，就输出评估指标
    metrics = {}
    if "y" in test_pred_df.columns and test_pred_df["y"].isin([0, 1]).any():
        metrics = evaluate_predictions(
            train_df=train_pred_df,
            test_df=test_pred_df,
            train_pred=train_pred_df["pred_ensemble"].to_numpy(),
            test_pred=test_pred_df["pred_ensemble"].to_numpy(),
            target_col="y",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_out = OUTPUT_DIR / "train_predictions.csv"
    test_out = OUTPUT_DIR / "test_predictions.csv"
    metrics_out = OUTPUT_DIR / "metrics.json"
    meta_out = OUTPUT_DIR / "model_meta.json"

    train_pred_df.to_csv(train_out, index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(test_out, index=False, encoding="utf-8-sig")
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== 结果输出 ==========")
    print("train 预测:", train_out)
    print("test 预测:", test_out)
    print("指标:", metrics_out)
    print("模型信息:", meta_out)
    if metrics:
        print("test AUC:", round(float(metrics.get("auc_test", float("nan"))), 6))
        print("test KS:", round(float(metrics.get("ks_test", float("nan"))), 6))
        print("test vs train PSI:", round(float(metrics.get("psi_test_vs_train", float("nan"))), 6))


if __name__ == "__main__":
    run_pipeline()

