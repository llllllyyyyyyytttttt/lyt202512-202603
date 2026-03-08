from __future__ import annotations

import numpy as np
import pandas as pd

from simple_project.model import DynamicCreditModel
from simple_project.tools import evaluate_predictions


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def build_demo_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    手动构造最小可运行示例数据（train/test DataFrame）。

    必需列：
    - time: 时间列（可转为 datetime）
    - y: train 必需，test 可选（这里保留，便于演示评估）
    - 至少一个 x_ 开头特征（这里给 3 个）

    可选列：
    - user_id / region / province（这里也给出，便于对齐真实项目）
    """
    rng = np.random.default_rng(seed)

    # 训练集：6 个季度，每季度 20 条，总计 120 条
    train_quarters = pd.to_datetime(
        ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30"]
    )
    train_times = np.repeat(train_quarters, 20)
    n_train = train_times.size

    # 测试集：2 个季度，每季度 20 条，总计 40 条
    test_quarters = pd.to_datetime(["2025-09-30", "2025-12-31"])
    test_times = np.repeat(test_quarters, 20)
    n_test = test_times.size

    # 构造 3 个行为特征（x_ 开头）
    x1_train = rng.normal(loc=0.0, scale=1.0, size=n_train)
    x2_train = rng.normal(loc=0.0, scale=1.2, size=n_train)
    x3_train = rng.normal(loc=0.2, scale=0.8, size=n_train)

    x1_test = rng.normal(loc=0.2, scale=1.0, size=n_test)
    x2_test = rng.normal(loc=-0.1, scale=1.2, size=n_test)
    x3_test = rng.normal(loc=0.3, scale=0.8, size=n_test)

    # 用逻辑模型模拟标签，保证是 0/1
    # 这里加入简单时间效应，让模型更接近真实时变场景
    t_eff_train = np.array([(t.year - 2024) * 0.12 + t.quarter * 0.03 for t in train_times])
    t_eff_test = np.array([(t.year - 2024) * 0.12 + t.quarter * 0.03 for t in test_times])

    p_train = _sigmoid(-1.0 + 1.1 * x1_train - 0.8 * x2_train + 0.5 * x3_train + t_eff_train)
    p_test = _sigmoid(-1.0 + 1.1 * x1_test - 0.8 * x2_test + 0.5 * x3_test + t_eff_test)

    y_train = rng.binomial(1, p_train).astype(int)
    y_test = rng.binomial(1, p_test).astype(int)

    # region/province 故意给空，演示“直接喂 model”最小输入并不依赖地区列
    train_df = pd.DataFrame(
        {
            "user_id": [f"TR_{i:04d}" for i in range(n_train)],
            "time": train_times,
            "region": [np.nan] * n_train,
            "province": [np.nan] * n_train,
            "x_001": x1_train,
            "x_002": x2_train,
            "x_003": x3_train,
            "y": y_train,
        }
    )

    test_df = pd.DataFrame(
        {
            "user_id": [f"TE_{i:04d}" for i in range(n_test)],
            "time": test_times,
            "region": [np.nan] * n_test,
            "province": [np.nan] * n_test,
            "x_001": x1_test,
            "x_002": x2_test,
            "x_003": x3_test,
            "y": y_test,  # 演示评估时保留；若线上推理可去掉此列
        }
    )
    return train_df, test_df


def main() -> None:
    # 1) 手工造数据（不读取任何 CSV）
    train_df, test_df = build_demo_data(seed=2026)

    # 2) 直接把 train/test DataFrame 输入模型
    model = DynamicCreditModel(random_state=42)
    output = model.fit_predict(train_df=train_df, test_df=test_df, target_col="y", time_col="time")

    train_pred_df: pd.DataFrame = output["train_pred_df"]  # type: ignore[assignment]
    test_pred_df: pd.DataFrame = output["test_pred_df"]  # type: ignore[assignment]
    meta: dict[str, str] = output["meta"]  # type: ignore[assignment]

    # 3) 评估演示
    metrics = evaluate_predictions(
        train_df=train_pred_df,
        test_df=test_pred_df,
        train_pred=train_pred_df["pred_ensemble"].to_numpy(),
        test_pred=test_pred_df["pred_ensemble"].to_numpy(),
        target_col="y",
    )

    print("===== demo_run: 模型信息 =====")
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\n===== demo_run: 指标 =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n===== demo_run: 预测结果示例（test 前5行） =====")
    show_cols = ["user_id", "time", "y", "pred_base", "pred_gas", "pred_bs", "pred_ensemble"]
    print(test_pred_df[show_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()

