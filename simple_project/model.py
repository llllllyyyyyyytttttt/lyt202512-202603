from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer


EPS = 1e-9


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


class GASDynamicLogit:
    """
    简化版 GAS 时变参数 Logit（时变截距）：
        eta_{i,t} = x_{i,t}' beta + f_t
        f_t = omega + gamma' z_t + alpha * score_{t-1} + phi * f_{t-1}
    """

    def __init__(self, l2: float = 5e-3, maxiter: int = 320):
        self.l2 = l2
        self.maxiter = maxiter

    @staticmethod
    def _split(theta: np.ndarray, p: int, m: int):
        beta = theta[:p]
        gamma = theta[p : p + m]
        omega = theta[p + m]
        alpha = theta[p + m + 1]
        phi = theta[p + m + 2]
        return beta, gamma, omega, alpha, phi

    @staticmethod
    def _build_idx_cache(t_idx: np.ndarray, T: int) -> list[np.ndarray]:
        return [np.where(t_idx == t)[0] for t in range(T)]

    def _forward(
        self,
        X: np.ndarray,
        y_for_state: np.ndarray,
        z_by_t: np.ndarray,
        theta: np.ndarray,
        idx_cache: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = X.shape[1]
        m = z_by_t.shape[1]
        beta, gamma, omega, alpha, phi = self._split(theta, p, m)
        T = z_by_t.shape[0]

        pred = np.zeros(X.shape[0], dtype=float)
        states = np.full(T, np.nan, dtype=float)
        scores = np.full(T, np.nan, dtype=float)

        # 使用长期均值初始化
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

    def _loss(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray, z_by_t: np.ndarray) -> float:
        pred, _, _ = self._forward(X, y, z_by_t, theta, self._fit_idx_cache)
        nll = -np.sum(y * np.log(pred + EPS) + (1.0 - y) * np.log(1.0 - pred + EPS))
        p = X.shape[1]
        m = z_by_t.shape[1]
        beta, gamma, _, _, _ = self._split(theta, p, m)
        penalty = self.l2 * (np.sum(beta**2) + np.sum(gamma**2))
        return float(nll + penalty)

    def fit(self, X: np.ndarray, y: np.ndarray, t_idx: np.ndarray, z_by_t: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        t_idx = np.asarray(t_idx, dtype=int)
        z_by_t = np.asarray(z_by_t, dtype=float)

        p = X.shape[1]
        m = z_by_t.shape[1]
        T = z_by_t.shape[0]
        self._fit_idx_cache = self._build_idx_cache(t_idx, T)

        # 用静态逻辑回归初始化 beta
        base = LogisticRegression(max_iter=300, solver="lbfgs")
        base.fit(X, y)
        beta0 = base.coef_.reshape(-1)
        y_bar = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        omega0 = float(np.log(y_bar / (1.0 - y_bar)))
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
        return self

    def predict_proba(
        self,
        X: np.ndarray,
        y_for_state: np.ndarray,
        t_idx: np.ndarray,
        z_by_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y_for_state = np.asarray(y_for_state, dtype=float)
        t_idx = np.asarray(t_idx, dtype=int)
        z_by_t = np.asarray(z_by_t, dtype=float)
        idx_cache = self._build_idx_cache(t_idx, z_by_t.shape[0])
        return self._forward(X, y_for_state, z_by_t, self.theta_, idx_cache)


class BSplineVaryingLogit:
    """
    简化版 B-spline 变系数 Logit：
        eta = X beta + B(t) theta0 + key*B(t) theta1
    """

    def __init__(
        self,
        n_knots: int = 7,
        degree: int = 3,
        lam: float = 1.2,
        ridge: float = 5e-3,
        maxiter: int = 220,
    ):
        self.n_knots = n_knots
        self.degree = degree
        self.lam = lam
        self.ridge = ridge
        self.maxiter = maxiter

    def _build_design(
        self,
        X: np.ndarray,
        t_num: np.ndarray,
        key_feature: np.ndarray,
        fit: bool = False,
    ) -> tuple[np.ndarray, int]:
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

    def _loss(self, w: np.ndarray, Xd: np.ndarray, y: np.ndarray, p: int, K: int) -> float:
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

    def fit(self, X: np.ndarray, y: np.ndarray, t_num: np.ndarray, key_feature: np.ndarray):
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

    def predict_proba(self, X: np.ndarray, t_num: np.ndarray, key_feature: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xd, _ = self._build_design(X, t_num, key_feature, fit=False)
        return sigmoid(Xd.dot(self.w_))


class DynamicCreditModel:
    """
    项目主模型（简化版）：
    - 直接输入 train_df / test_df
    - 输出 train/test 预测结果与模型信息
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"输入数据缺少必要字段: {miss}")

    def _build_feature_columns(self, train_df: pd.DataFrame) -> list[str]:
        # 模型特征优先用 x_ 开头字段 + 宏观字段
        x_cols = [c for c in train_df.columns if c.startswith("x_")]
        macro_cols = sorted([c for c in train_df.columns if c.startswith("gov_macro_") or c.startswith("nat_macro_")])
        base_cols = x_cols + macro_cols
        if len(base_cols) == 0:
            raise ValueError("未找到可用特征（至少需要 x_ 或宏观字段）。")

        # 只保留训练集里有波动的特征
        valid_cols: list[str] = []
        for c in base_cols:
            s = pd.to_numeric(train_df[c], errors="coerce")
            if s.notna().sum() < 50:
                continue
            if s.var(skipna=True) <= 1e-10:
                continue
            valid_cols.append(c)

        if len(valid_cols) == 0:
            raise ValueError("特征过滤后为空，请检查输入数据。")
        return valid_cols

    @staticmethod
    def _build_time_index(series: pd.Series) -> tuple[np.ndarray, np.ndarray, dict[pd.Timestamp, int]]:
        ts = pd.to_datetime(series, errors="coerce")
        order = np.sort(ts.dropna().unique())
        mapper = {t: i for i, t in enumerate(order)}
        t_idx = ts.map(mapper).to_numpy()
        return ts.to_numpy(), t_idx, mapper

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "y",
        time_col: str = "time",
    ) -> dict[str, pd.DataFrame | dict[str, str]]:
        """
        直接输入 train_df / test_df，返回预测结果。

        返回：
        - train_pred_df: 训练集预测
        - test_pred_df: 测试集预测
        - meta: 训练信息（特征数、收敛状态等）
        """
        train = train_df.copy()
        test = test_df.copy()

        self._check_required_columns(train, [target_col, time_col])
        self._check_required_columns(test, [time_col])

        train[time_col] = pd.to_datetime(train[time_col], errors="coerce")
        test[time_col] = pd.to_datetime(test[time_col], errors="coerce")

        # 训练集必须有 y；测试集可以有也可以没有
        train[target_col] = pd.to_numeric(train[target_col], errors="coerce")
        train = train[train[target_col].isin([0, 1])].copy()
        if train.empty:
            raise ValueError("训练集没有有效 y（必须为 0/1）。")

        features = self._build_feature_columns(train)
        self.feature_cols_ = features

        # 对齐 test 特征列（缺失列补 NaN）
        for c in features:
            if c not in test.columns:
                test[c] = np.nan

        # 预处理：impute + scale
        X_train = self.scaler.fit_transform(self.imputer.fit_transform(train[features]))
        X_test = self.scaler.transform(self.imputer.transform(test[features]))
        y_train = train[target_col].to_numpy()

        # Baseline：静态逻辑回归
        self.base_model_ = LogisticRegression(max_iter=400, solver="lbfgs", random_state=self.random_state)
        self.base_model_.fit(X_train, y_train)
        p_base_train = self.base_model_.predict_proba(X_train)[:, 1]
        p_base_test = self.base_model_.predict_proba(X_test)[:, 1]

        # 构建全样本（train+test）用于时序动态预测
        train["_is_train_"] = 1
        test["_is_train_"] = 0
        all_df = pd.concat([train, test], axis=0, ignore_index=True)
        all_df = all_df.sort_values(time_col).reset_index(drop=True)

        X_all = self.scaler.transform(self.imputer.transform(all_df[features]))
        _, t_idx_all, time_mapper = self._build_time_index(all_df[time_col])
        self.time_mapper_ = time_mapper

        # GAS 状态变量：优先省级宏观，否则 national 宏观
        state_cols = [c for c in features if c.startswith("gov_macro_")][:4]
        if len(state_cols) == 0:
            state_cols = [c for c in features if c.startswith("nat_macro_")][:4]
        if len(state_cols) == 0:
            state_cols = features[:3]

        z_df = all_df.groupby(time_col)[state_cols].mean().sort_index()
        z_df = z_df.ffill().bfill().fillna(z_df.median())
        z_all = z_df.to_numpy()

        # GAS 拟合只用 train 时段
        train_times = np.sort(train[time_col].dropna().unique())
        train_time_map = {t: i for i, t in enumerate(train_times)}
        t_idx_train = train[time_col].map(train_time_map).to_numpy()
        z_train = z_df.loc[train_times].to_numpy()

        self.gas_model_ = GASDynamicLogit()
        self.gas_model_.fit(X_train, y_train, t_idx_train, z_train)

        # test 没有真实标签时，使用 baseline 预测转 0/1 作为递推状态
        y_state_all = np.zeros(len(all_df), dtype=float)
        # 先填 train 真实标签
        train_mask_all = all_df["_is_train_"].to_numpy() == 1
        y_state_all[train_mask_all] = all_df.loc[train_mask_all, target_col].to_numpy()
        # 再填 test 伪标签
        test_mask_all = ~train_mask_all
        p_base_all = self.base_model_.predict_proba(X_all)[:, 1]
        y_state_all[test_mask_all] = (p_base_all[test_mask_all] >= 0.5).astype(float)

        p_gas_all, _, _ = self.gas_model_.predict_proba(X_all, y_state_all, t_idx_all, z_all)

        # B-spline 模型
        key_idx = 0
        key_train = X_train[:, key_idx]
        t_train_num = train[time_col].map(train_time_map).to_numpy()
        self.bs_model_ = BSplineVaryingLogit()
        self.bs_model_.fit(X_train, y_train, t_train_num, key_train)

        # 对 all 时段预测
        all_times = np.sort(all_df[time_col].dropna().unique())
        all_time_map = {t: i for i, t in enumerate(all_times)}
        t_all_num = all_df[time_col].map(all_time_map).to_numpy()
        p_bs_all = self.bs_model_.predict_proba(X_all, t_all_num, X_all[:, key_idx])

        # 简化集成（固定权重）
        p_ens_all = 0.5 * p_gas_all + 0.5 * p_bs_all

        # 拆回 train/test 顺序
        all_df = all_df.assign(
            pred_base=p_base_all,
            pred_gas=p_gas_all,
            pred_bs=p_bs_all,
            pred_ensemble=p_ens_all,
        )

        train_pred_df = all_df[all_df["_is_train_"] == 1].copy()
        test_pred_df = all_df[all_df["_is_train_"] == 0].copy()

        # 恢复原 test 行顺序（按 time 可能已经重排）
        test_pred_df = test_pred_df.sort_index()

        meta = {
            "feature_count": str(len(features)),
            "gas_converged": str(self.gas_model_.success_),
            "gas_message": self.gas_model_.message_,
            "bs_converged": str(self.bs_model_.success_),
            "bs_message": self.bs_model_.message_,
            "state_cols": ",".join(state_cols),
        }

        return {
            "train_pred_df": train_pred_df,
            "test_pred_df": test_pred_df,
            "meta": meta,
        }

