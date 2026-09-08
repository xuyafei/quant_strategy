"""
机器学习打分因子。

本模块把已有多因子面板转换成监督学习样本，用未来收益作为标签，
按时间滚动训练梯度提升类模型，并输出每个 (date, symbol) 的 `ML_SCORE`。
它只生成候选因子，不直接替代回测、风控或实盘执行逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backtest.backtest_utils import prices_to_wide_close
from config import Settings


ML_SCORE_NAME = "ML_SCORE"


@dataclass(frozen=True)
class MLScoreConfig:
    model: str
    forward_days: int
    train_lookback_days: int
    min_train_days: int
    min_train_rows: int
    refit_every_days: int
    random_state: int


def _settings_to_ml_config(settings: Settings) -> MLScoreConfig:
    return MLScoreConfig(
        model=str(getattr(settings, "ml_score_model", "hist_gradient_boosting")),
        forward_days=max(1, int(getattr(settings, "ml_score_forward_days", 20))),
        train_lookback_days=max(1, int(getattr(settings, "ml_score_train_lookback_days", 252))),
        min_train_days=max(1, int(getattr(settings, "ml_score_min_train_days", 60))),
        min_train_rows=max(1, int(getattr(settings, "ml_score_min_train_rows", 100))),
        refit_every_days=max(1, int(getattr(settings, "ml_score_refit_every_days", 20))),
        random_state=int(getattr(settings, "ml_score_random_state", 42)),
    )


def _make_regressor(config: MLScoreConfig) -> tuple[Any, str]:
    """按配置创建回归模型；缺少可选依赖时回退到 sklearn 实现。"""
    model = config.model.strip().lower()
    choices = ["lightgbm", "catboost", "xgboost", "hist_gradient_boosting"]
    if model == "auto":
        choices = ["lightgbm", "catboost", "xgboost", "hist_gradient_boosting"]
    elif model in choices:
        choices = [model]
    else:
        choices = ["hist_gradient_boosting"]

    for choice in choices:
        if choice == "lightgbm":
            try:
                from lightgbm import LGBMRegressor  # type: ignore

                return (
                    LGBMRegressor(
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=4,
                        num_leaves=15,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=config.random_state,
                        verbose=-1,
                    ),
                    "lightgbm",
                )
            except Exception:
                continue
        if choice == "catboost":
            try:
                from catboost import CatBoostRegressor  # type: ignore

                return (
                    CatBoostRegressor(
                        iterations=80,
                        learning_rate=0.05,
                        depth=4,
                        loss_function="RMSE",
                        random_seed=config.random_state,
                        verbose=False,
                    ),
                    "catboost",
                )
            except Exception:
                continue
        if choice == "xgboost":
            try:
                from xgboost import XGBRegressor  # type: ignore

                return (
                    XGBRegressor(
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=config.random_state,
                        n_jobs=1,
                    ),
                    "xgboost",
                )
            except Exception:
                continue

    from sklearn.ensemble import HistGradientBoostingRegressor

    return (
        HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.01,
            random_state=config.random_state,
        ),
        "hist_gradient_boosting",
    )


def forward_return_label(
    prices: pd.DataFrame,
    *,
    forward_days: int,
    price_col: str = "close",
) -> pd.Series:
    """生成未来 N 日收益标签，索引为 MultiIndex(date, symbol)。"""
    wide = prices_to_wide_close(
        prices,
        date_col="trade_date",
        symbol_col="ts_code",
        close_col=price_col,
    ).sort_index()
    future = wide.shift(-forward_days) / wide - 1.0
    out = future.stack()
    out.index = out.index.set_names(["date", "symbol"])
    out.name = "forward_return"
    return out.sort_index()


def _date_mask(panel: pd.DataFrame, dates: pd.Index) -> np.ndarray:
    return panel.index.get_level_values("date").isin(pd.Index(dates))


def _prepare_matrix(
    frame: pd.DataFrame,
    feature_cols: list[str],
    medians: pd.Series | None = None,
) -> tuple[np.ndarray, pd.Series]:
    x = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(axis=0, skipna=True).fillna(0.0)
    x = x.fillna(medians).fillna(0.0)
    return x.to_numpy(dtype=float), medians


def _valid_feature_columns(panel: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    cols = [str(c) for c in (feature_cols or list(panel.columns)) if str(c) in panel.columns]
    out: list[str] = []
    for col in cols:
        ser = panel[col].replace([np.inf, -np.inf], np.nan)
        if ser.notna().sum() > 0:
            out.append(col)
    return out


def build_ml_score_factor(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    *,
    feature_cols: list[str] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    构造机器学习打分因子。

    训练约束：
    - 预测日为 t；
    - 标签为 feature_date 之后 `forward_days` 的未来收益；
    - 训练集只使用 feature_date + forward_days <= t 的样本。

    返回：
    - `ML_SCORE` Series，索引与 panel 对齐；
    - 训练日志 DataFrame，记录每次 refit 的训练窗口和模型后端。
    """
    if panel.empty or not isinstance(panel.index, pd.MultiIndex):
        return pd.Series(index=panel.index, dtype=float, name=ML_SCORE_NAME), pd.DataFrame()

    cfg = _settings_to_ml_config(settings)
    features = _valid_feature_columns(panel, feature_cols)
    if not features:
        return pd.Series(index=panel.index, dtype=float, name=ML_SCORE_NAME), pd.DataFrame()

    panel = panel.sort_index()
    labels = forward_return_label(
        prices,
        forward_days=cfg.forward_days,
        price_col=settings.price_col,
    ).reindex(panel.index)
    dataset = panel[features].copy()
    dataset["__label__"] = labels

    price_dates = pd.Index(pd.to_datetime(prices.index)).sort_values()
    panel_dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    dates = price_dates.intersection(panel_dates)
    if len(dates) <= cfg.forward_days + cfg.min_train_days:
        return pd.Series(index=panel.index, dtype=float, name=ML_SCORE_NAME), pd.DataFrame()

    pred = pd.Series(index=panel.index, dtype=float, name=ML_SCORE_NAME)
    logs: list[dict[str, Any]] = []
    model_obj: Any | None = None
    model_backend = ""
    medians: pd.Series | None = None
    next_refit_pos = 0

    for pos, dt in enumerate(dates):
        cutoff_pos = pos - cfg.forward_days
        if cutoff_pos < cfg.min_train_days:
            continue

        should_refit = model_obj is None or pos >= next_refit_pos
        if should_refit:
            start_pos = max(0, cutoff_pos - cfg.train_lookback_days + 1)
            train_dates = dates[start_pos : cutoff_pos + 1]
            train = dataset.loc[_date_mask(dataset, train_dates)].dropna(subset=["__label__"])
            train = train.dropna(how="all", subset=features)
            if len(train_dates) < cfg.min_train_days or len(train) < cfg.min_train_rows:
                continue
            x_train, medians = _prepare_matrix(train, features)
            y_train = train["__label__"].to_numpy(dtype=float)
            if not np.isfinite(y_train).any():
                continue
            try:
                model_obj, model_backend = _make_regressor(cfg)
                model_obj.fit(x_train, y_train)
            except Exception:
                model_obj = None
                medians = None
                continue
            logs.append(
                {
                    "prediction_date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "train_start": pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
                    "train_end": pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"),
                    "label_forward_days": cfg.forward_days,
                    "n_train_days": int(len(train_dates)),
                    "n_train_rows": int(len(train)),
                    "n_features": int(len(features)),
                    "features": ",".join(features),
                    "configured_model": cfg.model,
                    "model_backend": model_backend,
                }
            )
            next_refit_pos = pos + cfg.refit_every_days

        if model_obj is None or medians is None:
            continue
        pred_frame = dataset.loc[_date_mask(dataset, pd.Index([dt])), features]
        # 时点股票池掩码会让非成员的全部基础特征均为 NaN。此类行不能用
        # 训练期中位数“造出”预测，否则 ML_SCORE 会重新把非成分股带回候选池。
        pred_frame = pred_frame.loc[pred_frame.notna().any(axis=1)]
        if pred_frame.empty:
            continue
        x_pred, _ = _prepare_matrix(pred_frame, features, medians=medians)
        try:
            pred.loc[pred_frame.index] = model_obj.predict(x_pred)
        except Exception:
            continue

    log_frame = pd.DataFrame(logs)
    return pred.sort_index(), log_frame
