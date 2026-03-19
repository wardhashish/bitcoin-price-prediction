"""
src/evaluate.py
Evaluation utilities for all models and horizons.

Public API
----------
chronological_split(df, train, val)      -> (train_df, val_df, test_df)
split_windows(X, y, train, val)          -> ((Xtr,ytr), (Xvl,yvl), (Xte,yte))
compute_metrics(y_true, y_pred, y_prob)  -> dict
evaluate_model(model, X_test, y_test)    -> dict
plot_confusion_matrix(y_true, y_pred, title, ax)
plot_roc_curves(results, ax)
plot_calibration(results, ax)
aggregate_results(all_results)           -> pd.DataFrame
print_results_table(df)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-series DataFrame chronologically into train / val / test.

    No shuffling — strict temporal order is preserved.

    Parameters
    ----------
    df : pd.DataFrame
    train_ratio : float
        Fraction of rows for training (default 0.70).
    val_ratio : float
        Fraction of rows for validation (default 0.15).
        Test gets the remainder (~0.15).

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(
        f"[eval] Split → train {len(train):,}  val {len(val):,}  test {len(test):,}  "
        f"({train.index[0].date()} / {val.index[0].date()} / {test.index[0].date()})"
    )
    return train, val, test


def split_windows(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """Split pre-built windowed arrays chronologically.

    Parameters
    ----------
    X : np.ndarray, shape (n, window, features)  or  (n, features)
    y : np.ndarray, shape (n,)

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = (
        (X[:train_end],       y[:train_end]),
        (X[train_end:val_end], y[train_end:val_end]),
        (X[val_end:],          y[val_end:]),
    )
    names = ("train", "val", "test")
    for name, (Xs, ys) in zip(names, splits):
        print(f"[eval]   {name:5s}: X={Xs.shape}  y={ys.shape}  "
              f"pos_rate={ys.mean():.3f}")
    return splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute binary classification metrics.

    Parameters
    ----------
    y_true : array of 0/1
    y_pred : array of 0/1
    y_prob : array of float in [0, 1]  (probability of class 1)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


def evaluate_model(model, X_test, y_test: np.ndarray) -> dict:
    """Run a trained model on the test set and return metrics + raw outputs.

    Works with flat (LR/LGBM) and windowed (LSTM) inputs.
    For TFT, pass a feature DataFrame as X_test.

    Parameters
    ----------
    model
        Any model with .predict() and .predict_proba() methods.
    X_test
        2-D ndarray, 3-D ndarray, or DataFrame depending on model_type.
    y_test : np.ndarray

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, y_pred, y_prob
    """
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    ax=None,
) -> "matplotlib.axes.Axes":
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : 0/1 arrays
    title : str
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_roc_curves(
    results: dict[str, dict],
    ax=None,
    title: str = "ROC Curves",
) -> "matplotlib.axes.Axes":
    """Plot ROC curves for multiple models on one axes.

    Parameters
    ----------
    results : dict
        {model_name: {"y_prob": ..., "roc_auc": ..., "y_true": ...}}
    ax : matplotlib Axes or None

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_calibration(
    results: dict[str, dict],
    ax=None,
    n_bins: int = 10,
    title: str = "Calibration Curves",
) -> "matplotlib.axes.Axes":
    """Plot probability calibration curves for multiple models.

    Parameters
    ----------
    results : dict
        {model_name: {"y_prob": ..., "y_true": ...}}
    ax : matplotlib Axes or None
    n_bins : int

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    for name, res in results.items():
        frac_pos, mean_pred = calibration_curve(
            res["y_true"], res["y_prob"], n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_pred, frac_pos, marker="o", markersize=4, label=name)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    ax=None,
) -> "matplotlib.axes.Axes":
    """Horizontal bar chart of top-N feature importances (LightGBM or similar).

    Parameters
    ----------
    importances : np.ndarray
    feature_names : list[str]
    top_n : int
    title : str
    ax : matplotlib Axes or None
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(4, top_n * 0.35)))

    idx = np.argsort(importances)[-top_n:]
    ax.barh(np.array(feature_names)[idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)
    return ax


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    all_results: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """Aggregate per-model, per-horizon metrics into a tidy DataFrame.

    Parameters
    ----------
    all_results : dict
        Structure: {horizon: {model_name: metrics_dict}}
        e.g. {"5m": {"lr": {"accuracy": 0.52, ...}, "lgbm": {...}}, "30m": ...}

    Returns
    -------
    pd.DataFrame
        MultiIndex (horizon, model) with metric columns.
    """
    rows = []
    for horizon, models in all_results.items():
        for model_name, metrics in models.items():
            row = {"horizon": horizon, "model": model_name}
            row.update({k: v for k, v in metrics.items()
                        if k not in ("y_pred", "y_prob", "y_true")})
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["horizon", "model"])
    return df


def print_results_table(df: pd.DataFrame) -> None:
    """Pretty-print the aggregated results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`aggregate_results`.
    """
    try:
        from tabulate import tabulate
        print(tabulate(df.reset_index(), headers="keys", tablefmt="github",
                       floatfmt=".4f"))
    except ImportError:
        print(df.to_string())
