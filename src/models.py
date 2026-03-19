"""
src/models.py
Model definitions and training routines.

Models
------
LightGBMModel  — LightGBM, flat features, tabular benchmark

Common interface:
    train(...)           train model
    predict(X)           return 0/1 array
    predict_proba(X)     return float array in [0, 1]
    save(path)           persist model to disk
    load(path)           restore from disk

model_type attribute
    "flat"  — expects 2-D input  (n_samples, n_features)
"""

import pickle
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMModel:
    """LightGBM gradient-boosted tree classifier.

    Tree models are scale-invariant; no scaler is applied.
    Uses early stopping on the validation set when provided.
    """

    model_type = "flat"

    def __init__(self, **lgbm_kwargs):
        self.params = {**LGBM_PARAMS, **lgbm_kwargs}
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int = 30,
    ) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("Run: pip install lightgbm") from exc

        print("[LGBM] Training …")

        callbacks = [lgb.log_evaluation(period=50)]

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_train,
            y_train,
            callbacks=callbacks,
            **fit_kwargs,
        )
        print(f"[LGBM] Best iteration: {self.model.best_iteration_}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[LGBM] Saved → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[LGBM] Loaded ← {path}")

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "lgbm": LightGBMModel,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of: "lgbm".
    **kwargs
        Passed to the model constructor.

    Returns
    -------
    LightGBMModel instance.
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
