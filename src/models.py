"""
src/models.py
Model definitions and training routines.

Models
------
LogisticRegressionModel  — scikit-learn, flat features, sanity baseline
LightGBMModel            — LightGBM, flat features, tabular benchmark
LSTMModel                — TensorFlow/Keras, windowed 3-D input
TFTModel                 — pytorch-forecasting, DataFrame input

All models share a common interface:
    train(...)           train and fit internal scaler
    predict(X)           return 0/1 array
    predict_proba(X)     return float array in [0, 1]
    save(path)           persist model + scaler to disk
    load(path)           restore from disk

model_type attribute
    "flat"       — expects 2-D input  (n_samples, n_features)
    "windowed"   — expects 3-D input  (n_samples, window, n_features)
    "sequential" — expects a feature DataFrame (TFT only)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

LR_C = 0.01
LR_MAX_ITER = 1000

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

LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 50
LSTM_BATCH = 64
LSTM_PATIENCE = 7

TFT_HIDDEN = 32
TFT_ATTENTION_HEADS = 2
TFT_DROPOUT = 0.1
TFT_HIDDEN_CONT = 16
TFT_LR = 0.03
TFT_EPOCHS = 30
TFT_BATCH = 64
TFT_PATIENCE = 5


# ---------------------------------------------------------------------------
# 1. Logistic Regression
# ---------------------------------------------------------------------------

class LogisticRegressionModel:
    """Logistic Regression with StandardScaler pre-processing."""

    model_type = "flat"

    def __init__(self, C: float = LR_C, max_iter: int = LR_MAX_ITER):
        from sklearn.preprocessing import StandardScaler
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        print("[LR] Fitting …")
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self.model.fit(X_scaled, y_train)

        train_acc = self.model.score(X_scaled, y_train)
        print(f"[LR] Train accuracy: {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            val_acc = self.model.score(self.scaler.transform(X_val), y_val)
            print(f"[LR] Val   accuracy: {val_acc:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        print(f"[LR] Saved → {path}")

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        print(f"[LR] Loaded ← {path}")


# ---------------------------------------------------------------------------
# 2. LightGBM
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
# 3. LSTM
# ---------------------------------------------------------------------------

class LSTMModel:
    """Two-layer stacked LSTM with dropout, trained via TensorFlow/Keras.

    Input shape: (n_samples, window_size, n_features)
    Features are scaled to [0, 1] with a MinMaxScaler fitted on training data.
    """

    model_type = "windowed"

    def __init__(
        self,
        window_size: int,
        n_features: int,
        units: list[int] = LSTM_UNITS,
        dropout: float = LSTM_DROPOUT,
    ):
        from sklearn.preprocessing import MinMaxScaler

        self.window_size = window_size
        self.n_features = n_features
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()

    # ---- internal ----

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Reshape to 2-D, scale, reshape back to 3-D."""
        n, w, f = X.shape
        X2d = X.reshape(-1, f)
        X2d = self.scaler.fit_transform(X2d) if fit else self.scaler.transform(X2d)
        return X2d.reshape(n, w, f)

    def _build(self):
        import tensorflow as tf

        model = tf.keras.Sequential()
        for i, u in enumerate(self.units):
            return_seq = i < len(self.units) - 1
            if i == 0:
                model.add(tf.keras.layers.LSTM(
                    u, return_sequences=return_seq,
                    input_shape=(self.window_size, self.n_features),
                ))
            else:
                model.add(tf.keras.layers.LSTM(u, return_sequences=return_seq))
            model.add(tf.keras.layers.Dropout(self.dropout))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ---- public ----

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH,
        patience: int = LSTM_PATIENCE,
        checkpoint_path: str | Path | None = None,
    ):
        import tensorflow as tf

        X_tr = self._scale(X_train, fit=True)
        X_vl = self._scale(X_val, fit=False)

        self.model = self._build()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1,
            ),
        ]
        if checkpoint_path:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path),
                    monitor="val_loss", save_best_only=True, verbose=0,
                )
            )

        print(f"[LSTM] Training  epochs={epochs}  batch={batch_size} …")
        history = self.model.fit(
            X_tr, y_train,
            validation_data=(X_vl, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._scale(X, fit=False), verbose=0).flatten()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path) + ".keras")
        with open(str(path) + "_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[LSTM] Saved → {path}.keras  +  {path}_scaler.pkl")

    def load(self, path: str | Path) -> None:
        import tensorflow as tf

        path = Path(path)
        self.model = tf.keras.models.load_model(str(path) + ".keras")
        with open(str(path) + "_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[LSTM] Loaded ← {path}.keras")


# ---------------------------------------------------------------------------
# 4. Temporal Fusion Transformer
# ---------------------------------------------------------------------------

class TFTModel:
    """Temporal Fusion Transformer via pytorch-forecasting.

    Unlike the other models, TFTModel takes the full feature DataFrame as
    input to train() rather than pre-windowed numpy arrays.  The
    TimeSeriesDataSet inside pytorch-forecasting handles the windowing.

    Target is treated as a binary float (0 / 1).  A QuantileLoss with
    quantile [0.5] is used so the model learns to predict the median
    direction probability; outputs are clipped to [0, 1] and thresholded
    at 0.5 for class predictions.
    """

    model_type = "sequential"

    def __init__(
        self,
        window_size: int,
        feature_cols: list[str],
        hidden_size: int = TFT_HIDDEN,
        attention_head_size: int = TFT_ATTENTION_HEADS,
        dropout: float = TFT_DROPOUT,
        hidden_continuous_size: int = TFT_HIDDEN_CONT,
    ):
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.model = None
        self._training_dataset = None  # kept for inference dataset creation

    # ---- internal helpers ----

    def _make_timeseries_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert a feature DataFrame to the format expected by TimeSeriesDataSet."""
        data = df[self.feature_cols + ["target"]].copy()
        data = data.reset_index(drop=True)
        data["time_idx"] = np.arange(len(data), dtype=int)
        data["group_id"] = "BTC"
        data["target"] = data["target"].astype(float)
        return data

    def _build_dataset(
        self,
        data: pd.DataFrame,
        predict: bool = False,
    ):
        from pytorch_forecasting import TimeSeriesDataSet

        return TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            min_encoder_length=self.window_size // 2,
            max_encoder_length=self.window_size,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_unknown_reals=self.feature_cols,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=True,
            predict_mode=predict,
        )

    # ---- public ----

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        epochs: int = TFT_EPOCHS,
        batch_size: int = TFT_BATCH,
        lr: float = TFT_LR,
        patience: int = TFT_PATIENCE,
    ) -> None:
        try:
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.metrics import QuantileLoss
            import pytorch_lightning as pl
            from pytorch_lightning.callbacks import EarlyStopping
        except ImportError as exc:
            raise ImportError(
                "pytorch-forecasting and pytorch-lightning are required. "
                "Run: pip install pytorch-forecasting pytorch-lightning"
            ) from exc

        # Combine train + val with a continuous time_idx
        train_data = self._make_timeseries_df(df_train)
        val_data = self._make_timeseries_df(df_val)

        # Offset val time_idx so it continues from train
        val_data["time_idx"] = val_data["time_idx"] + len(train_data)
        combined = pd.concat([train_data, val_data], ignore_index=True)

        training_cutoff = train_data["time_idx"].max()

        training_ds = self._build_dataset(combined[combined.time_idx <= training_cutoff])
        validation_ds = TimeSeriesDataSet.from_dataset(
            training_ds, combined, predict=True, stop_randomization=True
        )
        self._training_dataset = training_ds

        train_loader = training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = validation_ds.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=0)

        self.model = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=lr,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=QuantileLoss(quantiles=[0.5]),
            log_interval=10,
            reduce_on_plateau_patience=patience // 2,
        )

        print(f"[TFT] Parameters: {self.model.size() / 1e3:.1f}k")

        early_stop = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=patience, mode="min",
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            gradient_clip_val=0.1,
            callbacks=[early_stop],
            enable_progress_bar=True,
            logger=False,
        )
        print(f"[TFT] Training  epochs={epochs}  batch={batch_size} …")
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Run inference on a feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have the same feature columns as training data.

        Returns
        -------
        np.ndarray
            Array of probabilities in [0, 1].
        """
        from pytorch_forecasting import TimeSeriesDataSet

        data = self._make_timeseries_df(df)
        pred_ds = TimeSeriesDataSet.from_dataset(
            self._training_dataset, data, predict=True, stop_randomization=True,
        )
        loader = pred_ds.to_dataloader(train=False, batch_size=256, num_workers=0)
        preds = self.model.predict(loader, mode="prediction").numpy()
        # QuantileLoss with [0.5] returns shape (n,) or (n,1)
        preds = preds.flatten()
        return np.clip(preds, 0.0, 1.0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= 0.5).astype(int)

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hparams": self.model.hparams,
                "feature_cols": self.feature_cols,
                "window_size": self.window_size,
            },
            str(path) + ".pt",
        )
        with open(str(path) + "_train_ds.pkl", "wb") as f:
            pickle.dump(self._training_dataset, f)
        print(f"[TFT] Saved → {path}.pt")

    def load(self, path: str | Path) -> None:
        import torch
        from pytorch_forecasting import TemporalFusionTransformer

        path = Path(path)
        checkpoint = torch.load(str(path) + ".pt", map_location="cpu")
        with open(str(path) + "_train_ds.pkl", "rb") as f:
            self._training_dataset = pickle.load(f)
        self.model = TemporalFusionTransformer.load_from_checkpoint(
            str(path) + ".pt"
        )
        self.feature_cols = checkpoint["feature_cols"]
        self.window_size = checkpoint["window_size"]
        print(f"[TFT] Loaded ← {path}.pt")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "lr": LogisticRegressionModel,
    "lgbm": LightGBMModel,
    "lstm": LSTMModel,
    "tft": TFTModel,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of: "lr", "lgbm", "lstm", "tft".
    **kwargs
        Passed to the model constructor.

    Returns
    -------
    Model instance.
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
