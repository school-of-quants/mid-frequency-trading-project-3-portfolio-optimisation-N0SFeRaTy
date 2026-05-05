import os
from pathlib import Path
from typing import Dict, Iterator, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

from equity_project.src.utils import save_dict

project_path = Path(__file__).parent.parent

# Keep in sync with labeling horizon in get_data.py (LABEL_HORIZON_DAYS)
LABEL_HORIZON_DAYS = 10


class PurgedKFold:
    """Purged K-Fold cross-validation for panel financial data.

    Splits are made by dates, not by individual rows. Around every validation
    period, a purge/embargo window is removed from the training set to reduce
    label leakage caused by overlapping forward-looking labels.
    """

    def __init__(self, n_splits: int = 5, purge_n: int = 12, embargo_n: int = 3):
        self.n_splits = n_splits
        self.purge_n = purge_n
        self.embargo_n = embargo_n

    def split(self, x: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/validation indices.

        Args:
            x (pd.DataFrame): Feature matrix indexed by Date/Ticker MultiIndex.

        Yields:
            Tuple[np.ndarray, np.ndarray]: Train and validation row indices.
        """
        dates = pd.Index(sorted(x.index.get_level_values("Date").unique()))
        date_folds = np.array_split(dates.to_numpy(), self.n_splits)

        row_dates = x.index.get_level_values("Date")

        for val_dates_arr in date_folds:
            val_dates = pd.Index(val_dates_arr)

            val_start = val_dates.min()
            val_end = val_dates.max()

            val_start_pos = int(dates.get_indexer([val_start])[0])
            val_end_pos = int(dates.get_indexer([val_end])[0])

            purge_start_pos = max(0, val_start_pos - int(self.purge_n))
            embargo_end_pos = min(len(dates) - 1, val_end_pos + int(self.embargo_n))

            purged_dates = dates[purge_start_pos : embargo_end_pos + 1]

            is_val = row_dates.isin(val_dates)
            is_purged = row_dates.isin(purged_dates)

            train_idx = np.where(~is_val & ~is_purged)[0]
            val_idx = np.where(is_val)[0]

            yield train_idx, val_idx


def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    """Compute class weights with lower emphasis on neutral class 0."""
    vc = y.value_counts()
    total = float(vc.sum())
    inv = {int(k): total / float(v) for k, v in vc.items()}

    # De-emphasize neutral class to encourage learning direction when signal exists.
    if 0 in inv:
        inv[0] *= 0.5

    # Normalize average weight ~ 1.0
    mean_w = float(np.mean(list(inv.values())))
    if mean_w > 0:
        inv = {k: float(v / mean_w) for k, v in inv.items()}
    return inv


def instantiate_model(random_seed: int = 42, class_weights: Dict[int, float] | None = None) -> CatBoostClassifier:
    """Instantiate CatBoost multiclass classifier."""
    return CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8,
        random_seed=random_seed,
        class_weights=class_weights,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        early_stopping_rounds=80,
        allow_writing_files=False,
        verbose=100,
    )


def prepare_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and clean training dataset."""
    x = pd.read_parquet(project_path / "data/processed/X_train.parquet")
    y = pd.read_parquet(project_path / "data/processed/y_train.parquet")["target"]

    dataset = x.join(y).replace([np.inf, -np.inf], np.nan).dropna()

    x = dataset.drop(columns="target")
    y = dataset["target"].astype(int)

    return x, y


def train() -> None:
    """Train CatBoost model with PurgedKFold validation and save final model."""
    os.makedirs(project_path / "models", exist_ok=True)
    os.makedirs(project_path / "artifacts/metrics", exist_ok=True)

    x, y = prepare_dataset()

    # Purge/embargo should cover the forward-looking label horizon to reduce overlap leakage.
    cv = PurgedKFold(
        n_splits=5,
        purge_n=LABEL_HORIZON_DAYS,
        embargo_n=LABEL_HORIZON_DAYS,
    )
    # For overall accuracy, avoid aggressive inverse-frequency weighting.
    # (If you care more about directional minority classes, switch back to compute_class_weights.)
    class_weights = None

    fold_metrics = []
    oof_pred = pd.Series(index=y.index, dtype=float)

    for fold, (train_idx, val_idx) in enumerate(cv.split(x), start=1):
        x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = instantiate_model(random_seed=42 + fold, class_weights=class_weights)

        model.fit(
            X=x_train,
            y=y_train,
            eval_set=(x_val, y_val),
            use_best_model=True,
        )

        pred = model.predict(x_val).reshape(-1).astype(int)

        acc = accuracy_score(y_val, pred)
        score = balanced_accuracy_score(y_val, pred)
        macro_f1 = f1_score(y_val, pred, average="macro")

        fold_metrics.append(
            {
                "fold": fold,
                "n_train": len(x_train),
                "n_val": len(x_val),
                "accuracy": float(acc),
                "balanced_accuracy": float(score),
                "macro_f1": float(macro_f1),
                "best_iteration": int(model.get_best_iteration()),
            }
        )

        oof_pred.iloc[val_idx] = pred

        print(
            f"Fold {fold}: accuracy={acc:.4f} balanced_accuracy={score:.4f} macro_f1={macro_f1:.4f}"
        )

    valid_oof = oof_pred.notna()
    oof_accuracy = accuracy_score(
        y.loc[valid_oof],
        oof_pred.loc[valid_oof].astype(int),
    )
    oof_balanced_accuracy = balanced_accuracy_score(
        y.loc[valid_oof],
        oof_pred.loc[valid_oof].astype(int),
    )
    oof_macro_f1 = f1_score(
        y.loc[valid_oof],
        oof_pred.loc[valid_oof].astype(int),
        average="macro",
    )

    report = classification_report(
        y.loc[valid_oof],
        oof_pred.loc[valid_oof].astype(int),
        output_dict=True,
    )

    # Final time-based holdout for early stopping.
    dates = pd.Index(sorted(x.index.get_level_values("Date").unique()))
    split_date = dates[int(len(dates) * 0.8)]

    train_mask = x.index.get_level_values("Date") <= split_date
    val_mask = x.index.get_level_values("Date") > split_date

    x_train, x_val = x.loc[train_mask], x.loc[val_mask]
    y_train, y_val = y.loc[train_mask], y.loc[val_mask]

    final_model = instantiate_model(random_seed=777, class_weights=class_weights)

    final_model.fit(
        X=x_train,
        y=y_train,
        eval_set=(x_val, y_val),
        use_best_model=True,
    )

    bundle = {
        "model": final_model,
        "feature_columns": list(x.columns),
        "classes": list(final_model.classes_),
    }

    joblib.dump(bundle, project_path / "models/model.joblib")

    save_dict(
        {
            "fold_metrics": fold_metrics,
            "oof_accuracy": float(oof_accuracy),
            "oof_balanced_accuracy": float(oof_balanced_accuracy),
            "oof_macro_f1": float(oof_macro_f1),
            "classification_report": report,
            "final_model_best_iteration": int(final_model.get_best_iteration()),
            "n_samples": len(x),
            "n_features": x.shape[1],
            "class_weights": class_weights,
            "classes": list(final_model.classes_),
        },
        project_path / "artifacts/metrics/train_metrics.json",
    )

    print("Модель сохранена.")
    print(f"OOF accuracy: {oof_accuracy:.4f}")
    print(f"OOF balanced accuracy: {oof_balanced_accuracy:.4f}")
    print(f"OOF macro F1: {oof_macro_f1:.4f}")


if __name__ == "__main__":
    train()
