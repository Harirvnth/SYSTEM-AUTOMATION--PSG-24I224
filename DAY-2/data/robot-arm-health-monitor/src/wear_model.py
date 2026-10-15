"""LAYER 2a -- predictive wear model + the physics prior used by Layer 1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config as C


class WearModel:
    """RandomForest over the 18 raw sensor channels."""

    def __init__(self, **params):
        self.feature_names = list(C.READING_COLUMNS)
        self.label_col = "wear_label"
        self.params = dict(n_estimators=400, max_depth=7, min_samples_leaf=2,
                           class_weight="balanced", random_state=0, n_jobs=-1)
        self.params.update(params)
        self.clf: RandomForestClassifier | None = None
        self.metrics_: dict = {}

    def fit(self, df, evaluate: bool = True) -> "WearModel":
        X = df[self.feature_names].to_numpy(float)
        y = df[self.label_col].to_numpy(int)

        if evaluate and len(df) >= 60 and len(set(y)) == 2:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.25, random_state=0, stratify=y)
            probe = RandomForestClassifier(**self.params).fit(Xtr, ytr)
            prob = probe.predict_proba(Xte)[:, 1]
            self.metrics_ = {
                "roc_auc": float(roc_auc_score(yte, prob)),
                "report": classification_report(
                    yte, (prob >= 0.5).astype(int), digits=3, zero_division=0),
            }
        # Refit on the full dataset for production use.
        self.clf = RandomForestClassifier(**self.params).fit(X, y)
        return self

    def predict_proba(self, reading: dict) -> float:
        if self.clf is None:
            raise RuntimeError("WearModel has not been fitted")
        x = np.array([[float(reading[c]) for c in self.feature_names]])
        return float(self.clf.predict_proba(x)[0, 1])


def fit_torque_prior(df) -> dict:
    """Fit torque ~ a*temperature + b*vibration + c per joint.

    This is the physical cross-check the spoof detector needs: a genuine
    torque reading must be consistent with the heat and vibration the joint
    is actually producing.
    """
    prior = {}
    for j in C.JOINTS:
        T = df[C.sensor_key(j, "temperature")].to_numpy(float)
        V = df[C.sensor_key(j, "vibration")].to_numpy(float)
        Y = df[C.sensor_key(j, "torque")].to_numpy(float)
        A = np.column_stack([T, V, np.ones_like(T)])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        sigma = float(np.std(Y - A @ coef))
        prior[j] = {
            "coef": [float(c) for c in coef],
            "sigma": sigma,
            "temp_ref": float(np.median(T)),
            "vib_ref": float(np.median(V)),
        }
    return prior


def save_prior(prior: dict, path: str = "data/torque_prior.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prior, indent=2), encoding="utf-8")


def load_prior(path: str = "data/torque_prior.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))