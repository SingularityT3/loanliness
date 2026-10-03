"""Shared helpers for the loan pipeline: loading the saved splits, balancing, metrics, preprocessing new data."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

ARTIFACTS = Path("artifacts")
SEED = 42


def load_split(name, artifacts=ARTIFACTS):
    # name: "train", "test", "train_balanced" or "test_balanced" -> (X, y) with the feature order from schema.json
    schema = json.load(open(Path(artifacts) / "schema.json"))
    df = pd.read_parquet(Path(artifacts) / f"{name}.parquet").set_index(schema["id"])
    return df[schema["features"]], df[schema["target"]]


def downsample(X, y, seed=SEED):
    # keep every positive, sample the same number of negatives
    pos = y.index[y == 1]
    neg = y.index[y == 0].to_series().sample(len(pos), random_state=seed).index
    idx = pos.union(neg)
    idx = pd.Index(np.random.default_rng(seed).permutation(idx))
    return X.loc[idx], y.loc[idx]


def smote(X, y, seed=SEED):
    # needs imputed, encoded numeric input; tree models recognise the synthetic rows, so use it with linear models only
    Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
    return pd.DataFrame(Xr, columns=X.columns).astype("float32"), pd.Series(yr, name=y.name)


def class_weights(y):
    # w_i = n / (2 n_i)
    n = len(y)
    return {int(c): n / (2 * k) for c, k in y.value_counts().items()}


def metrics(y_true, y_pred, y_prob=None):
    # the paper's layout: accuracy, then precision / recall / F1 for class 0 and class 1
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "prec_0": p[0], "prec_1": p[1],
           "rec_0": r[0], "rec_1": r[1], "f1_0": f[0], "f1_1": f[1], "n": len(y_true), "pos_rate": float(np.mean(y_true))}
    if y_prob is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_prob)
    return out


def transform_new(raw: pd.DataFrame, artifacts=ARTIFACTS) -> pd.DataFrame:
    # raw: joined features after the DAYS_EMPLOYED fix and count-fill in the notebook, without ID/TARGET columns
    art = joblib.load(Path(artifacts) / "preprocessor.joblib")
    x = raw[art["keep_cols"]]
    out = art["scaler"].transform(art["encoder"].transform(art["imputer"].transform(x)))[art["raw_names"]]
    out.columns = art["feature_names"]
    return out
