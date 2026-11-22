"""Cross-validation berulang untuk LightGBM, regresi logistik, dan model teks TF-IDF pada string URL."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8)


def lgb_model(seed: int = 0, **kw):
    return lgb.LGBMClassifier(**{**LGB_PARAMS, **kw}, random_state=seed, verbose=-1)


def logreg_model(seed: int = 0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def tfidf_model(seed: int = 0):
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                         max_features=300_000),
                         LogisticRegression(C=5.0, max_iter=3000))


def cv_predict(make_model, X, y, seeds=range(3), n_splits: int = 5) -> np.ndarray:
    """Probabilitas out-of-fold untuk tiap seed. Hasil: array (n_seed, n_baris)."""
    y = np.asarray(y)
    out = np.zeros((len(list(seeds)), len(y)))
    for i, s in enumerate(seeds):
        for trn, val in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(np.zeros(len(y)), y):
            Xt = X.iloc[trn] if hasattr(X, "iloc") else X[trn]
            Xv = X.iloc[val] if hasattr(X, "iloc") else X[val]
            out[i, val] = make_model(s).fit(Xt, y[trn]).predict_proba(Xv)[:, 1]
    return out


def auc_by_seed(oof: np.ndarray, y) -> list[float]:
    return [roc_auc_score(y, p) for p in np.atleast_2d(oof)]


def rank_mean(*preds) -> np.ndarray:
    return np.mean([pd.Series(p).rank(pct=True).values for p in preds], axis=0)
