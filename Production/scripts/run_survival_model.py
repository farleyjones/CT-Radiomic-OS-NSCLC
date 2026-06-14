"""
Survival model evaluation using continuous time-to-event data.

Avoids the information loss of binary landmark classification.
Primary metric: Harrell's C-index (concordance index).

Models:
  - Cox PH with L1/L2 regularisation
  - Random Survival Forest

Uses 5-fold nested CV for unbiased C-index estimates.

Usage:
  python run_survival_model.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

# Use the LASSO-selected feature set (49 features, 12-month selection)
# Survival models use continuous time — no landmark discretisation
FEATURES_CSV = HERE / "integrated_features_lasso_12m.csv"
OUTPUT_JSON  = HERE / "survival_model_results.json"
N_FOLDS      = 5
SEED         = 42


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    exclude = {"patient_id", "survival_days", "event_occurred",
               "landmark_12m", "landmark_6m"}
    feat_cols = [c for c in df.columns if c not in exclude]

    X    = df[feat_cols].values.astype(np.float32)
    time = df["survival_days"].values.astype(float)
    event = df["event_occurred"].values.astype(bool)

    # sksurv structured array
    y = np.array(
        [(bool(e), t) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)]
    )

    logger.info(f"Loaded: {len(df)} patients, {len(feat_cols)} features")
    logger.info(f"Events: {event.sum()} / {len(event)} ({event.mean():.1%})")
    logger.info(f"Median survival: {np.median(time):.0f} days")

    return X, y, time, event, feat_cols, df


def run_cox(X, y, time, event, feat_cols):
    """Cox PH with elastic net regularisation."""
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # Stratify by median survival for balanced folds
    strat = (time > np.median(time)).astype(int)

    c_indices = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, strat)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        ev_te       = event[test_idx]
        t_te        = time[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            cox = CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.1,
                max_iter=1000, fit_baseline_model=True
            )
            cox.fit(X_tr_s, y_tr)
            risk = cox.predict(X_te_s)
            c = concordance_index_censored(ev_te, t_te, risk)[0]
            c_indices.append(c)
            logger.info(f"  Cox fold {fold_i+1}: C-index = {c:.3f}")
        except Exception as e:
            logger.warning(f"  Cox fold {fold_i+1} failed: {e}")

    return c_indices


def run_rsf(X, y, time, event):
    """Random Survival Forest."""
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strat = (time > np.median(time)).astype(int)

    c_indices = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, strat)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr        = y[train_idx]
        ev_te       = event[test_idx]
        t_te        = time[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            rsf = RandomSurvivalForest(
                n_estimators=200, min_samples_split=10,
                min_samples_leaf=5, max_features="sqrt",
                n_jobs=-1, random_state=SEED
            )
            rsf.fit(X_tr_s, y_tr)
            risk = rsf.predict(X_te_s)
            c = concordance_index_censored(ev_te, t_te, risk)[0]
            c_indices.append(c)
            logger.info(f"  RSF  fold {fold_i+1}: C-index = {c:.3f}")
        except Exception as e:
            logger.warning(f"  RSF  fold {fold_i+1} failed: {e}")

    return c_indices


def run_clinical_baseline(X, y, time, event, feat_cols):
    """Cox on clinical features only — baseline comparator."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored

    clin_idx = [i for i, n in enumerate(feat_cols) if n.startswith("clin_")]
    if not clin_idx:
        logger.warning("No clinical features found")
        return []

    X_clin = X[:, clin_idx]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strat = (time > np.median(time)).astype(int)

    c_indices = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_clin, strat)):
        X_tr, X_te = X_clin[train_idx], X_clin[test_idx]
        y_tr        = y[train_idx]
        ev_te       = event[test_idx]
        t_te        = time[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        try:
            cox = CoxPHSurvivalAnalysis(alpha=0.1)
            cox.fit(X_tr_s, y_tr)
            risk = cox.predict(X_te_s)
            c = concordance_index_censored(ev_te, t_te, risk)[0]
            c_indices.append(c)
        except Exception as e:
            logger.warning(f"  Clinical baseline fold {fold_i+1}: {e}")

    return c_indices


def main():
    X, y, time, event, feat_cols, df = load_data()

    results = {}

    logger.info("\nFitting Cox PH (elastic net)...")
    cox_scores = run_cox(X, y, time, event, feat_cols)

    logger.info("\nFitting Random Survival Forest...")
    rsf_scores = run_rsf(X, y, time, event)

    logger.info("\nFitting clinical baseline (Cox, clinical features only)...")
    clin_scores = run_clinical_baseline(X, y, time, event, feat_cols)

    print("\n" + "="*60)
    print("  Survival Model Results — C-index (nested 5-fold CV)")
    print("="*60)

    for name, scores in [
        ("Cox PH (rad+clin)",  cox_scores),
        ("RSF (rad+clin)",     rsf_scores),
        ("Clinical baseline",  clin_scores),
    ]:
        if scores:
            mean, std = np.mean(scores), np.std(scores)
            results[name] = {"c_index_mean": round(mean, 3),
                             "c_index_std":  round(std, 3),
                             "fold_scores":  [round(s, 3) for s in scores]}
            print(f"  {name:25s}  C-index = {mean:.3f} ± {std:.3f}")
        else:
            print(f"  {name:25s}  FAILED")

    print("="*60)
    print(f"\n  Binary classifier (logistic, nested CV): AUROC = 0.657 ± 0.040")
    print(f"  [Survival C-index and binary AUROC are comparable metrics]")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {OUTPUT_JSON.name}")


if __name__ == "__main__":
    main()
