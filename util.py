import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))

def accuracy_at_thresh(y_true: np.ndarray, p: np.ndarray, thresh: float = 0.5) -> float:
    y_pred = (p >= thresh).astype(int)
    return float(np.mean(y_pred == y_true))

def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> tuple[float, pd.DataFrame]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    rows = []
    total_n = len(y)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(np.sum(mask))
        if n == 0:
            rows.append((bins[b], bins[b+1], n, np.nan, np.nan))
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        weight = n / total_n
        ece += weight * abs(acc - conf)
        rows.append((bins[b], bins[b+1], n, conf, acc))

    calib_df = pd.DataFrame(rows, columns=["bin_lo","bin_hi","n","mean_confidence","empirical_accuracy"])
    return float(ece), calib_df

def decile_lift_table(y_true: np.ndarray, p: np.ndarray, k: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": p})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    n = len(df)
    rows = []
    base_rate = df["y"].mean() if n > 0 else np.nan
    for i in range(k):
        lo = int(i * n / k)
        hi = int((i + 1) * n / k)
        chunk = df.iloc[lo:hi]
        if len(chunk) == 0:
            rows.append((i+1, lo, hi, np.nan, np.nan))
            continue
        avg_p = float(chunk["p"].mean())
        acc = float((chunk["y"] == (chunk["p"] >= 0.5).astype(int)).mean())
        lift = (acc / base_rate) if (base_rate and not np.isnan(base_rate) and base_rate > 0) else np.nan
        rows.append((i+1, lo, hi, avg_p, acc, lift))
    return pd.DataFrame(rows, columns=["decile","start_idx","end_idx","avg_pred","acc@0.5","lift_vs_base_rate"])

def bootstrap_metric_ci(y_true: np.ndarray, p: np.ndarray, fn, n_boot: int = 1000, seed: int = 42) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    pr = np.asarray(p).astype(float)
    n = len(y)
    stat = fn(y, pr)
    if n == 0:
        return np.nan, (np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(fn(y[idx], pr[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(stat), (float(lo), float(hi))

def safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return log_loss(y_true, p, labels=[0,1])
