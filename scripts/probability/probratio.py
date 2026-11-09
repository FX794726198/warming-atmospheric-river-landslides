"""Ratios of average predicted probabilities (marginal standardization) from fitted logistic models,
with 95% intervals from the delta method on the log ratio using the model's (cluster-robust) covariance."""
import numpy as np, pandas as pd, patsy
from scipy.special import expit

def design(di, data):
    return np.asarray(patsy.build_design_matrices([di], data, return_type='dataframe')[0], dtype=float)

def ratio(b, V, X1, X0):
    """b: params, V: covariance, X1/X0: counterfactual design matrices (same rows). Returns P1, P0, ratio, lo, hi, se_log."""
    p1 = expit(X1 @ b); p0 = expit(X0 @ b)
    P1 = p1.mean(); P0 = p0.mean()
    g = (X1 * (p1 * (1 - p1))[:, None]).mean(0) / P1 - (X0 * (p0 * (1 - p0))[:, None]).mean(0) / P0
    se = float(np.sqrt(g @ V @ g)); r = P1 / P0
    return dict(P1=P1, P0=P0, RR=r, lo=r * np.exp(-1.96 * se), hi=r * np.exp(1.96 * se), se_log=se)

def prob(b, V, X1):
    p1 = expit(X1 @ b); P1 = p1.mean(); g = (X1 * (p1 * (1 - p1))[:, None]).mean(0)
    se = float(np.sqrt(g @ V @ g)); return P1, P1 - 1.96 * se, P1 + 1.96 * se
