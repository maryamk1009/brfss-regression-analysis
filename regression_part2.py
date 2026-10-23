"""
part 2 - regression math (person 2)

single pass multiple linear regression over the brfss records from part 1.
we never store rows, we only keep running sums:
    A = sum of x x^T      ((k+1) x (k+1), this is X^T X)
    z = sum of x * y      (k+1, this is X^T y)
    sum_y, sum_y2, n
then beta = A^-1 z and adjusted R^2 comes straight from the same sums.

in the final notebook paste this cell right under part 1's code
(drop the import line below since iter_health_records will already be defined).
"""

import numpy as np

from data_processing import iter_health_records  # remove this line in the notebook


def init_stats(k):
    """empty running sums for k predictors (+1 for the intercept)."""
    return {
        "n": 0,
        "A": np.zeros((k + 1, k + 1)),
        "z": np.zeros(k + 1),
        "sum_y": 0.0,
        "sum_y2": 0.0,
    }


def update_stats(stats, predictors, y):
    """add one observation to the running sums."""
    x = np.array([1.0] + list(predictors))  # intercept goes first
    stats["n"] += 1
    stats["A"] += np.outer(x, x)
    stats["z"] += x * y
    stats["sum_y"] += y
    stats["sum_y2"] += y * y


def accumulate_stats(records, k):
    """one pass over any (predictors, y) iterable, returns the sums."""
    stats = init_stats(k)
    for predictors, y in records:
        update_stats(stats, predictors, y)
    return stats


def compute_beta(stats):
    """beta = (X^T X)^-1 X^T y. solve() instead of inv() for stability."""
    return np.linalg.solve(stats["A"], stats["z"])


def compute_adj_r2(stats, beta):
    """
    SSE = y'y - beta' X'y      (true at the least squares solution)
    SST = y'y - n * ybar^2
    R^2 = 1 - SSE / SST
    adj R^2 = 1 - (1 - R^2) (n - 1) / (n - k - 1)
    """
    n = stats["n"]
    k = len(beta) - 1
    if n <= k + 1:
        raise ValueError(f"need more than {k + 1} observations, got {n}")
    sse = stats["sum_y2"] - beta @ stats["z"]
    sst = stats["sum_y2"] - stats["sum_y"] ** 2 / n
    r2 = 1 - sse / sst
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def regression_health(data_dir, columns, require_all=False):
    """
    returns (beta, adj_r2)
      beta: numpy array, intercept first then one coef per column in `columns` order
      adj_r2: adjusted R^2 of the model
    """
    stats = accumulate_stats(iter_health_records(data_dir, columns, require_all), len(columns))
    beta = compute_beta(stats)
    adj_r2 = compute_adj_r2(stats, beta)
    return beta, adj_r2
