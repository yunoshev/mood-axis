"""Statistical utilities for Mood Axis analysis.

Provides bootstrap confidence intervals, significance testing,
ICC, d-prime, and multiple comparison corrections.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "mean": self.mean, "ci_lower": self.ci_lower, "ci_upper": self.ci_upper,
            "std": self.std, "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


@dataclass
class SignificanceResult:
    mean: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    test_value: float
    effect_size: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "mean": self.mean, "ci_lower": self.ci_lower, "ci_upper": self.ci_upper,
            "p_value": self.p_value, "is_significant": self.is_significant,
            "test_value": self.test_value, "effect_size": self.effect_size,
        }


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for a statistic."""
    if random_state is not None:
        np.random.seed(random_state)

    values = np.array(values)
    n = len(values)

    if n < 2:
        return BootstrapResult(
            mean=float(values[0]) if n == 1 else np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
            std=0.0 if n == 1 else np.nan, n_samples=n,
        )

    stat_funcs = {"mean": np.mean, "median": np.median, "std": np.std}
    stat_func = stat_funcs.get(statistic, np.mean)

    bootstraps = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstraps.append(stat_func(sample))

    bootstraps = np.array(bootstraps)
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstraps, alpha * 100)
    upper = np.percentile(bootstraps, (1 - alpha) * 100)

    return BootstrapResult(
        mean=float(stat_func(values)),
        ci_lower=float(lower), ci_upper=float(upper),
        std=float(np.std(values)), n_samples=n,
    )


def test_against_value(
    values: List[float],
    test_value: float = 0.0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> SignificanceResult:
    """Test if mean is significantly different from a value using bootstrap."""
    if random_state is not None:
        np.random.seed(random_state)

    values = np.array(values)
    n = len(values)

    if n < 2:
        return SignificanceResult(
            mean=float(values[0]) if n == 1 else np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
            p_value=1.0, is_significant=False, test_value=test_value,
        )

    bootstraps = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstraps.append(np.mean(sample))

    bootstraps = np.array(bootstraps)
    ci_lower = np.percentile(bootstraps, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstraps, (1 - alpha / 2) * 100)

    sample_mean = np.mean(values)
    if sample_mean >= test_value:
        p_value = 2 * np.mean(bootstraps <= test_value)
    else:
        p_value = 2 * np.mean(bootstraps >= test_value)
    p_value = min(p_value, 1.0)

    std = np.std(values, ddof=1)
    effect_size = (sample_mean - test_value) / std if std > 0 else 0.0

    return SignificanceResult(
        mean=float(sample_mean),
        ci_lower=float(ci_lower), ci_upper=float(ci_upper),
        p_value=float(p_value), is_significant=p_value < alpha,
        test_value=test_value, effect_size=float(effect_size),
    )


def compute_dprime(positive_values: List[float], negative_values: List[float]) -> float:
    """Compute d-prime (Cohen's d) between two distributions."""
    pos = np.array(positive_values)
    neg = np.array(negative_values)
    mean_diff = np.mean(pos) - np.mean(neg)

    n_pos, n_neg = len(pos), len(neg)
    var_pos = np.var(pos, ddof=1) if n_pos > 1 else 0
    var_neg = np.var(neg, ddof=1) if n_neg > 1 else 0

    if n_pos + n_neg > 2:
        pooled_var = ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        pooled_std = 1.0

    if pooled_std < 1e-10:
        return 0.0
    return float(mean_diff / pooled_std)


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Apply Holm-Bonferroni step-down correction."""
    n = len(p_values)
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    significant = [False] * n
    adjusted = [1.0] * n

    for rank, (p, orig_idx) in enumerate(indexed):
        threshold = alpha / (n - rank)
        adjusted[orig_idx] = min(p * (n - rank), 1.0)
        if p < threshold:
            significant[orig_idx] = True
        else:
            break

    return significant, adjusted


def compute_icc(measurements: List[List[float]]) -> float:
    """Compute Intraclass Correlation Coefficient ICC(2,1)."""
    data = np.array(measurements).T  # [n_subjects, n_runs]

    n_subjects, n_runs = data.shape
    if n_subjects < 2 or n_runs < 2:
        return np.nan

    grand_mean = np.mean(data)
    subject_means = np.mean(data, axis=1)
    run_means = np.mean(data, axis=0)

    ss_between = n_runs * np.sum((subject_means - grand_mean) ** 2)
    ss_within = np.sum((data - subject_means.reshape(-1, 1)) ** 2)
    ss_runs = n_subjects * np.sum((run_means - grand_mean) ** 2)
    ss_error = ss_within - ss_runs

    ms_between = ss_between / (n_subjects - 1)
    ms_runs = ss_runs / (n_runs - 1) if n_runs > 1 else 0
    ms_error = ss_error / ((n_subjects - 1) * (n_runs - 1))

    if ms_error < 1e-10:
        return 1.0

    icc = (ms_between - ms_error) / (ms_between + (n_runs - 1) * ms_error +
                                      n_runs * (ms_runs - ms_error) / n_subjects)

    return float(np.clip(icc, -1, 1))


def compute_axis_statistics(
    baseline_values: List[Dict[str, float]],
    axes: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Compute comprehensive statistics for baseline measurements."""
    if not baseline_values:
        return {}

    if axes is None:
        axes = list(baseline_values[0].keys())

    results = {}
    p_values = []

    for axis in axes:
        values = [v[axis] for v in baseline_values if axis in v]
        if len(values) < 2:
            continue

        ci_result = bootstrap_ci(values, n_bootstrap=n_bootstrap)
        sig_result = test_against_value(values, test_value=0.0, n_bootstrap=n_bootstrap, alpha=alpha)

        results[axis] = {
            "mean": ci_result.mean, "std": ci_result.std,
            "ci_lower": ci_result.ci_lower, "ci_upper": ci_result.ci_upper,
            "p_value": sig_result.p_value, "effect_size": sig_result.effect_size,
            "n_samples": ci_result.n_samples,
        }
        p_values.append((axis, sig_result.p_value))

    if p_values:
        axes_list = [a for a, _ in p_values]
        pvals_list = [p for _, p in p_values]
        significant_corrected, adjusted_pvals = holm_bonferroni_correction(pvals_list, alpha)

        for i, axis in enumerate(axes_list):
            results[axis]["p_value_adjusted"] = adjusted_pvals[i]
            results[axis]["significant_corrected"] = significant_corrected[i]

    return results
