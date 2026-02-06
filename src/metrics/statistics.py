"""Statistical utilities for Mood Axis analysis.

Provides bootstrap confidence intervals, significance testing,
and multiple comparison corrections.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval computation."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_samples: int

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std": self.std,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.ci_lower:.3f}, {self.ci_upper:.3f}]"


@dataclass
class SignificanceResult:
    """Result of significance test against zero or another value."""
    mean: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    test_value: float  # Value tested against (usually 0)
    effect_size: Optional[float] = None  # Cohen's d if applicable

    def to_dict(self) -> dict:
        return {
            "mean": self.mean,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "test_value": self.test_value,
            "effect_size": self.effect_size,
        }


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: str = "mean",
    random_state: Optional[int] = None,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        values: Sample values
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence level (default 0.95 for 95% CI)
        statistic: Which statistic to compute ("mean", "median", "std")
        random_state: Random seed for reproducibility

    Returns:
        BootstrapResult with mean, CI bounds, and sample info
    """
    if random_state is not None:
        np.random.seed(random_state)

    values = np.array(values)
    n = len(values)

    if n < 2:
        return BootstrapResult(
            mean=float(values[0]) if n == 1 else np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            std=0.0 if n == 1 else np.nan,
            n_samples=n,
        )

    # Select statistic function
    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
    }
    stat_func = stat_funcs.get(statistic, np.mean)

    # Bootstrap resampling
    bootstraps = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstraps.append(stat_func(sample))

    bootstraps = np.array(bootstraps)

    # Compute percentile CI
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstraps, alpha * 100)
    upper = np.percentile(bootstraps, (1 - alpha) * 100)

    return BootstrapResult(
        mean=float(stat_func(values)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        std=float(np.std(values)),
        n_samples=n,
    )


def test_against_value(
    values: List[float],
    test_value: float = 0.0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> SignificanceResult:
    """Test if mean is significantly different from a value using bootstrap.

    Uses bootstrap percentile method: if test_value is outside CI,
    result is significant.

    Args:
        values: Sample values
        test_value: Value to test against (default 0)
        n_bootstrap: Number of bootstrap iterations
        alpha: Significance level
        random_state: Random seed

    Returns:
        SignificanceResult with p-value and significance
    """
    if random_state is not None:
        np.random.seed(random_state)

    values = np.array(values)
    n = len(values)

    if n < 2:
        return SignificanceResult(
            mean=float(values[0]) if n == 1 else np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            p_value=1.0,
            is_significant=False,
            test_value=test_value,
        )

    # Bootstrap for CI
    bootstraps = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstraps.append(np.mean(sample))

    bootstraps = np.array(bootstraps)

    # CI bounds
    ci_lower = np.percentile(bootstraps, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstraps, (1 - alpha / 2) * 100)

    # P-value: proportion of bootstraps on "wrong" side of test_value
    # Two-tailed test
    sample_mean = np.mean(values)
    if sample_mean >= test_value:
        p_value = 2 * np.mean(bootstraps <= test_value)
    else:
        p_value = 2 * np.mean(bootstraps >= test_value)

    p_value = min(p_value, 1.0)  # Cap at 1.0

    # Effect size (Cohen's d relative to test_value)
    std = np.std(values, ddof=1)
    effect_size = (sample_mean - test_value) / std if std > 0 else 0.0

    return SignificanceResult(
        mean=float(sample_mean),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        test_value=test_value,
        effect_size=float(effect_size),
    )


def compute_dprime(
    positive_values: List[float],
    negative_values: List[float],
) -> float:
    """Compute d-prime (Cohen's d) between two distributions.

    d' = (mean_pos - mean_neg) / pooled_std

    Interpretation:
    - d' > 0.2: small effect
    - d' > 0.5: medium effect
    - d' > 0.8: large effect
    - d' > 1.2: very large effect

    Args:
        positive_values: Values from positive pole
        negative_values: Values from negative pole

    Returns:
        d-prime value
    """
    pos = np.array(positive_values)
    neg = np.array(negative_values)

    mean_diff = np.mean(pos) - np.mean(neg)

    # Pooled standard deviation
    n_pos, n_neg = len(pos), len(neg)
    var_pos = np.var(pos, ddof=1) if n_pos > 1 else 0
    var_neg = np.var(neg, ddof=1) if n_neg > 1 else 0

    # Weighted pooled variance
    if n_pos + n_neg > 2:
        pooled_var = ((n_pos - 1) * var_pos + (n_neg - 1) * var_neg) / (n_pos + n_neg - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        pooled_std = 1.0  # Fallback

    if pooled_std < 1e-10:
        return 0.0

    return float(mean_diff / pooled_std)


def dprime_interpretation(dprime: float) -> str:
    """Human-readable interpretation of d-prime."""
    abs_d = abs(dprime)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    elif abs_d < 1.2:
        magnitude = "large"
    else:
        magnitude = "very large"

    direction = "positive" if dprime > 0 else "negative" if dprime < 0 else "zero"
    return f"{magnitude} {direction} effect (d'={dprime:.2f})"


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], float]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        Tuple of (list of significant flags, corrected alpha)
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]
    return significant, corrected_alpha


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Apply Holm-Bonferroni step-down correction.

    More powerful than standard Bonferroni while still controlling FWER.

    Args:
        p_values: List of p-values
        alpha: Family-wise error rate

    Returns:
        Tuple of (list of significant flags, list of adjusted p-values)
    """
    n = len(p_values)

    # Sort p-values and keep track of original indices
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])

    significant = [False] * n
    adjusted = [1.0] * n

    for rank, (p, orig_idx) in enumerate(indexed):
        # Holm correction: compare to alpha / (n - rank)
        threshold = alpha / (n - rank)
        adjusted[orig_idx] = min(p * (n - rank), 1.0)

        if p < threshold:
            significant[orig_idx] = True
        else:
            # Once we fail to reject, stop rejecting
            break

    return significant, adjusted


def compute_icc(
    measurements: List[List[float]],
    icc_type: str = "ICC(2,1)",
) -> float:
    """Compute Intraclass Correlation Coefficient for test-retest reliability.

    ICC(2,1) - Two-way random effects, single measurement
    Used when raters are a random sample and we want to generalize.

    Interpretation:
    - ICC < 0.5: poor reliability
    - 0.5 <= ICC < 0.75: moderate reliability
    - 0.75 <= ICC < 0.9: good reliability
    - ICC >= 0.9: excellent reliability

    Args:
        measurements: List of measurement sets (runs), each containing values for subjects
                     Shape: [n_runs][n_subjects]
        icc_type: Type of ICC (currently only ICC(2,1) implemented)

    Returns:
        ICC value
    """
    # Convert to numpy array: rows = subjects, columns = measurements
    data = np.array(measurements).T  # Shape: [n_subjects, n_runs]

    n_subjects, n_runs = data.shape

    if n_subjects < 2 or n_runs < 2:
        return np.nan

    # Grand mean
    grand_mean = np.mean(data)

    # Subject means (across runs)
    subject_means = np.mean(data, axis=1)

    # Run means (across subjects)
    run_means = np.mean(data, axis=0)

    # Sum of squares
    # Between subjects
    ss_between = n_runs * np.sum((subject_means - grand_mean) ** 2)

    # Within subjects (error + systematic)
    ss_within = np.sum((data - subject_means.reshape(-1, 1)) ** 2)

    # Between runs (systematic)
    ss_runs = n_subjects * np.sum((run_means - grand_mean) ** 2)

    # Residual error
    ss_error = ss_within - ss_runs

    # Mean squares
    ms_between = ss_between / (n_subjects - 1)
    ms_runs = ss_runs / (n_runs - 1) if n_runs > 1 else 0
    ms_error = ss_error / ((n_subjects - 1) * (n_runs - 1))

    # ICC(2,1)
    if ms_error < 1e-10:
        return 1.0  # Perfect agreement

    icc = (ms_between - ms_error) / (ms_between + (n_runs - 1) * ms_error +
                                      n_runs * (ms_runs - ms_error) / n_subjects)

    return float(np.clip(icc, -1, 1))


def icc_interpretation(icc: float) -> str:
    """Human-readable interpretation of ICC."""
    if np.isnan(icc):
        return "insufficient data"
    elif icc < 0.5:
        return f"poor reliability (ICC={icc:.2f})"
    elif icc < 0.75:
        return f"moderate reliability (ICC={icc:.2f})"
    elif icc < 0.9:
        return f"good reliability (ICC={icc:.2f})"
    else:
        return f"excellent reliability (ICC={icc:.2f})"


def compute_axis_statistics(
    baseline_values: List[Dict[str, float]],
    axes: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Dict]:
    """Compute comprehensive statistics for baseline measurements.

    For each axis, computes:
    - Mean with bootstrap CI
    - Significance test against 0
    - Effect size

    Args:
        baseline_values: List of mood readings (dicts of axis->value)
        axes: List of axes to analyze (default: all axes in data)
        n_bootstrap: Bootstrap iterations
        alpha: Significance level

    Returns:
        Dict mapping axis name to statistics
    """
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

        # Bootstrap CI
        ci_result = bootstrap_ci(values, n_bootstrap=n_bootstrap)

        # Significance against 0
        sig_result = test_against_value(
            values, test_value=0.0,
            n_bootstrap=n_bootstrap, alpha=alpha
        )

        results[axis] = {
            "mean": ci_result.mean,
            "std": ci_result.std,
            "ci_lower": ci_result.ci_lower,
            "ci_upper": ci_result.ci_upper,
            "p_value": sig_result.p_value,
            "effect_size": sig_result.effect_size,
            "n_samples": ci_result.n_samples,
        }

        p_values.append((axis, sig_result.p_value))

    # Apply Holm-Bonferroni correction
    if p_values:
        axes_list = [a for a, _ in p_values]
        pvals_list = [p for _, p in p_values]
        significant_corrected, adjusted_pvals = holm_bonferroni_correction(pvals_list, alpha)

        for i, axis in enumerate(axes_list):
            results[axis]["p_value_adjusted"] = adjusted_pvals[i]
            results[axis]["significant_corrected"] = significant_corrected[i]

    return results


def format_statistics_table(
    stats: Dict[str, Dict],
    format_type: str = "markdown",
) -> str:
    """Format statistics as a table.

    Args:
        stats: Output from compute_axis_statistics
        format_type: "markdown" or "latex"

    Returns:
        Formatted table string
    """
    if format_type == "markdown":
        header = "| Axis | Mean | 95% CI | p-value | p (adj) | Sig? | Effect |"
        sep = "|------|------|--------|---------|---------|------|--------|"
        rows = [header, sep]

        for axis, s in sorted(stats.items()):
            sig = "**Yes**" if s.get("significant_corrected", False) else "No"
            effect = f"{s['effect_size']:.2f}" if s.get('effect_size') else "-"
            row = (f"| {axis} | {s['mean']:.3f} | "
                   f"[{s['ci_lower']:.3f}, {s['ci_upper']:.3f}] | "
                   f"{s['p_value']:.4f} | {s.get('p_value_adjusted', s['p_value']):.4f} | "
                   f"{sig} | {effect} |")
            rows.append(row)

        return "\n".join(rows)

    elif format_type == "latex":
        rows = [
            r"\begin{tabular}{lcccccc}",
            r"\hline",
            r"Axis & Mean & 95\% CI & p-value & p (adj) & Sig? & Effect \\",
            r"\hline",
        ]

        for axis, s in sorted(stats.items()):
            sig = r"\textbf{Yes}" if s.get("significant_corrected", False) else "No"
            effect = f"{s['effect_size']:.2f}" if s.get('effect_size') else "-"
            row = (f"{axis} & {s['mean']:.3f} & "
                   f"[{s['ci_lower']:.3f}, {s['ci_upper']:.3f}] & "
                   f"{s['p_value']:.4f} & {s.get('p_value_adjusted', s['p_value']):.4f} & "
                   f"{sig} & {effect} \\\\")
            rows.append(row)

        rows.extend([r"\hline", r"\end{tabular}"])
        return "\n".join(rows)

    return str(stats)
