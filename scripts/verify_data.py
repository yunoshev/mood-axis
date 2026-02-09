#!/usr/bin/env python3
"""Verify pipeline output data quality.

Runs post-hoc checks on calibration, baseline, and drift data.
Catches garbage output (zero vectors, missing spaces, NaN projections)
before it reaches articles.

Usage:
    # All checks for a model
    python scripts/verify_data.py --model yi_9b

    # Specific check
    python scripts/verify_data.py --check tokenizer --model yi_9b
    python scripts/verify_data.py --check calibration --model yi_9b
    python scripts/verify_data.py --check baseline --model yi_9b
    python scripts/verify_data.py --check drift --model yi_9b

    # Cross-model comparison
    python scripts/verify_data.py --check compare
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MOOD_AXES, AXES_DIR, DATA_DIR
from config.models import MODELS, get_model_config

BASELINES_DIR = DATA_DIR / "article" / "baselines"
DRIFT_DIR = DATA_DIR / "article" / "extended_drift"

# Counters
_pass = 0
_warn = 0
_fail = 0


def PASS(msg: str):
    global _pass
    _pass += 1
    print(f"  [PASS] {msg}")


def WARN(msg: str):
    global _warn
    _warn += 1
    print(f"  [WARN] {msg}")


def FAIL(msg: str):
    global _fail
    _fail += 1
    print(f"  [FAIL] {msg}")


def INFO(msg: str):
    print(f"  [INFO] {msg}")


def section(title: str):
    print(f"\n=== VERIFY: {title} ===")


def text_has_spaces(text: str) -> bool:
    return " " in text and len(text) > 20


# ── Tokenizer ──────────────────────────────────────────────────────────────

def check_tokenizer(model_key: str):
    section(f"{model_key} tokenizer")
    config = get_model_config(model_key)

    try:
        from transformers import AutoTokenizer, AutoConfig
    except ImportError:
        WARN("transformers not installed, skipping tokenizer check")
        return

    # Load tokenizer
    try:
        tok = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    except Exception as e:
        FAIL(f"Cannot load tokenizer: {e}")
        return

    # Decode test
    test_text = "Hello, how are you?"
    ids = tok.encode(test_text, add_special_tokens=False)
    decoded = tok.decode(ids)
    if " " in decoded:
        PASS(f'Decode preserves spaces: "{decoded}"')
    else:
        FAIL(f'Decode strips spaces: "{decoded}"')

    # EOS consistency
    try:
        model_cfg = AutoConfig.from_pretrained(config.model_id, trust_remote_code=True)
        cfg_eos = model_cfg.eos_token_id
        tok_eos = tok.eos_token_id
        # cfg_eos can be int or list
        if isinstance(cfg_eos, list):
            cfg_eos_set = set(cfg_eos)
        else:
            cfg_eos_set = {cfg_eos}
        if tok_eos in cfg_eos_set or cfg_eos == tok_eos:
            PASS(f"EOS token: config={cfg_eos}, tokenizer={tok_eos} (match)")
        else:
            FAIL(f"EOS mismatch: config={cfg_eos}, tokenizer={tok_eos}")
    except Exception as e:
        WARN(f"Cannot check EOS: {e}")

    # Chat template
    try:
        tok.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True
        )
        PASS("Chat template: OK")
    except Exception as e:
        FAIL(f"Chat template failed: {e}")


# ── Calibration ────────────────────────────────────────────────────────────

def check_calibration(model_key: str):
    section(f"{model_key} calibration")
    axes_file = AXES_DIR / f"{model_key}_axes.npz"

    if not axes_file.exists():
        FAIL(f"Axes file not found: {axes_file}")
        return

    data = np.load(axes_file, allow_pickle=True)
    axes_in_file = list(data["_axes"])

    # All 7 axes present
    missing = set(MOOD_AXES) - set(axes_in_file)
    if not missing:
        PASS(f"{len(MOOD_AXES)}/{len(MOOD_AXES)} axes present")
    else:
        FAIL(f"Missing axes: {sorted(missing)}")

    # Check vectors and scales
    norms = []
    scales = []
    has_nan = False
    for axis in MOOD_AXES:
        if axis not in axes_in_file:
            continue
        vec = data[axis]
        scale_key = f"{axis}_scale"
        scale = float(data[scale_key]) if scale_key in data else None

        norm = np.linalg.norm(vec)
        norms.append(norm)

        if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
            has_nan = True
        if scale is not None:
            if np.isnan(scale) or np.isinf(scale):
                has_nan = True
            scales.append(scale)

    # No NaN/inf
    if has_nan:
        FAIL("NaN or inf found in vectors or scales")
    else:
        PASS("No NaN/inf in vectors or scales")

    # Vector norms > 0.01
    if norms and min(norms) > 0.01:
        PASS(f"Vector norms: {min(norms):.2f}-{max(norms):.2f} (all non-zero)")
    else:
        FAIL(f"Zero vector detected (min norm={min(norms):.4f})")

    # Scales in sane range (varies with hidden dim, 500 is generous upper bound)
    if scales:
        if all(0.01 < s < 500 for s in scales):
            PASS(f"Scales: {min(scales):.1f}-{max(scales):.1f} (sane range)")
        else:
            FAIL(f"Scales out of range: {min(scales):.4f}-{max(scales):.4f}")

    # Check calibration JSONL if exists
    jsonl_file = AXES_DIR / f"{model_key}_calibration.jsonl"
    if jsonl_file.exists():
        lines = jsonl_file.read_text().strip().split("\n")
        records = [json.loads(l) for l in lines if l.strip()]
        expected = len(MOOD_AXES) * 2 * 30  # 7 axes × 2 poles × 30 questions
        if len(records) == expected:
            PASS(f"{len(records)} calibration responses (expected {expected})")
        else:
            WARN(f"{len(records)} calibration responses (expected {expected})")

        # Sample text quality
        import random
        samples = random.sample(records, min(5, len(records)))
        good = sum(1 for s in samples if text_has_spaces(s.get("response", "")))
        if good == len(samples):
            PASS(f"Text quality: spaces present in {good}/{len(samples)} samples")
        else:
            FAIL(f"Text quality: spaces in only {good}/{len(samples)} samples")


# ── Baseline ───────────────────────────────────────────────────────────────

def check_baseline(model_key: str):
    section(f"{model_key} baseline")
    baseline_file = BASELINES_DIR / f"{model_key}_baseline.json"

    if not baseline_file.exists():
        FAIL(f"Baseline file not found: {baseline_file}")
        return

    data = json.loads(baseline_file.read_text())
    axes_data = data.get("axes", {})
    responses = data.get("responses", [])

    # All 7 axes present
    missing = set(MOOD_AXES) - set(axes_data.keys())
    if not missing:
        PASS(f"{len(MOOD_AXES)}/{len(MOOD_AXES)} axes present")
    else:
        FAIL(f"Missing axes: {sorted(missing)}")

    # Response count
    n_responses = data.get("n_questions", len(responses))
    if n_responses == 30:
        PASS(f"{n_responses} responses")
    else:
        WARN(f"{n_responses} responses (expected 30)")

    # Per-axis checks
    all_have_variance = True
    dead_zones = []
    min_std = float("inf")
    has_nan = False

    for axis in MOOD_AXES:
        if axis not in axes_data:
            continue
        ad = axes_data[axis]
        mean = ad["mean"]
        std = ad["std"]
        values = ad.get("values", [])

        if np.isnan(mean) or np.isnan(std) or any(np.isnan(v) for v in values):
            has_nan = True

        min_std = min(min_std, std)

        if std == 0:
            all_have_variance = False
        elif abs(mean) < 0.01 and std < 0.05:
            all_have_variance = False

        if abs(mean) < 0.15 and std < 0.10:
            dead_zones.append(axis)

    if has_nan:
        FAIL("NaN found in axis values")
    else:
        PASS("No NaN in axis values")

    if all_have_variance:
        PASS(f"All axes have variance (min std={min_std:.3f})")
    else:
        FAIL("Some axes have zero or near-zero variance")

    # Text quality from responses
    if responses:
        import random
        samples = random.sample(responses, min(3, len(responses)))
        good = sum(1 for s in samples if text_has_spaces(s.get("response", "")))
        if good == len(samples):
            PASS(f"Text quality: spaces in {good}/{len(samples)} samples")
        else:
            FAIL(f"Text quality: spaces in only {good}/{len(samples)} samples")

        # Token counts
        token_counts = [r.get("n_tokens", 0) for r in responses if "n_tokens" in r]
        if token_counts:
            tmin, tmax = min(token_counts), max(token_counts)
            if all(10 <= t <= 400 for t in token_counts):
                PASS(f"Token counts: {tmin}-{tmax} (range OK)")
            else:
                WARN(f"Token counts: {tmin}-{tmax} (some outside 10-400)")

        # Text length
        lengths = [len(r.get("response", "")) for r in responses]
        short = sum(1 for l in lengths if l < 50)
        if short == 0:
            PASS(f"Response lengths: {min(lengths)}-{max(lengths)} chars (all >50)")
        else:
            WARN(f"{short}/{len(lengths)} responses shorter than 50 chars")

    # PCA
    all_values = []
    for axis in MOOD_AXES:
        if axis in axes_data and "values" in axes_data[axis]:
            all_values.append(axes_data[axis]["values"])
    if len(all_values) == len(MOOD_AXES):
        matrix = np.array(all_values).T  # (n_responses, n_axes)
        matrix_centered = matrix - matrix.mean(axis=0)
        cov = np.cov(matrix_centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) > 0:
            total_var = eigenvalues.sum()
            pc1_pct = eigenvalues[0] / total_var * 100
            # Effective dimensionality (participation ratio)
            probs = eigenvalues / total_var
            eff_dim = 1.0 / np.sum(probs ** 2) if np.sum(probs ** 2) > 0 else 0
            INFO(f"PCA: PC1={pc1_pct:.1f}%, eff dim={eff_dim:.2f}")
            if eff_dim < 1.0:
                FAIL(f"PCA collapsed: eff dim={eff_dim:.2f}")

    # Dead zones
    INFO(f"Dead zones: {len(dead_zones)}/{len(MOOD_AXES)}" +
         (f" ({', '.join(dead_zones)})" if dead_zones else ""))


# ── Drift ──────────────────────────────────────────────────────────────────

def check_drift(model_key: str):
    section(f"{model_key} drift")
    drift_file = DRIFT_DIR / f"{model_key}_drift.json"

    if not drift_file.exists():
        FAIL(f"Drift file not found: {drift_file}")
        return

    data = json.loads(drift_file.read_text())
    scenarios = data.get("scenarios", [])
    n_scenarios = data.get("n_scenarios", len(scenarios))

    # Scenario count
    if n_scenarios >= 20:
        PASS(f"{n_scenarios} scenarios")
    else:
        WARN(f"{n_scenarios} scenarios (expected 20)")

    # Turn counts
    turn_counts = [len(s.get("turns", [])) for s in scenarios]
    short_scenarios = sum(1 for t in turn_counts if t < 3)
    if short_scenarios == 0:
        PASS(f"Turn counts: {min(turn_counts)}-{max(turn_counts)} (all >= 3)")
    else:
        FAIL(f"{short_scenarios} scenarios with < 3 turns")

    # NaN check and constant check
    has_nan = False
    all_constant = 0
    for scenario in scenarios:
        axis_series = {a: [] for a in MOOD_AXES}
        for turn in scenario.get("turns", []):
            values = turn.get("values", {})
            for axis in MOOD_AXES:
                v = values.get(axis)
                if v is not None:
                    if np.isnan(v):
                        has_nan = True
                    axis_series[axis].append(v)

        # Check if all axes are constant (no change at all)
        constant_axes = 0
        for axis, series in axis_series.items():
            if len(series) > 1 and len(set(f"{v:.6f}" for v in series)) == 1:
                constant_axes += 1
        if constant_axes == len(MOOD_AXES):
            all_constant += 1

    if has_nan:
        FAIL("NaN found in drift values")
    else:
        PASS("No NaN in drift values")

    if all_constant == 0:
        PASS("All scenarios show variation across turns")
    else:
        WARN(f"{all_constant} scenarios with all axes constant")

    # Sample text quality
    if scenarios:
        import random
        s = random.choice(scenarios)
        turns = s.get("turns", [])
        if turns:
            t = random.choice(turns)
            text = t.get("assistant", "")
            if text_has_spaces(text):
                PASS(f"Sample text has spaces ({len(text)} chars)")
            else:
                FAIL(f"Sample text missing spaces: '{text[:80]}...'")


# ── Cross-model comparison ─────────────────────────────────────────────────

def check_compare():
    section("cross-model comparison")

    # Find all baseline files
    baseline_files = sorted(BASELINES_DIR.glob("*_baseline.json"))
    if not baseline_files:
        FAIL(f"No baseline files found in {BASELINES_DIR}")
        return

    rows = []
    for bf in baseline_files:
        data = json.loads(bf.read_text())
        model = data.get("model", bf.stem.replace("_baseline", ""))
        axes_data = data.get("axes", {})

        # Count dead zones
        dead_zones = 0
        all_zero = True
        for axis in MOOD_AXES:
            if axis not in axes_data:
                continue
            mean = axes_data[axis].get("mean", 0)
            std = axes_data[axis].get("std", 0)
            if abs(mean) < 0.15 and std < 0.10:
                dead_zones += 1
            if abs(mean) > 0.01 or std > 0.01:
                all_zero = False

        # PCA
        all_values = []
        for axis in MOOD_AXES:
            if axis in axes_data and "values" in axes_data[axis]:
                all_values.append(axes_data[axis]["values"])

        pc1_pct = None
        eff_dim = None
        if len(all_values) == len(MOOD_AXES):
            matrix = np.array(all_values).T
            matrix_centered = matrix - matrix.mean(axis=0)
            cov = np.cov(matrix_centered, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(cov)[::-1]
            eigenvalues = eigenvalues[eigenvalues > 0]
            if len(eigenvalues) > 0:
                total_var = eigenvalues.sum()
                pc1_pct = eigenvalues[0] / total_var * 100
                probs = eigenvalues / total_var
                eff_dim = 1.0 / np.sum(probs ** 2) if np.sum(probs ** 2) > 0 else 0

        rows.append({
            "model": model,
            "dead_zones": dead_zones,
            "pc1_pct": pc1_pct,
            "eff_dim": eff_dim,
            "all_zero": all_zero,
        })

    # Print table
    print(f"\n  {'Model':<22} {'Dead zones':>10} {'PC1%':>8} {'Eff dim':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8}")
    for r in rows:
        pc1 = f"{r['pc1_pct']:.1f}" if r['pc1_pct'] is not None else "N/A"
        ed = f"{r['eff_dim']:.2f}" if r['eff_dim'] is not None else "N/A"
        print(f"  {r['model']:<22} {r['dead_zones']:>10} {pc1:>8} {ed:>8}")

    # Flag outliers
    flagged = False
    for r in rows:
        if r["all_zero"]:
            FAIL(f"{r['model']}: all axes at zero")
            flagged = True
        if r["eff_dim"] is not None and r["eff_dim"] < 1.1:
            WARN(f"{r['model']}: eff dim={r['eff_dim']:.2f} (< 1.1, collapsed)")
            flagged = True
        if r["eff_dim"] is not None and r["eff_dim"] > 5.0:
            WARN(f"{r['model']}: eff dim={r['eff_dim']:.2f} (> 5.0, unusually high)")
            flagged = True

    if not flagged:
        PASS("No outliers detected")


# ── Main ───────────────────────────────────────────────────────────────────

ALL_CHECKS = ["tokenizer", "calibration", "baseline", "drift"]


def run_checks(checks: list[str], model_key: str | None):
    for check in checks:
        if check == "compare":
            check_compare()
        elif model_key is None:
            print(f"Error: --model required for '{check}' check")
            sys.exit(1)
        elif check == "tokenizer":
            check_tokenizer(model_key)
        elif check == "calibration":
            check_calibration(model_key)
        elif check == "baseline":
            check_baseline(model_key)
        elif check == "drift":
            check_drift(model_key)


def main():
    global _pass, _warn, _fail

    parser = argparse.ArgumentParser(description="Verify pipeline output data quality")
    parser.add_argument("--model", help="Model key (e.g., yi_9b)")
    parser.add_argument("--check", choices=ALL_CHECKS + ["compare", "all"],
                        default="all", help="Which check to run (default: all)")
    args = parser.parse_args()

    _pass = _warn = _fail = 0

    if args.check == "compare":
        checks = ["compare"]
    elif args.check == "all":
        if args.model:
            checks = ALL_CHECKS
        else:
            checks = ["compare"]
    else:
        checks = [args.check]

    # Validate model
    if args.model and args.model not in MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)

    run_checks(checks, args.model)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"PASS: {_pass}, WARN: {_warn}, FAIL: {_fail}")

    sys.exit(1 if _fail > 0 else 0)


if __name__ == "__main__":
    main()
