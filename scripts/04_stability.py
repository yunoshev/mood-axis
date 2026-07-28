#!/usr/bin/env python3
"""Step 04: Axis stability test (3 independent calibration sets).

Proves axis directions are real measurement instruments, not artifacts of
specific question sets. Calibrates axes independently on sets A/B/C with
different questions and style paraphrases, then measures cosine similarity.

Produces:
    data/stability/<model>_set_{A,B,C}_axes.npz   — axis vectors per set
    data/stability/<model>_set_{A,B,C}_meta.json   — metrics per set
    data/stability/<model>.json                    — cross-set analysis (CPU)

Does NOT depend on 01_calibrate (calibrates its own axes).

Usage:
    python scripts/04_stability.py --model qwen_7b calibrate              # all sets
    python scripts/04_stability.py --model qwen_7b calibrate --set B      # one set
    python scripts/04_stability.py --model qwen_7b calibrate --backend mlx
    python scripts/04_stability.py --model qwen_7b analyze                # CPU only
    python scripts/04_stability.py --model qwen_7b --check
"""

import argparse
import gc
import json
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))  # repo root for framework imports

# Optional integration with the private `fw` framework. When it is not
# installed (public/standalone mode), fall back to a minimal emitter that
# keeps the #PROGRESS stdout protocol working.
try:
    from fw.progress import log_progress
except ImportError:
    import json as _progress_json

    def log_progress(done, total, elapsed_s, n_elapsed=None, extra=None):
        """Standalone fallback: emit #PROGRESS lines to stdout."""
        n = n_elapsed if n_elapsed is not None else done
        rate = (elapsed_s / n) if (n and n > 0 and elapsed_s > 0) else None
        eta = rate * (total - done) if rate is not None else None
        data = {"done": done, "total": total,
                "elapsed_s": round(elapsed_s, 1),
                "rate_s": round(rate, 1) if rate is not None else None,
                "eta_s": round(eta, 1) if eta is not None else None}
        if extra:
            data.update(extra)
        print(f"#PROGRESS {_progress_json.dumps(data)}", flush=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
for handler in logging.root.handlers:
    handler.flush = lambda h=handler: h.stream.flush()
logger = logging.getLogger(__name__)

from config.settings import (
    MOOD_AXES, CALIBRATION_MAX_NEW_TOKENS, TEMPERATURE, TOP_P, DO_SAMPLE,
)
from config.models import MODELS
from config.stability_prompts import get_stability_set
from src.calibration.axis_computer import (
    compute_axis_vector, compute_normalization_scale, compute_normalization_center,
    compute_validation_metrics, save_axis_vectors,
)
from src.backend import add_backend_args, resolve_backend, load_generate_fn, run_on_vast

OUTPUT_DIR = PROJECT_ROOT / "data" / "stability"
SET_NAMES = ["A", "B", "C"]

# Global progress tracker for fw status (#PROGRESS protocol)
_progress = {"total": 0, "t_start": 0}
_stab_budget = None

# Sample-granular totals: {(set_name, axis): n_questions}. Computed once from
# config/stability_prompts.py (deterministic, model-independent — the same
# 3 sets x 5 axes x 30 questions for every model) and cached for the whole
# process; never recomputed on the fly so total stays a true constant.
_stab_n_q = {}
_stab_total = None


def _init_stab_totals():
    """Compute and cache the fixed sample-granular total (once per process)."""
    global _stab_total
    if _stab_total is not None:
        return _stab_total
    total = 0
    for s in SET_NAMES:
        questions = get_stability_set(s)["questions"]
        for axis in MOOD_AXES:
            n_q = len(questions.get(axis, []))
            _stab_n_q[(s, axis)] = n_q
            total += n_q
    _stab_total = total
    return total


def _stab_done_total(model_key):
    """(done, total) at SAMPLE granularity, derived from checkpoints on disk.

    done is global across ALL sets/axes, monotonic, and never resets on
    resume: a completed axis contributes its full n_q; an in-progress axis
    contributes its clamped samples_per_axis count; a not-yet-started axis
    contributes 0 (so a completed axis is never double-counted against its
    own samples_per_axis entry). total is the fixed constant from
    _init_stab_totals() — never recomputed mid-run.
    """
    total = _init_stab_totals()
    done = 0
    for s in SET_NAMES:
        ckpt_path = _set_checkpoint_path(model_key, s)
        # A finished set deletes its checkpoint and leaves the final npz —
        # count it in full, or done would DROP by a whole set (150) right
        # after set completion (observed: qwen27b stab 159 -> 63).
        if (not ckpt_path.exists()
                and (OUTPUT_DIR / f"{model_key}_set_{s}_axes.npz").exists()):
            done += sum(_stab_n_q.get((s, axis), 0) for axis in MOOD_AXES)
            continue
        ckpt = _load_set_checkpoint(model_key, s)
        completed = set(ckpt.get("completed_axes", []))
        samples_per_axis = ckpt.get("samples_per_axis", {})
        for axis in MOOD_AXES:
            n_q = _stab_n_q.get((s, axis), 0)
            if axis in completed:
                done += n_q
            else:
                done += min(samples_per_axis.get(axis, 0), n_q)
    return min(done, total), total


def _emit_stab_progress(model_key, item=None, phase_done=None, phase_total=None):
    """Emit #PROGRESS derived from checkpoints, not a mutable accumulator.

    done/total (see _stab_done_total) are integer sample counts — global,
    monotonic, never reset between axes/sets/resume. Local within-axis
    position is carried only in `extra` (item/phase_done/phase_total), never
    folded into done itself.
    """
    done, total = _stab_done_total(model_key)
    el = time.monotonic() - _progress["t_start"] if _progress["t_start"] else 0
    extra = {}
    if item is not None:
        extra["item"] = item
    if phase_done is not None:
        extra["phase_done"] = phase_done
    if phase_total is not None:
        extra["phase_total"] = phase_total
    log_progress(done, total, el,
                 n_elapsed=max(done - _progress.get("done_at_start", 0), 0),
                 extra=extra or None)


def _log(msg):
    logger.info(msg)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "_progress.txt", "w") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
    except Exception:
        pass


def preflight_check(model_key: str):
    assert model_key in MODELS, f"Unknown model: {model_key}"
    print(f"Model: {MODELS[model_key].model_id}", flush=True)
    print(f"Axes: {len(MOOD_AXES)} ({', '.join(MOOD_AXES)})", flush=True)
    print(f"Sets: {SET_NAMES}", flush=True)
    for s in SET_NAMES:
        ss = get_stability_set(s)
        q_per_axis = {a: len(ss["questions"].get(a, [])) for a in MOOD_AXES}
        print(f"  Set {s}: {q_per_axis}", flush=True)
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    print("Preflight OK", flush=True)


# ---------------------------------------------------------------------------
#  Calibration (GPU)
# ---------------------------------------------------------------------------

def _set_checkpoint_path(model_key: str, set_name: str) -> Path:
    return OUTPUT_DIR / f"{model_key}_set_{set_name}_checkpoint.json"


def _load_set_checkpoint(model_key: str, set_name: str) -> dict:
    path = _set_checkpoint_path(model_key, set_name)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_axes": [], "samples_per_axis": {}}


def _save_set_checkpoint(model_key: str, set_name: str, ckpt: dict):
    # Atomic: a kill mid-dump must not truncate the checkpoint (deepseek
    # drift incident 2026-07-23 — plain open("w") left an unparseable stub).
    path = _set_checkpoint_path(model_key, set_name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_partial_states(model_key: str, set_name: str) -> dict:
    """Load partial hidden states NPZ for incremental saving."""
    path = OUTPUT_DIR / f"{model_key}_set_{set_name}_states.npz"
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    return {}


def _save_partial_states(model_key: str, set_name: str, hs_dict: dict):
    path = OUTPUT_DIR / f"{model_key}_set_{set_name}_states.npz"
    # tmp name MUST end in .npz — np.savez appends it otherwise and os.replace
    # then misses the file (35b_a3b crash 2026-07-23 03:19).
    tmp = OUTPUT_DIR / f"{model_key}_set_{set_name}_states.tmp.npz"
    np.savez_compressed(tmp, **hs_dict)
    os.replace(tmp, path)


def calibrate_set(generate_fn, model_key: str, set_name: str, force: bool = False) -> dict:
    """Calibrate all axes using a specific question+style set.

    Saves texts to JSONL after every sample and checkpoints hidden states
    incrementally. Supports resume on crash.
    """
    stability_set = get_stability_set(set_name)
    questions = stability_set["questions"]
    styles = stability_set["styles"]
    yi_overrides = stability_set.get("yi_overrides", {})

    model_config = MODELS.get(model_key)
    if model_config and getattr(model_config, "style_overrides", None) == "yi":
        styles = dict(styles)
        for axis, overrides in yi_overrides.items():
            styles[axis] = overrides

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    texts_file = OUTPUT_DIR / f"{model_key}_set_{set_name}_texts.jsonl"

    # Load checkpoint for resume
    ckpt = _load_set_checkpoint(model_key, set_name)
    hs_dict = _load_partial_states(model_key, set_name)

    if force:
        ckpt = {"completed_axes": [], "samples_per_axis": {}}
        hs_dict = {}
        texts_file.unlink(missing_ok=True)

    results = {
        "model": model_key, "set": set_name,
        "timestamp": datetime.now().isoformat(), "axes": {},
    }
    axis_vectors = {}
    scales = {}
    centers = {}

    for axis_idx, axis in enumerate(MOOD_AXES):
        # Skip completed axes
        if axis in ckpt.get("completed_axes", []):
            _log(f"  [{set_name}] {axis}: already done (resume), loading from disk")
            # Recover axis vector from partial states
            pos_key = f"{axis}_pos"
            neg_key = f"{axis}_neg"
            if pos_key in hs_dict and neg_key in hs_dict:
                pos_states = list(hs_dict[pos_key])
                neg_states = list(hs_dict[neg_key])
                av = compute_axis_vector(pos_states, neg_states)
                axis_vectors[axis] = av
                all_proj = [float(np.dot(s, av)) for s in pos_states + neg_states]
                scales[axis] = compute_normalization_scale(all_proj)
                centers[axis] = compute_normalization_center(all_proj)
                n_val = max(1, len(pos_states) // 5)
                val_metrics = compute_validation_metrics(av, pos_states[-n_val:], neg_states[-n_val:])
                results["axes"][axis] = {
                    "n_questions": len(pos_states),
                    "scale": float(scales[axis]),
                    "accuracy": val_metrics["accuracy"],
                    "dprime": val_metrics["dprime"],
                    "truncated": 0,  # unknown for resumed, but axis is complete
                }
            continue

        axis_questions = questions.get(axis, [])
        if not axis_questions:
            _log(f"  [{set_name}] {axis}: no questions, skipping")
            continue

        style_pos = styles[axis]["positive"]
        style_neg = styles[axis]["negative"]
        n_q = len(axis_questions)

        # Check how many samples are already done for this axis
        done_samples = ckpt.get("samples_per_axis", {}).get(axis, 0)
        pos_key = f"{axis}_pos"
        neg_key = f"{axis}_neg"

        # m5: initialize per-axis so the 'already on disk' branch doesn't
        # inherit values from the previous axis via stale locals
        truncated = 0
        t_axis = time.monotonic()

        # Validate checkpoint against saved states
        if pos_key in hs_dict:
            saved_n = hs_dict[pos_key].shape[0]
            if saved_n < done_samples:
                done_samples = saved_n
            elif saved_n > done_samples:
                done_samples = saved_n

        if done_samples >= n_q:
            # All samples done but axis not marked complete — recompute axis
            _log(f"  [{set_name}] {axis}: all {n_q} samples on disk, computing axis")
            positive_states = list(hs_dict[pos_key])
            negative_states = list(hs_dict[neg_key])
        else:
            _log(f"  [{set_name}] Axis {axis_idx+1}/{len(MOOD_AXES)}: {axis} "
                 f"({n_q} questions x 2 poles"
                 f"{f', resuming from {done_samples}' if done_samples > 0 else ''})")

            # Pre-load completed samples from disk
            if done_samples > 0 and pos_key in hs_dict:
                positive_states = list(hs_dict[pos_key][:done_samples])
                negative_states = list(hs_dict[neg_key][:done_samples])
            else:
                positive_states = []
                negative_states = []
                done_samples = 0

            for i in range(done_samples, n_q):
                question = axis_questions[i]

                # Positive pole
                result_pos = generate_fn(
                    [{"role": "system", "content": style_pos},
                     {"role": "user", "content": question}],
                    max_new_tokens=CALIBRATION_MAX_NEW_TOKENS,
                )
                positive_states.append(result_pos.hidden_state)

                # Negative pole
                result_neg = generate_fn(
                    [{"role": "system", "content": style_neg},
                     {"role": "user", "content": question}],
                    max_new_tokens=CALIBRATION_MAX_NEW_TOKENS,
                )
                negative_states.append(result_neg.hidden_state)

                # Save texts to JSONL immediately
                pos_truncated = result_pos.n_generated_tokens >= CALIBRATION_MAX_NEW_TOKENS
                neg_truncated = result_neg.n_generated_tokens >= CALIBRATION_MAX_NEW_TOKENS
                for pole, result, is_trunc in [
                    ("positive", result_pos, pos_truncated),
                    ("negative", result_neg, neg_truncated),
                ]:
                    entry = {
                        "axis": axis, "pole": pole, "question_idx": i,
                        "question": question, "response": result.text,
                        "n_tokens": result.n_generated_tokens,
                        "generation_time_s": round(result.generation_time_s, 2),
                        "truncated": is_trunc,
                    }
                    with open(texts_file, "a") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        f.flush()

                if pos_truncated:
                    truncated += 1
                if neg_truncated:
                    truncated += 1

                # Save hidden states incrementally
                hs_dict[pos_key] = np.stack(positive_states)
                hs_dict[neg_key] = np.stack(negative_states)
                _save_partial_states(model_key, set_name, hs_dict)

                # Update checkpoint
                ckpt.setdefault("samples_per_axis", {})[axis] = i + 1
                _save_set_checkpoint(model_key, set_name, ckpt)

                # Chunk boundary at SAMPLE granularity (slow models: one axis can
                # exceed the whole budget; per-sample checkpoint makes this safe).
                if _stab_budget is not None and _stab_budget.expired():
                    _emit_stab_progress(model_key, item=f"{set_name}/{axis}",
                                        phase_done=i + 1, phase_total=n_q)
                    done_now, total_now = _stab_done_total(model_key)
                    _stab_budget.check_and_exit(done_now, total_now, log=_log)

                # Per-sample progress emit (bar moves every sample even on 27B+
                # where one axis takes 1-2h); log line stays every 5 samples.
                if _progress["t_start"] > 0:
                    _emit_stab_progress(model_key, item=f"{set_name}/{axis}",
                                        phase_done=i + 1, phase_total=n_q)
                n_done = i + 1 - done_samples
                if n_done % 5 == 0 or i + 1 == n_q:
                    elapsed = time.monotonic() - t_axis
                    per_q = elapsed / n_done
                    eta = per_q * (n_q - i - 1)
                    _log(f"    {i+1}/{n_q} ({per_q:.1f}s/q, ETA {eta:.0f}s)")

        # Compute axis
        axis_vector = compute_axis_vector(positive_states, negative_states)
        axis_vectors[axis] = axis_vector

        all_proj = [float(np.dot(s, axis_vector)) for s in positive_states + negative_states]
        scale = compute_normalization_scale(all_proj)
        scales[axis] = scale
        centers[axis] = compute_normalization_center(all_proj)

        # Validation on last 20% of samples
        n_val = max(1, len(positive_states) // 5)
        val_metrics = compute_validation_metrics(
            axis_vector, positive_states[-n_val:], negative_states[-n_val:],
        )

        axis_truncated = truncated
        results["axes"][axis] = {
            "n_questions": n_q,
            "scale": float(scale),
            "accuracy": val_metrics["accuracy"],
            "dprime": val_metrics["dprime"],
            "truncated": axis_truncated,
        }

        axis_time = time.monotonic() - t_axis
        _log(f"  [{set_name}] {axis}: d'={val_metrics['dprime']:.2f}, "
             f"acc={val_metrics['accuracy']:.0%}, "
             f"truncated={axis_truncated}/{n_q*2} "
             f"({axis_time:.0f}s)")

        # Mark axis as completed in checkpoint
        ckpt.setdefault("completed_axes", []).append(axis)
        _save_set_checkpoint(model_key, set_name, ckpt)

        _emit_stab_progress(model_key, item=f"{set_name}/{axis}",
                            phase_done=n_q, phase_total=n_q)

        # Clean chunk boundary for fw2 --repeat-until-exit (resume is per-axis)
        if _stab_budget is not None:
            done, total = _stab_done_total(model_key)
            _stab_budget.check_and_exit(done, total, log=_log)

    # Save final outputs
    npz_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
    save_axis_vectors(axis_vectors, normalization_scales=scales,
                      normalization_centers=centers, output_path=npz_path)

    json_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_meta.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Clean up checkpoint and partial states (final data is in npz + meta + texts)
    _set_checkpoint_path(model_key, set_name).unlink(missing_ok=True)
    partial_states_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_states.npz"
    partial_states_path.unlink(missing_ok=True)

    _log(f"  [{set_name}] Saved: {npz_path.name}, {json_path.name}, {texts_file.name}")
    return results


def calibrate_set_modal(model_key: str, set_name: str, gpu: str = "A100") -> dict:
    """Calibrate one set on Modal GPU."""
    stability_set = get_stability_set(set_name)
    questions = stability_set["questions"]
    styles = stability_set["styles"]

    # Build sample list for Modal
    samples = []
    for axis in MOOD_AXES:
        axis_questions = questions.get(axis, [])
        style_pos = styles[axis]["positive"]
        style_neg = styles[axis]["negative"]

        for i, q in enumerate(axis_questions):
            samples.append({
                "id": f"{axis}_pos_{i}",
                "axis": axis,
                "pole": "positive",
                "messages": [
                    {"role": "system", "content": style_pos},
                    {"role": "user", "content": q},
                ],
            })
            samples.append({
                "id": f"{axis}_neg_{i}",
                "axis": axis,
                "pole": "negative",
                "messages": [
                    {"role": "system", "content": style_neg},
                    {"role": "user", "content": q},
                ],
            })

    _log(f"[{set_name}] Sending {len(samples)} samples to Modal ({gpu})...")

    from modal_runner import app, run_calibration_samples

    with app.run():
        results_list = run_calibration_samples.remote(
            model_key=model_key, samples=samples,
            max_new_tokens=CALIBRATION_MAX_NEW_TOKENS,
        )

    _log(f"[{set_name}] Received {len(results_list)} results from Modal")

    # Organize by axis and pole
    states = {axis: {"positive": [], "negative": []} for axis in MOOD_AXES}
    for r in results_list:
        sid = r["id"]
        axis = "_".join(sid.split("_")[:-2])  # axis_pos_0 -> axis
        pole = sid.split("_")[-2]  # positive or negative
        states[axis][pole].append(r["hidden_state"])

    # Compute axes
    results = {
        "model": model_key, "set": set_name,
        "timestamp": datetime.now().isoformat(), "axes": {},
    }
    axis_vectors = {}
    scales = {}
    centers = {}

    for axis in MOOD_AXES:
        pos = states[axis]["positive"]
        neg = states[axis]["negative"]
        if not pos or not neg:
            continue

        axis_vector = compute_axis_vector(pos, neg)
        axis_vectors[axis] = axis_vector

        all_proj = [float(np.dot(s, axis_vector)) for s in pos + neg]
        scale = compute_normalization_scale(all_proj)
        scales[axis] = scale
        centers[axis] = compute_normalization_center(all_proj)

        n_val = max(1, len(pos) // 5)
        val_metrics = compute_validation_metrics(
            axis_vector, pos[-n_val:], neg[-n_val:],
        )

        results["axes"][axis] = {
            "n_questions": len(pos),
            "scale": float(scale),
            "accuracy": val_metrics["accuracy"],
            "dprime": val_metrics["dprime"],
        }
        _log(f"  [{set_name}] {axis}: d'={val_metrics['dprime']:.2f}, "
             f"acc={val_metrics['accuracy']:.0%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
    save_axis_vectors(axis_vectors, normalization_scales=scales,
                      normalization_centers=centers, output_path=npz_path)

    json_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_meta.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    _log(f"  [{set_name}] Saved: {npz_path.name}")
    return results


# ---------------------------------------------------------------------------
#  Analysis (CPU only)
# ---------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _icc_from_matrix(ratings: np.ndarray) -> float:
    """Core ICC(2,1) computation from a ratings matrix.

    Args:
        ratings: (n_items, k_raters) matrix of scores.
    Returns:
        ICC value in [-1, 1]. Higher = more reliable.
    """
    n, k = ratings.shape
    if n < 2 or k < 2:
        return 0.0

    grand_mean = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)

    # Sum of squares
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1) if k > 1 else 0.0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10

    # ICC(2,1)
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-10:
        return 0.0
    return float((ms_rows - ms_error) / denom)


def compute_icc(ratings: np.ndarray) -> float:
    """ICC(2,1) on raw scores — absolute agreement."""
    return _icc_from_matrix(ratings)


def compute_rank_icc(ratings: np.ndarray) -> float:
    """ICC(2,1) on rank-transformed scores — consistency of ordering.

    Removes scale differences between raters: each rater's scores are
    replaced by their ranks within that rater. This answers "do sets
    agree on which texts are more X?" rather than "do they assign the
    same number?"
    """
    from scipy.stats import rankdata
    ranked = np.column_stack([
        rankdata(ratings[:, j]) for j in range(ratings.shape[1])
    ])
    return _icc_from_matrix(ranked)


def procrustes_congruence(V1: np.ndarray, V2: np.ndarray) -> float:
    """Orthogonal Procrustes alignment + congruence coefficient.

    Args:
        V1, V2: (k_axes, hidden_dim) matrices of axis vectors.
    Returns:
        Congruence coefficient in [0, 1]. Higher = more similar spaces.
    """
    # Normalize rows (each axis vector to unit length)
    V1n = V1 / (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-10)
    V2n = V2 / (np.linalg.norm(V2, axis=1, keepdims=True) + 1e-10)

    # Orthogonal Procrustes: find R minimizing ||V1n - V2n @ R||
    M = V1n.T @ V2n  # (d, d)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = Vt.T @ U.T  # optimal rotation

    V2_aligned = V2n @ R

    # Per-axis congruence after alignment (cosine between corresponding rows)
    per_axis = np.sum(V1n * V2_aligned, axis=1)  # (k_axes,)

    # Overall congruence: mean absolute per-axis congruence
    return float(np.mean(np.abs(per_axis)))


def load_hidden_states_for_icc(model_key: str) -> Dict[str, np.ndarray]:
    """Load validation hidden states from main calibration for ICC.

    Returns: {axis: (n_val*2, hidden_dim)} — val pos + val neg states.
    """
    hs_path = PROJECT_ROOT / "data" / "axes" / f"{model_key}_hidden_states.npz"
    if not hs_path.exists():
        return {}

    data = np.load(hs_path, allow_pickle=True)
    states = {}
    for axis in MOOD_AXES:
        pos_key = f"{axis}_val_pos_aggregated"
        neg_key = f"{axis}_val_neg_aggregated"
        if pos_key in data and neg_key in data:
            states[axis] = np.concatenate([data[pos_key], data[neg_key]], axis=0)
    return states


def analyze_model(model_key: str) -> dict:
    """Analyze axis stability across sets (CPU only).

    Computes:
    1. Per-axis cosine similarity (original metric)
    2. Procrustes congruence (full-space alignment)
    3. ICC(2,1) on held-out hidden states (psychometric reliability)
    """
    vectors: Dict[str, Dict[str, np.ndarray]] = {}
    for set_name in SET_NAMES:
        npz_path = OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            axis_names = data["_axes"].tolist()
            vectors[set_name] = {a: data[a] for a in axis_names}
            _log(f"Loaded set {set_name}: {len(axis_names)} axes")

    if len(vectors) < 2:
        print(f"ERROR: Need at least 2 sets, found {len(vectors)}. "
              f"Run 'calibrate' first.", flush=True)
        sys.exit(1)

    available_sets = sorted(vectors.keys())
    set_pairs = [(a, b) for i, a in enumerate(available_sets)
                 for b in available_sets[i+1:]]

    results = {
        "_meta": {
            "script": "scripts/04_stability.py",
            "model": model_key,
            "model_id": MODELS[model_key].model_id,
            "timestamp": datetime.now().isoformat(),
            "sets_compared": available_sets,
            "max_new_tokens": CALIBRATION_MAX_NEW_TOKENS,
            "generation_settings": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": DO_SAMPLE,
            },
        },
        "cosine_similarity": {},
        "per_axis_summary": {},
    }

    # --- 1. Per-axis cosine similarity (original) ---
    for set_a, set_b in set_pairs:
        pair_key = f"{set_a}_vs_{set_b}"
        results["cosine_similarity"][pair_key] = {}
        common_axes = sorted(set(vectors[set_a].keys()) & set(vectors[set_b].keys()))
        for axis in common_axes:
            cos = cosine_similarity(vectors[set_a][axis], vectors[set_b][axis])
            results["cosine_similarity"][pair_key][axis] = cos

    all_axes = set()
    for pair_data in results["cosine_similarity"].values():
        all_axes.update(pair_data.keys())

    for axis in sorted(all_axes):
        cos_values = [pair_data[axis] for pair_data in results["cosine_similarity"].values()
                      if axis in pair_data]
        mean_cos = float(np.mean(cos_values)) if cos_values else None
        results["per_axis_summary"][axis] = {
            "mean_cosine": mean_cos,
            "min_cosine": float(np.min(cos_values)) if cos_values else None,
            "stable": mean_cos > 0.8 if mean_cos is not None else None,
        }

    # --- 2. Procrustes congruence (full-space) ---
    common_axes_all = sorted(set.intersection(
        *[set(vectors[s].keys()) for s in available_sets]
    ))

    procrustes_results = {}
    for set_a, set_b in set_pairs:
        V1 = np.stack([vectors[set_a][a] for a in common_axes_all])
        V2 = np.stack([vectors[set_b][a] for a in common_axes_all])
        cong = procrustes_congruence(V1, V2)
        procrustes_results[f"{set_a}_vs_{set_b}"] = cong

    procrustes_values = list(procrustes_results.values())
    results["procrustes"] = {
        "pairs": procrustes_results,
        "mean_congruence": float(np.mean(procrustes_values)),
        "min_congruence": float(np.min(procrustes_values)),
        "axes_used": common_axes_all,
    }

    # --- 3. ICC on held-out hidden states ---
    _log("Loading hidden states for ICC...")
    hidden_states = load_hidden_states_for_icc(model_key)
    if hidden_states and len(available_sets) >= 2:
        # Pre-load stability NPZ scales/centers (avoid reloading inside loop)
        set_npz_cache = {}
        for set_name in available_sets:
            set_npz_cache[set_name] = np.load(
                OUTPUT_DIR / f"{model_key}_set_{set_name}_axes.npz"
            )

        icc_results = {}
        rank_icc_results = {}
        for axis_idx, axis in enumerate(common_axes_all):
            if axis not in hidden_states:
                continue
            _log(f"  ICC axis {axis_idx+1}/{len(common_axes_all)}: {axis}")
            states = hidden_states[axis]  # (n_items, hidden_dim)
            n_items = states.shape[0]

            # Project each item onto axis vectors from each set (vectorized)
            ratings = np.zeros((n_items, len(available_sets)))
            for j, set_name in enumerate(available_sets):
                ax_vec = vectors[set_name][axis]
                npz = set_npz_cache[set_name]
                scale_key = f"{axis}_scale"
                scale = float(npz[scale_key]) if scale_key in npz else 2.0
                projs = states @ ax_vec  # vectorized dot product
                center_key = f"{axis}_center"
                if center_key in npz:
                    center = float(npz[center_key])
                else:
                    # Legacy set NPZs don't store a center; per methodology (§3.2)
                    # scores are median-centered. center=0 would clip offset
                    # projections to ±1 (e.g. Gemma 3 offsets by -40k..-80k)
                    # and destroy ICC.
                    logger.warning(
                        f"legacy set NPZ (set {set_name}, axis {axis}) without center — "
                        f"using median-of-projections fallback; "
                        f"ICC is consistency-type, not absolute-agreement"
                    )
                    center = float(np.median(projs))
                ratings[:, j] = np.clip((projs - center) / scale, -1.0, 1.0)

            icc_val = compute_icc(ratings)
            rank_icc_val = compute_rank_icc(ratings)
            icc_results[axis] = round(icc_val, 4)
            rank_icc_results[axis] = round(rank_icc_val, 4)

        results["icc"] = {
            "per_axis": icc_results,
            "mean_icc": round(float(np.mean(list(icc_results.values()))), 4) if icc_results else None,
            "n_items_per_axis": next(iter(hidden_states.values())).shape[0] if hidden_states else 0,
            "note": "ICC(2,1) on validation hidden states from main calibration projected onto each set's axes",
        }
        results["rank_icc"] = {
            "per_axis": rank_icc_results,
            "mean_rank_icc": round(float(np.mean(list(rank_icc_results.values()))), 4) if rank_icc_results else None,
            "note": "ICC(2,1) on rank-transformed projections — measures ordering consistency, ignores scale differences",
        }
        _log(f"ICC computed: {len(icc_results)} axes, mean={results['icc']['mean_icc']}, "
             f"rank_icc mean={results['rank_icc']['mean_rank_icc']}")
    else:
        results["icc"] = {"note": "Skipped: no hidden states found for ICC computation"}
        _log("ICC skipped: no hidden_states.npz found")

    # --- Save ---
    output_file = OUTPUT_DIR / f"{model_key}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"Saved to {output_file}")

    # --- Print summary ---
    print(f"\nAxis Stability: {model_key}", flush=True)
    print(f"  {'Axis':<25s} {'Cosine':>7s} {'ICC':>7s} {'RankICC':>8s}  Status", flush=True)
    print(f"  {'-'*60}", flush=True)
    icc_per = results.get("icc", {}).get("per_axis", {})
    ricc_per = results.get("rank_icc", {}).get("per_axis", {})
    for axis in sorted(all_axes):
        summary = results["per_axis_summary"][axis]
        cos_str = f"{summary['mean_cosine']:.3f}" if summary["mean_cosine"] is not None else "  N/A"
        icc_str = f"{icc_per[axis]:.3f}" if axis in icc_per else "    N/A"
        ricc_str = f"{ricc_per[axis]:.3f}" if axis in ricc_per else "     N/A"
        stable = "STABLE" if summary.get("stable") else "UNSTABLE"
        print(f"  {axis:<25s} {cos_str:>7s} {icc_str:>7s} {ricc_str:>8s}  {stable}", flush=True)

    proc = results.get("procrustes", {})
    if proc:
        print(f"\n  Procrustes congruence: {proc['mean_congruence']:.3f} "
              f"(min={proc['min_congruence']:.3f})", flush=True)

    icc_data = results.get("icc", {})
    if icc_data.get("mean_icc") is not None:
        ricc_data = results.get("rank_icc", {})
        print(f"  ICC(2,1) mean: {icc_data['mean_icc']:.3f}, "
              f"Rank ICC mean: {ricc_data.get('mean_rank_icc', 'N/A'):.3f} "
              f"(n={icc_data['n_items_per_axis']} items/axis)", flush=True)

    return results


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 04: Axis stability")
    parser.add_argument("--model", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    from src.time_budget import add_time_budget_args, TimeBudget
    add_time_budget_args(parser)
    add_backend_args(parser)
    sub = parser.add_subparsers(dest="command")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--set", choices=SET_NAMES, help="Calibrate only this set")
    sub.add_parser("analyze")
    args = parser.parse_args()

    global _stab_budget
    _stab_budget = TimeBudget(getattr(args, "time_budget_min", 0),
                              getattr(args, "time_budget_exit_code", 75))

    if args.check:
        preflight_check(args.model)
        return

    if args.command == "calibrate":
        backend = resolve_backend(args.backend)
        _log(f"Backend: {backend}")

        set_arg = args.set if hasattr(args, "set") else None
        sets = [set_arg] if set_arg else SET_NAMES

        if backend == "vast":
            extra = f"calibrate"
            if set_arg:
                extra += f" --set {set_arg}"
            run_on_vast(
                __file__, args.model, gpu=args.gpu, extra_args=extra,
                sync_paths=[
                    f"data/stability/{args.model}_set_*_meta.json",
                    f"data/stability/{args.model}_set_*_texts.jsonl",
                    f"data/stability/{args.model}_set_*_checkpoint.json",
                    f"data/stability/{args.model}_set_*_states.npz",
                ],
                final_paths=[
                    f"data/stability/{args.model}_set_*.npz",
                    f"data/stability/{args.model}_set_*.json",
                    f"data/stability/{args.model}_set_*_texts.jsonl",
                ],
            )
            return

        if backend == "modal":
            for s in sets:
                _log(f"=== Calibrating set {s} on Modal ({args.gpu}) ===")
                calibrate_set_modal(args.model, s, gpu=args.gpu)
            return

        # Local (pytorch / mlx)
        # Skip sets that already have output, unless --force
        # Sets with a checkpoint file are resumable (not "already done")
        sets_to_run = []
        for s in sets:
            npz_path = OUTPUT_DIR / f"{args.model}_set_{s}_axes.npz"
            ckpt_path = _set_checkpoint_path(args.model, s)
            if npz_path.exists() and not ckpt_path.exists() and not args.force:
                _log(f"=== Set {s}: already done (skip, use --force to rerun) ===")
            else:
                sets_to_run.append(s)

        if not sets_to_run:
            _log("All sets already done. Use --force to rerun.")
            return

        # Progress is in SAMPLE units over ALL sets (fixed total = 3 sets x 5
        # axes x 30 questions = 450), derived from checkpoints each emit —
        # never a per-attempt accumulator (would break on chunked resume: the
        # "200/150" bug).
        _progress["total"] = _init_stab_totals()
        _progress["t_start"] = time.monotonic()
        _progress["done_at_start"], _ = _stab_done_total(args.model)
        _emit_stab_progress(args.model)

        model_config = MODELS[args.model]
        generate_fn, cleanup_fn = load_generate_fn(
            model_config, backend,
            max_new_tokens=CALIBRATION_MAX_NEW_TOKENS, log_fn=_log)

        for s in sets_to_run:
            _log(f"=== Calibrating set {s} ===")
            calibrate_set(generate_fn, args.model, s, force=args.force)

        cleanup_fn()

    elif args.command == "analyze":
        analyze_model(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
