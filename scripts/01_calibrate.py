#!/usr/bin/env python3
"""Step 01: Calibrate axis vectors for a model.

Inputs:
    data/calibration/<axis>.json    — 50 cal + 15 eval questions + style instructions per axis
    config/settings.py              — MOOD_AXES (5 axes), generation params
    config/models.py                — model registry (HuggingFace IDs)

Outputs:
    data/axes/<model>_axes.npz              — axis direction vectors + normalization scales
    data/axes/<model>_axes_meta.json        — accuracy, d-prime, per-axis metrics
    data/axes/<model>_calibration.jsonl     — all generated responses (appended per-sample)
    data/axes/<model>_hidden_states.npz     — per-layer + aggregated hidden states

Incremental design (D21):
    - Responses appended to JSONL after EVERY sample (not batched)
    - Hidden states checkpoint saved every 10 samples per group
    - Axes saved after each axis completes
    - Resume support: skips completed axes/samples on restart

Train/Val split:
    Train = 50 calibration questions (data/calibration/<axis>.json → calibration_questions)
    Val   = 15 eval questions (data/calibration/<axis>.json → eval_questions, zero overlap)

Backend:
    --backend pytorch   Use PyTorch (HF transformers). Default on CUDA/CPU.
    --backend mlx       Use MLX (native bf16 weights). 2-5x faster on Apple Silicon. Requires: pip install mlx mlx-lm

Usage:
    python scripts/01_calibrate.py --model qwen_7b
    python scripts/01_calibrate.py --model qwen_7b --backend mlx    # 2-5x faster on Mac
    python scripts/01_calibrate.py --model qwen_7b --axes warm_cold,confident_cautious
    python scripts/01_calibrate.py --model qwen_7b --check          # preflight only
    python scripts/01_calibrate.py --model qwen_7b --force          # recompute all
    python scripts/01_calibrate.py --model qwen_7b --save-tokens    # per-token data (~500MB)
"""

import argparse
import json
import sys
import gc
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# Force unbuffered output for background/remote runs
import os
_log_file = os.environ.get("MOOD_LOG_FILE")


class _AutoFlush:
    """Wraps a file object to flush after every write."""
    def __init__(self, f):
        self._f = f
    def write(self, s):
        self._f.write(s)
        self._f.flush()
    def flush(self):
        self._f.flush()
    def fileno(self):
        return self._f.fileno()


if _log_file:
    # Script handles its own logging via MOOD_LOG_FILE.
    # Shell stdout is redirected to /dev/null (see ssh.py).
    _log_f = open(_log_file, 'a')
    _writer = _AutoFlush(_log_f)
    sys.stdout = _writer
    sys.stderr = _writer
    _handler = logging.StreamHandler(_writer)
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[_handler])
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr,
    )
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# Optional integration with the global `fw` framework. When it is not
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

# Global progress tracker for fw status (#PROGRESS protocol)
_global_progress = {"done": 0, "total": 0, "t_start": 0}
_time_budget_deadline = None  # monotonic deadline; set from --time-budget-min
_time_budget_exit_code = 0  # exit code on budget stop (75 for fw2 repeat-until-exit)

# Stall watchdog: MLX generation can wedge forever (observed on qwen3.6_27b —
# process spins at 100% CPU, generation never returns). If no sample completes
# within MOOD_STALL_TIMEOUT seconds, dump all thread stacks (root-cause
# evidence in the log) and exit with the time-budget code so the fw2
# repeat-until-exit loop restarts cleanly from the checkpoint.
_stall_s = int(os.environ.get("MOOD_STALL_TIMEOUT", "1500"))


def _touch_heartbeat():
    """Re-arm the C-level stall watchdog.

    MLX wedges inside native code WITHOUT releasing the GIL, so a Python
    watchdog thread never gets scheduled. faulthandler's watchdog runs at C
    level (no GIL needed): if not re-armed within MOOD_STALL_TIMEOUT seconds
    it dumps all thread stacks to stderr and hard-exits (code 1), letting the
    fw2 repeat-until-exit loop restart from the checkpoint.
    """
    if _stall_s > 0:
        import faulthandler
        faulthandler.dump_traceback_later(_stall_s, exit=True)


def _start_stall_watchdog():
    if _stall_s > 0:
        _log(f"Stall watchdog armed: faulthandler dump+exit after {_stall_s}s "
             f"without a finished sample")
        _touch_heartbeat()

from config.settings import (
    MOOD_AXES, AXES_DIR,
    CALIBRATION_SAMPLES_PER_STYLE, CALIBRATION_EVAL_SAMPLES,
    CALIBRATION_MAX_NEW_TOKENS, CALIBRATION_MAX_NEW_TOKENS_PER_AXIS,
    TEMPERATURE, TOP_P, DO_SAMPLE,
)
from config.models import MODELS
from src.calibration.dataset import (
    generate_calibration_dataset,
    get_eval_questions,
    get_style_instruction,
    CalibrationSample,
)
from src.calibration.axis_computer import (
    compute_axis_vector,
    compute_normalization_scale,
    compute_normalization_center,
    compute_dprime,
    save_axis_vectors,
)
from src.backend import add_backend_args, resolve_backend, run_on_vast


def _log(msg):
    """Log + write to progress file (bypasses pipe buffering)."""
    logger.info(msg)
    try:
        progress_file = AXES_DIR / "_progress.txt"
        with open(progress_file, "w") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
    except Exception:
        pass


def preflight_check(model_key: str):
    """Run preflight checks without GPU."""
    assert model_key in MODELS, f"Unknown model: {model_key}. Available: {list(MODELS.keys())}"
    config = MODELS[model_key]

    from config.settings import CALIBRATION_DIR
    for axis in MOOD_AXES:
        axis_file = CALIBRATION_DIR / f"{axis}.json"
        assert axis_file.exists(), f"Missing calibration data: {axis_file}"

    print(f"Model:      {config.model_id} ({config.display_name})", flush=True)
    print(f"Hidden:     {config.hidden_dim}", flush=True)
    print(f"Axes:       {len(MOOD_AXES)} ({', '.join(MOOD_AXES)})", flush=True)
    print(f"Train:      {CALIBRATION_SAMPLES_PER_STYLE} cal questions per pole", flush=True)
    print(f"Val:        {CALIBRATION_EVAL_SAMPLES} eval questions per pole (independent)", flush=True)
    print(f"Max tokens: {CALIBRATION_MAX_NEW_TOKENS}", flush=True)
    output_file = AXES_DIR / f"{model_key}_axes.npz"
    print(f"Output:     {output_file} (exists: {output_file.exists()})", flush=True)
    print("Preflight OK", flush=True)


def _build_eval_samples(axis: str) -> list:
    """Build CalibrationSample list from eval questions for an axis."""
    eval_questions = get_eval_questions(axis, CALIBRATION_EVAL_SAMPLES)
    samples = []
    for pole in ["positive", "negative"]:
        style = get_style_instruction(axis, pole)
        for q in eval_questions:
            samples.append(CalibrationSample(
                axis=axis, pole=pole, question=q,
                system_prompt=style, user_prompt=q,
            ))
    return samples


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(model_key: str) -> Path:
    return AXES_DIR / f"{model_key}_checkpoint.json"


def _load_checkpoint(model_key: str) -> dict:
    """Load checkpoint. Returns dict with completed axes and per-group sample counts."""
    path = _checkpoint_path(model_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_axes": [], "groups": {}}


def _save_checkpoint(model_key: str, ckpt: dict):
    """Save checkpoint to disk."""
    path = _checkpoint_path(model_key)
    with open(path, "w") as f:
        json.dump(ckpt, f, indent=2)
        f.flush()


def _hidden_states_path(model_key: str) -> Path:
    return AXES_DIR / f"{model_key}_hidden_states.npz"


def _load_partial_hidden_states(model_key: str) -> dict:
    """Load existing partial hidden states from disk."""
    path = _hidden_states_path(model_key)
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    return {}


def _save_partial_hidden_states(model_key: str, hs_dict: dict):
    """Save hidden states dict to disk (compressed)."""
    path = _hidden_states_path(model_key)
    np.savez_compressed(path, **hs_dict)


# ---------------------------------------------------------------------------
#  Core collection with incremental saves
# ---------------------------------------------------------------------------

def _collect_states_incremental(
    generate_fn, samples, group_key, model_key,
    save_tokens=False, checkpoint_every=10, max_new_tokens=None,
):
    """Collect hidden states with per-sample saves.

    Every sample result is saved to disk immediately after generation.
    On resume: loads completed samples from disk, continues from where it stopped.

    Args:
        generate_fn: callable(messages, save_extra=True) -> GenerationResult
        group_key: e.g. "warm_cold_train_pos" — used for checkpoint tracking
        model_key: e.g. "qwen_7b" — used for file paths
        checkpoint_every: log progress every N samples

    Returns same tuple as old _collect_states.
    """
    total = len(samples)

    # Effective token cap for this group (used for truncation check + resume guard)
    effective_max_tokens = max_new_tokens if max_new_tokens is not None else CALIBRATION_MAX_NEW_TOKENS

    # Check how many samples are already done on disk
    ckpt = _load_checkpoint(model_key)
    done_count = ckpt.get("groups", {}).get(group_key, 0)
    # Samples of this group already counted into _global_progress["done"] at
    # startup (done_from_partial) — used below to avoid double counting on resume.
    _counted_at_init = min(done_count, total)

    # M5: refuse to resume across a token-cap change (states would be inconsistent)
    recorded_cap = ckpt.get("group_max_tokens", {}).get(group_key)
    if recorded_cap is not None and done_count > 0 and recorded_cap != effective_max_tokens:
        axis_name = group_key.rsplit("_", 2)[0]
        logger.error(
            f"  {group_key}: checkpoint was created with max_new_tokens={recorded_cap}, "
            f"current effective cap is {effective_max_tokens}. Refusing to resume — "
            f"partial states would mix token caps. "
            f"Rerun with --force --axes {axis_name} to recompute this axis."
        )
        raise RuntimeError(
            f"Token cap changed for {group_key} ({recorded_cap} -> {effective_max_tokens}); "
            f"use --force --axes {axis_name}"
        )

    # Load existing hidden states from disk
    hs = _load_partial_hidden_states(model_key)
    agg_key = f"{group_key}_aggregated"

    # Validate: checkpoint says N done, verify hidden states match
    if agg_key in hs:
        saved_n = hs[agg_key].shape[0]
        if done_count == 0:
            # No checkpoint but hidden states exist — trust states
            _log(f"  {group_key}: no checkpoint but found {saved_n} states on disk, using them")
            done_count = saved_n
        elif saved_n < done_count:
            _log(f"  {group_key}: checkpoint={done_count} but only {saved_n} states on disk, resuming from {saved_n}")
            done_count = saved_n
        elif saved_n > done_count:
            # States ahead of checkpoint — trust states
            done_count = saved_n
    elif done_count > 0:
        _log(f"  {group_key}: checkpoint={done_count} but no states on disk, restarting")
        done_count = 0

    # Reconcile startup accounting (done_from_partial counted the checkpoint
    # value) with the disk-validated done_count — prevents done > total on
    # resumed attempts (double counting) and corrects stale checkpoints.
    _global_progress["done"] += min(done_count, total) - _counted_at_init

    if done_count >= total:
        _log(f"  {group_key}: loaded from disk ({total} samples)")
        g = _global_progress
        g_elapsed = time.monotonic() - g["t_start"] if g["t_start"] else 0
        log_progress(g["done"], g["total"], g_elapsed,
                     n_elapsed=g["done"] - g["done_at_start"])
        states = list(hs[agg_key])
        lt_key = f"{group_key}_last_token"
        last_token_states = list(hs[lt_key]) if lt_key in hs else []
        ly_key = f"{group_key}_layers"
        per_layer_list = list(hs[ly_key]) if ly_key in hs else []
        return states, last_token_states, per_layer_list, [], [], 0

    # Pre-load completed samples from disk
    if done_count > 0:
        _log(f"  {group_key}: resuming from sample {done_count}/{total}")
        states = list(hs[agg_key][:done_count])
        lt_key = f"{group_key}_last_token"
        last_token_states = list(hs[lt_key][:done_count]) if lt_key in hs else []
        ly_key = f"{group_key}_layers"
        per_layer_list = list(hs[ly_key][:done_count]) if ly_key in hs else []
    else:
        states = []
        last_token_states = []
        per_layer_list = []

    token_data = []
    responses = []
    truncated_count = 0
    t_start = time.monotonic()

    responses_file = AXES_DIR / f"{model_key}_calibration.jsonl"
    pending_rows = []  # JSONL rows buffered until their states are persisted (M3)

    # Warm the shared-prefix prompt cache on the pole's system prompt
    # (no-op unless MOOD_PROMPT_CACHE=1 and the backend supports it)
    if done_count < total and hasattr(generate_fn, "warm"):
        first = samples[done_count]
        n_warmed = generate_fn.warm([
            {"role": "system", "content": first.system_prompt},
            {"role": "user", "content": first.user_prompt},
        ])
        if n_warmed:
            _log(f"  {group_key}: prompt cache warmed ({n_warmed} tokens)")

    for i in range(done_count, total):
        sample = samples[i]
        messages = [
            {"role": "system", "content": sample.system_prompt},
            {"role": "user", "content": sample.user_prompt},
        ]
        _touch_heartbeat()
        result = generate_fn(messages, save_extra=True, max_new_tokens=max_new_tokens)
        _touch_heartbeat()

        states.append(result.hidden_state)

        if result.hidden_state_last_token is not None:
            last_token_states.append(result.hidden_state_last_token)

        if result.per_layer_states is not None:
            per_layer_list.append(result.per_layer_states)

        is_truncated = result.n_generated_tokens >= effective_max_tokens
        if is_truncated:
            truncated_count += 1
            logger.warning(
                f"  TRUNCATED: {group_key} #{i} — "
                f"{result.n_generated_tokens} tokens (max={effective_max_tokens})"
            )

        if save_tokens:
            td = {}
            if result.token_states is not None:
                td["token_states"] = result.token_states
            if result.top_k_logprobs is not None:
                td["top_k_logprobs"] = result.top_k_logprobs
            if result.top_k_ids is not None:
                td["top_k_ids"] = result.top_k_ids
            token_data.append(td)

        resp_entry = {
            "axis": sample.axis, "pole": sample.pole,
            "split": group_key.split("_")[-2] if "_" in group_key else "unknown",
            "question": sample.user_prompt,
            "response": result.text,
            "n_tokens": result.n_generated_tokens,
            "truncated": is_truncated,
            "generation_time_s": round(result.generation_time_s, 2),
        }
        responses.append(resp_entry)
        pending_rows.append(resp_entry)

        # Update global progress
        _global_progress["done"] += 1

        n = i + 1
        # Checkpoint every checkpoint_every samples and at group end (m7):
        # npz + checkpoint are saved first, then the matching JSONL rows are
        # flushed — rows on disk always correspond to persisted states (M3).
        if n == total or n % checkpoint_every == 0:
            # 1. Save hidden states to disk
            hs[agg_key] = np.stack(states)
            if last_token_states:
                hs[f"{group_key}_last_token"] = np.stack(last_token_states)
            if per_layer_list:
                try:
                    hs[f"{group_key}_layers"] = np.stack(per_layer_list)
                except ValueError:
                    pass  # Shape mismatch in per-layer data (non-essential)
            _save_partial_hidden_states(model_key, hs)

            # 2. Update checkpoint count + effective token cap (M5)
            ckpt = _load_checkpoint(model_key)
            ckpt.setdefault("groups", {})[group_key] = n
            ckpt.setdefault("group_max_tokens", {})[group_key] = effective_max_tokens
            _save_checkpoint(model_key, ckpt)

            # 3. Flush buffered JSONL rows (states for them are now persisted)
            with open(responses_file, "a") as f:
                for row in pending_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            pending_rows = []

            elapsed = time.monotonic() - t_start
            n_new = n - done_count
            per_sample = elapsed / n_new
            remaining = total - n
            eta = per_sample * remaining
            eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta >= 60 else f"{eta:.0f}s"
            _log(f"  {group_key}: {n}/{total} ({per_sample:.1f}s/sample, ETA {eta_str})")
            # Emit global progress for fw status
            g = _global_progress
            g_elapsed = time.monotonic() - g["t_start"] if g["t_start"] else elapsed
            log_progress(g["done"], g["total"], g_elapsed,
                         n_elapsed=g["done"] - g["done_at_start"])

            # Time budget: exit cleanly at a checkpoint so short queue slots
            # free up for other projects; a re-queued job resumes from here.
            if _time_budget_deadline and time.monotonic() > _time_budget_deadline and n < total:
                _log(f"  {group_key}: TIME BUDGET reached at {n}/{total} — "
                     f"exiting (code {_time_budget_exit_code}), resume will continue")
                sys.exit(_time_budget_exit_code)

    if truncated_count > 0:
        pct = truncated_count / total * 100
        logger.warning(
            f"  {group_key}: {truncated_count}/{total} ({pct:.0f}%) truncated "
            f"at {effective_max_tokens} tokens"
        )

    _log(f"  {group_key}: DONE ({total} samples saved)")

    return states, last_token_states, per_layer_list, token_data, responses, truncated_count


# ---------------------------------------------------------------------------
#  Per-axis calibration
# ---------------------------------------------------------------------------

def calibrate_axis(generate_fn, axis: str, model_key: str,
                   save_tokens: bool = False,
                   is_base_model: bool = False) -> dict:
    """Calibrate a single axis with incremental saves."""
    n_train = CALIBRATION_SAMPLES_PER_STYLE
    n_val = CALIBRATION_EVAL_SAMPLES
    # Per-axis token limit (D19). Base models keep their 1024 cap (EOS control):
    # the closure default (base_max_tokens) applies when we pass None.
    axis_max_tokens = None
    if not is_base_model:
        axis_max_tokens = CALIBRATION_MAX_NEW_TOKENS_PER_AXIS.get(axis)
    _log(f"Calibrating axis: {axis} (train={n_train}, val={n_val} per pole)")

    # Train samples: 50 calibration questions × 2 poles
    train_dataset = generate_calibration_dataset(
        num_samples_per_style=n_train, axes=[axis],
    )
    train_pos = [s for s in train_dataset if s.pole == "positive"]
    train_neg = [s for s in train_dataset if s.pole == "negative"]

    # Val samples: 15 eval questions × 2 poles (independent, zero overlap)
    val_samples = _build_eval_samples(axis)
    val_pos = [s for s in val_samples if s.pole == "positive"]
    val_neg = [s for s in val_samples if s.pole == "negative"]

    _log(f"  Collecting train positive ({len(train_pos)} samples)...")
    tp_st, tp_lt, tp_ly, tp_tk, tp_rsp, tp_tr = _collect_states_incremental(
        generate_fn, train_pos, f"{axis}_train_pos", model_key,
        save_tokens=save_tokens, max_new_tokens=axis_max_tokens,
    )
    _log(f"  Collecting train negative ({len(train_neg)} samples)...")
    tn_st, tn_lt, tn_ly, tn_tk, tn_rsp, tn_tr = _collect_states_incremental(
        generate_fn, train_neg, f"{axis}_train_neg", model_key,
        save_tokens=save_tokens, max_new_tokens=axis_max_tokens,
    )
    _log(f"  Collecting val positive ({len(val_pos)} samples)...")
    vp_st, vp_lt, vp_ly, vp_tk, vp_rsp, vp_tr = _collect_states_incremental(
        generate_fn, val_pos, f"{axis}_val_pos", model_key,
        save_tokens=save_tokens, max_new_tokens=axis_max_tokens,
    )
    _log(f"  Collecting val negative ({len(val_neg)} samples)...")
    vn_st, vn_lt, vn_ly, vn_tk, vn_rsp, vn_tr = _collect_states_incremental(
        generate_fn, val_neg, f"{axis}_val_neg", model_key,
        save_tokens=save_tokens, max_new_tokens=axis_max_tokens,
    )

    total_truncated = tp_tr + tn_tr + vp_tr + vn_tr
    total_samples = 2 * (n_train + n_val)

    # Compute axis vector from ALL train data
    axis_vector = compute_axis_vector(tp_st, tn_st)

    # Normalization scale and center from train projections
    all_train_proj = [float(np.dot(s, axis_vector)) for s in tp_st + tn_st]
    scale = compute_normalization_scale(all_train_proj)
    center = compute_normalization_center(all_train_proj)

    # Train metrics
    train_pos_proj = [float(np.dot(s, axis_vector)) for s in tp_st]
    train_neg_proj = [float(np.dot(s, axis_vector)) for s in tn_st]
    train_dprime = compute_dprime(train_pos_proj, train_neg_proj)
    train_separation = float(np.mean(train_pos_proj) - np.mean(train_neg_proj))

    # Val metrics (independent eval questions)
    val_pos_proj = [float(np.dot(s, axis_vector)) for s in vp_st]
    val_neg_proj = [float(np.dot(s, axis_vector)) for s in vn_st]
    val_dprime = compute_dprime(val_pos_proj, val_neg_proj)
    val_separation = float(np.mean(val_pos_proj) - np.mean(val_neg_proj))

    correct = sum(1 for p in val_pos_proj for n in val_neg_proj if p > n)
    total_pairs = len(val_pos_proj) * len(val_neg_proj)
    val_accuracy = correct / total_pairs if total_pairs > 0 else 0

    _log(
        f"  {axis}: train_d'={train_dprime:.2f}, val_d'={val_dprime:.2f}, "
        f"val_acc={val_accuracy*100:.1f}%, truncated={total_truncated}/{total_samples}"
    )

    # Collect hidden states into arrays
    hs = {}
    for prefix, st, lt, ly in [
        ("train_pos", tp_st, tp_lt, tp_ly),
        ("train_neg", tn_st, tn_lt, tn_ly),
        ("val_pos", vp_st, vp_lt, vp_ly),
        ("val_neg", vn_st, vn_lt, vn_ly),
    ]:
        if ly:
            try:
                hs[f"{prefix}_layers"] = np.stack(ly)
            except ValueError:
                pass  # Shape mismatch in per-layer data (non-essential)
        if st:
            hs[f"{prefix}_aggregated"] = np.stack(st)
        if lt:
            hs[f"{prefix}_last_token"] = np.stack(lt)

    # Collect token data
    token_data = {}
    if save_tokens:
        for prefix, tk in [("train_pos", tp_tk), ("train_neg", tn_tk),
                           ("val_pos", vp_tk), ("val_neg", vn_tk)]:
            if tk:
                for idx, td in enumerate(tk):
                    for k, v in td.items():
                        token_data[f"{prefix}_{idx}_{k}"] = v

    return {
        "axis_vector": axis_vector,
        "scale": scale,
        "center": center,
        "metrics": {
            "train_separation": train_separation,
            "train_dprime": train_dprime,
            "val_separation": val_separation,
            "val_dprime": val_dprime,
            "val_accuracy": val_accuracy,
            "n_train_per_pole": n_train,
            "n_val_per_pole": n_val,
            "n_truncated": total_truncated,
            "pct_truncated": round(total_truncated / total_samples * 100, 1),
            "max_new_tokens": axis_max_tokens or CALIBRATION_MAX_NEW_TOKENS,
        },
        "responses": tp_rsp + tn_rsp + vp_rsp + vn_rsp,
        "hidden_states": hs,
        "token_data": token_data,
    }


# ---------------------------------------------------------------------------
#  Top-level: calibrate all axes with per-axis saves
# ---------------------------------------------------------------------------

def calibrate_model(model_key: str, axes: list = None, force: bool = False,
                    save_tokens: bool = False, backend: str = "pytorch"):
    """Calibrate all axes for a model with incremental saves."""
    model_config = MODELS[model_key]
    axes = axes or list(MOOD_AXES)

    AXES_DIR.mkdir(parents=True, exist_ok=True)
    output_file = AXES_DIR / f"{model_key}_axes.npz"
    responses_file = AXES_DIR / f"{model_key}_calibration.jsonl"

    # Check what's already done
    ckpt = _load_checkpoint(model_key)
    completed_axes = ckpt.get("completed_axes", [])

    if force:
        if axes == list(MOOD_AXES):
            # Full force: clear everything
            for f in [output_file, responses_file, _checkpoint_path(model_key),
                      _hidden_states_path(model_key)]:
                f.unlink(missing_ok=True)
            completed_axes = []
            ckpt = {"completed_axes": [], "groups": {}}
        else:
            # Partial force: only clear specified axes, keep the rest
            completed_axes = [a for a in completed_axes if a not in axes]
            groups_to_remove = [k for k in ckpt.get("groups", {})
                                if any(k.startswith(a + "_") for a in axes)]
            for k in groups_to_remove:
                ckpt["groups"].pop(k, None)
                ckpt.get("group_max_tokens", {}).pop(k, None)
            ckpt["completed_axes"] = completed_axes
            # Remove axis-specific hidden states from NPZ
            hs = _load_partial_hidden_states(model_key)
            keys_to_remove = [k for k in hs if any(k.startswith(a + "_") for a in axes)]
            for k in keys_to_remove:
                del hs[k]
            if hs:
                _save_partial_hidden_states(model_key, hs)
        _save_checkpoint(model_key, ckpt)
    else:
        # Skip completed axes
        remaining = [a for a in axes if a not in completed_axes]
        if not remaining:
            _log(f"All axes already calibrated. Use --force to recompute.")
            return {}
        if completed_axes:
            _log(f"Resuming: {completed_axes} done, {remaining} remaining")
        axes = remaining

    # Load model and create generate_fn based on backend
    template_kwargs = model_config.chat_template_kwargs
    # Base models use completion format (Q:/A:) — stop when model starts a new question
    # or at the end of the first paragraph (\n\n). Also cap max_new_tokens to 1024:
    # 300-400 words is more than enough to capture personality signal for calibration.
    # Without this cap, base models generate until 4096 token limit (26%+ truncation).
    if model_config.is_base_model:
        base_stop_strings = ["\nQ:", "\nQuestion:", "\n\n"]
        base_max_tokens = 1024
    else:
        base_stop_strings = None
        base_max_tokens = CALIBRATION_MAX_NEW_TOKENS
    _cleanup_fn = None

    if backend == "mlx":
        from src.model.mlx_inference import (
            load_mlx_model,
            get_full_result_for_prompt as mlx_get_result,
            warm_prompt_cache,
        )
        _log(f"Loading model: {model_config.model_id} (MLX)")
        ctx = load_mlx_model(model_config.model_id, quantize=model_config.quantize,
                            mlx_model_id=model_config.mlx_model_id)

        def generate_fn(messages, save_extra=True, max_new_tokens=None):
            return mlx_get_result(
                ctx, messages,
                max_new_tokens=max_new_tokens or base_max_tokens,
                chat_template_kwargs=template_kwargs,
                save_extra=save_extra,
                stop_strings=base_stop_strings,
            )

        # Shared-prefix prompt cache warm-up (no-op unless MOOD_PROMPT_CACHE=1)
        generate_fn.warm = lambda messages: warm_prompt_cache(
            ctx, messages, chat_template_kwargs=template_kwargs)

        def _cleanup_fn():
            del ctx["model"]
            gc.collect()
    else:
        from src.model.loader import load_model
        from src.model.inference import get_full_result_for_prompt
        import torch

        _log(f"Loading model: {model_config.model_id} (PyTorch)")
        model, tokenizer = load_model(model_config.model_id)

        def generate_fn(messages, save_extra=True, max_new_tokens=None):
            return get_full_result_for_prompt(
                model, tokenizer, messages,
                max_new_tokens=max_new_tokens or base_max_tokens,
                chat_template_kwargs=template_kwargs,
                save_extra=save_extra,
                stop_strings=base_stop_strings,
            )

        def _cleanup_fn():
            nonlocal model, tokenizer
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load any previously saved hidden states (from completed axes)
    hidden_states_dict = _load_partial_hidden_states(model_key)
    all_metrics = {}

    # Global progress for fw status
    n_train = CALIBRATION_SAMPLES_PER_STYLE
    n_val = CALIBRATION_EVAL_SAMPLES
    samples_per_axis = 2 * n_train + 2 * n_val  # train_pos + train_neg + val_pos + val_neg
    _global_progress["total"] = (len(axes) + len(completed_axes)) * samples_per_axis
    _global_progress["t_start"] = time.monotonic()
    # Count already-completed samples from checkpoint
    ckpt = _load_checkpoint(model_key)
    completed_axes = ckpt.get("completed_axes", [])
    done_from_completed = len(completed_axes) * samples_per_axis
    groups = ckpt.get("groups", {})
    # Count only partials of THIS run's axes: the shared checkpoint may hold
    # leftovers of other axes (chunked jobs) which are outside `total`.
    done_from_partial = sum(v for k, v in groups.items()
                           if any(k.startswith(f"{a}_") for a in axes))
    _global_progress["done"] = done_from_completed + done_from_partial
    _global_progress["done_at_start"] = _global_progress["done"]
    log_progress(_global_progress["done"], _global_progress["total"], 0)

    for i_axis, axis in enumerate(axes):
        _log(f"=== Axis {i_axis+1}/{len(axes)}: {axis} ===")

        result = calibrate_axis(
            generate_fn, axis, model_key,
            save_tokens=save_tokens,
            is_base_model=model_config.is_base_model,
        )

        all_metrics[axis] = result["metrics"]

        # Merge hidden states
        for key, arr in result["hidden_states"].items():
            hidden_states_dict[f"{axis}_{key}"] = arr

        # --- SAVE AFTER EACH AXIS (D21) ---

        # 1. Axis vectors: accumulate from existing + new
        all_axis_vectors = {}
        all_scales = {}
        all_centers = {}
        if output_file.exists():
            old_data = np.load(output_file, allow_pickle=True)
            old_axes = old_data["_axes"].tolist() if "_axes" in old_data else []
            # The _axes index has historically gone out of sync with the file
            # contents (6 models found on 2026-07-24 with calibrated axes
            # missing from the index, silently dropped by every consumer).
            # Merge from the union so a stale index can never discard data.
            old_axes = list(dict.fromkeys(
                old_axes + [a for a in MOOD_AXES if a in old_data]))
            for a in old_axes:
                if a in old_data:
                    all_axis_vectors[a] = old_data[a]
                if f"{a}_scale" in old_data:
                    all_scales[a] = float(old_data[f"{a}_scale"])
                if f"{a}_center" in old_data:
                    all_centers[a] = float(old_data[f"{a}_center"])
        all_axis_vectors[axis] = result["axis_vector"]
        all_scales[axis] = result["scale"]
        all_centers[axis] = result["center"]
        save_axis_vectors(all_axis_vectors, all_scales, all_centers, output_file)

        # 2. Hidden states
        sample_layers = next(
            (v for k, v in hidden_states_dict.items() if k.endswith("_layers")), None
        )
        hs_save = dict(hidden_states_dict)
        if sample_layers is not None:
            hs_save["_n_layers"] = np.array(sample_layers.shape[1])
            hs_save["_hidden_dim"] = np.array(sample_layers.shape[2])
        hs_save["_train_size"] = np.array(CALIBRATION_SAMPLES_PER_STYLE)
        hs_save["_val_size"] = np.array(CALIBRATION_EVAL_SAMPLES)
        hs_save["_axes"] = np.array(list(all_axis_vectors.keys()))
        _save_partial_hidden_states(model_key, hs_save)
        hs_mb = _hidden_states_path(model_key).stat().st_size / 1024 / 1024
        _log(f"  Saved hidden states ({hs_mb:.1f} MB)")

        # 3. Metadata (incremental — preserve existing metrics)
        meta_file = AXES_DIR / f"{model_key}_axes_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
        else:
            meta = {}
        meta["_meta"] = {
            "script": "scripts/01_calibrate.py",
            "backend": backend,
            "model": model_key,
            "model_id": model_config.model_id,
            "hidden_dim": model_config.hidden_dim,
            "timestamp": datetime.now().isoformat(),
            "axes": list(all_axis_vectors.keys()),
            "train_questions": CALIBRATION_SAMPLES_PER_STYLE,
            "val_questions": CALIBRATION_EVAL_SAMPLES,
            "val_source": "eval_questions (independent, zero overlap with train)",
            "max_new_tokens": CALIBRATION_MAX_NEW_TOKENS,
            "saved_token_states": save_tokens,
            "generation_settings": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": DO_SAMPLE,
            },
        }
        # Add new/updated metrics (keep existing metrics from prior runs)
        for a in all_axis_vectors:
            if a in all_metrics:
                meta[a] = all_metrics[a]
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        # 4. Mark axis as completed in checkpoint
        ckpt = _load_checkpoint(model_key)
        if axis not in ckpt.get("completed_axes", []):
            ckpt.setdefault("completed_axes", []).append(axis)
        _save_checkpoint(model_key, ckpt)

        _log(f"  Axis {axis} DONE and saved to disk")

        gc.collect()

    # Final summary
    total_truncated = sum(m.get("n_truncated", 0) for m in all_metrics.values())
    total_samples = sum(
        (m["n_train_per_pole"] + m["n_val_per_pole"]) * 2
        for m in all_metrics.values()
    )
    if total_truncated > 0:
        pct = total_truncated / total_samples * 100
        logger.warning(
            f"WARNING: {total_truncated}/{total_samples} ({pct:.1f}%) responses truncated. "
            f"Consider increasing CALIBRATION_MAX_NEW_TOKENS (currently {CALIBRATION_MAX_NEW_TOKENS})."
        )

    # Clean up checkpoint file on successful completion
    _checkpoint_path(model_key).unlink(missing_ok=True)

    if _cleanup_fn:
        _cleanup_fn()

    return all_metrics


def _detect_backend() -> str:
    """Auto-detect best backend: mlx on Apple Silicon, pytorch elsewhere."""
    try:
        import mlx.core
        return "mlx"
    except ImportError:
        return "pytorch"


def main():
    parser = argparse.ArgumentParser(description="Step 01: Calibrate axis vectors")
    parser.add_argument("--model", required=True, help="Model key from config/models.py")
    parser.add_argument("--axes", help="Comma-separated axes (default: all 5)")
    parser.add_argument("--force", action="store_true", help="Recompute all from scratch")
    parser.add_argument("--check", action="store_true", help="Preflight check only (no GPU)")
    parser.add_argument("--time-budget-min", type=float, default=0,
                        help="Exit cleanly at the next checkpoint after N minutes "
                             "(0 = no limit); re-queue the same job to resume")
    parser.add_argument("--time-budget-exit-code", type=int, default=0,
                        help="Exit code to use when stopping on time budget "
                             "(default 0; use 75 with fw2 --repeat-until-exit 0)")
    parser.add_argument("--save-tokens", action="store_true",
                        help="Also save per-token states + logprobs (~500MB per model)")
    add_backend_args(parser)
    args = parser.parse_args()

    if args.time_budget_min > 0:
        global _time_budget_deadline, _time_budget_exit_code
        _time_budget_deadline = time.monotonic() + args.time_budget_min * 60
        _time_budget_exit_code = args.time_budget_exit_code
        _log(f"Time budget: {args.time_budget_min:.0f} min (clean exit at checkpoint)")

    if args.check:
        preflight_check(args.model)
        return

    _start_stall_watchdog()

    # --- Vast.ai dispatch ---
    if args.backend == "vast":
        model_key = args.model
        extra_args_parts = []
        if args.force:
            extra_args_parts.append("--force")
        if args.axes:
            extra_args_parts.append(f"--axes {args.axes}")

        # Upload existing partial data so remote can resume
        upload_extra = [
            f"data/axes/{model_key}_checkpoint.json",
            f"data/axes/{model_key}_hidden_states.npz",
            f"data/axes/{model_key}_axes_meta.json",
        ]

        run_on_vast(
            __file__, model_key, args.gpu,
            extra_args=" ".join(extra_args_parts),
            sync_paths=[
                f"data/axes/{model_key}_axes_meta.json",
                f"data/axes/{model_key}_checkpoint.json",
                f"data/axes/_progress.txt",
            ],
            final_paths=[
                f"data/axes/{model_key}_axes.npz",
                f"data/axes/{model_key}_axes_meta.json",
                f"data/axes/{model_key}_hidden_states.npz",
                f"data/axes/{model_key}_calibration.jsonl",
            ],
            upload_extra=upload_extra,
        )
        return

    backend = resolve_backend(args.backend)
    _log(f"Backend: {backend}")

    axes = args.axes.split(",") if args.axes else None
    metrics = calibrate_model(args.model, axes, force=args.force,
                              save_tokens=args.save_tokens, backend=backend)

    if metrics:
        print("\n=== Calibration Results ===", flush=True)
        for axis, m in metrics.items():
            trunc_str = f", truncated={m['n_truncated']}" if m.get('n_truncated', 0) > 0 else ""
            print(
                f"  {axis}: d'={m['val_dprime']:.2f}, acc={m['val_accuracy']*100:.1f}%, "
                f"sep={m['val_separation']:.4f}{trunc_str}",
                flush=True,
            )


if __name__ == "__main__":
    main()
