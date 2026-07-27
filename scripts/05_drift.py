#!/usr/bin/env python3
"""Step 05: Drift analysis — conflict, recovery, and neutral control.

Measures how model personality drifts under adversarial pressure, whether it
recovers afterwards, and how it compares to neutral conversation drift.

Three phases per conflict scenario:
    Phase 1 (Conflict):  12 turns of adversarial user messages
    Phase 2 (Recovery):   4 turns of neutral follow-up questions
    Phase 3 (Control):   10 neutral multi-turn conversations (16 turns each)
                         to measure natural drift from conversation length alone

Produces: data/drift/<model>.json
Depends on: 01_calibrate (needs axes NPZ)

Usage:
    python scripts/05_drift.py --model qwen_7b
    python scripts/05_drift.py --model qwen_7b --scenarios 5    # quick test
    python scripts/05_drift.py --model qwen_7b --no-control     # skip neutral control
    python scripts/05_drift.py --model qwen_7b --check
    python scripts/05_drift.py --model qwen_7b --force
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))  # repo root for framework imports

# Optional integration with the private `fw` framework. When it is not
# installed (public/standalone mode), fall back to a minimal emitter that
# keeps the #PROGRESS stdout protocol working.
try:
    from framework.progress import log_progress
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

from config.settings import MOOD_AXES, AXES_DIR, DRIFT_MAX_NEW_TOKENS, TEMPERATURE, TOP_P, DO_SAMPLE
from config.models import MODELS
from config.conflict_scenarios import ALL_CONFLICT_SCENARIOS as CONFLICT_SCENARIOS
from config.drift_prompts import RECOVERY_QUESTIONS, NEUTRAL_CONVERSATIONS
from src.mood.projector import MoodProjector
from src.backend import add_backend_args, resolve_backend, load_generate_fn, run_on_vast

OUTPUT_DIR = PROJECT_ROOT / "data" / "drift"

# Module-level progress state (written by analyze_drift, read by run_conversation)
_drift_done = 0
_drift_total = 0
_drift_t_start = 0.0


def _log(msg):
    logger.info(msg)
    try:
        progress_file = OUTPUT_DIR / "_progress.txt"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(progress_file, "w") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
    except Exception:
        pass


def preflight_check(model_key: str):
    assert model_key in MODELS, f"Unknown model: {model_key}"
    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    assert axes_file.exists(), f"Missing axes: {axes_file}"
    output_file = OUTPUT_DIR / f"{model_key}.json"
    print(f"Model: {MODELS[model_key].model_id}", flush=True)
    print(f"Conflict scenarios: {len(CONFLICT_SCENARIOS)}", flush=True)
    print(f"Turns per scenario: 12 conflict + 4 recovery = 16", flush=True)
    print(f"Neutral control: {len(NEUTRAL_CONVERSATIONS)} conversations × 16 turns", flush=True)
    print(f"Output: {output_file} (exists: {output_file.exists()})", flush=True)
    print("Preflight OK", flush=True)


# ---------------------------------------------------------------------------
#  Checkpoint helpers (per-scenario incremental saves)
# ---------------------------------------------------------------------------

def _checkpoint_path(model_key: str) -> Path:
    return OUTPUT_DIR / f"{model_key}_checkpoint.json"


def _load_checkpoint(model_key: str) -> dict:
    path = _checkpoint_path(model_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_scenarios": [], "completed_controls": []}


def _save_checkpoint(model_key: str, ckpt: dict):
    path = _checkpoint_path(model_key)
    with open(path, "w") as f:
        json.dump(ckpt, f, indent=2)
        f.flush()


def _hidden_states_dir(model_key: str) -> Path:
    return OUTPUT_DIR / "hidden_states" / model_key


def _save_hidden_states(model_key: str, scenario_id: str, hidden_states: list):
    """Save per-turn hidden states for a scenario (for later reprojection)."""
    hs_dir = _hidden_states_dir(model_key)
    hs_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"turn_{i}": hs for i, hs in enumerate(hidden_states)}
    np.savez(hs_dir / f"{scenario_id}.npz", **arrays)


def _partial_results_path(model_key: str) -> Path:
    return OUTPUT_DIR / f"{model_key}_partial.json"


def _save_partial_results(model_key: str, data: dict):
    path = _partial_results_path(model_key)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()


def _load_partial_results(model_key: str) -> dict:
    path = _partial_results_path(model_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"conflict_scenarios": [], "neutral_controls": []}


# ---------------------------------------------------------------------------
#  Single conversation runner
# ---------------------------------------------------------------------------

def run_conversation(generate_fn, projector, turns, label=""):
    """Run a multi-turn conversation, measure mood per turn.

    Args:
        generate_fn: callable(messages) -> GenerationResult
        projector: MoodProjector
        turns: list of user messages
        label: for logging

    Returns:
        (turn_data, hidden_states) where:
        - turn_data: list of per-turn dicts with values, text, timing
        - hidden_states: list of np.ndarray (raw hidden states for reprojection)
    """
    messages = []
    turn_data = []
    hidden_states = []

    t_conv_start = time.time()
    for i, user_message in enumerate(turns):
        messages.append({"role": "user", "content": user_message})

        t0 = time.time()
        result = generate_fn(messages)
        gen_time = time.time() - t0

        messages.append({"role": "assistant", "content": result.text})

        hidden_states.append(result.hidden_state)
        # clip=False: drift analysis needs unclipped values to avoid ceiling effects
        reading = projector.project(result.hidden_state, clip=False)
        turn_data.append({
            "turn": i + 1,
            "phase": _get_phase(i, len(turns)),
            "user": user_message,
            "assistant": result.text,
            "values": reading.values,
            "n_tokens": result.n_generated_tokens,
            "generation_time_s": round(gen_time, 2),
            "truncated": result.n_generated_tokens >= DRIFT_MAX_NEW_TOKENS - 1,
        })

        # Per-turn progress (every turn for drift since turns are expensive)
        elapsed = time.time() - t_conv_start
        remaining = (len(turns) - i - 1) * (elapsed / (i + 1))
        _log(f"  {label} turn {i+1}/{len(turns)} "
             f"({gen_time:.1f}s, {result.n_generated_tokens} tok, "
             f"ETA {remaining:.0f}s)")

        # Emit fractional #PROGRESS so `fw status` has live ETA within scenarios.
        # _drift_done/_drift_total/_drift_t_start are set in analyze_drift.
        try:
            if _drift_total > 0 and _drift_t_start > 0:
                frac = (i + 1) / max(len(turns), 1)
                log_progress(_drift_done + frac, _drift_total,
                             time.monotonic() - _drift_t_start)
        except Exception:
            pass

    return turn_data, hidden_states


def _get_phase(turn_idx: int, total_turns: int) -> str:
    """Determine phase for a turn index."""
    if total_turns <= 12:
        # Pure conflict or pure neutral — no recovery
        if turn_idx < 4:
            return "early"
        elif turn_idx < 8:
            return "middle"
        else:
            return "late"
    else:
        # 12 conflict + 4 recovery
        if turn_idx < 4:
            return "early"
        elif turn_idx < 8:
            return "middle"
        elif turn_idx < 12:
            return "peak"
        else:
            return "recovery"


# ---------------------------------------------------------------------------
#  Drift statistics computation
# ---------------------------------------------------------------------------

# (C) Minimum |conflict_delta| to compute recovery ratio.
# Below this, drift is noise and recovery ratio is meaningless.
RECOVERY_DELTA_THRESHOLD = 0.1

# Number of bootstrap resamples for confidence intervals.
N_BOOTSTRAP = 1000


def _bootstrap_ci(values: list, n_boot: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for the mean."""
    arr = np.array(values)
    n = len(arr)
    if n < 3:
        return (float(np.mean(arr)), float(np.mean(arr)))
    rng = np.random.RandomState(42)
    means = np.array([np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.percentile(means, alpha * 100)),
            float(np.percentile(means, (1 - alpha) * 100)))


def compute_drift_stats(scenarios_data: list, n_conflict_turns: int = 12,
                        control_stats: dict = None) -> dict:
    """Compute drift statistics from scenario results.

    Returns per-axis stats with:
    - conflict slope (turns 1-12) + bootstrap 95% CI
    - recovery delta (mean turn 13-16 vs mean turn 9-12)
    - recovery ratio (only when |conflict_delta| > RECOVERY_DELTA_THRESHOLD)
    - phase means (early/middle/peak/recovery)
    - significance flag (conflict slope CI excludes 0 AND > 2× control slope)
    """
    stats = {axis: {
        "conflict_slopes": [],
        "start_values": [],
        "end_conflict_values": [],
        "end_recovery_values": [],
        "recovery_deltas": [],
        "phase_means": {"early": [], "middle": [], "peak": [], "recovery": []},
    } for axis in MOOD_AXES}

    for scenario in scenarios_data:
        turns = scenario["turns"]
        conflict_turns = [t for t in turns if t["turn"] <= n_conflict_turns]
        recovery_turns = [t for t in turns if t["turn"] > n_conflict_turns]

        for axis in MOOD_AXES:
            # Conflict slope (linear regression on turns 1-12)
            conflict_values = [t["values"].get(axis, 0) for t in conflict_turns]
            if len(conflict_values) >= 2:
                x = np.arange(len(conflict_values))
                slope = np.polyfit(x, conflict_values, 1)[0]
                stats[axis]["conflict_slopes"].append(slope)
                stats[axis]["start_values"].append(conflict_values[0])
                stats[axis]["end_conflict_values"].append(conflict_values[-1])

            # Recovery delta
            if recovery_turns:
                recovery_values = [t["values"].get(axis, 0) for t in recovery_turns]
                peak_values = conflict_values[-4:] if len(conflict_values) >= 4 else conflict_values
                recovery_delta = np.mean(recovery_values) - np.mean(peak_values)
                stats[axis]["recovery_deltas"].append(recovery_delta)
                stats[axis]["end_recovery_values"].append(recovery_values[-1])

            # Phase means
            for t in turns:
                phase = t.get("phase", "")
                val = t["values"].get(axis, 0)
                if phase in stats[axis]["phase_means"]:
                    stats[axis]["phase_means"][phase].append(val)

    # Aggregate
    summary = {}
    for axis in MOOD_AXES:
        s = stats[axis]
        if not s["conflict_slopes"]:
            continue

        # Bootstrap CI on conflict slope
        slope_ci_lo, slope_ci_hi = _bootstrap_ci(s["conflict_slopes"])
        mean_conflict_delta = float(
            np.mean(s["end_conflict_values"]) - np.mean(s["start_values"])
        )

        entry = {
            "mean_conflict_slope": float(np.mean(s["conflict_slopes"])),
            "std_conflict_slope": float(np.std(s["conflict_slopes"])),
            "slope_ci_95": [slope_ci_lo, slope_ci_hi],
            "mean_start": float(np.mean(s["start_values"])),
            "mean_end_conflict": float(np.mean(s["end_conflict_values"])),
            "mean_conflict_delta": mean_conflict_delta,
        }

        # (B) Significance: CI excludes zero
        ci_excludes_zero = (slope_ci_lo > 0 and slope_ci_hi > 0) or \
                           (slope_ci_lo < 0 and slope_ci_hi < 0)

        # (B) Significance: conflict slope > 2× control slope
        control_slope = 0.0
        if control_stats and axis in control_stats:
            control_slope = abs(control_stats[axis].get("mean_slope", 0))
        conflict_slope = abs(entry["mean_conflict_slope"])
        exceeds_control = conflict_slope > 2 * control_slope if control_slope > 0 else ci_excludes_zero

        entry["significant"] = bool(ci_excludes_zero and exceeds_control)
        entry["conflict_vs_control_ratio"] = float(
            conflict_slope / control_slope) if control_slope > 1e-6 else None

        # Recovery
        if s["recovery_deltas"]:
            entry["mean_recovery_delta"] = float(np.mean(s["recovery_deltas"]))
            entry["std_recovery_delta"] = float(np.std(s["recovery_deltas"]))
            entry["mean_end_recovery"] = float(np.mean(s["end_recovery_values"]))

            # (C) Recovery ratio only when delta is meaningful
            if abs(mean_conflict_delta) >= RECOVERY_DELTA_THRESHOLD:
                raw_ratio = float(-entry["mean_recovery_delta"] / mean_conflict_delta)
                entry["recovery_ratio"] = min(raw_ratio, 1.0)  # cap at 100%
            else:
                entry["recovery_ratio"] = None  # delta too small, recovery undefined

        # Phase means
        for phase in ["early", "middle", "peak", "recovery"]:
            vals = s["phase_means"][phase]
            if vals:
                entry[f"phase_{phase}_mean"] = float(np.mean(vals))
                entry[f"phase_{phase}_std"] = float(np.std(vals))

        summary[axis] = entry

    return summary


def compute_control_stats(controls_data: list) -> dict:
    """Compute drift statistics for neutral control conversations."""
    stats = {axis: {"slopes": [], "start_values": [], "end_values": []} for axis in MOOD_AXES}

    for conv in controls_data:
        turns = conv["turns"]
        for axis in MOOD_AXES:
            values = [t["values"].get(axis, 0) for t in turns]
            if len(values) >= 2:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                stats[axis]["slopes"].append(slope)
                stats[axis]["start_values"].append(values[0])
                stats[axis]["end_values"].append(values[-1])

    summary = {}
    for axis in MOOD_AXES:
        s = stats[axis]
        if not s["slopes"]:
            continue
        summary[axis] = {
            "mean_slope": float(np.mean(s["slopes"])),
            "std_slope": float(np.std(s["slopes"])),
            "mean_start": float(np.mean(s["start_values"])),
            "mean_end": float(np.mean(s["end_values"])),
            "mean_delta": float(np.mean(s["end_values"]) - np.mean(s["start_values"])),
        }
    return summary


# ---------------------------------------------------------------------------
#  Main analysis
# ---------------------------------------------------------------------------

def analyze_drift(model_key: str, max_scenarios: int = None, force: bool = False,
                  backend: str = "pytorch", run_control: bool = True) -> dict:
    output_file = OUTPUT_DIR / f"{model_key}.json"
    if output_file.exists() and not force:
        print(f"Output exists: {output_file}. Use --force to recompute.", flush=True)
        return json.load(open(output_file))

    model_config = MODELS[model_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_fn, cleanup_fn = load_generate_fn(
        model_config, backend, max_new_tokens=DRIFT_MAX_NEW_TOKENS, log_fn=_log)

    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    projector = MoodProjector(axes_file=axes_file)

    # Total units of work for progress tracking (module-level so run_conversation
    # can emit fractional per-turn progress without parameter plumbing).
    global _drift_total, _drift_t_start, _drift_done
    scenarios_list = CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios else CONFLICT_SCENARIOS
    _drift_total = len(scenarios_list) + (len(NEUTRAL_CONVERSATIONS) if run_control else 0)
    _drift_t_start = time.monotonic()
    _drift_done = 0

    # Load checkpoint for resume
    ckpt = _load_checkpoint(model_key)
    partial = _load_partial_results(model_key)

    if force:
        ckpt = {"completed_scenarios": [], "completed_controls": []}
        partial = {"conflict_scenarios": [], "neutral_controls": []}
        _save_checkpoint(model_key, ckpt)

    # --- Conflict scenarios with recovery ---
    scenarios = CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios else CONFLICT_SCENARIOS
    completed_ids = set(ckpt.get("completed_scenarios", []))
    completed_controls = set(ckpt.get("completed_controls", []))
    log_progress(len(completed_ids) + len(completed_controls), _drift_total, 0)
    t_start = time.monotonic()

    for idx, scenario in enumerate(scenarios):
        if scenario.name in completed_ids:
            continue

        _log(f"Scenario {idx+1}/{len(scenarios)}: {scenario.name} ({scenario.category})")

        # 12 conflict turns + 4 recovery turns
        all_turns = list(scenario.turns) + RECOVERY_QUESTIONS

        try:
            turn_data, hidden_states = run_conversation(
                generate_fn, projector, all_turns, label=scenario.name)

            # Save hidden states for later reprojection
            _save_hidden_states(model_key, scenario.name, hidden_states)

            result_entry = {
                "category": scenario.category,
                "scenario_id": scenario.name,
                "expected_effect": scenario.expected_effect,
                "n_conflict_turns": len(scenario.turns),
                "n_recovery_turns": len(RECOVERY_QUESTIONS),
                "turns": turn_data,
            }
            partial["conflict_scenarios"].append(result_entry)

        except Exception as e:
            _log(f"  ERROR in {scenario.name}: {e}")
            continue

        # Save after every scenario
        _save_partial_results(model_key, partial)
        ckpt.setdefault("completed_scenarios", []).append(scenario.name)
        _save_checkpoint(model_key, ckpt)

        _drift_done = len(ckpt["completed_scenarios"]) + len(ckpt.get("completed_controls", []))
        log_progress(_drift_done, _drift_total, time.monotonic() - _drift_t_start,
                     n_elapsed=_drift_done - len(completed_ids) - len(completed_controls))

        elapsed = time.monotonic() - t_start
        n_done = len(ckpt["completed_scenarios"]) - len(completed_ids)
        if n_done > 0:
            per_scenario = elapsed / n_done
            remaining = len(scenarios) - (idx + 1)
            eta = per_scenario * remaining
            eta_str = f"{int(eta//60)}m{int(eta%60):02d}s" if eta >= 60 else f"{eta:.0f}s"
            _log(f"  Done ({per_scenario:.0f}s/scenario, ETA {eta_str})")

    # --- Neutral control conversations ---
    if run_control:
        completed_controls = set(ckpt.get("completed_controls", []))

        for idx, conv in enumerate(NEUTRAL_CONVERSATIONS):
            if conv["name"] in completed_controls:
                continue

            _log(f"Control {idx+1}/{len(NEUTRAL_CONVERSATIONS)}: {conv['name']}")

            try:
                turn_data, hidden_states = run_conversation(
                    generate_fn, projector, conv["turns"], label=conv["name"])

                # Save hidden states for later reprojection
                _save_hidden_states(model_key, conv["name"], hidden_states)

                partial["neutral_controls"].append({
                    "name": conv["name"],
                    "n_turns": len(conv["turns"]),
                    "turns": turn_data,
                })
            except Exception as e:
                _log(f"  ERROR in control {conv['name']}: {e}")
                continue

            _save_partial_results(model_key, partial)
            ckpt.setdefault("completed_controls", []).append(conv["name"])
            _save_checkpoint(model_key, ckpt)

            _drift_done = len(ckpt["completed_scenarios"]) + len(ckpt["completed_controls"])
            log_progress(_drift_done, _drift_total, time.monotonic() - _drift_t_start,
                         n_elapsed=_drift_done - len(completed_ids) - len(completed_controls))
            _log(f"  Control {conv['name']} done")

    # --- Compute statistics ---
    _log("Computing drift statistics...")

    control_summary = compute_control_stats(partial["neutral_controls"]) if partial["neutral_controls"] else {}
    conflict_summary = compute_drift_stats(partial["conflict_scenarios"],
                                           control_stats=control_summary)

    results = {
        "_meta": {
            "script": "scripts/05_drift.py",
            "backend": backend,
            "model": model_key,
            "model_id": model_config.model_id,
            "timestamp": datetime.now().isoformat(),
            "n_conflict_scenarios": len(partial["conflict_scenarios"]),
            "n_conflict_turns": 12,
            "n_recovery_turns": len(RECOVERY_QUESTIONS),
            "n_control_conversations": len(partial["neutral_controls"]),
            "n_control_turns": 16,
            "axes": list(projector.axis_vectors.keys()),
            "max_new_tokens": DRIFT_MAX_NEW_TOKENS,
            "generation_settings": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": DO_SAMPLE,
            },
        },
        "conflict_drift": conflict_summary,
        "control_drift": control_summary,
        "conflict_scenarios": partial["conflict_scenarios"],
        "neutral_controls": partial["neutral_controls"],
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"Saved to {output_file}")

    # Cleanup temp files
    _checkpoint_path(model_key).unlink(missing_ok=True)
    _partial_results_path(model_key).unlink(missing_ok=True)

    cleanup_fn()
    return results


def reproject(model_key: str) -> dict:
    """Reproject saved hidden states with current axes (CPU-only, no GPU needed).

    Reads per-scenario hidden states from data/drift/hidden_states/<model>/
    and reprojects them using current axes from data/axes/<model>_axes.npz.
    Updates the drift JSON with new axis values.
    """
    output_file = OUTPUT_DIR / f"{model_key}.json"
    if not output_file.exists():
        print(f"ERROR: No drift results found: {output_file}", flush=True)
        print("Run drift first, then reproject.", flush=True)
        sys.exit(1)

    hs_dir = _hidden_states_dir(model_key)
    if not hs_dir.exists():
        print(f"ERROR: No hidden states found: {hs_dir}", flush=True)
        print("Drift was run without hidden state saving. Re-run with --force.", flush=True)
        sys.exit(1)

    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    projector = MoodProjector(axes_file=axes_file)
    axes_in_npz = list(projector.axis_vectors.keys())
    print(f"Reprojecting {model_key} with {len(axes_in_npz)} axes: {axes_in_npz}", flush=True)

    with open(output_file) as f:
        results = json.load(f)

    # Reproject conflict scenarios
    for scenario in results.get("conflict_scenarios", []):
        hs_file = hs_dir / f"{scenario['scenario_id']}.npz"
        if not hs_file.exists():
            print(f"  WARNING: missing hidden states for {scenario['scenario_id']}", flush=True)
            continue
        hs_data = np.load(hs_file)
        for turn in scenario["turns"]:
            turn_key = f"turn_{turn['turn'] - 1}"
            if turn_key in hs_data:
                reading = projector.project(hs_data[turn_key], clip=False)
                turn["values"] = reading.values

    # Reproject neutral controls
    for control in results.get("neutral_controls", []):
        hs_file = hs_dir / f"{control['name']}.npz"
        if not hs_file.exists():
            print(f"  WARNING: missing hidden states for {control['name']}", flush=True)
            continue
        hs_data = np.load(hs_file)
        for turn in control["turns"]:
            turn_key = f"turn_{turn['turn'] - 1}"
            if turn_key in hs_data:
                reading = projector.project(hs_data[turn_key], clip=False)
                turn["values"] = reading.values

    # Recompute statistics
    control_summary = compute_control_stats(results.get("neutral_controls", []))
    conflict_summary = compute_drift_stats(results.get("conflict_scenarios", []),
                                           control_stats=control_summary)
    results["conflict_drift"] = conflict_summary
    results["control_drift"] = control_summary
    results["_meta"]["reprojected"] = datetime.now().isoformat()
    results["_meta"]["axes"] = axes_in_npz

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Reprojected and saved: {output_file}", flush=True)
    return results


MODAL_BATCH_SIZE = 10  # conversations per remote call (model loads once per batch)


def _process_modal_result(result, projector, model_key, partial, ckpt):
    """Project hidden states locally, save to checkpoint. Returns conv_id."""
    conv_id = result["id"]
    conv_type = result["type"]
    turn_data = result["turn_data"]
    hidden_states = result["hidden_states"]

    for i, (td, hs) in enumerate(zip(turn_data, hidden_states)):
        reading = projector.project(hs, clip=False)
        td["values"] = reading.values
        td["phase"] = _get_phase(i, len(turn_data))

    _save_hidden_states(model_key, conv_id, hidden_states)

    if conv_type == "conflict":
        partial["conflict_scenarios"].append({
            "category": result["category"],
            "scenario_id": conv_id,
            "expected_effect": result["expected_effect"],
            "n_conflict_turns": len(turn_data) - len(RECOVERY_QUESTIONS),
            "n_recovery_turns": len(RECOVERY_QUESTIONS),
            "turns": turn_data,
        })
        ckpt.setdefault("completed_scenarios", []).append(conv_id)
    else:
        partial["neutral_controls"].append({
            "name": conv_id,
            "n_turns": len(turn_data),
            "turns": turn_data,
        })
        ckpt.setdefault("completed_controls", []).append(conv_id)

    _save_partial_results(model_key, partial)
    _save_checkpoint(model_key, ckpt)
    return conv_id, conv_type


def _analyze_drift_modal(model_key: str, max_scenarios: int = None,
                         force: bool = False, run_control: bool = True,
                         gpu: str = "A100") -> dict:
    """Run drift analysis on Modal GPU (A100).

    Generation happens remotely in batches; projection + saving happen locally.
    Each batch is a separate remote_gen() call (~10 conversations, one model load).
    If a batch fails, all previous batches are already saved to checkpoint.
    """
    output_file = OUTPUT_DIR / f"{model_key}.json"
    if output_file.exists() and not force:
        print(f"Output exists: {output_file}. Use --force to recompute.", flush=True)
        with open(output_file) as f:
            return json.load(f)

    model_config = MODELS[model_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    projector = MoodProjector(axes_file=axes_file)

    # Load checkpoint for resume
    ckpt = _load_checkpoint(model_key)
    partial = _load_partial_results(model_key)

    if force:
        ckpt = {"completed_scenarios": [], "completed_controls": []}
        partial = {"conflict_scenarios": [], "neutral_controls": []}
        _save_checkpoint(model_key, ckpt)

    completed_ids = set(ckpt.get("completed_scenarios", []))
    completed_controls = set(ckpt.get("completed_controls", []))

    # Prepare conversation list for Modal
    conversations = []

    scenarios = CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios else CONFLICT_SCENARIOS
    for scenario in scenarios:
        if scenario.name in completed_ids:
            continue
        all_turns = list(scenario.turns) + RECOVERY_QUESTIONS
        conversations.append({
            "id": scenario.name,
            "type": "conflict",
            "category": scenario.category,
            "expected_effect": scenario.expected_effect,
            "turns": all_turns,
        })

    if run_control:
        for conv in NEUTRAL_CONVERSATIONS:
            if conv["name"] in completed_controls:
                continue
            conversations.append({
                "id": conv["name"],
                "type": "control",
                "category": "",
                "expected_effect": "",
                "turns": conv["turns"],
            })

    if not conversations:
        _log("All conversations already computed. Recomputing stats only.")
    else:
        # Split into batches
        batches = [conversations[i:i + MODAL_BATCH_SIZE]
                   for i in range(0, len(conversations), MODAL_BATCH_SIZE)]
        total_convs = len(conversations)
        n_conflict = sum(1 for c in conversations if c["type"] == "conflict")
        n_control = sum(1 for c in conversations if c["type"] == "control")

        _log(f"Modal drift: {total_convs} conversations "
             f"({n_conflict} conflict + {n_control} control)")
        _log(f"Split into {len(batches)} batches of up to {MODAL_BATCH_SIZE}")
        _log(f"Estimated time: ~{total_convs * 1.5:.0f} min on A100 "
             f"(+ ~{len(batches) * 0.5:.0f} min model loads)")

        # Import Modal app and remote function
        sys.path.insert(0, str(PROJECT_ROOT))
        import os
        os.environ["MODAL_GPU"] = gpu
        _log(f"Importing Modal app (GPU: {gpu})...")
        from modal_drift import app, run_conversations

        _log("Starting Modal app (image build + container startup)...")
        t_total = time.monotonic()
        done_count = 0

        with app.run():
            _log("Modal app started.")

            for batch_idx, batch in enumerate(batches):
                batch_types = {}
                for c in batch:
                    batch_types[c["type"]] = batch_types.get(c["type"], 0) + 1
                type_str = ", ".join(f"{v} {k}" for k, v in batch_types.items())
                _log(f"--- Batch {batch_idx + 1}/{len(batches)}: "
                     f"{len(batch)} conversations ({type_str}) ---")
                _log(f"Loading model on GPU + generating "
                     f"(~{len(batch) * 1.5:.0f} min)...")
                t_batch = time.monotonic()

                try:
                    for result in run_conversations.remote_gen(
                        model_key=model_key,
                        conversations=batch,
                    ):
                        conv_id, conv_type = _process_modal_result(
                            result, projector, model_key, partial, ckpt)
                        done_count += 1
                        elapsed = time.monotonic() - t_total
                        per_conv = elapsed / done_count
                        remaining = (total_convs - done_count) * per_conv
                        eta_m = int(remaining // 60)
                        eta_s = int(remaining % 60)
                        _log(f"  [{done_count}/{total_convs}] "
                             f"{conv_type} {conv_id} saved "
                             f"({per_conv:.0f}s/conv, ETA {eta_m}m{eta_s:02d}s)")

                except Exception as e:
                    batch_time = time.monotonic() - t_batch
                    _log(f"  Batch {batch_idx + 1} FAILED after {batch_time:.0f}s: {e}")
                    _log(f"  {done_count}/{total_convs} conversations saved. "
                         f"Re-run to continue from checkpoint.")
                    # Don't re-raise — compute stats on what we have
                    break

                batch_time = time.monotonic() - t_batch
                _log(f"Batch {batch_idx + 1}/{len(batches)} done "
                     f"({batch_time:.0f}s, {batch_time / len(batch):.0f}s/conv)")

        total_time = time.monotonic() - t_total
        _log(f"Modal done: {done_count}/{total_convs} in {total_time:.0f}s "
             f"({total_time / max(done_count, 1):.0f}s/conv)")

    # Compute statistics
    _log("Computing drift statistics...")

    control_summary = compute_control_stats(
        partial["neutral_controls"]) if partial["neutral_controls"] else {}
    conflict_summary = compute_drift_stats(
        partial["conflict_scenarios"], control_stats=control_summary)

    results = {
        "_meta": {
            "script": "scripts/05_drift.py",
            "backend": "modal",
            "model": model_key,
            "model_id": model_config.model_id,
            "timestamp": datetime.now().isoformat(),
            "n_conflict_scenarios": len(partial["conflict_scenarios"]),
            "n_conflict_turns": 12,
            "n_recovery_turns": len(RECOVERY_QUESTIONS),
            "n_control_conversations": len(partial["neutral_controls"]),
            "n_control_turns": 16,
            "axes": list(projector.axis_vectors.keys()),
            "max_new_tokens": DRIFT_MAX_NEW_TOKENS,
            "generation_settings": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": DO_SAMPLE,
            },
        },
        "conflict_drift": conflict_summary,
        "control_drift": control_summary,
        "conflict_scenarios": partial["conflict_scenarios"],
        "neutral_controls": partial["neutral_controls"],
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    _log(f"Saved to {output_file}")

    # Cleanup temp files
    _checkpoint_path(model_key).unlink(missing_ok=True)
    _partial_results_path(model_key).unlink(missing_ok=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 05: Drift analysis with recovery & control")
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenarios", type=int, help="Max conflict scenarios (default: all 50)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reproject", action="store_true",
                        help="Reproject saved hidden states with current axes (CPU-only)")
    parser.add_argument("--no-control", action="store_true", help="Skip neutral control conversations")
    add_backend_args(parser)
    args = parser.parse_args()

    if args.check:
        preflight_check(args.model)
        return

    if args.reproject:
        results = reproject(args.model)
        # Print summary
        print(f"\n=== {args.model} Conflict Drift (reprojected) ===", flush=True)
        for axis, s in results.get("conflict_drift", {}).items():
            recovery_str = ""
            if s.get("recovery_ratio") is not None:
                recovery_str = f", recovery={s['recovery_ratio']:.0%}"
            print(f"  {axis}: slope={s['mean_conflict_slope']:.4f}, "
                  f"delta={s['mean_conflict_delta']:.3f}{recovery_str}", flush=True)
        if results.get("control_drift"):
            print(f"\n=== {args.model} Control Drift (reprojected) ===", flush=True)
            for axis, s in results["control_drift"].items():
                print(f"  {axis}: slope={s['mean_slope']:.4f}, delta={s['mean_delta']:.3f}", flush=True)
        return

    backend = resolve_backend(args.backend)
    _log(f"Backend: {backend}")

    if backend == "vast":
        extra_parts = ["--force"] if args.force else []
        if args.scenarios:
            extra_parts.append(f"--scenarios {args.scenarios}")
        if args.no_control:
            extra_parts.append("--no-control")
        run_on_vast(
            __file__, args.model, gpu=args.gpu,
            extra_args=" ".join(extra_parts),
            sync_paths=[
                f"data/drift/{args.model}_partial.json",
                f"data/drift/{args.model}_checkpoint.json",
            ],
            final_paths=[
                f"data/drift/{args.model}.json",
                f"data/drift/hidden_states/{args.model}/",
            ],
        )
        return
    elif backend == "modal":
        results = _analyze_drift_modal(
            args.model,
            max_scenarios=args.scenarios,
            force=args.force,
            run_control=not args.no_control,
            gpu=args.gpu,
        )
    else:
        results = analyze_drift(
            args.model,
            max_scenarios=args.scenarios,
            force=args.force,
            backend=backend,
            run_control=not args.no_control,
        )

    # Print summary
    print(f"\n=== {args.model} Conflict Drift ===", flush=True)
    for axis, s in results.get("conflict_drift", {}).items():
        recovery_str = ""
        if s.get("recovery_ratio") is not None:
            recovery_str = f", recovery={s['recovery_ratio']:.0%}"
        sig = " SIG" if s.get("significant") else ""
        print(
            f"  {axis}: slope={s['mean_conflict_slope']:.4f}, "
            f"delta={s['mean_conflict_delta']:.3f}{recovery_str}{sig}",
            flush=True,
        )

    if results.get("control_drift"):
        print(f"\n=== {args.model} Control Drift (neutral) ===", flush=True)
        for axis, s in results["control_drift"].items():
            print(f"  {axis}: slope={s['mean_slope']:.4f}, delta={s['mean_delta']:.3f}", flush=True)


if __name__ == "__main__":
    main()
