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
import os
import sys
import gc
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

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
# Turn-granular progress: unit = one dialogue turn (not one scenario), so the
# bar moves every 2-5 min even on 27B+ models where a scenario takes 30-80 min.
_scen_units = {}   # scenario name -> n turns (conflict + recovery)
_ctrl_units = {}   # control name -> n turns
_drift_units_start = 0


def _units_done(ckpt):
    """Completed work in TURN units, derived from the checkpoint each time
    (never an accumulator — survives chunked resume, see D-progress rule).

    Includes fully-completed scenarios/controls AND the in-progress
    ("inflight") scenario's durably-saved turns, so a resume after a
    mid-scenario kill reports the correct global count immediately —
    it does NOT wait for the scenario to fully finish."""
    done = sum(_scen_units.get(n, 0) for n in ckpt.get("completed_scenarios", []))
    done += sum(_ctrl_units.get(n, 0) for n in ckpt.get("completed_controls", []))
    inflight = ckpt.get("inflight")
    if inflight:
        done += inflight.get("turns_done", 0)
    return done
_drift_budget = None


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


def _atomic_write_json(path: Path, data: dict):
    # open(path, "w") truncates in place: a kill mid-dump leaves an unparseable
    # stub and loses every prior record (deepseek incident 2026-07-23).
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_checkpoint(model_key: str, ckpt: dict):
    _atomic_write_json(_checkpoint_path(model_key), ckpt)


def _hidden_states_dir(model_key: str) -> Path:
    return OUTPUT_DIR / "hidden_states" / model_key


def _save_hidden_states(model_key: str, scenario_id: str, hidden_states: list):
    """Save per-turn hidden states for a FINISHED scenario (for later
    reprojection)."""
    hs_dir = _hidden_states_dir(model_key)
    hs_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"turn_{i}": hs for i, hs in enumerate(hidden_states)}
    final = hs_dir / f"{scenario_id}.npz"
    # tmp name MUST end in .npz — np.savez appends it otherwise and os.replace
    # then misses the file.
    tmp = hs_dir / f"{scenario_id}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, final)


def _partial_hidden_states_path(model_key: str, scenario_id: str) -> Path:
    return _hidden_states_dir(model_key) / f"{scenario_id}.partial.npz"


def _save_partial_hidden_states(model_key: str, scenario_id: str, hidden_states: list):
    """Atomically save in-progress (not-yet-finished) per-turn hidden states
    for a scenario. Overwritten after every turn; once the scenario finishes,
    the final <scenario_id>.npz is written via _save_hidden_states and this
    partial file is deleted (see _clear_partial_hidden_states)."""
    hs_dir = _hidden_states_dir(model_key)
    hs_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"turn_{i}": hs for i, hs in enumerate(hidden_states)}
    final = _partial_hidden_states_path(model_key, scenario_id)
    # tmp name MUST end in .npz — np.savez appends it otherwise and os.replace
    # then misses the file.
    tmp = hs_dir / f"{scenario_id}.partial.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, final)


def _load_partial_hidden_states(model_key: str, scenario_id: str) -> list:
    """Load in-progress per-turn hidden states saved by
    _save_partial_hidden_states, in turn order. Empty list if none saved."""
    path = _partial_hidden_states_path(model_key, scenario_id)
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    n = len(data.files)
    return [data[f"turn_{i}"] for i in range(n)]


def _clear_partial_hidden_states(model_key: str, scenario_id: str):
    _partial_hidden_states_path(model_key, scenario_id).unlink(missing_ok=True)


def _partial_results_path(model_key: str) -> Path:
    return OUTPUT_DIR / f"{model_key}_partial.json"


def _save_partial_results(model_key: str, data: dict):
    _atomic_write_json(_partial_results_path(model_key), data)


def _load_partial_results(model_key: str) -> dict:
    path = _partial_results_path(model_key)
    if path.exists():
        with open(path) as f:
            partial = json.load(f)
        # Fail loud on checkpoint/partial divergence: silently proceeding would
        # skip "completed" scenarios whose results are gone from the partial.
        ckpt = _load_checkpoint(model_key)
        n_scen = len(partial.get("conflict_scenarios", []))
        n_done = len(ckpt.get("completed_scenarios", []))
        if n_scen < n_done:
            raise RuntimeError(
                f"partial/checkpoint mismatch for {model_key}: partial has "
                f"{n_scen} scenarios but checkpoint marks {n_done} complete. "
                f"Restore {path} from a backup or delete the checkpoint to recompute."
            )
        return partial
    return {"conflict_scenarios": [], "neutral_controls": []}


# ---------------------------------------------------------------------------
#  Inflight (in-progress scenario) checkpointing — enables mid-scenario resume
# ---------------------------------------------------------------------------
#
# A scenario/control conversation is 12-16 turns and a single turn can take
# 1-2 min on 27B+ models, so waiting for the WHOLE conversation to finish
# before checkpointing (the old behavior) risks losing up to ~30-80 min of
# work on a kill. Instead, after every turn we durably save:
#   1. partial hidden states (_save_partial_hidden_states, atomic NPZ)
#   2. the "inflight" file: turn texts + full message history (atomic JSON) —
#      the message history is what lets resume continue the SAME dialogue
#      (the model needs the prior turns as context to keep generating).
#   3. checkpoint["inflight"] = {"scenario": name, "type": ..., "turns_done": N}
#
# On startup, _validate_inflight cross-checks all three sources agree before
# trusting them; any mismatch discards the inflight state (with a log) and
# the scenario restarts from turn 0 — completed scenarios are never affected.


def _inflight_path(model_key: str) -> Path:
    return OUTPUT_DIR / f"{model_key}_inflight.json"


def _save_inflight(model_key: str, name: str, conv_type: str,
                   turn_data: list, messages: list):
    _atomic_write_json(_inflight_path(model_key), {
        "name": name, "type": conv_type,
        "turn_data": turn_data, "messages": messages,
    })


def _load_inflight(model_key: str) -> dict:
    path = _inflight_path(model_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _clear_inflight(model_key: str):
    _inflight_path(model_key).unlink(missing_ok=True)


def _validate_inflight(ckpt: dict, model_key: str) -> Optional[dict]:
    """Cross-check checkpoint["inflight"] against the inflight JSON file and
    the saved partial hidden states. Returns a dict ready to resume from, or
    None (after logging why) if anything is inconsistent — the caller then
    discards the inflight marker and the scenario restarts from turn 0.
    Never raises. Completed scenarios/controls are untouched either way —
    only the single in-progress conversation is ever at risk.
    """
    ck_inflight = ckpt.get("inflight")
    if not ck_inflight:
        return None
    name = ck_inflight.get("scenario")
    conv_type = ck_inflight.get("type")
    turns_done = ck_inflight.get("turns_done", 0)
    if not name or turns_done <= 0:
        return None

    inflight = _load_inflight(model_key)
    if not inflight or inflight.get("name") != name or inflight.get("type") != conv_type:
        _log(f"inflight mismatch: checkpoint marks '{name}' ({conv_type}) but "
             f"inflight file has {inflight.get('name')!r} ({inflight.get('type')!r}) "
             f"— discarding, scenario will restart from turn 0")
        return None

    turn_data = inflight.get("turn_data", [])
    messages = inflight.get("messages", [])
    if len(turn_data) != turns_done or len(messages) != 2 * turns_done:
        _log(f"inflight mismatch for '{name}': checkpoint turns_done={turns_done} "
             f"but {len(turn_data)} turn_data / {len(messages)} messages saved "
             f"— discarding, scenario will restart from turn 0")
        return None

    hidden_states = _load_partial_hidden_states(model_key, name)
    if len(hidden_states) != turns_done:
        _log(f"inflight mismatch for '{name}': turns_done={turns_done} but "
             f"{len(hidden_states)} partial hidden states saved — discarding, "
             f"scenario will restart from turn 0")
        return None

    _log(f"Resuming '{name}' ({conv_type}) from turn {turns_done + 1}")
    return {
        "name": name, "type": conv_type, "turns_done": turns_done,
        "turn_data": turn_data, "messages": messages,
        "hidden_states": hidden_states,
    }


# ---------------------------------------------------------------------------
#  Single conversation runner
# ---------------------------------------------------------------------------

def run_conversation(generate_fn, projector, turns, label="", model_key=None,
                     ckpt=None, conv_type=None, start_turn=0,
                     resume_messages=None, resume_turn_data=None,
                     resume_hidden_states=None):
    """Run a multi-turn conversation, measuring mood per turn.

    Durably checkpoints after EVERY turn (not just at conversation end): the
    per-turn hidden states so far (atomic NPZ), the per-turn texts + full
    message history (atomic "inflight" JSON — the history is required to
    keep generating the SAME dialogue on resume), and a checkpoint marker
    {"inflight": {"scenario", "type", "turns_done"}}. A mid-conversation kill
    therefore loses at most the turn that was in flight, and resume can
    continue from start_turn using resume_messages/resume_turn_data/
    resume_hidden_states (see _validate_inflight for how these are restored).

    Args:
        generate_fn: callable(messages) -> GenerationResult
        projector: MoodProjector
        turns: list of user messages for the FULL conversation (0-indexed;
            when resuming, this is the same full list — only start_turn moves)
        label: scenario/control name — used for logging, as the inflight
            identity, and as the (partial/final) hidden-states filename
        model_key, ckpt, conv_type: identify the job for incremental
            checkpointing. If model_key is None, per-turn durable saves and
            progress emission are skipped (kept optional for tests/other
            callers that don't want on-disk side effects).
        start_turn: 0-indexed turn to resume from (0 = fresh start)
        resume_messages/resume_turn_data/resume_hidden_states: state restored
            from a previous crashed run (see _validate_inflight); ignored
            when start_turn == 0

    Returns:
        (turn_data, hidden_states) for ALL turns 0..len(turns)-1 (restored
        prefix + newly generated suffix).
    """
    messages = list(resume_messages) if resume_messages else []
    turn_data = list(resume_turn_data) if resume_turn_data else []
    hidden_states = list(resume_hidden_states) if resume_hidden_states else []

    t_conv_start = time.time()
    for i in range(start_turn, len(turns)):
        user_message = turns[i]
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

        # --- Durable per-turn checkpoint — MUST happen before progress emit ---
        if model_key is not None:
            _save_partial_hidden_states(model_key, label, hidden_states)
            _save_inflight(model_key, label, conv_type, turn_data, messages)
            if ckpt is not None:
                ckpt["inflight"] = {"scenario": label, "type": conv_type,
                                    "turns_done": i + 1}
                _save_checkpoint(model_key, ckpt)

        # Per-turn progress (every turn for drift since turns are expensive)
        elapsed = time.time() - t_conv_start
        remaining = (len(turns) - i - 1) * (elapsed / (i + 1 - start_turn))
        _log(f"  {label} turn {i+1}/{len(turns)} "
             f"({gen_time:.1f}s, {result.n_generated_tokens} tok, "
             f"ETA {remaining:.0f}s)")

        # #PROGRESS in TURN units: integer, global, monotonic — derived from
        # the checkpoint (which by now includes this turn) each time, never
        # an accumulator. Emitted AFTER the durable writes above so a crash
        # right after emit never overstates what's actually on disk.
        try:
            if _drift_total > 0 and _drift_t_start > 0 and ckpt is not None:
                done_units = min(_units_done(ckpt), _drift_total)
                log_progress(done_units, _drift_total,
                             time.monotonic() - _drift_t_start,
                             n_elapsed=max(done_units - _drift_units_start, 0),
                             extra={"item": label, "phase_done": i + 1,
                                    "phase_total": len(turns)})
        except Exception:
            pass

        # Per-turn budget check: a 16-turn scenario at ~100s/turn (~27 min)
        # can overshoot a scenario-boundary-only check past the hard
        # --timeout-seconds (observed: exit 124 instead of 75 on mistral
        # drift). The turn is already durable (inflight + partial HS +
        # checkpoint saved above), so exiting mid-scenario loses nothing.
        if (_drift_budget is not None and ckpt is not None
                and i + 1 < len(turns)):
            _drift_budget.check_and_exit(
                min(_units_done(ckpt), _drift_total), _drift_total, log=_log)

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
        # An existing final output only means THIS config's units are done if
        # it covers them all. When the scenario/control roster grows (e.g.
        # ctrl30 expansion) or a stale final sits in a reused chain workspace,
        # fall through to the normal checkpoint/resume path instead.
        ckpt = _load_checkpoint(model_key)
        n_scen_target = len(CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios
                            else CONFLICT_SCENARIOS)
        n_ctrl_target = len(NEUTRAL_CONVERSATIONS) if run_control else 0
        # Count coverage as the union of checkpoint and the final output
        # itself — a full output with a missing/stale checkpoint must NOT
        # trigger a from-scratch rerun (resume follows the checkpoint).
        try:
            prev = json.load(open(output_file))
        except Exception:
            prev = {}
        done_scen = len({s["scenario_id"] for s in prev.get("conflict_scenarios", [])}
                        | set(ckpt.get("completed_scenarios", [])))
        done_ctrl = len({c["name"] for c in prev.get("neutral_controls", [])}
                        | set(ckpt.get("completed_controls", [])))
        if done_scen >= n_scen_target and done_ctrl >= n_ctrl_target:
            print(f"Output exists: {output_file}. Use --force to recompute.",
                  flush=True)
            return json.load(open(output_file))
        # Resume follows checkpoint+partial; if they lag behind the final
        # output (e.g. wiped workspace), rebuild both from it so completed
        # units are never re-generated.
        if prev and (len(ckpt.get("completed_scenarios", []))
                     < len(prev.get("conflict_scenarios", []))
                     or len(ckpt.get("completed_controls", []))
                     < len(prev.get("neutral_controls", []))):
            _atomic_write_json(_partial_results_path(model_key), {
                "conflict_scenarios": prev.get("conflict_scenarios", []),
                "neutral_controls": prev.get("neutral_controls", [])})
            ckpt = {"completed_scenarios":
                    [s["scenario_id"] for s in prev.get("conflict_scenarios", [])],
                    "completed_controls":
                    [c["name"] for c in prev.get("neutral_controls", [])]}
            _save_checkpoint(model_key, ckpt)
            print("Rebuilt checkpoint+partial from existing output.", flush=True)
        print(f"Output exists but roster incomplete "
              f"(scenarios {done_scen}/{n_scen_target}, controls "
              f"{done_ctrl}/{n_ctrl_target}) — resuming remaining units.",
              flush=True)

    model_config = MODELS[model_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_fn, cleanup_fn = load_generate_fn(
        model_config, backend, max_new_tokens=DRIFT_MAX_NEW_TOKENS, log_fn=_log)

    axes_file = AXES_DIR / f"{model_key}_axes.npz"
    projector = MoodProjector(axes_file=axes_file)

    # Total units of work for progress tracking (module-level so run_conversation
    # can emit fractional per-turn progress without parameter plumbing).
    global _drift_total, _drift_t_start, _drift_done, _drift_units_start
    scenarios_list = CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios else CONFLICT_SCENARIOS
    # Turn-granular totals (see _units_done): unit = one dialogue turn.
    _scen_units.clear()
    _scen_units.update({s.name: len(list(s.turns)) + len(RECOVERY_QUESTIONS)
                        for s in scenarios_list})
    _ctrl_units.clear()
    if run_control:
        _ctrl_units.update({c["name"]: len(c["turns"])
                            for c in NEUTRAL_CONVERSATIONS})
    _drift_total = sum(_scen_units.values()) + sum(_ctrl_units.values())
    _drift_t_start = time.monotonic()
    _drift_done = 0

    # Load checkpoint for resume
    ckpt = _load_checkpoint(model_key)
    partial = _load_partial_results(model_key)

    if force:
        ckpt = {"completed_scenarios": [], "completed_controls": []}
        partial = {"conflict_scenarios": [], "neutral_controls": []}
        _clear_inflight(model_key)
        _save_checkpoint(model_key, ckpt)

    # Reconcile any in-progress scenario/control left over from a
    # mid-conversation kill. resume_state is consumed (set back to None) the
    # moment the matching scenario/control is reached below; if it's never
    # matched (e.g. --scenarios truncated the run), it's discarded after both
    # loops with a warning. Invalid/inconsistent inflight state is discarded
    # here — never blocks progress, never loses a completed scenario.
    resume_state = _validate_inflight(ckpt, model_key) if not force else None
    if ckpt.get("inflight") and resume_state is None:
        ckpt.pop("inflight", None)
        _clear_inflight(model_key)
        _save_checkpoint(model_key, ckpt)

    # --- Conflict scenarios with recovery ---
    scenarios = CONFLICT_SCENARIOS[:max_scenarios] if max_scenarios else CONFLICT_SCENARIOS
    completed_ids = set(ckpt.get("completed_scenarios", []))
    completed_controls = set(ckpt.get("completed_controls", []))
    _drift_done = _units_done(ckpt)
    _drift_units_start = _drift_done
    log_progress(_drift_done, _drift_total, 0)
    t_start = time.monotonic()

    for idx, scenario in enumerate(scenarios):
        if scenario.name in completed_ids:
            continue

        # 12 conflict turns + 4 recovery turns
        all_turns = list(scenario.turns) + RECOVERY_QUESTIONS

        if (resume_state is not None and resume_state["type"] == "conflict"
                and resume_state["name"] == scenario.name):
            start_turn = resume_state["turns_done"]
            r_messages = resume_state["messages"]
            r_turn_data = resume_state["turn_data"]
            r_hidden_states = resume_state["hidden_states"]
            resume_state = None  # only one conversation can be inflight at a time
            _log(f"Scenario {idx+1}/{len(scenarios)}: {scenario.name} "
                 f"({scenario.category}) — RESUMING from turn {start_turn + 1}/{len(all_turns)}")
        else:
            start_turn = 0
            r_messages = r_turn_data = r_hidden_states = None
            _log(f"Scenario {idx+1}/{len(scenarios)}: {scenario.name} ({scenario.category})")

        try:
            turn_data, hidden_states = run_conversation(
                generate_fn, projector, all_turns, label=scenario.name,
                model_key=model_key, ckpt=ckpt, conv_type="conflict",
                start_turn=start_turn, resume_messages=r_messages,
                resume_turn_data=r_turn_data, resume_hidden_states=r_hidden_states)

            # Scenario fully done: promote hidden states from partial to
            # final and drop the inflight marker — completed_scenarios
            # (below) is the durable "done" record from here on.
            _save_hidden_states(model_key, scenario.name, hidden_states)
            _clear_partial_hidden_states(model_key, scenario.name)
            _clear_inflight(model_key)
            ckpt.pop("inflight", None)

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

        _drift_done = _units_done(ckpt)
        log_progress(_drift_done, _drift_total, time.monotonic() - _drift_t_start,
                     n_elapsed=max(_drift_done - _drift_units_start, 0),
                     extra={"item": scenario.name, "phase_done": len(all_turns),
                            "phase_total": len(all_turns)})
        if _drift_budget is not None:
            _drift_budget.check_and_exit(_drift_done, _drift_total, log=_log)

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

            if (resume_state is not None and resume_state["type"] == "control"
                    and resume_state["name"] == conv["name"]):
                start_turn = resume_state["turns_done"]
                r_messages = resume_state["messages"]
                r_turn_data = resume_state["turn_data"]
                r_hidden_states = resume_state["hidden_states"]
                resume_state = None
                _log(f"Control {idx+1}/{len(NEUTRAL_CONVERSATIONS)}: {conv['name']} "
                     f"— RESUMING from turn {start_turn + 1}/{len(conv['turns'])}")
            else:
                start_turn = 0
                r_messages = r_turn_data = r_hidden_states = None
                _log(f"Control {idx+1}/{len(NEUTRAL_CONVERSATIONS)}: {conv['name']}")

            try:
                turn_data, hidden_states = run_conversation(
                    generate_fn, projector, conv["turns"], label=conv["name"],
                    model_key=model_key, ckpt=ckpt, conv_type="control",
                    start_turn=start_turn, resume_messages=r_messages,
                    resume_turn_data=r_turn_data, resume_hidden_states=r_hidden_states)

                # Save hidden states for later reprojection
                _save_hidden_states(model_key, conv["name"], hidden_states)
                _clear_partial_hidden_states(model_key, conv["name"])
                _clear_inflight(model_key)
                ckpt.pop("inflight", None)

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

            _drift_done = _units_done(ckpt)
            log_progress(_drift_done, _drift_total, time.monotonic() - _drift_t_start,
                         n_elapsed=max(_drift_done - _drift_units_start, 0),
                         extra={"item": conv["name"], "phase_done": len(conv["turns"]),
                                "phase_total": len(conv["turns"])})
            _log(f"  Control {conv['name']} done")
            if _drift_budget is not None:
                _drift_budget.check_and_exit(_drift_done, _drift_total, log=_log)

    # Inflight scenario/control was never matched (e.g. a truncated
    # --scenarios run, or config changed between crash and resume) — never
    # silently keep stale state around; drop it with a clear log line.
    if resume_state is not None:
        _log(f"WARNING: inflight '{resume_state['name']}' ({resume_state['type']}) "
             f"not found in this run's scenario/control lists — discarding "
             f"inflight state (it will restart from turn 0 if it reappears)")
        ckpt.pop("inflight", None)
        _clear_inflight(model_key)
        _save_checkpoint(model_key, ckpt)

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
    from src.time_budget import add_time_budget_args
    add_time_budget_args(parser)
    add_backend_args(parser)
    args = parser.parse_args()
    global _drift_budget
    from src.time_budget import TimeBudget
    _drift_budget = TimeBudget(args.time_budget_min, args.time_budget_exit_code)

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
