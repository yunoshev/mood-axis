#!/usr/bin/env python3
"""Re-encode drift conversation hidden states (forward pass, no generation).

Early drift runs for qwen3.5_9b / phi4 saved per-turn hidden states only for
neutral controls (20/21 of 80 scenarios). 06m (format-partialled drift) needs
conflict-turn hidden states too. This script rebuilds the missing NPZ files
from the stored conversations in data/drift/<model>.json: same chat template,
same forward pass, same decay aggregation as generation time.

Startup gate: re-encodes a few turns of scenarios that DO have stored NPZ and
compares cosine similarity. If median cosine < 0.98 the re-encode path is not
equivalent to the generation-time path — abort instead of contaminating 06m.

Produces: data/drift/hidden_states/<model>/<scenario_id>.npz  (turn_<i> keys)
Depends on: data/drift/<model>.json (05_drift output)

Usage:
    python3 scripts/05b_reencode_drift_hs.py --model qwen3.5_9b --backend mlx
"""

import argparse
import json
import os
import sys
import gc
import time
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent.parent))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:
    from fw.progress import log_progress
except ImportError:
    import json as _pj

    def log_progress(done, total, elapsed_s, n_elapsed=None, extra=None):
        n = n_elapsed if n_elapsed is not None else done
        rate = (elapsed_s / n) if (n and n > 0 and elapsed_s > 0) else None
        eta = rate * (total - done) if rate is not None else None
        data = {"done": done, "total": total, "elapsed_s": round(elapsed_s, 1),
                "rate_s": round(rate, 1) if rate is not None else None,
                "eta_s": round(eta, 1) if eta is not None else None}
        if extra:
            data.update(extra)
        print(f"#PROGRESS {_pj.dumps(data)}", flush=True)

from config.models import MODELS


class _AutoFlush:
    def __init__(self, f):
        self._f = f
    def write(self, s):
        self._f.write(s)
        self._f.flush()
    def flush(self):
        self._f.flush()
    def fileno(self):
        return self._f.fileno()


_log_file = os.environ.get("MOOD_LOG_FILE")
if _log_file:
    _writer = _AutoFlush(open(_log_file, "a"))
    sys.stdout = _writer
    sys.stderr = _writer
    _handler = logging.StreamHandler(_writer)
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[_handler])
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DRIFT_DIR = PROJECT_ROOT / "data" / "drift"
VERIFY_COS_GATE = 0.99
VERIFY_N_SCENARIOS = 3
VERIFY_N_TURNS = 2


def _hs_dir(model_key: str) -> Path:
    return DRIFT_DIR / "hidden_states" / model_key


def _atomic_savez(path: Path, arrays: dict):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_scenarios(model_key: str) -> list:
    """[(scenario_id, [turn dicts])] for all 80 scenarios, conflict first."""
    d = json.load(open(DRIFT_DIR / f"{model_key}.json"))
    out = []
    for c in d["conflict_scenarios"]:
        out.append((c["scenario_id"], c["turns"]))
    for c in d["neutral_controls"]:
        out.append((c.get("scenario_id") or c["name"], c["turns"]))
    return out


def _reencode_turn(ctx, tokenizer, template_kwargs, messages, assistant_text):
    """Hidden state for one assistant turn given the message prefix.

    messages ends with the current user turn. Returns (hidden_state, full_msgs).

    Token construction mirrors generation time EXACTLY (see
    generate_with_hidden_states: full_sequence = prompt_tokens +
    generated_tokens, where generated_tokens ends with EOS): re-rendering the
    assistant turn through the chat template instead appends trailing
    template tokens after EOS, and the recency-weighted decay aggregation is
    sensitive to the final positions (qwen3.5 verify cosine 0.80 vs 0.99+).
    """
    from src.model.mlx_inference import (
        build_prompt,
        _forward_extract_hidden,
        _aggregate_decay,
    )
    full_msgs = messages + [{"role": "assistant", "content": assistant_text}]
    prompt_text = build_prompt(ctx, messages, template_kwargs)
    prompt_ids = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        response_ids = response_ids + [eos_id]
    full_ids = prompt_ids + response_ids
    _, gen_last_n = _forward_extract_hidden(ctx, full_ids, len(prompt_ids))
    return _aggregate_decay(gen_last_n), full_msgs


def _reencode_scenario(ctx, tokenizer, template_kwargs, turns,
                       start_turn=0, prior_states=None,
                       on_turn_done=None):
    """Re-encode all turns of one conversation. Returns list of (hidden_dim,)."""
    import mlx.core as mx
    states = list(prior_states) if prior_states else []
    messages = []
    for t in turns[:start_turn]:
        messages.append({"role": "user", "content": t["user"]})
        messages.append({"role": "assistant", "content": t["assistant"]})
    for i in range(start_turn, len(turns)):
        t = turns[i]
        messages.append({"role": "user", "content": t["user"]})
        hs, messages = _reencode_turn(ctx, tokenizer, template_kwargs,
                                      messages, t["assistant"])
        states.append(hs)
        mx.metal.clear_cache()
        gc.collect()
        if on_turn_done is not None:
            on_turn_done(states)
    return states


def _verify_gate(ctx, tokenizer, template_kwargs, scenarios, hs_dir) -> bool:
    """Compare re-encoded vs stored generation-time hidden states."""
    cosines = []
    checked = 0
    for sid, turns in scenarios:
        p = hs_dir / f"{sid}.npz"
        if not p.exists():
            continue
        stored = np.load(p)
        states = _reencode_scenario(ctx, tokenizer, template_kwargs,
                                    turns[:VERIFY_N_TURNS])
        for i, hs in enumerate(states):
            ref = stored[f"turn_{i}"]
            cos = float(np.dot(hs, ref) /
                        (np.linalg.norm(hs) * np.linalg.norm(ref) + 1e-10))
            cosines.append(cos)
            logger.info(f"  verify {sid} turn_{i}: cos={cos:.4f}")
        checked += 1
        if checked >= VERIFY_N_SCENARIOS:
            break
    if not cosines:
        logger.warning("No stored NPZ available for verification — proceeding blind")
        return True
    med = float(np.median(cosines))
    logger.info(f"Verify gate: median cos={med:.4f} over {len(cosines)} turns "
                f"(gate {VERIFY_COS_GATE})")
    return med >= VERIFY_COS_GATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["mlx"], default="mlx")
    ap.add_argument("--time-budget-min", type=int, default=0)
    ap.add_argument("--time-budget-exit-code", type=int, default=75)
    args = ap.parse_args()

    model_key = args.model
    cfg = MODELS[model_key]
    scenarios = _load_scenarios(model_key)
    hs_dir = _hs_dir(model_key)
    hs_dir.mkdir(parents=True, exist_ok=True)

    total_turns = sum(len(t) for _, t in scenarios)

    def done_turns():
        """Derived from disk each time (never an accumulator)."""
        n = 0
        for sid, turns in scenarios:
            if (hs_dir / f"{sid}.npz").exists():
                n += len(turns)
            else:
                part = hs_dir / f"{sid}.partial.npz"
                if part.exists():
                    try:
                        n += len(np.load(part).files)
                    except Exception:
                        pass
        return min(n, total_turns)

    todo = [(sid, turns) for sid, turns in scenarios
            if not (hs_dir / f"{sid}.npz").exists()]
    logger.info(f"{model_key}: {len(todo)}/{len(scenarios)} scenarios missing, "
                f"{total_turns - done_turns()} turns to re-encode")
    # Initial emit so a no-op run still satisfies --require-progress
    log_progress(done_turns(), total_turns, 0.0, n_elapsed=0)
    if not todo:
        print("Nothing to do — all NPZ present", flush=True)
        return

    from src.model.mlx_inference import load_mlx_model
    logger.info(f"Loading model: {cfg.model_id} (MLX"
                f"{', ' + cfg.quantize if cfg.quantize else ''})")
    ctx = load_mlx_model(cfg.model_id, quantize=cfg.quantize,
                         mlx_model_id=cfg.mlx_model_id)
    tokenizer = ctx["tokenizer"]
    template_kwargs = cfg.chat_template_kwargs

    if not _verify_gate(ctx, tokenizer, template_kwargs, scenarios, hs_dir):
        logger.error("Verify gate FAILED: re-encode path is not equivalent to "
                     "generation-time hidden states. Aborting — do not use for 06m.")
        sys.exit(1)

    t_start = time.monotonic()
    done_at_start = done_turns()
    for sid, turns in todo:
        partial_path = hs_dir / f"{sid}.partial.npz"
        prior = []
        if partial_path.exists():
            try:
                data = np.load(partial_path)
                prior = [data[f"turn_{i}"] for i in range(len(data.files))]
            except Exception:
                prior = []
        start_turn = len(prior)
        logger.info(f"{sid}: turns {start_turn}->{len(turns)}")

        def on_turn_done(states, _sid=sid, _pp=partial_path, _n=len(turns)):
            _atomic_savez(_pp, {f"turn_{i}": s for i, s in enumerate(states)})
            d = done_turns()
            log_progress(d, total_turns, time.monotonic() - t_start,
                         n_elapsed=max(d - done_at_start, 0),
                         extra={"item": _sid, "phase_done": len(states),
                                "phase_total": _n})
            if (args.time_budget_min
                    and (time.monotonic() - t_start) > args.time_budget_min * 60
                    and len(states) < _n):
                logger.info(f"Time budget {args.time_budget_min} min reached at "
                            f"{d}/{total_turns}; clean exit "
                            f"{args.time_budget_exit_code}")
                sys.exit(args.time_budget_exit_code)

        states = _reencode_scenario(ctx, tokenizer, template_kwargs, turns,
                                    start_turn=start_turn, prior_states=prior,
                                    on_turn_done=on_turn_done)
        _atomic_savez(hs_dir / f"{sid}.npz",
                      {f"turn_{i}": s for i, s in enumerate(states)})
        partial_path.unlink(missing_ok=True)
        logger.info(f"{sid}: saved {len(states)} turns")

        if (args.time_budget_min
                and (time.monotonic() - t_start) > args.time_budget_min * 60):
            logger.info(f"Time budget {args.time_budget_min} min reached after "
                        f"{sid}; clean exit {args.time_budget_exit_code}")
            sys.exit(args.time_budget_exit_code)

    logger.info(f"Done: {done_turns()}/{total_turns} turns on disk")


if __name__ == "__main__":
    main()
