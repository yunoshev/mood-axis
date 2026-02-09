#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import MODELS
from config.settings import AXES_DIR, AXIS_LABELS, CALIBRATION_MAX_NEW_TOKENS, MOOD_AXES
from src.calibration.axis_computer import compute_axis_vector
from src.calibration.dataset import CalibrationSample, generate_calibration_dataset, iterate_by_axis, get_messages_for_sample
from src.model.inference import get_hidden_state_for_prompt
from src.model.loader import load_model


def _safe_json_extract(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _anthropic_messages_create(api_key: str, payload: dict) -> dict:
    req = Request(
        url="https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    with urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def anthropic_judge_pairwise(
    *,
    api_key: str,
    judge_model: str,
    axis: str,
    positive_label: str,
    negative_label: str,
    question: str,
    response_a: str,
    response_b: str,
    max_retries: int = 5,
    retry_sleep_s: float = 1.0,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    system = (
        "You are an impartial evaluator of assistant responses. "
        f"Decide which response is more {positive_label} (as opposed to {negative_label}). "
        "Return ONLY valid JSON with keys: choice (A or B), confidence (0 to 1)."
    )

    user = (
        f"Axis: {axis}\n"
        f"Definition: {positive_label} vs {negative_label}\n\n"
        f"Question: {question}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n"
    )

    payload = {
        "model": judge_model,
        "max_tokens": 128,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            data = _anthropic_messages_create(api_key, payload)
            content = data.get("content", [])
            text = ""
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and first.get("type") == "text":
                    text = first.get("text", "")
            parsed = _safe_json_extract(text)
            if not parsed:
                return None, None, text

            choice = parsed.get("choice")
            confidence = parsed.get("confidence")
            if isinstance(choice, str):
                choice = choice.strip().upper()
            if isinstance(confidence, (int, float)):
                confidence = float(confidence)
            else:
                confidence = None

            if choice not in {"A", "B"}:
                return None, confidence, text

            return choice, confidence, text
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(retry_sleep_s * (2**attempt))
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(retry_sleep_s * (2**attempt))

    return None, None, last_err


def _pearsonr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _spearmanr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return None

    r, _p = spearmanr(x_arr, y_arr)
    if r is None or np.isnan(r):
        return None
    return float(r)


def _collect_states_and_texts(
    model,
    tokenizer,
    samples: List[CalibrationSample],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in samples:
        messages = get_messages_for_sample(s)
        text, hidden_state = get_hidden_state_for_prompt(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        out.append(
            {
                "sample": asdict(s),
                "text": text,
                "hidden_state": hidden_state,
            }
        )
    return out


def _split_pairs(
    pos: List[CalibrationSample],
    neg: List[CalibrationSample],
    train_frac: float,
    rng: random.Random,
) -> Tuple[List[Tuple[CalibrationSample, CalibrationSample]], List[Tuple[CalibrationSample, CalibrationSample]]]:
    by_q_pos = {s.question: s for s in pos}
    by_q_neg = {s.question: s for s in neg}
    common_questions = sorted(set(by_q_pos.keys()) & set(by_q_neg.keys()))
    pairs = [(by_q_pos[q], by_q_neg[q]) for q in common_questions]
    rng.shuffle(pairs)

    if not pairs:
        return [], []

    train_n = int(round(len(pairs) * train_frac))
    train_n = max(1, min(train_n, len(pairs) - 1)) if len(pairs) >= 2 else len(pairs)

    train_pairs = pairs[:train_n]
    val_pairs = pairs[train_n:]
    return train_pairs, val_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key from config/models.py (e.g. qwen_7b) or HF model id")
    parser.add_argument("--axes", help="Comma-separated axes (default: all)")
    parser.add_argument("--samples_per_pole", type=int, default=20)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=CALIBRATION_MAX_NEW_TOKENS)
    parser.add_argument("--judge", choices=["none", "anthropic"], default="none")
    parser.add_argument("--judge_model", default="claude-3-haiku-20240307")
    parser.add_argument("--anthropic_api_key", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_pairs", type=int, default=None)
    parser.add_argument("--rate_limit_s", type=float, default=0.0)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    rng = random.Random(args.seed)

    axes = args.axes.split(",") if args.axes else list(MOOD_AXES)

    if args.model in MODELS:
        model_id = MODELS[args.model].model_id
        model_short = MODELS[args.model].model_short
    else:
        model_id = args.model
        model_short = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (AXES_DIR.parent / "judge" / "calibration")
        out_path = out_dir / f"judge_calibration_{model_short}_{args.judge}_{ts}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.judge == "anthropic":
        api_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for --judge anthropic")
    else:
        api_key = None

    model, tokenizer = load_model(model_id)

    dataset = generate_calibration_dataset(
        num_samples_per_style=args.samples_per_pole,
        axes=axes,
    )

    report: Dict[str, Any] = {
        "model": {"model_id": model_id, "model_short": model_short},
        "config": {
            "axes": axes,
            "samples_per_pole": args.samples_per_pole,
            "train_frac": args.train_frac,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "judge": args.judge,
            "judge_model": args.judge_model if args.judge == "anthropic" else None,
            "limit_pairs": args.limit_pairs,
        },
        "per_axis": {},
    }

    for axis, pos_samples, neg_samples in iterate_by_axis(dataset):
        if axis not in axes:
            continue

        pos_label, neg_label = AXIS_LABELS.get(axis, ("Positive", "Negative"))

        train_pairs, val_pairs = _split_pairs(pos_samples, neg_samples, args.train_frac, rng)
        if args.limit_pairs is not None:
            val_pairs = val_pairs[: args.limit_pairs]

        train_pos = [p for p, _ in train_pairs]
        train_neg = [n for _, n in train_pairs]
        val_pos = [p for p, _ in val_pairs]
        val_neg = [n for _, n in val_pairs]

        train_pos_items = _collect_states_and_texts(model, tokenizer, train_pos, args.max_new_tokens)
        train_neg_items = _collect_states_and_texts(model, tokenizer, train_neg, args.max_new_tokens)

        axis_vector = compute_axis_vector(
            [d["hidden_state"] for d in train_pos_items],
            [d["hidden_state"] for d in train_neg_items],
        )

        val_pos_items = _collect_states_and_texts(model, tokenizer, val_pos, args.max_new_tokens)
        val_neg_items = _collect_states_and_texts(model, tokenizer, val_neg, args.max_new_tokens)

        val_pairs_items = list(zip(val_pos_items, val_neg_items))

        proj_pos = [float(np.dot(d["hidden_state"], axis_vector)) for d in val_pos_items]
        proj_neg = [float(np.dot(d["hidden_state"], axis_vector)) for d in val_neg_items]

        projection_matched_pair_acc = None
        if val_pairs_items:
            projection_matched_pair_acc = float(
                sum(1 for p, n in zip(proj_pos, proj_neg) if p > n) / len(val_pairs_items)
            )

        projection_cartesian_pair_acc = None
        if len(proj_pos) > 0 and len(proj_neg) > 0:
            correct = sum(1 for p in proj_pos for n in proj_neg if p > n)
            total = len(proj_pos) * len(proj_neg)
            projection_cartesian_pair_acc = float(correct / total) if total > 0 else None

        corr_label = [1.0] * len(proj_pos) + [-1.0] * len(proj_neg)
        corr_proj = proj_pos + proj_neg
        projection_label_pearson = _pearsonr(corr_proj, corr_label)
        projection_label_spearman = _spearmanr(corr_proj, corr_label)

        judged = []
        judge_matched_pair_acc = None

        if args.judge == "anthropic" and api_key:
            correct = 0
            total = 0
            for pos_item, neg_item in val_pairs_items:
                question = pos_item["sample"]["question"]

                if rng.random() < 0.5:
                    resp_a = pos_item["text"]
                    resp_b = neg_item["text"]
                    correct_choice = "A"
                else:
                    resp_a = neg_item["text"]
                    resp_b = pos_item["text"]
                    correct_choice = "B"

                choice, confidence, raw = anthropic_judge_pairwise(
                    api_key=api_key,
                    judge_model=args.judge_model,
                    axis=axis,
                    positive_label=pos_label,
                    negative_label=neg_label,
                    question=question,
                    response_a=resp_a,
                    response_b=resp_b,
                )

                ok = (choice == correct_choice)
                if choice in {"A", "B"}:
                    total += 1
                    if ok:
                        correct += 1

                judged.append(
                    {
                        "axis": axis,
                        "question": question,
                        "positive_text": pos_item["text"],
                        "negative_text": neg_item["text"],
                        "order": {"A": "positive" if correct_choice == "A" else "negative", "B": "negative" if correct_choice == "A" else "positive"},
                        "judge": {"choice": choice, "confidence": confidence, "raw": raw},
                        "correct": ok,
                    }
                )

                if args.rate_limit_s > 0:
                    time.sleep(args.rate_limit_s)

            judge_matched_pair_acc = float(correct / total) if total > 0 else None

        axis_result = {
            "axis": axis,
            "labels": {"positive": pos_label, "negative": neg_label},
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs_items),
            "projection_matched_pair_accuracy": projection_matched_pair_acc,
            "projection_cartesian_pair_accuracy": projection_cartesian_pair_acc,
            "projection_pairwise_accuracy": projection_matched_pair_acc,
            "projection_label_pearson": projection_label_pearson,
            "projection_label_spearman": projection_label_spearman,
            "judge_matched_pair_accuracy": judge_matched_pair_acc,
            "judge_pairwise_accuracy": judge_matched_pair_acc,
            "val_pairs": [
                {
                    "question": pos_item["sample"]["question"],
                    "positive_text": pos_item["text"],
                    "negative_text": neg_item["text"],
                    "projection": {
                        "positive": float(np.dot(pos_item["hidden_state"], axis_vector)),
                        "negative": float(np.dot(neg_item["hidden_state"], axis_vector)),
                    },
                }
                for pos_item, neg_item in val_pairs_items
            ],
        }

        if judged:
            axis_result["judge_details"] = judged

        report["per_axis"][axis] = axis_result

    def _mean(xs: List[Optional[float]]) -> Optional[float]:
        vals = [x for x in xs if isinstance(x, (int, float))]
        return float(sum(vals) / len(vals)) if vals else None

    per_axis = list(report["per_axis"].values())
    report["summary"] = {
        "n_axes": len(per_axis),
        "projection_matched_pair_accuracy_mean": _mean(
            [a.get("projection_matched_pair_accuracy") for a in per_axis]
        ),
        "projection_cartesian_pair_accuracy_mean": _mean(
            [a.get("projection_cartesian_pair_accuracy") for a in per_axis]
        ),
        "projection_label_pearson_mean": _mean(
            [a.get("projection_label_pearson") for a in per_axis]
        ),
        "projection_label_spearman_mean": _mean(
            [a.get("projection_label_spearman") for a in per_axis]
        ),
        "judge_matched_pair_accuracy_mean": _mean(
            [a.get("judge_matched_pair_accuracy") for a in per_axis]
        ),
    }

    print("\n=== Calibration Judge Report (summary) ===")
    print(f"Model: {model_short} ({model_id})")
    print(f"Judge: {args.judge}{' / ' + args.judge_model if args.judge == 'anthropic' else ''}")
    for axis in axes:
        a = report["per_axis"].get(axis)
        if not a:
            continue
        p_acc = a.get("projection_matched_pair_accuracy")
        p_cart = a.get("projection_cartesian_pair_accuracy")
        p_sp = a.get("projection_label_spearman")
        j_acc = a.get("judge_matched_pair_accuracy")

        def fmt(x: Optional[float]) -> str:
            return "N/A" if x is None else f"{x:.3f}"

        print(
            f"- {axis}: proj_acc={fmt(p_acc)} proj_cart={fmt(p_cart)} spearman={fmt(p_sp)} judge_acc={fmt(j_acc)}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
