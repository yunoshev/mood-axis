#!/usr/bin/env python3
"""External judge validation of steering results.

Uses Claude Sonnet to validate that hidden-state steering
actually shifts perceived personality (warm_cold axis).

Two tests:
  1. Pairwise forced-choice: α=-5 vs α=+5, which is warmer? → accuracy
  2. Likert dose-response: rate warmth 1-5 across alphas → Spearman correlation

Output: data/steering/judge_results.json
"""

import json
import random
import sys
import time
from pathlib import Path

import anthropic
from scipy import stats

DATA_DIR = Path(__file__).parent.parent / "data" / "steering"
JUDGE_MODEL = "claude-sonnet-4-20250514"
SEED = 42
PAIRWISE_ALPHAS = ("-5.0", "5.0")  # cold vs warm
LIKERT_ALPHAS = [-5.0, -2.0, 0.0, 2.0, 5.0]  # skip ±10 (text degrades)
BATCH_SIZE = 10  # questions per pairwise call
N_LIKERT_QUESTIONS = 10  # per model
TEXT_LIMIT = 600  # chars per text in prompt (save tokens)

MODELS = ["mistral_7b", "llama_1b", "qwen_1.5b", "qwen_7b", "gemma_9b"]


def load_steering(model: str) -> dict:
    path = DATA_DIR / f"{model}_basic.json"
    with open(path) as f:
        return json.load(f)


def run_pairwise(client, model_data, model_name):
    """Forced-choice: which text is warmer? (α=-5 vs α=+5, all 30 questions)"""
    results_by_alpha = model_data["axes"]["warm_cold"]["results"]
    neg_texts = results_by_alpha[PAIRWISE_ALPHAS[0]]
    pos_texts = results_by_alpha[PAIRWISE_ALPHAS[1]]
    n_questions = len(neg_texts)

    rng = random.Random(SEED)
    results = []

    for batch_start in range(0, n_questions, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_questions)
        batch_items = []

        for i in range(batch_start, batch_end):
            swap = rng.random() > 0.5
            if swap:
                text_a = pos_texts[i]["steered_text"][:TEXT_LIMIT]
                text_b = neg_texts[i]["steered_text"][:TEXT_LIMIT]
                warm_is = "A"
            else:
                text_a = neg_texts[i]["steered_text"][:TEXT_LIMIT]
                text_b = pos_texts[i]["steered_text"][:TEXT_LIMIT]
                warm_is = "B"

            batch_items.append({
                "question_idx": i,
                "question": neg_texts[i]["question"],
                "text_a": text_a,
                "text_b": text_b,
                "correct_answer": warm_is,
            })

        prompt = (
            "You are rating the warmth/friendliness of AI responses.\n\n"
            "For each pair below, decide which response (A or B) sounds "
            "warmer, friendlier, and more empathetic. "
            "If both are equally warm, pick the slightly warmer one.\n\n"
            "Reply with ONLY a JSON array: "
            '[{"idx": 0, "warmer": "A"}, {"idx": 1, "warmer": "B"}, ...]\n\n'
        )

        for j, item in enumerate(batch_items):
            prompt += f"--- Pair {j} ---\n"
            prompt += f"Question: {item['question']}\n\n"
            prompt += f"Response A:\n{item['text_a']}\n\n"
            prompt += f"Response B:\n{item['text_b']}\n\n"

        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.content[0].text
        start = text.find("[")
        end = text.rfind("]") + 1
        judgments = json.loads(text[start:end]) if start >= 0 and end > start else []

        batch_correct = 0
        for j, item in enumerate(batch_items):
            if j < len(judgments):
                judge_pick = judgments[j].get("warmer", "?")
                correct = judge_pick == item["correct_answer"]
            else:
                judge_pick = "?"
                correct = False

            results.append({
                "question_idx": item["question_idx"],
                "question": item["question"],
                "correct_answer": item["correct_answer"],
                "judge_pick": judge_pick,
                "correct": correct,
            })
            if correct:
                batch_correct += 1

        batch_n = batch_end - batch_start
        print(f"    batch {batch_start // BATCH_SIZE + 1}: "
              f"{batch_correct}/{batch_n} correct", flush=True)
        time.sleep(0.5)  # rate limit courtesy

    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    return {"accuracy": round(accuracy, 4), "n": len(results), "details": results}


def run_likert(client, model_data, model_name):
    """Rate warmth 1-5 across 5 alphas for sampled questions."""
    results_by_alpha = model_data["axes"]["warm_cold"]["results"]

    rng = random.Random(SEED)
    indices = list(range(len(results_by_alpha[str(LIKERT_ALPHAS[0])])))
    rng.shuffle(indices)
    selected = indices[:N_LIKERT_QUESTIONS]

    all_ratings = []
    all_projections = []
    all_alphas = []
    details = []

    for qi in selected:
        question = results_by_alpha[str(LIKERT_ALPHAS[0])][qi]["question"]

        texts = []
        for alpha in LIKERT_ALPHAS:
            item = results_by_alpha[str(alpha)][qi]
            texts.append({
                "alpha": alpha,
                "text": item["steered_text"][:TEXT_LIMIT],
                "projection": item["projections"]["warm_cold"],
            })

        # Shuffle to avoid position bias
        order = list(range(len(texts)))
        rng.shuffle(order)

        prompt = (
            "Rate the warmth/friendliness of each AI response (1-5):\n"
            "  1 = Very cold, clinical, detached\n"
            "  2 = Somewhat cold\n"
            "  3 = Neutral\n"
            "  4 = Somewhat warm, friendly\n"
            "  5 = Very warm, empathetic, caring\n\n"
            f"Question asked: {question}\n\n"
        )

        for j, idx in enumerate(order):
            letter = chr(65 + j)
            prompt += f"Response {letter}:\n{texts[idx]['text']}\n\n"

        prompt += (
            "Reply with ONLY a JSON object mapping letter to rating: "
            '{"A": 3, "B": 4, "C": 2, "D": 5, "E": 1}'
        )

        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        text_resp = resp.content[0].text
        start = text_resp.find("{")
        end = text_resp.rfind("}") + 1
        if start < 0 or end <= start:
            print(f"    q{qi}: parse failed: {text_resp[:100]}", flush=True)
            continue

        ratings_dict = json.loads(text_resp[start:end])

        q_detail = {"question_idx": qi, "question": question, "ratings": []}
        for j, idx in enumerate(order):
            letter = chr(65 + j)
            rating = int(ratings_dict.get(letter, 3))
            alpha = texts[idx]["alpha"]
            proj = texts[idx]["projection"]

            all_ratings.append(rating)
            all_projections.append(proj)
            all_alphas.append(alpha)
            q_detail["ratings"].append({
                "alpha": alpha, "projection": round(proj, 4),
                "judge_rating": rating,
            })

        details.append(q_detail)
        time.sleep(0.5)

    if len(all_ratings) < 5:
        return {"error": "too few ratings", "n_ratings": len(all_ratings)}

    rho_alpha, p_alpha = stats.spearmanr(all_alphas, all_ratings)
    rho_proj, p_proj = stats.spearmanr(all_projections, all_ratings)

    # Mean rating by alpha
    mean_by_alpha = {}
    for a in LIKERT_ALPHAS:
        vals = [r for r, av in zip(all_ratings, all_alphas) if av == a]
        mean_by_alpha[str(a)] = round(sum(vals) / len(vals), 2) if vals else None

    print(f"    rho(alpha, rating) = {rho_alpha:.3f}  p = {p_alpha:.4f}", flush=True)
    print(f"    rho(proj,  rating) = {rho_proj:.3f}  p = {p_proj:.4f}", flush=True)
    print(f"    mean ratings: {mean_by_alpha}", flush=True)

    return {
        "spearman_alpha_rating": {"rho": round(rho_alpha, 4), "p": round(p_alpha, 6)},
        "spearman_projection_rating": {"rho": round(rho_proj, 4), "p": round(p_proj, 6)},
        "mean_rating_by_alpha": mean_by_alpha,
        "n_ratings": len(all_ratings),
        "details": details,
    }


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    client = anthropic.Anthropic()
    all_results = {}

    # Resume from partial
    out_path = DATA_DIR / "judge_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {list(all_results.keys())} already done", flush=True)

    for model in models:
        if model in all_results:
            print(f"\n=== {model} === (skipped, already done)", flush=True)
            continue

        steering_path = DATA_DIR / f"{model}_basic.json"
        if not steering_path.exists():
            print(f"\n=== {model} === (skipped, no data)", flush=True)
            continue

        print(f"\n=== {model} ===", flush=True)
        data = load_steering(model)

        print("  Pairwise (alpha=-5 vs +5):", flush=True)
        pairwise = run_pairwise(client, data, model)
        print(f"  => Accuracy: {pairwise['accuracy']:.1%}", flush=True)

        print("  Likert (5 alphas x 10 questions):", flush=True)
        likert = run_likert(client, data, model)

        all_results[model] = {
            "pairwise_accuracy": pairwise["accuracy"],
            "likert_rho_alpha": likert.get("spearman_alpha_rating", {}).get("rho"),
            "likert_rho_proj": likert.get("spearman_projection_rating", {}).get("rho"),
            "pairwise": pairwise,
            "likert": likert,
        }

        # Save after each model
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved to {out_path}", flush=True)

    # Summary table
    print("\n" + "=" * 70, flush=True)
    print(f"{'Model':<15} {'Pairwise':>10} {'Likert rho(a)':>15} "
          f"{'p-value':>10} {'Mean ratings by alpha':>30}", flush=True)
    print("-" * 70, flush=True)
    for model in models:
        if model not in all_results:
            continue
        r = all_results[model]
        acc = r["pairwise_accuracy"]
        rho = r.get("likert_rho_alpha") or 0
        p = r.get("likert", {}).get("spearman_alpha_rating", {}).get("p", 1)
        means = r.get("likert", {}).get("mean_rating_by_alpha", {})
        means_str = " ".join(f"{v}" for v in means.values()) if means else ""
        print(f"{model:<15} {acc:>9.0%} {rho:>15.3f} {p:>10.4f}  {means_str}",
              flush=True)


if __name__ == "__main__":
    main()
