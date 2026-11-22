import argparse
import json
import math
import sys
import os
import sqlite3
import time
from pathlib import Path
import pandas as pd
from scipy.stats import kendalltau, pearsonr

# ==============================================================================
# 1. EXPERIMENT CONFIGURATION (Matching STATS 425 Setup, Adapted for Claude)
# ==============================================================================
SUBSET_PROMPT_IDS = {1, 4, 6, 11, 16, 17, 18, 20, 22, 24, 27, 30, 34, 45, 49}

TARGET_MODELS = [
    "gemini-2.5-pro-deep-research",
    "openai-deep-research",
    "perplexity-research",
    "claude-3-7-sonnet",
    "grok-deepersearch",
]

CLAUDE_JUDGES = ["claude-haiku-5-5"]

ROUNDS = [1, 2]

# Paths (expected to match DeepResearch Bench repo structure)
DATA_DIR = Path("data")
PROMPTS_FILE = DATA_DIR / "prompt_data" / "query.jsonl"
CRITERIA_FILE = DATA_DIR / "criteria_data" / "criteria.jsonl"
REFERENCE_DIR = DATA_DIR / "test_data" / "cleaned_data"
CLEANED_REPORTS_DIR = DATA_DIR / "test_data" / "cleaned_data"
GPT55_BASELINE_FILE = DATA_DIR / "baseline" / "gpt55_item_scores.csv"
OUTPUT_DIR = Path("results_claude_race")


def load_records(path, prompts=None):
    records = {}
    prompt_ids = {row["prompt"]: pid for pid, row in prompts.items()} if prompts else {}
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if prompts and "prompt" in row:
                pid = prompt_ids.get(row["prompt"])
            else:
                pid = int(row["id"]) if "id" in row else None
            if pid in SUBSET_PROMPT_IDS:
                if pid in records:
                    raise ValueError(f"{path}: duplicate task {pid}")
                records[pid] = row
    missing = SUBSET_PROMPT_IDS - records.keys()
    if missing:
        raise ValueError(f"{path}: missing tasks {sorted(missing)}")
    return records


def strict_race_score(output, criteria, calculator):
    expected = criteria["criterions"]
    if not isinstance(output, dict) or output.keys() != expected.keys():
        raise ValueError("Unexpected RACE dimensions")
    for dimension, criteria_rows in expected.items():
        rows = output[dimension]
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"{dimension}: expected a list of scoring objects")
        names = [row.get("criterion") for row in rows]
        if any(not isinstance(name, str) for name in names) or sorted(names) != sorted(row["criterion"] for row in criteria_rows):
            raise ValueError(f"{dimension}: missing, duplicate, or renamed criterion")
        for row in rows:
            for field in ("article_1_score", "article_2_score"):
                value = row.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 10:
                    raise ValueError(f"{dimension}: invalid {field}")
    totals = calculator(output, criteria, "zh")
    target = totals["target"]["total"]
    reference = totals["reference"]["total"]
    denominator = target + reference
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Undefined RACE score")
    return target / denominator


class BudgetExhausted(Exception):
    pass


class TokenBudget:
    """Track estimated exposure for one invocation, including retries."""

    def __init__(self, limit, interval):
        self.limit = limit
        self.interval = interval
        self.used = 0
        self.last_request = None

    def reserve(self, tokens):
        if self.used + tokens > self.limit:
            raise BudgetExhausted(
                f"Budget pause: {self.used:,} tokens accounted for, "
                f"next attempt reserves {tokens:,}, session limit {self.limit:,}."
            )
        if self.last_request is not None:
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
        self.used += tokens
        self.last_request = time.monotonic()


class SpendingGuard:
    # Reserve at the higher published Haiku 5.5 rates, in millionths of a dollar.
    CAP = 4_500_000
    MAX_INPUT_TOKENS = 1_000_000

    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level="IMMEDIATE")
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS attempts (cost INTEGER NOT NULL)")

    @staticmethod
    def cost(input_tokens, output_tokens):
        return (input_tokens + 5 * output_tokens + 1) // 2

    def reserve(self, max_output_tokens):
        amount = self.cost(self.MAX_INPUT_TOKENS, max_output_tokens)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.total() + amount > self.CAP:
                raise BudgetExhausted("Next Claude request would exceed the $4.50 spending guard.")
            return self.db.execute("INSERT INTO attempts VALUES (?)", (amount,)).lastrowid

    def settle(self, attempt_id, amount):
        with self.db:
            self.db.execute("UPDATE attempts SET cost = ? WHERE rowid = ?", (amount, attempt_id))

    def total(self):
        return self.db.execute("SELECT COALESCE(SUM(cost), 0) FROM attempts").fetchone()[0]


def score_single_item(client, judge_model, prompt_text, criteria, calculator, args, budget, spending):
    from anthropic import APIConnectionError, APIStatusError

    messages = [{"role": "user", "content": prompt_text}]
    estimated = client.messages.count_tokens(model=judge_model, messages=messages).input_tokens
    # The counting endpoint returns an estimate. Allow for small differences.
    reservation = math.ceil(estimated * 1.05) + 128 + args.max_output_tokens
    print(f"    Estimated input: {estimated:,} tokens", flush=True)
    for attempt in range(1, 3):
        budget.reserve(reservation)
        attempt_id = spending.reserve(args.max_output_tokens)
        try:
            response = client.messages.create(
                model=judge_model, messages=messages, max_tokens=args.max_output_tokens,
            )
        except APIStatusError as error:
            if error.status_code == 429:
                spending.settle(attempt_id, 0)
            if error.status_code != 429 or "retry-after" not in error.response.headers:
                raise
            # Anthropic supplies retry-after for a recoverable rate limit.
            budget.used -= reservation
            if attempt == 2:
                raise
            delay = float(error.response.headers["retry-after"]) + 1
            print(f"    Rate limited. Waiting {delay:g}s.", flush=True)
            time.sleep(delay)
            continue
        except APIConnectionError as error:
            # A timed-out submission may have been processed. Keep the reservation.
            raise RuntimeError("Connection failed; attempt kept in token budget.") from error

        usage = response.usage.model_dump(mode="json")
        spending.settle(attempt_id, spending.cost(response.usage.input_tokens, response.usage.output_tokens))
        budget.used += response.usage.input_tokens + response.usage.output_tokens - reservation
        try:
            if response.stop_reason != "end_turn":
                raise ValueError(f"Unexpected Claude stop reason: {response.stop_reason}")
            body = "".join(block.text for block in response.content if block.type == "text").strip()
            if body.startswith("```") and body.endswith("```"):
                body = body.split("\n", 1)[1].removesuffix("```").strip()
            parsed = json.loads(body)
            race_score = strict_race_score(parsed, criteria, calculator)
        except (ValueError, IndexError) as error:
            # Do not resend an entire report automatically on schema failure.
            raise RuntimeError(f"Invalid verdict: {error}") from error
        return {
            "race_score": race_score, "raw_evaluations": parsed, "served_model": response.model,
            "usage": usage, "attempts": attempt,
        }


def run_experiment(args):
    repo = args.bench_repo.resolve()
    paths = {target: repo / CLEANED_REPORTS_DIR / f"{target}.jsonl" for target in TARGET_MODELS}
    for target, path in args.report:
        if target not in paths:
            raise ValueError(f"Unknown report target: {target}")
        paths[target] = Path(path).resolve()
    required = [
        repo / PROMPTS_FILE, repo / CRITERIA_FILE, repo / REFERENCE_DIR / "reference.jsonl",
        repo / "prompt/score_prompt_zh.py", repo / "utils/score_calculator.py", *paths.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    problems = [f"Missing file: {path}" for path in missing]
    if not args.check and not os.getenv("ANTHROPIC_API_KEY"):
        problems.append("Missing Claude credentials: set ANTHROPIC_API_KEY.")
    if not args.baseline.is_file():
        print(f"Optional GPT-5.5 comparison CSV missing: {args.baseline}")
    if missing:
        raise ValueError("\n".join(problems))

    sys.path.insert(0, str(repo))
    from prompt.score_prompt_zh import generate_merged_score_prompt
    from utils.score_calculator import calculate_weighted_scores

    prompts = load_records(repo / PROMPTS_FILE)
    criteria_map = load_records(repo / CRITERIA_FILE, prompts)
    references = load_records(repo / REFERENCE_DIR / "reference.jsonl", prompts)
    targets = {name: load_records(path, prompts) for name, path in paths.items()}
    for pid in sorted(SUBSET_PROMPT_IDS):
        for name, records in [("reference", references), *targets.items()]:
            if not isinstance(records[pid].get("article"), str) or not records[pid]["article"].strip():
                raise ValueError(f"{name}: empty article for task {pid}")
        criteria = criteria_map[pid]
        if not criteria.get("criterions") or criteria["criterions"].keys() != criteria.get("dimension_weight", {}).keys():
            raise ValueError(f"Invalid criteria dimensions for task {pid}")
    if problems:
        raise ValueError("\n".join(problems))
    if args.check:
        print("Input preflight passed. No scoring requests sent.")
        return

    from anthropic import Anthropic, APIError

    judges = args.judge or CLAUDE_JUDGES
    if judges != CLAUDE_JUDGES:
        raise ValueError("The spending guard is priced for claude-haiku-5-5 only.")
    client = Anthropic(max_retries=0, base_url="https://api.anthropic.com")
    budget = TokenBudget(args.token_budget, args.min_interval)
    ledger = Path(__file__).resolve().with_name("claude-race-spend.sqlite3")
    spending = SpendingGuard(ledger)
    print(f"Claude spend accounted for: ${spending.total() / 1_000_000:.4f} / $4.50")
    prompt_ids = [min(SUBSET_PROMPT_IDS)] if args.pilot else sorted(SUBSET_PROMPT_IDS)
    rounds = [1] if args.pilot else ROUNDS
    selected_targets = TARGET_MODELS[:1] if args.pilot else TARGET_MODELS
    output_dir = args.output_dir / "claude_full_prompt" if args.full_prompt else args.output_dir / "claude_compact"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / ("pilot.jsonl" if args.pilot else "claude_judge_runs.jsonl")
    planned_keys = {
        (judge, rnd, target, pid)
        for judge in judges for rnd in rounds for target in selected_targets for pid in prompt_ids
    }
    completed_keys = set()

    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") == "success":
                    completed_keys.add((rec["judge"], rec["round"], rec["target"], rec["prompt_id"]))

    print(f"Completed: {len(completed_keys & planned_keys)} / {len(planned_keys)} evaluations.")

    try:
        with open(checkpoint_file, "a", encoding="utf-8") as out_f:
            for judge in judges:
                for rnd in rounds:
                    for target in selected_targets:
                        for pid in prompt_ids:
                            run_key = (judge, rnd, target, pid)
                            if run_key in completed_keys:
                                continue

                            print(f"Scoring | Judge: {judge} | R{rnd} | Target: {target} | Prompt: {pid}")
                            criteria = criteria_map[pid]
                            criteria_for_prompt = {
                                dimension: [
                                    {key: item[key] for key in ("criterion", "explanation")}
                                    for item in items
                                ]
                                for dimension, items in criteria["criterions"].items()
                            }
                            prompt_text = generate_merged_score_prompt.format(
                                task_prompt=prompts[pid]["prompt"],
                                criteria_list=json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2),
                                article_1=targets[target][pid]["article"],
                                article_2=references[pid]["article"],
                            )
                            if not args.full_prompt:
                                prompt_text += (
                                    "\n输出要求补充：先按原评分规则完整评估。"
                                    "每条标准的 analysis 只写一句简短的中文比较理由。"
                                    "保留全部维度和每条 criterion 的原文，以及两篇文章的分数。"
                                )
                            try:
                                result = score_single_item(client, judge, prompt_text, criteria, calculate_weighted_scores, args, budget, spending)
                            except RuntimeError as error:
                                print(f"    Skipped: {error}")
                                continue
                            record = {
                                "judge": judge,
                                "round": rnd,
                                "target": target,
                                "prompt_id": pid,
                                "race_score": result["race_score"],
                                "attempts": result["attempts"],
                                "served_model": result["served_model"],
                                "usage": result["usage"],
                                "status": "success",
                                "raw_evaluations": result["raw_evaluations"],
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            out_f.flush()
                            completed_keys.add(run_key)
    except (BudgetExhausted, APIError) as error:
        print(f"Stopped with checkpoints saved. {error}")

    complete = completed_keys & planned_keys
    print(f"Finished with {len(complete)} / {len(planned_keys)} accepted scores.")
    print(f"Claude spend accounted for: ${spending.total() / 1_000_000:.4f} / $4.50")
    spending.db.close()
    print(f"Tokens accounted for this invocation: {budget.used:,} / {budget.limit:,}")
    if complete:
        with checkpoint_file.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        scores = pd.DataFrame(
            row for row in rows
            if (row["judge"], row["round"], row["target"], row["prompt_id"]) in complete
        )
        csv_path = checkpoint_file.with_suffix(".csv")
        scores.drop(columns=["raw_evaluations", "usage"]).to_csv(csv_path, index=False)
        print(f"Scores exported to {csv_path}")
    if not args.pilot and complete == planned_keys:
        analyze_results(checkpoint_file, args.baseline, judges)


def analyze_results(checkpoint_file, baseline, judges):
    with checkpoint_file.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    df = pd.DataFrame(row for row in rows if row["judge"] in judges)

    # 1. Noise Floor: Mean Absolute Difference between Round 1 and Round 2
    piv_rounds = df.pivot(
        index=["judge", "target", "prompt_id"], columns="round", values="race_score"
    ).reset_index()
    piv_rounds["abs_diff"] = (piv_rounds[1] - piv_rounds[2]).abs()
    noise_floor = piv_rounds.groupby("judge")["abs_diff"].mean()

    print("\n=== 1. NOISE FLOOR (Round 1 vs Round 2 MAD) ===")
    print(noise_floor.round(4))

    # Average across both rounds per (judge, target, prompt_id)
    item_avg = df.groupby(["judge", "target", "prompt_id"])["race_score"].mean().reset_index()

    # 2. Target Averages & Score Spread
    target_means = item_avg.pivot(index="target", columns="judge", values="race_score")
    print("\n=== 2. TARGET MEAN SCORES (0-1 RACE SCALE) ===")
    print(target_means.round(3))

    print("\n=== 3. SCORE SPREAD (Max - Min Target) ===")
    for judge in judges:
        spread = target_means[judge].max() - target_means[judge].min()
        print(f"{judge}: {spread:.3f}")

    # 3. Direct Self-Preference Check (Gemini-2.5-Pro DR vs OpenAI DR margin)
    print("\n=== 4. DESCRIPTIVE SCORE MARGIN (Gemini DR - OpenAI DR) ===")
    for judge in judges:
        margin = (
            target_means.loc["gemini-2.5-pro-deep-research", judge]
            - target_means.loc["openai-deep-research", judge]
        )
        print(f"{judge} margin: {margin:+.3f} (GPT-5.5 baseline was +0.023)")

    # 4. Comparison against GPT-5.5 Item-Level Baseline (if available)
    if baseline.is_file():
        base_df = pd.read_csv(baseline)  # Expected cols: target, prompt_id, gpt55_score
        keys = ["target", "prompt_id"]
        base_df = base_df[base_df["target"].isin(TARGET_MODELS) & base_df["prompt_id"].isin(SUBSET_PROMPT_IDS)]
        if (len(base_df) != 75 or base_df.duplicated(keys).any()
                or not base_df["gpt55_score"].between(0, 1).all()):
            raise ValueError("Baseline must contain 75 distinct matched scores on the 0–1 scale")
        merged = item_avg.merge(base_df, on=keys, validate="many_to_one")

        print("\n=== 5. ALIGNMENT WITH GPT-5.5 OFFICIAL BASELINE ===")
        base_target_means = base_df.groupby("target")["gpt55_score"].mean()

        for judge in judges:
            sub = merged[merged["judge"] == judge]
            r_val, _ = pearsonr(sub["race_score"], sub["gpt55_score"])

            j_means = sub.groupby("target")["race_score"].mean()
            tau_val, _ = kendalltau(j_means.loc[base_target_means.index], base_target_means)

            # Pairwise win-rate agreement across all 150 head-to-head comparisons
            agreements = 0
            total_pairs = 0
            for pid, grp in sub.groupby("prompt_id"):
                grp = grp.set_index("target")
                models = list(grp.index)
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        m1, m2 = models[i], models[j]
                        j_diff = grp.loc[m1, "race_score"] - grp.loc[m2, "race_score"]
                        b_diff = grp.loc[m1, "gpt55_score"] - grp.loc[m2, "gpt55_score"]
                        if (j_diff > 0 and b_diff > 0) or (j_diff < 0 and b_diff < 0):
                            agreements += 1
                        total_pairs += 1

            print(
                f"{judge} | Kendall tau: {tau_val:.2f} | "
                f"Report Pearson r (n=75): {r_val:.2f} | "
                f"Pairwise Agreement: {agreements}/{total_pairs} ({agreements/total_pairs:.1%})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Claude judges with the official DeepResearch Bench I RACE rubric.")
    parser.add_argument("--bench-repo", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, default=GPT55_BASELINE_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report", action="append", nargs=2, metavar=("TARGET", "JSONL"), default=[])
    parser.add_argument("--judge", action="append", help="Override the Claude model IDs")
    parser.add_argument(
        "--full-prompt", action="store_true",
        help="Use the official prompt without the brief-justification suffix",
    )
    parser.add_argument("--token-budget", type=int, default=100000,
                        help="Estimated billable token budget for this invocation (default 100000)")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--min-interval", type=float, default=5,
                        help="Minimum seconds between scoring requests")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate local inputs without a scoring call")
    mode.add_argument("--pilot", action="store_true", help="Score one report per judge in a separate checkpoint")
    try:
        args = parser.parse_args()
        if (args.token_budget < 0 or args.max_output_tokens < 1
                or not math.isfinite(args.min_interval) or args.min_interval < 0):
            parser.error("Use a nonnegative budget and interval and a positive output limit.")
        run_experiment(args)
    except (OSError, ValueError, ImportError, KeyError) as error:
        raise SystemExit(str(error)) from None