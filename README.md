# Does the Judge Change the Score? Judge Sensitivity in DeepResearch Bench

Course project replicating and extending **DeepResearch Bench** ([arXiv:2506.11763](https://arxiv.org/abs/2506.11763)).

We hold every input fixed — same tasks, same reports, same reference, same evaluation
criteria — and vary **only the judge model**. The question: how much of a RACE score is a
property of the report, and how much is a property of whoever grades it?

## Motivation

RACE scores a target report against a reference report using criteria generated per task:

```
S = S_target / (S_target + S_reference)
```

An LLM does the scoring. The paper validated that this judge agrees with human experts in
aggregate, but never reported two things we think matter:

1. **Reproducibility** — the repo leaves sampling parameters unset, so the judge is sampled
   stochastically. How much does the same judge disagree with itself across runs?
2. **Judge dependence** — does the score move when you swap the judge?

This is not hypothetical. The paper's own judge (Gemini 2.5 Pro Preview) was **shut down on
2025-12-02**, forcing the authors to migrate to GPT-5.5. Comparing the paper's published
numbers against the repo's shipped results for the same reports shows the shift:

| Dimension | Paper (Gemini-2.5-Pro judge) | Repo (GPT-5.5 judge) | Δ |
|---|---|---|---|
| Comprehensiveness | 38.99 | 41.10 | +2.11 |
| Insight / Depth | 37.66 | 40.51 | +2.85 |
| Instruction-Following | 45.77 | 46.21 | +0.44 |
| Readability | 41.46 | 41.72 | +0.26 |
| **Overall** | **40.67** | **42.18** | **+1.51** |

(Claude-3.7-Sonnet w/Search, 100 tasks. Confounded — judge generation and the cleaning model
both changed — which is exactly why a controlled experiment is needed.)

For scale: the gap between the #1 and #2 agents on the leaderboard is **1.90 points**.

## Design

| Factor | Levels |
|---|---|
| Tasks | 15 Chinese tasks, stratified by input size across the full range |
| Target reports | 5 systems (4 Deep Research Agents + Claude-3.7-Sonnet w/Search) |
| Judges | `gpt-5-mini`, `gpt-5-nano` — same vendor, same generation, different tier |
| Rounds | 2 per judge, identical inputs, to measure run-to-run noise |

300 scoring calls total. Criteria are **loaded from the benchmark's shipped file, never
regenerated**, so every judge sees identical criteria and weights. `reasoning_effort` is held
constant across judges.

Because both judges are OpenAI models, this configuration isolates the **capability-tier**
effect. Adding a Google judge (`gemini-3.8-flash`, free tier) re-enables the cross-vendor /
self-preference analysis — one commented line in `exp_run.sh`.

### Targets and their model families

| Target | Vendor | Underlying model |
|---|---|---|
| Gemini-2.5-Pro Deep Research | Google | Gemini-2.5-Pro (stated in paper) |
| OpenAI Deep Research | OpenAI | o3 |
| Perplexity Research | Perplexity | DeepSeek-R1 (reported, not confirmed) |
| Grok Deeper Search | xAI | Grok 3 |
| Claude-3.7-Sonnet w/Search | Anthropic | Claude 3.7 Sonnet |

Note that Gemini-2.5-Pro Deep Research is **both a target and the author of the reference
reports**. Its score (48.88) is the closest to 50 of any agent, which is what partial
self-comparison would produce.

## Setup

```bash
git clone https://github.com/Ayanami0730/deep_research_bench
export DRB_PATH=$PWD/deep_research_bench
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

Download the four DRA report files from
[muset-ai/DeepResearch-Bench-Dataset](https://huggingface.co/datasets/muset-ai/DeepResearch-Bench-Dataset)
into `$DRB_PATH/data/test_data/raw_data/`. Claude-3.7-Sonnet ships with the benchmark repo.

## Running

```bash
python exp_prep.py       # select the 15 tasks, report what data is missing
python exp_pilot.py      # one call per judge: measure real usage, validate JSON
python exp_patch_raw.py  # keep the judges' raw responses (do this before scoring)
./exp_run.sh smoke all   # one task per target through both judges, prints a pass/fail table
./exp_run.sh score       # 2 judges x 2 rounds x 5 targets
python exp_analyze.py    # three-layer analysis
python exp_inspect.py --stats   # compare how the judges actually reason
```

Results are written into this repository, not into the benchmark clone:

```
results/race/<target>__<judge>__<round>/
  ├── raw_results.jsonl   per-task scores      (committed)
  ├── race_result.txt     aggregated summary   (committed)
  └── judge_outputs.jsonl raw judge text       (gitignored, ~9 MB total)
```

Set `OUT_ROOT` to write them elsewhere.

`exp_prep.py` writes 15-task copies of each target into `data/raw_subset/` and
`data/cleaned_subset/`, and every run points the benchmark at those. This matters for cost:
the benchmark never passes `task_ids` to its cleaner, so given a full report file it cleans
all 100 tasks — including 50 long English reports — instead of our 15 (about $4.64 versus
$0.56 for the four DRAs).

Citation cleaning happens automatically on the first scoring run for a target and is
skipped on every run after that, so there is no separate cleaning step. (`./exp_run.sh
clean` still exists if you would rather pay for it up front.) Do **not** pass
`--skip_cleaning` to the benchmark: upstream wraps the whole scoring branch in
`if not skip_cleaning`, so the flag skips the evaluation itself, not just the cleaning,
and the run finishes with "Total tasks processed: 0".

**Resuming is per task, not per cell.** The benchmark's own resume is misleading: it
reads the finished ids and logs "Will skip N already processed task IDs", but never passes
them to `process_language_data()`, which re-scores every task in the query file and appends
the results to the old ones. A cell that already holds one task (say, from a smoke test)
ends up re-billed for it and with that task written twice, so its averages are wrong. It
also writes `raw_results.jsonl` only once the whole cell finishes, so an interrupted cell
loses every task it had already paid for.

`exp_resume.py` runs before each cell: it merges `raw_results.jsonl` with the per-task
`progress.jsonl` that the patch writes, drops duplicates and errored tasks, and hands the
benchmark a query file containing only the tasks still missing. Checked against the real
benchmark code with a mock judge:

| Scenario | Billed calls | Rows | Duplicates |
|---|---|---|---|
| Upstream: smoke, then full run | 16 | 16 | task 4 twice |
| With `exp_resume.py`: smoke, then full run | 15 | 15 | none |
| With `exp_resume.py`: killed at task 5, then rerun | 15 | 15 | none |

`exp_patch_raw.py` must run **before** scoring. The benchmark keeps only the final
numbers, so without it the judges' written comparisons are discarded and cannot be
recovered. Saving them costs nothing — the text is already generated and paid for — and
adds roughly 9 MB over the full run.

Run `exp_pilot.py` before committing to the full run. It reports measured token usage,
checks that the cheap judge produces parseable JSON with the expected number of criteria,
and extrapolates the real cost. `exp_run.sh` omits `--force`, so an interrupted run resumes.

### Cost

Measured input sizes, estimated output (the pilot replaces the estimate with real numbers):

| Item | Cost |
|---|---|
| `gpt-5-mini`, 5 targets x 2 rounds | $2.70 |
| `gpt-5-nano`, 5 targets x 2 rounds | $0.54 |
| Citation cleaning (4 DRAs, once) | $0.42 |
| **Total** | **~$3.67** |

Restricting to Chinese tasks is ~33% cheaper per call than English, because the English
reference reports are far longer (68.7k vs 20.6k characters at the median).

## Analysis

`exp_analyze.py` reports three layers:

1. **Noise floor** — mean absolute difference between round 1 and round 2 for each judge.
   Everything else is interpreted against this.
2. **Judge differences** — per target, paired by task, with bootstrap 95% CIs. Reported per
   target rather than pooled: when judges favour different targets, pooling cancels the
   effect out.
3. **Ranking stability** — each judge's ordering of the 5 targets, and whether it flips.

With a cross-vendor judge enabled, a fourth section measures each judge's deviation from the
cross-judge mean, split by whether judge and target share a model family.

Scores alone do not show whether a judge is reasoning or going through the motions. A cheap
judge can emit perfectly valid JSON while writing "both reports cover this adequately" for
every criterion. `exp_inspect.py` surfaces that: it prints each judge's written comparison
for the same criterion side by side, and `--stats` reports comment length per judge.

## Limitations

- 15 tasks, Chinese only. A small model may be weaker in Chinese than in English, so part of
  any measured tier effect could be a language effect. An English cross-check is the fix.
- Both judges are OpenAI, so vendor effects are not separable in this configuration.
- Target report lengths for OpenAI DR, Perplexity and Grok are estimated from file sizes
  until the HuggingFace files are downloaded.
- RACE scores are relative to a reference report, so absolute values are not meaningful on
  their own — only comparisons are.

## Files

| File | Purpose |
|---|---|
| `exp_prep.py` | Select the 15-task subset; verify criteria, references and target files |
| `exp_pilot.py` | One call per judge: real token usage, JSON validation, cost projection |
| `exp_run.sh` | Batch runner (`clean` / `score`), resumable |
| `exp_analyze.py` | Three-layer analysis with bootstrap CIs |
| `exp_patch_raw.py` | Patch the benchmark to persist raw judge responses (idempotent, reversible with `--revert`) |
| `exp_resume.py` | Per-task resume: merge partial results, dedupe, write the query file of remaining tasks |
| `exp_inspect.py` | Put two judges' written comparisons side by side; `--stats` summarises comment length |
| `exp_export.py` | Flatten `results/race/` into the same per-call format as `results_claude_raw_reports/` (read-only on `results/`) |
| `results_openai_raw_reports/openai_compact/` | Final gpt-5-mini / gpt-5-nano scores, see below |

## Result files: `results_openai_raw_reports/openai_compact/`

One row per judge call, in the same layout as `results_claude_raw_reports/claude_compact/`, so
the three judge families can be joined on `target` + `prompt_id`. 300 rows: 2 judges
(`gpt-5-mini`, `gpt-5-nano`) x 2 rounds x 5 targets x 15 tasks.

- `openai_judge_runs.jsonl`: full version; `raw_evaluations` holds the judge's per-criterion
  analysis and article scores.
- `openai_judge_runs.csv`: the same rows without `raw_evaluations`.
- Columns: `judge, round, target, prompt_id, race_score` (overall), the four dimension scores
  (`comprehensiveness, insight, instruction_following, readability`), `status`.
- Not recorded by our runs, so absent (unlike the Claude files): `attempts`, `served_model`, `usage`.
- 10 rows have `raw_evaluations: null`: round 1, task 4, every target, both judges. These were the
  smoke-test calls, run before `exp_patch_raw.py` was applied, so only their scores were kept.
- Scores are the final, validated ones: two all-zero results caused by a malformed judge
  response (silently scored 0 by the benchmark) were dropped and re-run; see `exp_resume.py`.
- Regenerate with `python3 exp_export.py` after a run. Target names follow the Claude/Gemini
  files (for example `openai-deep-research`, `grok-deepersearch`).
