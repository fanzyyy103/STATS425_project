#!/usr/bin/env python3
"""把 results/race/ 整理成与 results_claude_raw_reports/claude_compact 相同的格式。

读取：results/race/<target>__<judge>__<round>/raw_results.jsonl   （分数，必需）
      results/race/<target>__<judge>__<round>/judge_outputs.jsonl （judge 原文，可选）
写出：results_openai_raw_reports/openai_compact/openai_judge_runs.jsonl   完整版（含 raw_evaluations）
      results_openai_raw_reports/openai_compact/openai_judge_runs.csv     精简版（不含 raw_evaluations）

只读 results/，只写新目录，不改任何已有文件。
用法：在 STATS425_project 目录下运行  python3 exp_export.py
"""
import csv, glob, json, os, re, sys
from collections import Counter

# 解析 judge 原文时与 benchmark 打分时使用同一个提取器，口径才一致
DRB = os.environ.get("DRB_PATH") or os.path.expanduser("~/Desktop/deep_research_bench")
sys.path.insert(0, DRB)
try:
    from utils.json_extractor import extract_json_from_markdown as _extract
    EXTRACTOR = "benchmark 自带提取器"
except Exception:
    _extract = None
    EXTRACTOR = "简易提取器（未找到 deep_research_bench，部分原文可能解析失败）"

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.environ.get("OUT_ROOT") or os.path.join(HERE, "results"), "race")
OUT = os.path.join(HERE, "results_openai_raw_reports", "openai_compact")
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]

# 与 Jennie 上传的 claude_judge_runs 保持同样的 target 拼写，三个 judge 的文件才能直接按 target+prompt_id 对齐
TARGET_NAME = {
    "gemini-2.5-pro-deepresearch": "gemini-2.5-pro-deep-research",
    "openai-deepresearch": "openai-deep-research",
    "perplexity-Research": "perplexity-research",
    "grok-deeper-search": "grok-deepersearch",
    "claude-3-7-sonnet-latest": "claude-3-7-sonnet",
}
ORDER = list(TARGET_NAME.values())


def read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def parse_eval(raw):
    """judge 原文 -> 评分 JSON；不合规（缺维度、criterion 不是字符串）返回 None。"""
    if _extract is not None:
        js = _extract(raw)
    else:
        m = re.search(r"\{.*\}", raw or "", re.S)
        js = m.group(0) if m else None
    if not js:
        return None
    try:
        obj = json.loads(js)
    except Exception:
        return None
    for d in DIMS:
        items = obj.get(d)
        if not isinstance(items, list) or not items:
            return None
        if not all(isinstance(i, dict) and isinstance(i.get("criterion"), str) for i in items):
            return None
    return obj


def valid(r):
    return "error" not in r and any(r.get(d) not in (0, 0.0, None) for d in DIMS + ["overall_score"])


rows, no_raw, bad_cells = [], 0, []
folders = sorted(glob.glob(os.path.join(SRC, "*")))
if not folders:
    sys.exit(f"{SRC} 下没有结果。请在 STATS425_project 目录里运行。")

for d in folders:
    m = re.match(r"^(.*)__(.*)__r(\d+)$", os.path.basename(d))
    if not m or m.group(1) not in TARGET_NAME:
        continue
    target, judge, rnd = m.group(1), m.group(2), int(m.group(3))

    # 原文：同一题可能有多条（早先的无效输出 + 重跑后的有效输出），取最后一条合规的
    evals = {}
    for rec in read_jsonl(os.path.join(d, "judge_outputs.jsonl")):
        obj = parse_eval(rec.get("raw"))
        if obj is not None:
            evals[rec["id"]] = obj

    scores = {}
    for r in read_jsonl(os.path.join(d, "raw_results.jsonl")):
        if valid(r):
            scores[r["id"]] = r
    if len(scores) != 15:
        bad_cells.append((os.path.basename(d), len(scores)))

    for pid, r in scores.items():
        row = {
            "judge": judge, "round": rnd, "target": TARGET_NAME[target], "prompt_id": pid,
            "race_score": r["overall_score"],
            **{k: r[k] for k in DIMS},
            "status": "success",
            "raw_evaluations": evals.get(pid),
        }
        if row["raw_evaluations"] is None:
            no_raw += 1
        rows.append(row)

rows.sort(key=lambda x: (x["judge"], x["round"], ORDER.index(x["target"]), x["prompt_id"]))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "openai_judge_runs.jsonl"), "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
cols = [c for c in rows[0] if c != "raw_evaluations"]
with open(os.path.join(OUT, "openai_judge_runs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in cols})

print(f"写入 {OUT}")
print(f"原文解析：{EXTRACTOR}")
print(f"共 {len(rows)} 行（应为 300）")
for (j, rd), n in sorted(Counter((r['judge'], r['round']) for r in rows).items()):
    print(f"  {j:<12} 第{rd}轮  {n} 行（应为 75）")
print(f"带 judge 原文：{len(rows) - no_raw} 行；缺原文：{no_raw} 行")
if bad_cells:
    print("⚠ 这些格子不是 15 题：", bad_cells)
for fn in ("openai_judge_runs.jsonl", "openai_judge_runs.csv"):
    print(f"  {fn}  {os.path.getsize(os.path.join(OUT, fn)) / 1e6:.1f} MB")
