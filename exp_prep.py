#!/usr/bin/env python3
"""实验准备：抽出 15 道中文题的子集，并检查数据齐不齐。

用法：
    python exp_prep.py

产出：
    data/prompt_data/query_sub15.jsonl   —— 跑实验时用 --query_file 指向它

注意：criteria 是按 prompt 原文匹配的（主脚本第 247 行），所以子集里的
prompt 字段必须和原文逐字一致，不能改写。
"""
import json, os, sys

# DRB_PATH 指向 deep_research_bench 仓库；不设则默认脚本就放在该仓库根目录
REPO = os.environ.get("DRB_PATH") or os.path.dirname(os.path.abspath(__file__))
QUERY      = os.path.join(REPO, "data/prompt_data/query.jsonl")
CRITERIA   = os.path.join(REPO, "data/criteria_data/criteria.jsonl")
REFERENCE  = os.path.join(REPO, "data/test_data/cleaned_data/reference.jsonl")
RAW_DIR    = os.path.join(REPO, "data/test_data/raw_data")
CLEAN_DIR  = os.path.join(REPO, "data/test_data/cleaned_data")
OUT        = os.path.join(REPO, "data/prompt_data/query_sub15.jsonl")

# 按输入体量十五分位分层抽样得到，覆盖从最短到最长的全范围
TASK_IDS = [4, 11, 45, 49, 20, 34, 27, 1, 17, 18, 22, 30, 16, 24, 6]

# 5 个 target。前 4 个从 HF 下载，第 5 个仓库自带。
TARGETS = [
    "gemini-2.5-pro-deepresearch",
    "openai-deepresearch",
    "perplexity-Research",
    "grok-deeper-search",
    "claude-3-7-sonnet-latest",
]


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def est_tokens(s):
    """字符数 -> token 估算。中文 0.7/字，拉丁 0.25/字符。"""
    cjk = sum(1 for c in s if "一" <= c <= "鿿")
    f = cjk / max(len(s), 1)
    return len(s) * (f * 0.7 + (1 - f) * 0.25)


def main():
    queries = {r["id"]: r for r in load_jsonl(QUERY)}
    criteria_by_prompt = {r["prompt"]: r for r in load_jsonl(CRITERIA)}
    refs = {r["id"]: r["article"] for r in load_jsonl(REFERENCE)}

    picked, problems = [], []
    for tid in TASK_IDS:
        q = queries.get(tid)
        if q is None:
            problems.append(f"  任务 {tid} 不在 query.jsonl 里")
            continue
        if q.get("language") != "zh":
            problems.append(f"  任务 {tid} 不是中文题（language={q.get('language')}）")
        if q["prompt"] not in criteria_by_prompt:
            problems.append(f"  任务 {tid} 在 criteria.jsonl 里找不到对应 prompt —— 这一题跑不了")
            continue
        if tid not in refs:
            problems.append(f"  任务 {tid} 没有参考报告")
            continue
        picked.append(q)

    if problems:
        print("发现问题：")
        print("\n".join(problems))
        if len(picked) < len(TASK_IDS):
            print(f"\n只有 {len(picked)}/{len(TASK_IDS)} 道题可用。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for q in picked:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\n已写出子集：{OUT}（{len(picked)} 道题）")

    # 每题的 criteria 条数与体量
    print(f"\n{'id':<5}{'topic':<26}{'标准数':>7}{'参考报告字符':>14}")
    n_crit_total = 0
    for q in picked:
        c = criteria_by_prompt[q["prompt"]]
        n = sum(len(v) for v in c["criterions"].values())
        n_crit_total += n
        print(f"{q['id']:<5}{q['topic'][:24]:<26}{n:>7}{len(refs[q['id']]):>14,}")
    print(f"{'':<31}{n_crit_total:>7}  <- 评分标准总条数")

    # target 文件齐不齐
    print("\ntarget 数据状态：")
    ids = {q["id"] for q in picked}
    need_clean = []
    for t in TARGETS:
        raw = os.path.join(RAW_DIR, f"{t}.jsonl")
        cln = os.path.join(CLEAN_DIR, f"{t}.jsonl")
        has_raw, has_cln = os.path.exists(raw), os.path.exists(cln)
        if has_cln:
            cov = len(ids & {r["id"] for r in load_jsonl(cln)})
            note = f"已清洗，覆盖 {cov}/{len(ids)} 题" + ("" if cov == len(ids) else "  ← 缺题！")
            mark = "OK  "
        elif has_raw:
            cov = len(ids & {r["id"] for r in load_jsonl(raw)})
            note = f"只有原始版，覆盖 {cov}/{len(ids)} 题 —— 需要先清洗"
            mark = "待清洗"
            need_clean.append(t)
        else:
            note = "文件不存在 —— 需要从 Hugging Face 下载"
            mark = "缺失"
        print(f"  [{mark:<5}] {t:<32} {note}")

    if need_clean:
        print("\n需要先跑清洗（只做一次，之后所有 judge 复用）：")
        for t in need_clean:
            print(f"  python deepresearch_bench_race.py {t} --only_zh \\")
            print(f"    --query_file data/prompt_data/query_sub15.jsonl \\")
            print(f"    --output_dir results/_cleanpass/{t}")
        print("  （这一步会顺带产生 cleaned_data/<model>.jsonl，之后就能用 --skip_cleaning）")

    missing = [t for t in TARGETS
               if not os.path.exists(os.path.join(RAW_DIR, f"{t}.jsonl"))
               and not os.path.exists(os.path.join(CLEAN_DIR, f"{t}.jsonl"))]
    if missing:
        print("\n从 Hugging Face 下载缺失的 target（muset-ai/DeepResearch-Bench-Dataset）：")
        for t in missing:
            print(f"  {t}.jsonl  ->  data/test_data/raw_data/{t}.jsonl")

    # 体量估算，供成本参考
    fixed = sum(est_tokens(refs[q["id"]]) for q in picked)
    print(f"\n体量估算：参考报告合计约 {fixed:,.0f} tokens（每次打分调用都要带上）")
    print(f"          评分标准 {n_crit_total} 条，输出 token 主要由这个数决定")


if __name__ == "__main__":
    main()
