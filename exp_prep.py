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

# 只含 15 题的 target 文件，放在本项目里。benchmark 清洗时不会按题目过滤
# （主脚本没把 task_ids 传给 clean_articles），喂它整份文件就会清洗全部 100 题，
# 包括 50 道英文长报告，成本约为只清洗 15 题的 8 倍。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUB_RAW    = os.path.join(SCRIPT_DIR, "data/raw_subset")
SUB_CLEAN  = os.path.join(SCRIPT_DIR, "data/cleaned_subset")

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

    # target 文件：抽出 15 题写到本项目的子集目录
    print("\ntarget 数据（写入只含这 15 题的子集文件）：")
    ids = {q["id"] for q in picked}
    os.makedirs(SUB_RAW, exist_ok=True)
    os.makedirs(SUB_CLEAN, exist_ok=True)

    order = {tid: i for i, tid in enumerate(TASK_IDS)}

    def write_subset(src, dst):
        # 必须与 query_sub15.jsonl 同序：--limit N 时，打分取 query 文件的前 N 题，
        # 清洗取原始文件的前 N 题。顺序不一致就会清洗 A 题、打分去找 B 题。
        rows = sorted((r for r in load_jsonl(src) if r.get("id") in ids),
                      key=lambda r: order[r["id"]])
        with open(dst, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    missing, need_clean = [], []
    for t in TARGETS:
        raw = os.path.join(RAW_DIR, f"{t}.jsonl")
        cln = os.path.join(CLEAN_DIR, f"{t}.jsonl")
        sub_raw = os.path.join(SUB_RAW, f"{t}.jsonl")
        sub_cln = os.path.join(SUB_CLEAN, f"{t}.jsonl")

        if not os.path.exists(raw) and not os.path.exists(cln):
            print(f"  [缺失  ] {t:<32} 文件不存在 —— 需要从 Hugging Face 下载")
            missing.append(t)
            continue

        n_raw = write_subset(raw, sub_raw) if os.path.exists(raw) else 0
        if os.path.exists(cln):
            # benchmark 已有清洗版：直接复制对应 15 题，首次打分时清洗会跳过它们
            n_cln = write_subset(cln, sub_cln)
            note = f"已清洗 {n_cln}/{len(ids)} 题（复制自 benchmark，不必重洗）"
            mark = "OK    "
        else:
            note = f"原始版 {n_raw}/{len(ids)} 题 —— 首次打分时自动清洗"
            mark = "待清洗"
            need_clean.append(t)
        if n_raw and n_raw < len(ids):
            note += f"  ← 缺 {len(ids) - n_raw} 题！"
        print(f"  [{mark}] {t:<32} {note}")

    print(f"\n  子集目录：{SUB_RAW}")
    print(f"            {SUB_CLEAN}")
    if need_clean:
        print(f"\n  {len(need_clean)} 个 target 会在首次打分时清洗，只清洗这 15 题。")

    if missing:
        print("\n从 Hugging Face 下载缺失的 target，放到 benchmark 的 raw_data 目录：")
        print(f"  cd {RAW_DIR}")
        base = "https://huggingface.co/datasets/muset-ai/DeepResearch-Bench-Dataset/resolve/main/generated_reports"
        for t in missing:
            print(f"  curl -L -O {base}/{t}.jsonl")
        print("  下载后重新运行 python exp_prep.py")

    # 体量估算，供成本参考
    fixed = sum(est_tokens(refs[q["id"]]) for q in picked)
    print(f"\n体量估算：参考报告合计约 {fixed:,.0f} tokens（每次打分调用都要带上）")
    print(f"          评分标准 {n_crit_total} 条，输出 token 主要由这个数决定")


if __name__ == "__main__":
    main()
