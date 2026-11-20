#!/usr/bin/env python3
"""按题续跑：算出一格（target × judge × 轮次）还剩哪些题没做。

benchmark 自带的"续跑"是假的：它读出已完成的 id、打印 "Will skip N ..."，
但这些 id 从没传给真正干活的 process_language_data()，后者会把 query 文件里
的题全部重做，再和旧结果拼在一起——同一题出现两次，均值被算偏。另外它只在
整格跑完后才写 raw_results.jsonl，中途中断就全丢。

这个脚本在每格运行前执行：
  1. 合并 raw_results.jsonl 和 progress.jsonl（补丁逐题写的），按 id 去重，
     丢掉 error 记录和全零记录（让它们重跑；全零见 is_valid 的说明）
  2. 写回去重后的 raw_results.jsonl；全部完成时顺带写 race_result.txt
  3. 只把还没做的题写进 _query_remaining.jsonl，给 benchmark 当 --query_file

benchmark 看到的 query 文件里只有剩下的题，于是新结果和旧结果不会重叠。

用法：python exp_resume.py OUT_DIR QUERY_FILE
输出：一行 "<已完成题数> <剩余题数>"
"""
import json, os, sys

DIMS = ["comprehensiveness", "insight", "instruction_following", "readability", "overall_score"]


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # 中断时可能留下半行
    return out


def is_valid(r):
    """有效记录：没有 error，且不是四个维度全为 0。

    全零不是 judge 的判断，而是格式故障：judge 把 criterion 写成列表等不合规输出时，
    benchmark 的 calculate_weighted_scores 会静默跳过每一条标准，总分记为 0，
    既不报错也不重试。实测 gpt-5-nano 有 1.4% 的调用如此，gpt-5-mini 为 0。"""
    if "error" in r:
        return False
    return any(r.get(d) not in (0, 0.0, None) for d in DIMS)


def main():
    out_dir, query_file = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    tasks = load(query_file)
    order = {t["id"]: i for i, t in enumerate(tasks)}

    done, zeros = {}, set()
    for r in load(os.path.join(out_dir, "raw_results.jsonl")) + \
             load(os.path.join(out_dir, "progress.jsonl")):
        rid = r.get("id")
        if rid not in order:
            continue
        if not is_valid(r):
            if "error" not in r:
                zeros.add(rid)
            continue
        if rid not in done:
            done[rid] = r

    merged = sorted(done.values(), key=lambda r: order[r["id"]])
    rr = os.path.join(out_dir, "raw_results.jsonl")
    if merged:
        with open(rr, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif os.path.exists(rr):
        os.remove(rr)                 # 只剩 error 记录：清掉，整格重跑

    remaining = [t for t in tasks if t["id"] not in done]
    with open(os.path.join(out_dir, "_query_remaining.jsonl"), "w", encoding="utf-8") as f:
        for t in remaining:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    if merged and not remaining:
        names = {"comprehensiveness": "Comprehensiveness", "insight": "Insight",
                 "instruction_following": "Instruction Following",
                 "readability": "Readability", "overall_score": "Overall Score"}
        with open(os.path.join(out_dir, "race_result.txt"), "w", encoding="utf-8") as f:
            for d in DIMS:
                f.write(f"{names[d]}: {sum(r[d] for r in merged) / len(merged):.4f}\n")

    zeros -= set(done)          # 已经重跑成功的就不再提示
    if zeros:
        print(f"⚠ {os.path.basename(out_dir)}: 任务 {sorted(zeros)} 是全零记录"
              f"（judge 输出格式不合规，benchmark 静默记 0），已剔除，将重跑",
              file=sys.stderr)
    print(len(done), len(remaining))


if __name__ == "__main__":
    main()
