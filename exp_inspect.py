#!/usr/bin/env python3
"""并排对比不同 judge 对同一条标准写的评语。

分数相同不等于判断质量相同。低档 judge 可能格式完全合法，但评语是空话。
这种问题自动校验查不出来，只能人读——这个脚本把两个 judge 的原文摆到一起。

依赖 exp_patch_raw.py 打过补丁后产生的 judge_outputs.jsonl。

用法：
    python exp_inspect.py --list                     # 看有哪些可比对的运行
    python exp_inspect.py --task 4                   # 对比任务 4
    python exp_inspect.py --task 4 --dim insight     # 只看某个维度
    python exp_inspect.py --stats                    # 评语长度的整体统计
"""
import argparse, glob, json, os, re, sys

REPO = os.environ.get("DRB_PATH") or os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 结果默认写在本项目的 results/ 下（见 exp_run.sh 的 OUT_ROOT）
OUT_ROOT = os.environ.get("OUT_ROOT") or os.path.join(SCRIPT_DIR, "results")
RESULTS = os.path.join(OUT_ROOT, "race")
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]


def load_dumps():
    """-> {(judge, round, target): {task_id: parsed_json}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "judge_outputs.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        m = re.match(r"^(.*)__(.*)__(r\d+)$", name)
        if not m:
            continue
        target, judge, rnd = m.groups()
        per_task = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw = rec.get("raw") or ""
                mm = re.search(r"\{.*\}", raw, re.S)
                if not mm:
                    continue
                try:
                    per_task[rec["id"]] = json.loads(mm.group(0))
                except Exception:
                    continue
        if per_task:
            out[(judge, rnd, target)] = per_task
    return out


def iter_items(dim_obj):
    """评分 JSON 里一个维度下的条目，dict 或 list 都接受。"""
    if isinstance(dim_obj, dict):
        return list(dim_obj.values())
    if isinstance(dim_obj, list):
        return dim_obj
    return []


def wrap(text, width, indent):
    words, line, lines = str(text).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return ("\n" + " " * indent).join(lines) if lines else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int)
    ap.add_argument("--dim", choices=DIMS)
    ap.add_argument("--round", default="r1")
    ap.add_argument("--target", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--max-criteria", type=int, default=3, help="每个维度最多展示几条")
    args = ap.parse_args()

    dumps = load_dumps()
    if not dumps:
        sys.exit(f"{RESULTS} 下没有 judge_outputs.jsonl。\n"
                 f"先跑 python exp_patch_raw.py，再重新运行 ./exp_run.sh score")

    if args.list:
        print(f"{'judge':<18}{'轮次':<8}{'target':<32}{'题数':>6}")
        for (j, r, t), v in sorted(dumps.items()):
            print(f"{j:<18}{r:<8}{t:<32}{len(v):>6}")
        return

    if args.stats:
        print("评语长度统计（字符数，反映判断的详略）\n")
        print(f"  {'judge':<18}{'条目数':>8}{'平均':>8}{'中位':>8}{'最短':>8}")
        agg = {}
        for (j, r, t), tasks in dumps.items():
            for obj in tasks.values():
                for d in DIMS:
                    for item in iter_items(obj.get(d)):
                        if isinstance(item, dict):
                            a = item.get("comparative_analysis") or ""
                            agg.setdefault(j, []).append(len(str(a)))
        for j, lens in sorted(agg.items()):
            lens.sort()
            n = len(lens)
            print(f"  {j:<18}{n:>8}{sum(lens)/n:>8.0f}{lens[n//2]:>8}{lens[0]:>8}")
        print("\n  平均长度明显偏短，往往意味着评语在空转而不是在比较。")
        return

    if args.task is None:
        sys.exit("指定 --task <id>，或用 --list / --stats")

    sel = {k: v for k, v in dumps.items()
           if k[1] == args.round and args.task in v
           and (args.target is None or k[2] == args.target)}
    if not sel:
        sys.exit(f"没有任务 {args.task} 在 {args.round} 的记录。用 --list 看有什么。")

    targets = sorted({k[2] for k in sel})
    tgt = args.target or targets[0]
    judges = sorted({k[0] for k in sel if k[2] == tgt})
    print(f"任务 {args.task} | target = {tgt} | 轮次 {args.round}")
    print(f"judge: {', '.join(judges)}\n")

    dims = [args.dim] if args.dim else DIMS
    for d in dims:
        print("=" * 78)
        print(f"维度：{d}")
        per_judge = {j: iter_items(sel[(j, args.round, tgt)][args.task].get(d))
                     for j in judges}
        n = min((len(v) for v in per_judge.values()), default=0)
        if n == 0:
            print("  （无条目）")
            continue
        for i in range(min(n, args.max_criteria)):
            crit = None
            for j in judges:
                if isinstance(per_judge[j][i], dict):
                    crit = per_judge[j][i].get("criterion") or crit
            print(f"\n  标准 {i+1}：{wrap(crit, 70, 4)}")
            for j in judges:
                item = per_judge[j][i]
                if not isinstance(item, dict):
                    continue
                s1 = item.get("article_1_score"); s2 = item.get("article_2_score")
                a = item.get("comparative_analysis") or ""
                print(f"    [{j}]  目标 {s1} / 参考 {s2}   ({len(str(a))} 字符)")
                print(f"      {wrap(a, 68, 6)}")
        if n > args.max_criteria:
            print(f"\n  （还有 {n - args.max_criteria} 条，用 --max-criteria 调整）")


if __name__ == "__main__":
    main()
