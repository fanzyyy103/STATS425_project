#!/usr/bin/env python3
"""分析：读取所有运行结果，做三层分析。

  第一层  噪声底 —— 同一个 judge 两轮之间差多少
  第二层  judge 之间的差异，并与噪声底对比
  第三层  排名是否翻转 + 同家族偏好

用法：
    python exp_analyze.py
    python exp_analyze.py --dim insight     # 换一个维度看
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np

# DRB_PATH 指向 deep_research_bench 仓库；不设则默认脚本就放在该仓库根目录
REPO = os.environ.get("DRB_PATH") or os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 结果默认写在本项目的 results/ 下（见 exp_run.sh 的 OUT_ROOT）
OUT_ROOT = os.environ.get("OUT_ROOT") or os.path.join(SCRIPT_DIR, "results")
RESULTS = os.path.join(OUT_ROOT, "race")
DIMS = ["overall_score", "comprehensiveness", "insight",
        "instruction_following", "readability"]

# 底层模型的家族归属（用于自我偏好分析）
TARGET_FAMILY = {
    "gemini-2.5-pro-deepresearch": "Google",
    "openai-deepresearch":         "OpenAI",      # o3
    "perplexity-Research":         "第三方",       # DeepSeek-R1
    "grok-deeper-search":          "xAI",
    "claude-3-7-sonnet-latest":    "Anthropic",
}
JUDGE_FAMILY = {"gpt-5-mini": "OpenAI", "gpt-5-nano": "OpenAI",
                "gemini-3.8-flash": "Google"}


def load_all(dim):
    """-> data[judge][target][round][task_id] = score"""
    data = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(RESULTS, "*", "raw_results.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        m = re.match(r"^(.*)__(.*)__(r\d+)$", name)
        if not m:
            continue
        target, judge, rnd = m.groups()
        scores = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r or dim not in r:
                    continue
                if all(r.get(d) in (0, 0.0, None) for d in DIMS):
                    # 格式故障而非判断：benchmark 把不合规输出静默记成 0（见 exp_resume.py）
                    INVALID.append((judge, target, rnd, r["id"]))
                    continue
                scores[r["id"]] = r[dim]
        if scores:
            data[judge][target][rnd] = scores
    return data


INVALID = []


def boot_ci(x, n=10000, seed=0):
    """配对差值的 bootstrap 置信区间（避免引入 scipy 依赖）。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default="overall_score", choices=DIMS)
    args = ap.parse_args()

    data = load_all(args.dim)
    if not data:
        raise SystemExit(f"{RESULTS} 下没有结果。先跑 ./exp_run.sh score")

    judges = sorted(data)
    targets = sorted({t for j in data for t in data[j]})
    if INVALID:
        print(f"⚠ 剔除 {len(INVALID)} 条全零记录（judge 输出格式不合规，benchmark 静默记 0，不是真实分数）：")
        for j, t, r, i in sorted(INVALID):
            print(f"    {j} × {t} × {r} × 任务 {i}")
        print("  重跑补齐：./exp_run.sh score <judge>（续跑脚本会只重跑这些题）\n")
    print(f"维度：{args.dim} | judge {len(judges)} 个 | target {len(targets)} 个\n")

    # ── 第一层：噪声底 ────────────────────────────────────────
    print("=" * 70)
    print("第一层  噪声底：同一个 judge 两轮之间的差异")
    print(f"\n  {'judge':<18}{'配对数':>7}{'平均|差|':>10}{'差值SD':>10}{'最大|差|':>10}")
    noise = {}
    for j in judges:
        diffs = []
        for t in targets:
            rounds = data[j].get(t, {})
            if "r1" in rounds and "r2" in rounds:
                common = set(rounds["r1"]) & set(rounds["r2"])
                diffs += [rounds["r1"][i] - rounds["r2"][i] for i in common]
        if diffs:
            d = np.array(diffs)
            noise[j] = np.abs(d).mean()
            print(f"  {j:<18}{len(d):>7}{np.abs(d).mean():>10.4f}{d.std(ddof=1):>10.4f}{np.abs(d).max():>10.4f}")
        else:
            print(f"  {j:<18}{'—':>7}{'只有一轮，无法估计噪声':>30}")
    if noise:
        print(f"\n  参考：论文排行榜上相邻 agent 的分差约 0.019–0.047（RACE 原始尺度）")

    # ── judge × target 均值 ──────────────────────────────────
    print("\n" + "=" * 70)
    print("第二层  各 judge 给各 target 的平均分（已对轮次取平均）")
    mean = defaultdict(dict)
    for j in judges:
        for t in targets:
            vals = [v for rnd in data[j].get(t, {}).values() for v in rnd.values()]
            if vals:
                mean[j][t] = float(np.mean(vals))
    print(f"\n  {'target':<30}" + "".join(f"{j:>18}" for j in judges))
    for t in targets:
        row = "".join(f"{mean[j].get(t, float('nan')):>18.4f}" for j in judges)
        print(f"  {t:<30}{row}")

    # ── judge 两两比较，按题配对 ──────────────────────────────
    print(f"\n  judge 两两差异（按 target×题 配对，bootstrap 95% CI）")
    for a in range(len(judges)):
        for b in range(a + 1, len(judges)):
            ja, jb = judges[a], judges[b]
            print(f"\n    {ja} − {jb}")
            pooled = []
            for t in targets:
                diffs = []
                for rnd in ("r1", "r2"):
                    sa = data[ja].get(t, {}).get(rnd, {})
                    sb = data[jb].get(t, {}).get(rnd, {})
                    for i in set(sa) & set(sb):
                        diffs.append(sa[i] - sb[i])
                if not diffs:
                    continue
                pooled += diffs
                d = np.array(diffs)
                lo, hi = boot_ci(d)
                flag = "CI 不含 0" if (lo > 0 or hi < 0) else "CI 含 0"
                print(f"      {t:<30}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  n={len(d):<4} {flag}")
            if pooled:
                d = np.array(pooled)
                lo, hi = boot_ci(d)
                nf = max(noise.get(ja, 0), noise.get(jb, 0))
                print(f"      {'（汇总，仅供参考）':<30}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")
                print(f"      注：汇总会把各 target 方向相反的偏向互相抵消，"
                      f"所以要看逐 target 的行。")
                if nf:
                    print(f"      噪声底 {nf:.4f} —— 逐 target 差值超过它才有解释价值。")

    # ── 第三层：排名稳定性 ───────────────────────────────────
    print("\n" + "=" * 70)
    print("第三层  排名是否随 judge 变化")
    def paired(j, a, b):
        """judge j 下，target a − b 的逐题差（两轮都用，按题配对）。"""
        out = []
        for rnd in ("r1", "r2"):
            sa, sb = data[j].get(a, {}).get(rnd, {}), data[j].get(b, {}).get(rnd, {})
            out += [sa[i] - sb[i] for i in set(sa) & set(sb)]
        return np.array(out)

    rankings = {}
    for j in judges:
        order = sorted((t for t in targets if t in mean[j]), key=lambda t: -mean[j][t])
        rankings[j] = order
        print(f"\n  {j:<18} " + "  >  ".join(f"{t.split('-')[0]}({mean[j][t]:.3f})" for t in order))

    if len(judges) > 1:
        print("\n  名次对调的 target 对，逐 judge 检验差距是否超出噪声（bootstrap 95% CI）：")
        found = False
        for ai in range(len(targets)):
            for bi in range(ai + 1, len(targets)):
                a, b = targets[ai], targets[bi]
                signs = {j: np.sign(mean[j][a] - mean[j][b]) for j in judges
                         if a in mean[j] and b in mean[j]}
                if len(set(signs.values())) < 2:
                    continue
                found = True
                print(f"\n    {a.split('-')[0]} vs {b.split('-')[0]}")
                sig = {}
                for j in judges:
                    d = paired(j, a, b)
                    lo, hi = boot_ci(d)
                    sig[j] = (lo > 0 or hi < 0)
                    print(f"      {j:<16} 差 {d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                          f"{a.split('-')[0]} 胜出 {(d > 0).mean():.0%} 的题  "
                          f"{'显著' if sig[j] else '不显著'}")
                if all(sig.values()):
                    print("      → 真实翻转：两个 judge 都显著且方向相反")
                elif any(sig.values()):
                    print("      → 不是翻转，而是区分能力差异：一个 judge 能分开，另一个分不开")
                else:
                    print("      → 两个 judge 都分不开，名次对调只是噪声")
        if not found:
            print("    无：所有 judge 给出相同排名。")

        # 压缩效应：以第一个 judge 为参照
        ref, other = judges[0], judges[1]
        common = [t for t in targets if t in mean[ref] and t in mean[other]]
        x = np.array([mean[ref][t] for t in common])
        y = np.array([mean[other][t] for t in common])
        sx, sy = x.max() - x.min(), y.max() - y.min()
        print(f"\n  分数跨度：{ref} {sx:.4f}，{other} {sy:.4f}"
              f"（{other} 相对{'压缩' if sy < sx else '拉伸'} {abs(1 - sy / sx):.0%}）")
        if len(common) >= 3:
            r = np.corrcoef(x, y - x)[0, 1]
            print(f"  {ref} 分数 与 ({other} − {ref}) 的相关系数 r = {r:.3f}"
                  f"{'  ← 低分 target 被抬得更多' if r < -0.5 else ''}")

    # ── 同家族偏好 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    fams = {JUDGE_FAMILY.get(j) for j in judges}
    if len(fams) < 2:
        print(f"同家族偏好：跳过")
        print(f"  当前 judge 全部来自 {fams.pop()}，没有跨厂商对比，这一层测不了。")
        print(f"  要启用：在 exp_run.sh 里取消 gemini-3.8-flash 那行的注释（免费层，$0），")
        print(f"  跑完后重新执行本脚本即可。")
        return
    print("同家族偏好：各 judge 相对其他 judge 均值的偏离")
    print("  （正值 = 这个 judge 给该 target 打得比别人高）")
    print(f"\n  {'target':<30}{'家族':<12}" + "".join(f"{j:>18}" for j in judges))
    for t in targets:
        vals = [mean[j][t] for j in judges if t in mean[j]]
        if len(vals) < 2:
            continue
        cross = float(np.mean(vals))
        cells = ""
        for j in judges:
            if t not in mean[j]:
                cells += f"{'—':>18}"; continue
            dev = mean[j][t] - cross
            same = JUDGE_FAMILY.get(j) == TARGET_FAMILY.get(t)
            cells += f"{(('%+.4f' % dev) + (' *' if same else '  ')):>18}"
        print(f"  {t:<30}{TARGET_FAMILY.get(t, '?'):<12}{cells}")
    print("\n  * = judge 与 target 属同一厂商家族")
    print("  若 * 格子系统性为正，就是自我偏好的证据；")
    print("  对照行（第三方/xAI/Anthropic）应当没有这种规律。")


if __name__ == "__main__":
    main()
