#!/usr/bin/env python3
"""试调用：每个 judge 打一次分，量出真实成本并验证可用性。

目的（花不到 $0.10，消掉三个不确定性）：
  1. 读到真实 token usage，把成本估算换成实测
  2. 验证便宜模型（尤其 nano）能不能吐出可解析的 JSON
  3. 验证 Gemini 兼容端点能不能用这套 payload 调通

用法：
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...
    python exp_pilot.py                      # 全部 judge
    python exp_pilot.py --judges gpt-5-nano  # 只测一个
    python exp_pilot.py --task-id 4 --target claude-3-7-sonnet-latest
"""
import argparse, json, os, sys, time
import requests

# DRB_PATH 指向 deep_research_bench 仓库；不设则默认脚本就放在该仓库根目录
REPO = os.environ.get("DRB_PATH") or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from prompt.score_prompt_zh import generate_merged_score_prompt as ZH_PROMPT
from utils.json_extractor import extract_json_from_markdown

# ── judge 配置。价格单位：美元 / 百万 token ────────────────────────
JUDGES = [
    dict(name="gpt-5-mini",       model="gpt-5-mini",
         base_url="https://api.openai.com/v1",
         key_env="OPENAI_API_KEY", price_in=0.25, price_out=2.00,
         role="OpenAI 主 judge"),
    dict(name="gpt-5-nano",       model="gpt-5-nano",
         base_url="https://api.openai.com/v1",
         key_env="OPENAI_API_KEY", price_in=0.05, price_out=0.40,
         role="同世代低档，档位对照"),
    # ↓ 先不跑 Gemini。要加回跨厂商/自我偏好分析时，取消这段注释即可（免费层，$0）
    # dict(name="gemini-3.8-flash", model="gemini-3.8-flash",
    #      base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    #      key_env="GEMINI_API_KEY", price_in=0.0, price_out=0.0,
    #      role="Google 侧（免费层）"),
]

SUB_CLEAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/cleaned_subset")


def find_cleaned(target):
    """先找本项目的子集，再退回 benchmark 自带的清洗版。"""
    for p in (os.path.join(SUB_CLEAN, f"{target}.jsonl"),
              os.path.join(REPO, f"data/test_data/cleaned_data/{target}.jsonl")):
        if os.path.exists(p):
            return p
    return None


TARGETS = ["gemini-2.5-pro-deepresearch", "openai-deepresearch",
           "perplexity-Research", "grok-deeper-search",
           "claude-3-7-sonnet-latest"]
ROUNDS = 2
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "16000"))
REASONING_EFFORT = "medium"      # 与 api.py 的 score 阶段一致，三个 judge 必须相同
EXPECTED_DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def est_tokens(s):
    cjk = sum(1 for c in s if "一" <= c <= "鿿")
    f = cjk / max(len(s), 1)
    return len(s) * (f * 0.7 + (1 - f) * 0.25)


def format_criteria_list(criteria_data):
    """与 deepresearch_bench_race.py 的 format_criteria_list 保持一致：
    只给 judge 看标准正文，不给权重（权重是之后本地施加的）。"""
    out = {}
    for dim, items in criteria_data["criterions"].items():
        out[dim] = [{"criterion": c["criterion"], "explanation": c["explanation"]}
                    for c in items]
    return json.dumps(out, ensure_ascii=False, indent=2)


def call_once(judge, user_prompt, drop_reasoning=False):
    """按 utils/api.py 的 payload 形状发请求，但读回 usage。"""
    key = os.environ.get(judge["key_env"], "")
    if not key:
        return None, f"环境变量 {judge['key_env']} 没设"
    payload = {
        "model": judge["model"],
        "messages": [{"role": "user", "content": user_prompt}],
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
    }
    if not drop_reasoning:
        payload["reasoning_effort"] = REASONING_EFFORT

    url = judge["base_url"].rstrip("/") + "/chat/completions"
    t0 = time.time()
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"},
                          json=payload, timeout=600)
    except Exception as e:
        return None, f"请求失败：{e}"
    dt = time.time() - t0

    if r.status_code != 200:
        body = r.text[:300]
        # Gemini 兼容端点可能不认 reasoning_effort，自动退一步再试
        if not drop_reasoning and r.status_code == 400 and "reasoning" in body.lower():
            print(f"    （该端点不接受 reasoning_effort，去掉后重试）")
            return call_once(judge, user_prompt, drop_reasoning=True)
        return None, f"HTTP {r.status_code}: {body}"

    d = r.json()
    u = d.get("usage", {}) or {}
    det = u.get("completion_tokens_details", {}) or {}
    return dict(
        latency=dt,
        prompt_tokens=u.get("prompt_tokens"),
        completion_tokens=u.get("completion_tokens"),
        reasoning_tokens=det.get("reasoning_tokens"),
        finish_reason=(d.get("choices") or [{}])[0].get("finish_reason"),
        content=((d.get("choices") or [{}])[0].get("message") or {}).get("content") or "",
        dropped_reasoning=drop_reasoning,
    ), None


def validate(content, n_criteria):
    """验证 judge 的输出能不能被主脚本正常解析。"""
    issues = []
    js = extract_json_from_markdown(content)
    if not js:
        return False, ["无法从回复中提取 JSON"], None
    try:
        obj = json.loads(js)
    except Exception as e:
        return False, [f"JSON 解析失败：{e}"], None

    missing = [d for d in EXPECTED_DIMS if d not in obj]
    if missing:
        issues.append(f"缺少维度：{missing}")

    got = 0
    for dim in EXPECTED_DIMS:
        for item in (obj.get(dim) or {}).values() if isinstance(obj.get(dim), dict) else (obj.get(dim) or []):
            if isinstance(item, dict):
                got += 1
                if "article_1_score" not in item or "article_2_score" not in item:
                    issues.append(f"{dim} 下有条目缺少打分字段")
                    break
    if got != n_criteria:
        issues.append(f"标准条数对不上：返回 {got} 条，应为 {n_criteria} 条")
    return (len(issues) == 0), issues, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=4)
    ap.add_argument("--target", default="claude-3-7-sonnet-latest")
    ap.add_argument("--judges", nargs="*", default=None, help="只测指定 judge")
    args = ap.parse_args()

    sub = os.path.join(REPO, "data/prompt_data/query_sub15.jsonl")
    if not os.path.exists(sub):
        sys.exit("先跑 python exp_prep.py 生成子集")
    tasks = {r["id"]: r for r in load_jsonl(sub)}
    if args.task_id not in tasks:
        sys.exit(f"任务 {args.task_id} 不在子集里。可选：{sorted(tasks)}")
    task = tasks[args.task_id]

    criteria = {r["prompt"]: r for r in load_jsonl(os.path.join(REPO, "data/criteria_data/criteria.jsonl"))}[task["prompt"]]
    n_criteria = sum(len(v) for v in criteria["criterions"].values())
    refs = {r["id"]: r["article"] for r in load_jsonl(os.path.join(REPO, "data/test_data/cleaned_data/reference.jsonl"))}
    tgt_path = find_cleaned(args.target)
    if not tgt_path:
        sys.exit(f"找不到 {args.target} 的清洗版。\n"
                 f"先 python exp_prep.py，再用 ./exp_run.sh smoke {args.target} 跑一次让它自动清洗")
    tgts = {r["id"]: r["article"] for r in load_jsonl(tgt_path)}

    user_prompt = ZH_PROMPT.format(
        task_prompt=task["prompt"],
        article_1=tgts[args.task_id],
        article_2=refs[args.task_id],
        criteria_list=format_criteria_list(criteria),
    )
    est_in = est_tokens(user_prompt)
    print(f"试调用：任务 {args.task_id}（{task['topic']}），target = {args.target}")
    print(f"标准 {n_criteria} 条 | prompt 字符 {len(user_prompt):,} | 我的输入估算 {est_in:,.0f} tokens")
    print(f"max_completion_tokens={MAX_OUTPUT_TOKENS}, reasoning_effort={REASONING_EFFORT}\n")

    judges = [j for j in JUDGES if not args.judges or j["name"] in args.judges]
    results = []
    for j in judges:
        print(f"── {j['name']}  ({j['role']})")
        res, err = call_once(j, user_prompt)
        if err:
            print(f"    失败：{err}\n")
            continue
        ok, issues, got = validate(res["content"], n_criteria)
        cost = (res["prompt_tokens"] or 0) / 1e6 * j["price_in"] + \
               (res["completion_tokens"] or 0) / 1e6 * j["price_out"]
        ratio = (res["prompt_tokens"] or 0) / est_in if est_in else 0
        per_crit = (res["completion_tokens"] or 0) / max(n_criteria, 1)

        print(f"    输入 {res['prompt_tokens']:,} tok（我的估算偏差 {ratio:.2f}x）")
        print(f"    输出 {res['completion_tokens']:,} tok"
              + (f"（其中推理 {res['reasoning_tokens']:,}）" if res.get("reasoning_tokens") else "")
              + f" = 每条标准 {per_crit:.0f} tok")
        print(f"    finish_reason={res['finish_reason']}  耗时 {res['latency']:.1f}s")
        if res["finish_reason"] == "length":
            print(f"    ⚠ 输出被截断！调大 MAX_OUTPUT_TOKENS 后重试，否则 JSON 必然不完整")
        if res["dropped_reasoning"]:
            print(f"    ⚠ 该端点不支持 reasoning_effort —— 三个 judge 的设置不一致，需在报告中说明")
        print(f"    JSON 校验：{'通过' if ok else '未通过'}")
        for i in issues:
            print(f"      - {i}")
        print(f"    本次花费 ${cost:.4f}" + ("（免费层）" if j["price_in"] == 0 else "") + "\n")
        results.append((j, res, per_crit, ratio, ok))

    if not results:
        return

    # ── 用实测值推算整个实验的花费 ──────────────────────────────
    print("=" * 66)
    print("按实测值推算全量成本")
    fixed = sum(est_tokens(refs[t["id"]]) + est_tokens(format_criteria_list(
        criteria if t["id"] == args.task_id else
        {r["prompt"]: r for r in load_jsonl(os.path.join(REPO, 'data/criteria_data/criteria.jsonl'))}[t["prompt"]]))
        + est_tokens(t["prompt"]) for t in tasks.values())
    all_crit = 0
    cmap = {r["prompt"]: r for r in load_jsonl(os.path.join(REPO, "data/criteria_data/criteria.jsonl"))}
    for t in tasks.values():
        all_crit += sum(len(v) for v in cmap[t["prompt"]]["criterions"].values())

    tgt_tokens = {}
    for t in TARGETS:
        p = find_cleaned(t)
        if p:
            d = {r["id"]: r["article"] for r in load_jsonl(p)}
            tgt_tokens[t] = sum(est_tokens(d[i]) for i in tasks if i in d)
    missing = [t for t in TARGETS if t not in tgt_tokens]

    print(f"  已有 {len(tgt_tokens)}/{len(TARGETS)} 个 target 的清洗数据"
          + (f"；缺 {', '.join(missing)}" if missing else ""))
    print(f"  15 题固定部分 {fixed:,.0f} tok，标准共 {all_crit} 条\n")
    print(f"  {'judge':<18}{'轮数':>5}{'输入 tok':>14}{'输出 tok':>12}{'花费':>10}")
    total = 0.0
    for j, res, per_crit, ratio, ok in results:
        rounds = ROUNDS
        tin = sum((fixed + v) * ratio for v in tgt_tokens.values()) * rounds
        tout = all_crit * len(tgt_tokens) * per_crit * rounds
        c = tin / 1e6 * j["price_in"] + tout / 1e6 * j["price_out"]
        total += c
        print(f"  {j['name']:<18}{rounds:>5}{tin:>14,.0f}{tout:>12,.0f}{'$%.2f' % c:>10}")
    print(f"  {'合计':<18}{'':<5}{'':<14}{'':<12}{'$%.2f' % total:>10}")
    if missing:
        print(f"\n  注：缺 {len(missing)} 个 target，下载并清洗后重跑本脚本可得完整报价")


if __name__ == "__main__":
    main()
