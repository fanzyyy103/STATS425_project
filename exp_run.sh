#!/usr/bin/env bash
# 批量运行：3 个 judge × 2 轮 × 5 个 target × 15 题
#
#   ./exp_run.sh smoke     冒烟测试：1 道题 × 2 个 judge，跑完打印结果文件
#   ./exp_run.sh clean     预先清洗（可选；不跑的话首次打分时会自动清洗）
#   ./exp_run.sh score     正式打分
#   ./exp_run.sh score gpt-5-nano    只跑指定 judge
#   DRY=1 ./exp_run.sh score         只打印命令不执行
#
# 不加 --force，所以中断后重跑会跳过已完成的题目，可以安全续跑。

set -uo pipefail

# DRB_PATH 指向 deep_research_bench 仓库；不设则默认脚本就放在该仓库根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRB="${DRB_PATH:-$SCRIPT_DIR}"
[[ -f "$DRB/deepresearch_bench_race.py" ]] || {
  echo "在 $DRB 下找不到 deepresearch_bench_race.py"
  echo "请设置 DRB_PATH 指向 deep_research_bench 仓库，例如："
  echo "  export DRB_PATH=~/code/deep_research_bench"
  exit 1
}
export DRB_PATH="$DRB"

# 结果写进本项目（脚本所在目录），而不是 benchmark 克隆，便于版本管理与交付
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/results}"
mkdir -p "$OUT_ROOT"

# 只含 15 题的 target 文件（exp_prep.py 生成）。必须用它们：benchmark 清洗时
# 不按题目过滤，喂整份文件会把全部 100 题都清洗一遍（约 $4.6 vs $0.6）
SUB_RAW="$SCRIPT_DIR/data/raw_subset"
SUB_CLEAN="$SCRIPT_DIR/data/cleaned_subset"
DATA_ARGS=(--raw_data_dir "$SUB_RAW" --cleaned_data_dir "$SUB_CLEAN")
has_target() { [[ -f "$SUB_RAW/$1.jsonl" || -f "$SUB_CLEAN/$1.jsonl" ]]; }

cd "$DRB"

MODE="${1:-score}"
ONLY_JUDGE="${2:-}"
DRY="${DRY:-0}"

QUERY="data/prompt_data/query_sub15.jsonl"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16000}"

TARGETS=(
  gemini-2.5-pro-deepresearch
  openai-deepresearch
  perplexity-Research
  grok-deeper-search
  claude-3-7-sonnet-latest
)

# judge 名 | 模型 id | 后端 | 并发数
JUDGES=(
  "gpt-5-mini|gpt-5-mini|openai|8"
  "gpt-5-nano|gpt-5-nano|openai|8"
  # ↓ 先不跑 Gemini。要加回跨厂商/自我偏好分析时取消注释（免费层，$0）
  # 免费层速率低，并发必须小，撞 429 就分两天跑
  # "gemini-3.8-flash|gemini-3.8-flash|gemini|2"
)
ROUNDS=(r1 r2)

[[ -f "$QUERY" ]] || { echo "找不到 ${QUERY}，先跑 python exp_prep.py"; exit 1; }

setup_env() {   # $1 = 后端
  if [[ "$1" == "gemini" ]]; then
    : "${GEMINI_API_KEY:?需要设置 GEMINI_API_KEY}"
    export LLM_BACKEND=openai
    export OPENAI_API_KEY="$GEMINI_API_KEY"
    export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
  else
    : "${OPENAI_API_KEY:?需要设置 OPENAI_API_KEY}"
    export LLM_BACKEND=openai
    export OPENAI_BASE_URL="https://api.openai.com/v1"
  fi
}

run() {
  echo "+ $*"
  [[ "$DRY" == "1" ]] || "$@"
}

# ── 冒烟测试：每个 target 1 道题 × 2 个 judge，验证整条流水线 ─────
#   ./exp_run.sh smoke                     只测 Claude（已清洗，只验打分）
#   ./exp_run.sh smoke all                 全部 5 个 target（也验证清洗路径）
#   ./exp_run.sh smoke grok-deeper-search  指定一个
if [[ "$MODE" == "smoke" ]]; then
  ARG="${2:-claude-3-7-sonnet-latest}"
  if [[ "$ARG" == "all" ]]; then SMOKE_TARGETS=("${TARGETS[@]}"); else SMOKE_TARGETS=("$ARG"); fi

  RUN_T=()
  for T in "${SMOKE_TARGETS[@]}"; do
    if has_target "$T"; then RUN_T+=("$T")
    else echo "跳过 ${T}：子集里没有（先下载并重跑 python exp_prep.py）"; fi
  done
  [[ ${#RUN_T[@]} -gt 0 ]] || { echo "没有可测的 target"; exit 1; }

  echo "冒烟测试：${#RUN_T[@]} 个 target × 1 道题 × 2 个 judge"
  echo "写入正式目录，之后跑全量会自动跳过这些题，不浪费。"
  for T in "${RUN_T[@]}"; do
    for J in "${JUDGES[@]}"; do
      IFS='|' read -r JNAME JMODEL JBACKEND JWORKERS <<< "$J"
      setup_env "$JBACKEND"
      export RACE_MODEL="$JMODEL"
      OUT="$OUT_ROOT/race/${T}__${JNAME}__r1"
      export JUDGE_DUMP_DIR="$OUT"
      echo ""
      echo "=== target=${T}  judge=${JNAME} ==="
      run python deepresearch_bench_race.py "$T" \
        --only_zh --limit 1 "${DATA_ARGS[@]}" \
        --query_file "$QUERY" --max_workers 1 --output_dir "$OUT"
    done
  done

  [[ "$DRY" == "1" ]] && exit 0
  echo ""
  echo "════════════════════ 冒烟测试结果 ════════════════════"
  python - "$OUT_ROOT/race" "$SUB_CLEAN" "${RUN_T[@]}" <<'PYEOF'
import json, os, sys
race, sub_clean, targets = sys.argv[1], sys.argv[2], sys.argv[3:]
judges = ["gpt-5-mini", "gpt-5-nano"]
ok_all = True
print(f"\n{'target':<30}{'清洗':<8}" + "".join(f"{j:>14}" for j in judges) + "   原文")
for t in targets:
    cf = os.path.join(sub_clean, f"{t}.jsonl")
    n_clean = sum(1 for _ in open(cf)) if os.path.exists(cf) else 0
    row, dumps = f"{t:<30}{(str(n_clean)+' 题'):<8}", []
    for j in judges:
        d = os.path.join(race, f"{t}__{j}__r1")
        rr, jo = os.path.join(d, "raw_results.jsonl"), os.path.join(d, "judge_outputs.jsonl")
        cell = "无结果"
        if os.path.exists(rr):
            recs = [json.loads(l) for l in open(rr) if l.strip()]
            good = [r for r in recs if "error" not in r]
            if good:
                cell = f"{good[0]['overall_score']:.4f}"
            elif recs:
                cell = "error"
        if cell in ("无结果", "error"):
            ok_all = False
        row += f"{cell:>14}"
        dumps.append("✓" if os.path.exists(jo) else "✗")
    if "✗" in dumps:
        ok_all = False
    print(row + "   " + " ".join(dumps))
print("\n清洗列：子集里已清洗的题数（DRA 首次应为 1，Claude 为 15）")
print("原文列：judge_outputs.jsonl 是否生成（✗ = 没跑 exp_patch_raw.py）")
print("参考：GPT-5.5 官方 judge 在任务 4 上给 Claude 的 overall 是 0.4690")
print("\n" + ("✅ 全部通过，可以跑全量：./exp_run.sh score" if ok_all
              else "⚠ 有格子失败，先别跑全量，把上面的输出发出来排查"))
PYEOF
  exit 0
fi

# ── 清洗：每个 target 只做一次，与 judge 无关 ───────────────────
if [[ "$MODE" == "clean" ]]; then
  setup_env openai
  export CLEAN_MODEL="${CLEAN_MODEL:-gpt-5.6-luna}"
  for T in "${TARGETS[@]}"; do
    if [[ -f "$SUB_CLEAN/${T}.jsonl" ]]; then
      echo "跳过 ${T}（已有清洗版）"; continue
    fi
    if [[ ! -f "$SUB_RAW/${T}.jsonl" ]]; then
      echo "跳过 ${T}（原始文件缺失，需先从 Hugging Face 下载）"; continue
    fi
    echo "=== 清洗 $T ==="
    run python deepresearch_bench_race.py "$T" \
      --only_zh "${DATA_ARGS[@]}" --query_file "$QUERY" \
      --max_workers 8 --output_dir "$OUT_ROOT/_cleanpass/${T}"
  done
  echo "清洗完成。接下来：./exp_run.sh score"
  echo "（这一步是可选的——不预清洗的话，首次打分时会自动完成）"
  exit 0
fi

# ── 打分 ────────────────────────────────────────────────────────
START=$(date +%s)
NDONE=0
for J in "${JUDGES[@]}"; do
  IFS='|' read -r JNAME JMODEL JBACKEND JWORKERS <<< "$J"
  [[ -n "$ONLY_JUDGE" && "$JNAME" != "$ONLY_JUDGE" ]] && continue

  setup_env "$JBACKEND"
  export RACE_MODEL="$JMODEL"

  for R in "${ROUNDS[@]}"; do
    for T in "${TARGETS[@]}"; do
      if ! has_target "$T"; then
        echo "跳过 ${T}（没有清洗版，先跑 ./exp_run.sh clean）"; continue
      fi
      OUT="$OUT_ROOT/race/${T}__${JNAME}__${R}"
      # judge 原文写到同一目录（需先跑 exp_patch_raw.py，否则此变量被忽略）
      export JUDGE_DUMP_DIR="$OUT"
      echo ""
      echo "=== judge=$JNAME  轮次=$R  target=$T ==="
      run python deepresearch_bench_race.py "$T" \
        --only_zh "${DATA_ARGS[@]}" \
        --query_file "$QUERY" \
        --max_workers "$JWORKERS" \
        --output_dir "$OUT"
      NDONE=$((NDONE+1))
    done
  done
done

echo ""
echo "完成 $NDONE 个运行，用时 $(( ($(date +%s)-START)/60 )) 分钟"
echo "结果在 $OUT_ROOT/race/<target>__<judge>__<round>/raw_results.jsonl"
echo "下一步：python exp_analyze.py"
