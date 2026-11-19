#!/usr/bin/env bash
# 批量运行：3 个 judge × 2 轮 × 5 个 target × 15 题
#
#   ./exp_run.sh smoke     冒烟测试：1 道题 × 2 个 judge，跑完打印结果文件
#   ./exp_run.sh clean     先清洗 4 个 DRA（只做一次，之后所有 judge 复用）
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

# ── 冒烟测试：1 道题跑通两个 judge，验证整条流水线 ──────────────
if [[ "$MODE" == "smoke" ]]; then
  T="${2:-claude-3-7-sonnet-latest}"
  if [[ ! -f "data/test_data/cleaned_data/${T}.jsonl" ]]; then
    echo "$T 还没有清洗版。先跑 ./exp_run.sh clean，或改用 claude-3-7-sonnet-latest（仓库自带）"
    exit 1
  fi
  echo "冒烟测试：target=${T}，1 道题，2 个 judge，写入正式目录（之后跑全量会自动跳过这题）"
  for J in "${JUDGES[@]}"; do
    IFS='|' read -r JNAME JMODEL JBACKEND JWORKERS <<< "$J"
    setup_env "$JBACKEND"
    export RACE_MODEL="$JMODEL"
    OUT="$OUT_ROOT/race/${T}__${JNAME}__r1"
    export JUDGE_DUMP_DIR="$OUT"
    echo ""
    echo "=== judge=$JNAME ==="
    run python deepresearch_bench_race.py "$T" \
      --skip_cleaning --only_zh --limit 1 \
      --query_file "$QUERY" --max_workers 1 --output_dir "$OUT"
  done

  echo ""
  echo "════════ 结果文件 ════════"
  for J in "${JUDGES[@]}"; do
    IFS='|' read -r JNAME _ _ _ <<< "$J"
    OUT="$OUT_ROOT/race/${T}__${JNAME}__r1"
    echo ""
    echo "── $JNAME  ($OUT)"
    if [[ -f "$OUT/raw_results.jsonl" ]]; then
      echo "   raw_results.jsonl   $(wc -l < "$OUT/raw_results.jsonl") 条"
      python - "$OUT/raw_results.jsonl" <<'PYEOF'
import json,sys
for line in open(sys.argv[1]):
    r=json.loads(line)
    if "error" in r:
        print(f"   ⚠ 任务 {r.get('id')} 失败：{r['error'][:80]}")
    else:
        print(f"   任务 {r['id']}: overall={r['overall_score']:.4f} "
              f"compre={r['comprehensiveness']:.4f} insight={r['insight']:.4f} "
              f"instr={r['instruction_following']:.4f} read={r['readability']:.4f}")
PYEOF
    else
      echo "   ⚠ 没有生成 raw_results.jsonl"
    fi
    if [[ -f "$OUT/judge_outputs.jsonl" ]]; then
      echo "   judge_outputs.jsonl $(wc -l < "$OUT/judge_outputs.jsonl") 条，$(du -h "$OUT/judge_outputs.jsonl" | cut -f1)"
    else
      echo "   judge_outputs.jsonl 没有 —— 是否忘了跑 python exp_patch_raw.py ?"
    fi
  done
  echo ""
  echo "参考：GPT-5.5（官方 judge）在同一题上给 claude 的 overall 是 0.4690"
  echo "两个 judge 都有分数且无 error，就可以跑全量了。"
  exit 0
fi

# ── 清洗：每个 target 只做一次，与 judge 无关 ───────────────────
if [[ "$MODE" == "clean" ]]; then
  setup_env openai
  export CLEAN_MODEL="${CLEAN_MODEL:-gpt-5.6-luna}"
  for T in "${TARGETS[@]}"; do
    if [[ -f "data/test_data/cleaned_data/${T}.jsonl" ]]; then
      echo "跳过 ${T}（已有清洗版）"; continue
    fi
    if [[ ! -f "data/test_data/raw_data/${T}.jsonl" ]]; then
      echo "跳过 ${T}（原始文件缺失，需先从 Hugging Face 下载）"; continue
    fi
    echo "=== 清洗 $T ==="
    run python deepresearch_bench_race.py "$T" \
      --only_zh --query_file "$QUERY" \
      --max_workers 8 --output_dir "$OUT_ROOT/_cleanpass/${T}"
  done
  echo "清洗完成。接下来：./exp_run.sh score"
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
      if [[ ! -f "data/test_data/cleaned_data/${T}.jsonl" ]]; then
        echo "跳过 ${T}（没有清洗版，先跑 ./exp_run.sh clean）"; continue
      fi
      OUT="$OUT_ROOT/race/${T}__${JNAME}__${R}"
      # judge 原文写到同一目录（需先跑 exp_patch_raw.py，否则此变量被忽略）
      export JUDGE_DUMP_DIR="$OUT"
      echo ""
      echo "=== judge=$JNAME  轮次=$R  target=$T ==="
      run python deepresearch_bench_race.py "$T" \
        --skip_cleaning --only_zh \
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
