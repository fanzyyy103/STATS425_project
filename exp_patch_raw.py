#!/usr/bin/env python3
"""给 deepresearch_bench_race.py 打补丁：把 judge 的原始回复存下来。

默认的 raw_results.jsonl 只保留最终分数，judge 写的逐条对比分析会被丢掉。
但"格式合法但内容敷衍"这种问题只能靠人读原文才能发现——尤其是低档 judge。
跑完就补不回来了，所以必须在正式运行前打这个补丁。

补丁只在设置了 JUDGE_DUMP_DIR 时生效，不影响原有行为和输出格式；
原文写到该目录下的 judge_outputs.jsonl，与 raw_results.jsonl 并列。

用法：
    export DRB_PATH=~/Desktop/deep_research_bench
    python exp_patch_raw.py           # 打补丁（可重复执行，不会叠加）
    python exp_patch_raw.py --revert  # 还原
"""
import argparse, os, shutil, sys

REPO = os.environ.get("DRB_PATH") or os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(REPO, "deepresearch_bench_race.py")
BACKUP = TARGET + ".orig"
MARK = "# --- exp patch: dump judge raw output ---"

ANCHOR = """    with lock:
        pbar.update(1)

    return final_result"""

PATCHED = """    # --- exp patch: dump judge raw output ---
    _dump_dir = os.environ.get("JUDGE_DUMP_DIR")
    if _dump_dir and llm_response_str:
        try:
            os.makedirs(_dump_dir, exist_ok=True)
            with lock:
                with open(os.path.join(_dump_dir, "judge_outputs.jsonl"),
                          "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "id": task_id,
                        "language": language,
                        "judge": os.environ.get("RACE_MODEL", ""),
                        "raw": llm_response_str,
                    }, ensure_ascii=False) + "\\n")
        except Exception as _e:
            logger.warning(f"ID {task_id}: judge dump failed - {_e}")
    # --- end exp patch ---

    with lock:
        pbar.update(1)

    return final_result"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true", help="还原为未打补丁的版本")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit(f"找不到 {TARGET}\n请设置 DRB_PATH 指向 deep_research_bench 仓库")

    if args.revert:
        if not os.path.exists(BACKUP):
            sys.exit("没有找到备份文件，无需还原")
        shutil.copy2(BACKUP, TARGET)
        os.remove(BACKUP)
        print(f"已还原 {TARGET}")
        return

    src = open(TARGET, encoding="utf-8").read()
    if MARK in src:
        print("已经打过补丁了，无需重复执行。")
        return
    if ANCHOR not in src:
        sys.exit("找不到插入位置——上游代码可能变了。\n"
                 "请检查 process_single_item 末尾是否仍是 'return final_result'。")

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print(f"已备份原文件到 {os.path.basename(BACKUP)}")

    open(TARGET, "w", encoding="utf-8").write(src.replace(ANCHOR, PATCHED, 1))
    print(f"补丁已应用到 {TARGET}")
    print("\n之后每次运行设置 JUDGE_DUMP_DIR 即可保存原文；exp_run.sh 已自动设置。")
    print("还原：python exp_patch_raw.py --revert")


if __name__ == "__main__":
    main()
