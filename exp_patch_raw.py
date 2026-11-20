#!/usr/bin/env python3
"""给 deepresearch_bench_race.py 打补丁：把 judge 的原始回复存下来。

默认的 raw_results.jsonl 只保留最终分数，judge 写的逐条对比分析会被丢掉。
但"格式合法但内容敷衍"这种问题只能靠人读原文才能发现——尤其是低档 judge。
跑完就补不回来了，所以必须在正式运行前打这个补丁。

补丁只在设置了 JUDGE_DUMP_DIR 时生效，不影响原有行为和输出格式。它在该目录写两个文件：
  judge_outputs.jsonl  judge 的原始回复
  progress.jsonl       每题打完立刻落盘的分数（benchmark 只在整格跑完才写结果，
                       中断就全丢；exp_resume.py 用这个文件实现按题续跑）

用法：
    export DRB_PATH=~/Desktop/deep_research_bench
    python exp_patch_raw.py           # 打补丁（可重复执行，不会叠加）
    python exp_patch_raw.py --revert  # 还原
"""
import argparse, os, shutil, sys

REPO = os.environ.get("DRB_PATH") or os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(REPO, "deepresearch_bench_race.py")
BACKUP = TARGET + ".orig"

# 第二处：清洗器把"整块都是参考文献"返回的空字符串当成失败，导致整篇被丢弃
CLEAN_TARGET = os.path.join(REPO, "utils", "clean_article.py")
CLEAN_BACKUP = CLEAN_TARGET + ".orig"
CLEAN_MARK = "# --- exp patch: accept empty chunk ---"
CLEAN_ANCHOR = """        if not result:
            logger.error(f"Chunk {idx+1}/{total} failed to clean")
            return None"""
CLEAN_PATCHED = """        # --- exp patch: accept empty chunk ---
        # "" 表示这一块全是参考文献（prompt 的约定；_clean_text 和
        # chunk_clean_article 都接受它）。上游这里写的是 `if not result`，
        # 把 "" 当成失败，整篇报告因此被丢弃。整篇的有效性仍由
        # _clean_chunked 末尾的 _is_valid_result 检查。
        if result is None:
            logger.error(f"Chunk {idx+1}/{total} failed to clean")
            return None"""
MARK = "# --- exp patch v2 ---"
MARK_V1 = "# --- exp patch: dump judge raw output ---"   # 旧版，检测到就升级

ANCHOR = """    with lock:
        pbar.update(1)

    return final_result"""

PATCHED = """    # --- exp patch v2 ---
    _dump_dir = os.environ.get("JUDGE_DUMP_DIR")
    if _dump_dir:
        try:
            os.makedirs(_dump_dir, exist_ok=True)
            with lock:
                if llm_response_str:
                    with open(os.path.join(_dump_dir, "judge_outputs.jsonl"),
                              "a", encoding="utf-8") as _f:
                        _f.write(json.dumps({
                            "id": task_id,
                            "language": language,
                            "judge": os.environ.get("RACE_MODEL", ""),
                            "raw": llm_response_str,
                        }, ensure_ascii=False) + "\\n")
                # 逐题落盘。benchmark 只在整格跑完后才写 raw_results.jsonl，
                # 中途中断则已付费的题全部丢失；exp_resume.py 会把它合并回去。
                with open(os.path.join(_dump_dir, "progress.jsonl"),
                          "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(final_result, ensure_ascii=False) + "\\n")
        except Exception as _e:
            logger.warning(f"ID {task_id}: exp patch write failed - {_e}")
    # --- end exp patch ---

    with lock:
        pbar.update(1)

    return final_result"""


def patch_cleaner():
    src = open(CLEAN_TARGET, encoding="utf-8").read()
    if CLEAN_MARK in src:
        print("清洗器补丁已是最新。")
        return
    if CLEAN_ANCHOR not in src:
        sys.exit("清洗器里找不到插入位置——上游代码可能变了，请检查 _clean_one_chunk。")
    if not os.path.exists(CLEAN_BACKUP):
        shutil.copy2(CLEAN_TARGET, CLEAN_BACKUP)
    open(CLEAN_TARGET, "w", encoding="utf-8").write(src.replace(CLEAN_ANCHOR, CLEAN_PATCHED, 1))
    print(f"清洗器补丁已应用到 {CLEAN_TARGET}")
    print("  （修复：只含参考文献的分块返回空字符串时，不再把整篇报告判为失败）")


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
        if os.path.exists(CLEAN_BACKUP):
            shutil.copy2(CLEAN_BACKUP, CLEAN_TARGET)
            os.remove(CLEAN_BACKUP)
            print(f"已还原 {CLEAN_TARGET}")
        return

    src = open(TARGET, encoding="utf-8").read()
    if MARK in src:
        print("主补丁已是最新。")
        patch_cleaner()
        return
    if MARK_V1 in src:
        if not os.path.exists(BACKUP):
            sys.exit("检测到旧版补丁但找不到备份，无法升级。请重新克隆 benchmark 后再运行。")
        print("检测到旧版补丁，先还原再升级到 v2（新增逐题落盘，支持中断续跑）")
        src = open(BACKUP, encoding="utf-8").read()
    if ANCHOR not in src:
        sys.exit("找不到插入位置——上游代码可能变了。\n"
                 "请检查 process_single_item 末尾是否仍是 'return final_result'。")

    if not os.path.exists(BACKUP):
        shutil.copy2(TARGET, BACKUP)
        print(f"已备份原文件到 {os.path.basename(BACKUP)}")

    open(TARGET, "w", encoding="utf-8").write(src.replace(ANCHOR, PATCHED, 1))
    print(f"补丁已应用到 {TARGET}")
    patch_cleaner()
    print("\n之后每次运行设置 JUDGE_DUMP_DIR 即可保存原文；exp_run.sh 已自动设置。")
    print("还原：python exp_patch_raw.py --revert")


if __name__ == "__main__":
    main()
