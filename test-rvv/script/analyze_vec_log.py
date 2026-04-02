#!/usr/bin/env python3
"""
分析 GCC -fopt-info-vec-missed（及同类）产生的 missed 日志：按原因归类、写汇总与按类拆分文件。

共用位置：test-rvv/script/（test-rvv/2d、test-rvv/common/common 等 Makefile 指向本文件。）

用法:
  python3 analyze_vec_log.py [-o OUT] [-s SPLIT_DIR] [-p PATH_FILTER] [--title TEXT] LOGFILE

说明：
  - 分类为启发式规则，按子串优先级匹配；GCC 版本不同措辞可能略有差异，未命中时归入 Other_*。
  - 若需新类别，在 classify_reason() 中按「更具体的短语优先」追加分支即可。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

# file:line:col: missed: reason  或  file:line: missed: reason（少数工具链）
LOG_LINE_RE = re.compile(
    r"^(.+?):(\d+)(?::\d+)?:\s*missed:\s*(.+)\s*$",
    re.IGNORECASE,
)


def simplify_path(path: str, filter_key: str = "pcl") -> str:
    """路径降噪：保留 filter_key 起至末尾；否则仅文件名。"""
    if filter_key and filter_key in path:
        idx = path.find(filter_key)
        return "..." + path[idx:]
    return os.path.basename(path)


def sanitize_filename(text: str) -> str:
    """分类名 -> 安全文件名。"""
    text = re.sub(r"\(.*?\)", "", text)
    safe_text = re.sub(r"[^\w\d-]", "_", text)
    safe_text = re.sub(r"_+", "_", safe_text).strip("_")
    return safe_text + ".log"


def classify_reason(full_reason: str) -> str:
    """
    将单条 missed 原因归到稳定类别名（用于统计与文件名）。
    匹配一律在 lower 串上进行；返回串保留简短中文说明便于人工扫读。
    """
    s = full_reason.strip()
    sl = s.lower()

    # --- 内存 / 副作用（优先）---
    if "statement clobbers memory" in sl:
        return "Memory_Side-Effects (函数调用或内存副作用)"
    if "clobber" in sl and "memory" in sl:
        return "Memory_Clobber (内存被弄脏)"

    # --- 别名 / 数据依赖 ---
    if "data ref analysis failed" in sl:
        return "Data_Dependency_Aliasing (data ref 分析失败)"
    if "possible aliasing" in sl or "may alias" in sl:
        return "Data_Dependency_May_Alias (可能别名)"
    if "dependence" in sl and (
        "vector" in sl or "vectoriz" in sl or "loop" in sl or "cannot" in sl
    ):
        return "Data_Dependency_Distance (循环依赖/距离)"
    if "unsafe" in sl and ("depend" in sl or "data" in sl):
        return "Data_Dependency_Unsafe (不安全依赖)"

    # --- 控制流：splitting region ---
    if "splitting region" in sl:
        if "dominance boundary" in sl:
            return "Control_Flow_Dominance_Boundary (复杂分支_跳转边界)"
        if "control altering definition" in sl:
            return "Control_Flow_Altering_Def (控制流改变定义)"
        if "dont-vectorize loop" in sl or "do not vectorize" in sl:
            return "Control_Flow_Loop_Marked_Dont_Vectorize (显式不向量化)"
        return "Control_Flow_Region_Splitting (控制流区域拆分)"

    # --- 对齐 / 访问模式 ---
    if "misalign" in sl or ("alignment" in sl and "vector" in sl):
        return "Alignment_Misaligned (对齐/向量化对齐限制)"
    if any(
        x in sl
        for x in (
            "strided",
            "strides",
            "not consecutive",
            "non-consecutive",
            "gap in access",
            "unaligned memory",
        )
    ):
        return "Access_Pattern_Stride_Or_Gap (跨步或不连续访问)"

    # --- volatile ---
    if "volatile" in sl:
        return "Memory_Volatile (volatile 阻碍向量化)"

    # --- not vectorized 族（细分）---
    if "not vectorized" in sl or "couldn't vectorize loop" in sl:
        if "couldn't vectorize loop" in sl and "not vectorized" not in sl:
            return "Generic_Failure_Could_Not_Vectorize (无法向量化循环)"
        if "complicated access pattern" in sl:
            return "Access_Pattern_Complicated (复杂访问模式)"
        if "loop nest containing two or more" in sl:
            return "Loop_Structure_Complex_Nest (多重连续内循环)"
        if "multiple exits" in sl:
            return "Loop_Structure_Multiple_Exits (多出口 break/return)"
        if "number of iterations cannot be computed" in sl or "unknown number of iterations" in sl:
            return "Loop_Structure_Unknown_Iterations (无法确定迭代次数)"
        if "control flow" in sl:
            return "Control_Flow_General (循环内复杂控制流)"
        if "throw an exception" in sl or "may throw" in sl:
            return "Exception_Risk (可能抛异常)"
        if "not profitable" in sl or "not useful" in sl:
            return "Cost_Model_Not_Profitable (代价模型认为不划算)"
        if "reduction" in sl:
            return "Reduction_Not_Vectorized (归约相关未向量化)"
        if "pattern" in sl and "recogniz" in sl:
            return "Pattern_Not_Recognized (模式未识别)"
        if "unsupported" in sl or "not supported" in sl:
            return "Unsupported_Operation_or_Type (不支持的操作或类型)"
        if "no vectype" in sl or "no vector type" in sl:
            return "No_Vector_Type (无向量类型)"
        if "feature not enabled" in sl:
            return "Feature_Not_Enabled (目标特性未开启)"
        return "Other_Not_Vectorized_Misc (其他未向量化)"

    if "couldn't vectorize loop" in sl:
        return "Generic_Failure_Could_Not_Vectorize (无法向量化循环)"

    # --- 散落的其它常见短语 ---
    if "versioning" in sl and "alias" in sl:
        return "Data_Dependency_Versioning (别名版本化)"
    if "peel" in sl and "loop" in sl:
        return "Loop_Peel_Epilogue (剥离/余数相关)"
    if "slp" in sl and "not vectorized" in sl:
        return "SLP_Not_Vectorized (SLP 超字未向量化)"

    if "alias" in sl and ("pointer" in sl or "restrict" in sl or "ref" in sl):
        return "Data_Dependency_Aliasing (指针别名_依赖不清)"

    if "not supported" in sl:
        return "Unsupported_Data_Type (不支持的数据类型或操作)"

    # --- 兜底：取首段前缀避免文件名过长 ---
    head = s.split(":", 1)[0].strip()
    head = re.sub(r"\s+", "_", head)[:48]
    return f"Other_{head}"


def analyze_log(
    input_file: str,
    output_file: str | None = None,
    split_dir: str | None = "logs/vec_logs",
    path_filter: str = "pcl",
    report_title: str = "GCC Auto-Vectorization Missed Report Analysis",
) -> None:
    report_db: dict[str, list[dict]] = defaultdict(list)

    try:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("--"):
                    continue
                match = LOG_LINE_RE.match(line)
                if not match:
                    continue
                filepath, line_num, reason = match.groups()
                short_path = simplify_path(filepath, path_filter)
                location = f"{short_path}:{line_num}"
                category = classify_reason(reason)
                report_db[category].append(
                    {
                        "loc": location,
                        "detail": reason,
                        "full_path": filepath,
                        "line_num": line_num,
                    }
                )
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Read log file: {input_file}")
    print(f"Path simplification filter: '{path_filter}'")

    if split_dir:
        if not os.path.exists(split_dir):
            try:
                os.makedirs(split_dir)
                print(f"Created directory: {split_dir}/")
            except OSError as e:
                print(f"Error creating directory {split_dir}: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Using existing directory: {split_dir}/")

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f" {report_title}")
    lines.append(f" Source: {input_file}")
    lines.append("=" * 80)

    sorted_categories = sorted(report_db.items(), key=lambda x: len(x[1]), reverse=True)

    for category, items in sorted_categories:
        count = len(items)
        lines.append(f"\n[Category]: {category}")
        lines.append(f"   Count: {count} occurrences")
        lines.append(f"   {'Location':<50} | {'Specific Reason Detail (Truncated)'}")
        lines.append("   " + "-" * 100)

        if split_dir:
            safe_filename = sanitize_filename(category)
            split_filepath = os.path.join(split_dir, safe_filename)
            try:
                with open(split_filepath, "w", encoding="utf-8") as split_f:
                    split_f.write(f"Category: {category}\n")
                    split_f.write(f"Total Count: {count}\n")
                    split_f.write("=" * 100 + "\n\n")
                    items_sorted = sorted(items, key=lambda x: x["loc"])
                    for item in items_sorted:
                        split_f.write(f"[{item['loc']}] : missed: {item['detail']}\n")
            except OSError as e:
                print(f"Warning: Could not write to {split_filepath}: {e}")

        seen_locs: set[str] = set()
        printed_count = 0
        for item in items:
            if item["loc"] in seen_locs:
                continue
            seen_locs.add(item["loc"])
            detail_short = (
                (item["detail"][:60] + "...")
                if len(item["detail"]) > 60
                else item["detail"]
            )
            lines.append(f"   {item['loc']:<50} | {detail_short}")
            printed_count += 1
            if printed_count >= 10:
                if split_dir:
                    safe_filename = sanitize_filename(category)
                    lines.append(
                        f"   ... (Check {split_dir}/{safe_filename} for all {count} entries)"
                    )
                else:
                    lines.append(f"   ... (and {count - printed_count} more entries)")
                break

    output_content = "\n".join(lines)

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_content)
            print(f"Summary report saved to: {output_file}")
        except OSError as e:
            print(f"Error writing to file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_content)

    if split_dir:
        print(f"Split logs saved to: {split_dir}/*.log")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze GCC -fopt-info-vec-missed style logs (categorize + split)."
    )
    parser.add_argument("logfile", help="Input log (e.g. filtered_report.log)")
    parser.add_argument(
        "-o",
        "--output",
        default="log/analyzed_vec_report.log",
        help="Summary report output path",
    )
    parser.add_argument(
        "-s",
        "--split-dir",
        default="log/vec_logs",
        help="Directory for per-category .log files (set empty string via shell to disable not supported; use --no-split)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not write per-category files (summary only)",
    )
    parser.add_argument(
        "-p",
        "--path-filter",
        default="pcl",
        help="Substring for path shortening in report (default: pcl)",
    )
    parser.add_argument(
        "--title",
        default="GCC Auto-Vectorization Missed Report Analysis",
        help="Report title line in summary",
    )
    args = parser.parse_args()

    split_dir = None if args.no_split else args.split_dir
    analyze_log(
        args.logfile,
        args.output,
        split_dir,
        args.path_filter,
        report_title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
