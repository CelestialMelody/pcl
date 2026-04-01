"""Usage:
    usage: analyze_vec_log.py [-h] [-o OUTPUT] [-s SPLIT_DIR] [-p PATH_FILTER] logfile

    Analyze GCC Auto-Vectorization Missed Report

    positional arguments:
      logfile               The input log file (e.g., clean_report.log)

    options:
      -h, --help            show this help message and exit
      -o, --output OUTPUT   Summary report output file path (optional)
      -s, --split-dir SPLIT_DIR
                            Directory to save split log files (default: vec_missed_log)
      -p, --path-filter PATH_FILTER
                            Substring to filter/simplify paths in the report (default: 'pcl')
"""

import re
import sys
import os
import argparse
import shutil
from collections import defaultdict

def simplify_path(path, filter_key="pcl"):
    """
    对编译器输出的绝对路径进行“视觉降噪”，只保留关键部分。

    Args:
        path (str): 原始文件的绝对路径 (例如: /home/user/workspace/pcl/2d/include/edge.h)。
        filter_key (str): 用于定位截取的关键词子串 (默认: "pcl")。

    Returns:
        str: 简化后的路径。
             - 如果路径包含 filter_key，返回 "...filter_key/..."
             - 否则，仅返回文件名 (os.path.basename)。
    """
    if filter_key and filter_key in path:
        # 使用 find 找到起始位置，保留 filter_key 本身，使路径更清晰
        # 例如: path="/home/user/pcl/2d/include/edge.h", filter="pcl"
        # 结果: "...pcl/2d/include/edge.h"
        idx = path.find(filter_key)
        return "..." + path[idx:]

    # 如果没找到关键字，回退到只显示文件名
    return os.path.basename(path)

def sanitize_filename(text):
    """
    将包含特殊字符的分类描述转换为合法的文件名。

    Args:
        text (str): 原始分类描述 (例如: "Control Flow: Dominance Boundary").

    Returns:
        str: 安全的文件名 (例如: "Control_Flow_Dominance_Boundary.log")。
             非字母数字字符会被替换为下划线，且去除连续下划线。
    """
    # 移除括号及其内容
    text = re.sub(r'\(.*?\)', '', text)
    # 将非字母数字字符替换为下划线
    safe_text = re.sub(r'[^\w\d-]', '_', text)
    # 去掉连续的下划线
    safe_text = re.sub(r'_+', '_', safe_text).strip('_')
    return safe_text + ".log"

def classify_reason(full_reason):
    """
    基于启发式规则，对向量化失败原因进行归类。

    Args:
        full_reason (str): 原始日志中的 missed 原因详情。

    Returns:
        str: 归一化后的类别名称。
             涵盖: 内存副作用、复杂控制流、循环结构嵌套、数据依赖等 6 大类。
    """
def classify_reason(full_reason):
    """
    核心逻辑：提取错误原因的共性部分
    """
    full_reason = full_reason.strip()

    # --- 1. 内存与函数副作用 (优先级最高) ---
    if "statement clobbers memory" in full_reason:
        return "Memory_Side-Effects (函数调用或内存副作用)"

    # --- 2. 复杂的控制流 (Splitting Region) ---
    if "splitting region" in full_reason:
        if "dominance boundary" in full_reason:
            return "Control_Flow_Dominance_Boundary (复杂分支_跳转边界)"
        if "control altering definition" in full_reason:
            return "Control_Flow_Altering_Def (异常抛出风险_如new_delete)"
        if "dont-vectorize loop" in full_reason:
            return "Control_Flow_Loop_Marked_Dont_Vectorize (编译器放弃)"
        return "Control_Flow_Region_Splitting (通用控制流拆分)"

    # --- 3. 针对 "not vectorized" 的深度细分 ---
    # 之前这些被散落在各处或归为 Other，现在集中处理
    if "not vectorized" in full_reason:
        # 3.1 循环结构问题
        if "loop nest containing two or more" in full_reason:
            return "Loop_Structure_Complex_Nest (多重连续内循环)"
        if "multiple exits" in full_reason:
            return "Loop_Structure_Multiple_Exits (循环存在多出口_break/return)"
        if "number of iterations cannot be computed" in full_reason:
            return "Loop_Structure_Unknown_Iterations (无法计算循环次数)"

        # 3.2 控制流问题
        if "control flow" in full_reason:
            return "Control_Flow_General (循环内存在复杂分支)"

        # 3.3 异常与代价模型 (Log中常见)
        if "throw an exception" in full_reason:
            return "Exception_Risk (语句可能抛出异常_阻碍向量化)"
        if "not profitable" in full_reason:
            return "Cost_Model_Not_Profitable (向量化收益不足)"

        # 3.4 硬件/编译器支持问题
        if "unsupported" in full_reason:
            return "Unsupported_Operation_or_Type (不支持的操作或数据类型)"
        if "no vectype" in full_reason:
            return "No_Vector_Type (无对应的向量类型)"
        if "feature not enabled" in full_reason:
            return "Feature_Not_Enabled (硬件特性未开启)"

        # 依然无法分类的 not vectorized
        return "Other_Not_Vectorized_Misc (其他未向量化原因)"

    # --- 4. 其他散落的分类 ---
    if "couldn't vectorize loop" in full_reason:
        return "Generic_Failure (无法向量化)"

    if "data ref analysis failed" in full_reason or "alias" in full_reason:
        return "Data_Dependency_Aliasing (指针别名_依赖不清)"

    if "not supported" in full_reason: # 处理不带 not vectorized 前缀的情况
        return "Unsupported_Data_Type (数据类型不支持)"

    # --- 5. 兜底分类 ---
    # 截取前30个字符作为分类名
    return "Other_" + full_reason.split(':')[0][:30].replace(' ', '_')

def analyze_log(input_file, output_file=None, split_dir="logs/vec_logs", path_filter="pcl"):
    """
    主分析逻辑：读取日志 -> 解析 -> 归类 -> 输出。

    Args:
        input_file (str): GCC 生成的原始 log 文件路径。
        output_file (str, optional): 汇总报告的保存路径。如果为 None，则打印到标准输出。
        split_dir (str, optional): 存放拆分日志 (.log) 的目录路径。
                                   如果为 None 或空字符串，则不生成拆分文件。
        path_filter (str, optional): 传递给 simplify_path 的过滤关键词 (默认: "pcl")。

    Side Effects:
        - 创建 split_dir 目录（如果不存在）。
        - 写入 output_file 文件。
        - 在 split_dir 下生成多个分类 .log 文件。
    """
    # 存储结构: { Category: [ (Location, Detail, FullLine), ... ] }
    report_db = defaultdict(list)

    log_pattern = re.compile(r'^(.+?):(\d+):(\d+):\s*missed:\s*(.+)$')

    print(f"Reading log file: {input_file} ...")
    print(f"Path simplification filter: '{path_filter}'")

    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('--'):
                    continue

                match = log_pattern.match(line)
                if match:
                    filepath, line_num, col, reason = match.groups()

                    # 使用传入的 filter 参数进行简化
                    short_path = simplify_path(filepath, path_filter)
                    location = f"{short_path}:{line_num}"

                    category = classify_reason(reason)

                    report_db[category].append({
                        'loc': location,
                        'detail': reason,
                        'full_path': filepath,
                        'line_num': line_num
                    })
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    # === 准备拆分文件夹 ===
    if split_dir:
        if not os.path.exists(split_dir):
            try:
                os.makedirs(split_dir)
                print(f"Created directory: {split_dir}/")
            except OSError as e:
                print(f"Error creating directory {split_dir}: {e}")
                sys.exit(1)
        else:
            print(f"Using existing directory: {split_dir}/")

    # === 生成报告内容 ===
    lines = []
    lines.append("=" * 80)
    lines.append(f" RISC-V Auto-Vectorization Missed Report Analysis")
    lines.append(f" Source: {input_file}")
    lines.append("=" * 80)

    sorted_categories = sorted(report_db.items(), key=lambda x: len(x[1]), reverse=True)

    for category, items in sorted_categories:
        count = len(items)

        # 1. 添加到总报告摘要
        lines.append(f"\n[Category]: {category}")
        lines.append(f"   Count: {count} occurrences")
        lines.append(f"   {'Location':<50} | {'Specific Reason Detail (Truncated)'}")
        lines.append("   " + "-"*100)

        # 2. 写入单独的 Log 文件 (如果指定了 split_dir)
        if split_dir:
            safe_filename = sanitize_filename(category)
            split_filepath = os.path.join(split_dir, safe_filename)

            try:
                with open(split_filepath, 'w', encoding='utf-8') as split_f:
                    split_f.write(f"Category: {category}\n")
                    split_f.write(f"Total Count: {count}\n")
                    split_f.write("=" * 100 + "\n\n")

                    # 按文件名和行号排序
                    items.sort(key=lambda x: x['loc'])

                    for item in items:
                        # 写入: [位置] : missed: [原因]
                        split_f.write(f"[{item['loc']}] : missed: {item['detail']}\n")
            except IOError as e:
                print(f"Warning: Could not write to {split_filepath}: {e}")

        # 3. 继续生成总报告摘要
        seen_locs = set()
        printed_count = 0

        for item in items:
            if item['loc'] in seen_locs:
                continue
            seen_locs.add(item['loc'])

            detail_short = (item['detail'][:60] + '...') if len(item['detail']) > 60 else item['detail']
            lines.append(f"   {item['loc']:<50} | {detail_short}")

            printed_count += 1
            if printed_count >= 10:
                if split_dir:
                    safe_filename = sanitize_filename(category)
                    lines.append(f"   ... (Check {split_dir}/{safe_filename} for all {count} entries)")
                else:
                    lines.append(f"   ... (and {count - printed_count} more entries)")
                break

    output_content = "\n".join(lines)

    # === 输出总报告 ===
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_content)
            print(f"Summary report saved to: {output_file}")
        except IOError as e:
            print(f"Error writing to file: {e}")
    else:
        print(output_content)

    if split_dir:
        print(f"Split logs saved to: {split_dir}/*.log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GCC Auto-Vectorization Missed Report")

    # 必选参数
    parser.add_argument("logfile", help="The input log file (e.g., clean_report.log)")

    # 可选参数
    parser.add_argument("-o", "--output", default="log/analyzed_vec_report.log", help="Summary report output file path (optional)")
    parser.add_argument("-s", "--split-dir", default="log/analyzed_vec_log", help="Directory to save split log files (default: log/analyzed_vec_log)")

    # 路径过滤器参数
    parser.add_argument("-p", "--path-filter",
                        default="pcl",
                        help="Substring to filter/simplify paths in the report (default: 'pcl')")

    args = parser.parse_args()

    analyze_log(args.logfile, args.output, args.split_dir, args.path_filter)
