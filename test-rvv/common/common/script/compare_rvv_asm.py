#!/usr/bin/env python3
"""
比较两份 RISC-V 反汇编文件中 RVV 指令的数量与分布，用于判断 Eigen RVV 是否生效。

输入：须为 objdump -d -C -S -l 生成的 .full.asm（带源码路径与行号），否则“Eigen 归属”统计无效。

输出：
  1) 全量 RVV 行数：所有反汇编指令行中的 RVV 助记符（含自动向量化 + Eigen Packet）。
  2) Eigen 归属 RVV：仅“当前源码路径或符号名含 Eigen”的 RVV 指令行，用于区分 Eigen 引入的 RVV 与
     编译器自动向量化引入的 RVV。

用法:
  python3 compare_rvv_asm.py asm-eigen-1/bench_common_app.full.asm asm-eigen-rvv/bench_common_app.full.asm
  python3 compare_rvv_asm.py ... --by-symbol
  python3 compare_rvv_asm.py ... --top 20
  python3 compare_rvv_asm.py ... --diff-only   # 仅列出有差异的符号

其他方式（不依赖本脚本）:
  grep -cE 'vsetvli|vle[0-9]+\\.v|vfadd|vfmul|vfmacc|vfred' <file>.full.asm
  grep -cE 'vfred|vfmacc|vfmul' <file>.full.asm   # 归约/乘加（Eigen 常用）
"""

import re
import argparse
from pathlib import Path
from collections import defaultdict


# RVV 指令助记符模式（objdump 输出中的指令行）
RVV_MNEMONIC_PATTERN = re.compile(
    r"\b("
    r"vsetvli?|"           # 设置 vl
    r"vle\d+\.v|vlse\d+\.v|vse\d+\.v|"  # load/store
    r"vfadd|vfmul|vfmacc|vfmsub|vfmsac|vfnmsac|vfnmacc|"
    r"vfredosum|vfredmax|vfredmin|vfredusum|"
    r"vmul\.vv|vadd\.vv|vsub\.vv|"
    r"vmv\.v\.|vmv\.s\.|vmv\.v\.x|vmv\.s\.x|"
    r"vmerge|vcompress|vcpop|"
    r"vreinterpret|vfncvt|vfwcvt|"
    r"vrgather|vslideup|vslidedown|"
    r"vmul\.vx|vadd\.vx|vsub\.vx|"
    r"vfmv\.f\.s|vfmv\.s\.f|vfmv\.v\.f"
    r")\b",
    re.IGNORECASE
)

# 仅统计“指令行”：以空白+十六进制地址+冒号+制表符 开头，避免把源码/路径里的 v 算进去
ASM_LINE_PATTERN = re.compile(r"^\s*[0-9a-f]+:\s+\S")

# objdump -S -l 输出的“源码路径”行，如 /path/to/Eigen/.../PacketMath.h:942（可与 strip 后匹配）
SOURCE_PATH_PATTERN = re.compile(r".*[/\\].+\.(h|hpp|cpp|cc):\d+")
# 符号行（带地址）:    13e20 <getMeanStdKernelRVV(...)>:
SYMBOL_WITH_ADDR_RE = re.compile(r"^\s*[0-9a-f]+\s+<([^>]+)>:")
# 符号行（仅 demangled，无地址）: Eigen::internal::predux<...>(...):
SYMBOL_DEMANGLED_RE = re.compile(r"^[^0-9\s].*::.*\(.*\)\s*:\s*$")


def _is_eigen_context(source_path: str | None, symbol: str | None) -> bool:
    """当前反汇编上下文是否来自 Eigen（源码路径或符号名含 Eigen）。"""
    if source_path and "eigen" in source_path.lower():
        return True
    if symbol and "Eigen::" in symbol:
        return True
    return False


def count_rvv_in_file(path: Path):
    """返回 (总 RVV 行数, 每行匹配到的助记符列表, 按助记符分类计数)"""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    total = 0
    by_mnemonic = defaultdict(int)
    rvv_lines = []
    for line in lines:
        if not ASM_LINE_PATTERN.match(line):  # 只统计反汇编指令行
            continue
        matches = RVV_MNEMONIC_PATTERN.findall(line)
        if matches:
            total += 1
            rvv_lines.append(line.strip())
            for m in matches:
                by_mnemonic[m] += 1
    return total, rvv_lines, dict(by_mnemonic)


def count_rvv_with_eigen_attribution(path: Path):
    """
    统计 RVV 指令行数，并区分「全量」与「归属到 Eigen 上下文的」RVV。
    依赖 objdump -d -S -l 的 .full.asm：通过紧邻的源码路径行或符号行判断当前指令是否在 Eigen 内。
    返回 (total_rvv_lines, eigen_attributed_rvv_lines, by_mnemonic_all, by_mnemonic_eigen)
    """
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    total = 0
    eigen_attributed = 0
    by_mnemonic_all = defaultdict(int)
    by_mnemonic_eigen = defaultdict(int)
    current_source_path: str | None = None
    current_symbol: str | None = None
    for line in lines:
        # 先区分指令行：指令行不参与“上下文”更新
        is_insn = ASM_LINE_PATTERN.match(line)
        if is_insn:
            # 指令行：统计 RVV，并按当前上下文判断是否归属 Eigen
            matches = RVV_MNEMONIC_PATTERN.findall(line)
            if matches:
                total += 1
                in_eigen = _is_eigen_context(current_source_path, current_symbol)
                if in_eigen:
                    eigen_attributed += 1
                for m in matches:
                    by_mnemonic_all[m] += 1
                    if in_eigen:
                        by_mnemonic_eigen[m] += 1
            continue
        # 非指令行：更新源码路径或符号上下文
        stripped = line.strip()
        if SOURCE_PATH_PATTERN.search(stripped):
            current_source_path = stripped
            continue
        m = SYMBOL_WITH_ADDR_RE.match(line)
        if m:
            current_symbol = m.group(1)
            continue
        if SYMBOL_DEMANGLED_RE.match(stripped) and "::" in line:
            current_symbol = stripped.rstrip(":").strip()
    return total, eigen_attributed, dict(by_mnemonic_all), dict(by_mnemonic_eigen)


def count_rvv_by_symbol(path: Path):
    """按符号（函数）统计 RVV 指令数。依赖 objdump -d -C 的符号行格式：<addr> <symbol>:"""
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    # 符号行类似: 0000000000012b40 <getMeanStdKernelRVV(...)>:
    symbol_re = re.compile(r"^\s*[0-9a-f]+\s+<([^>]+)>:")
    current_symbol = None
    symbol_counts = defaultdict(int)
    for line in lines:
        m = symbol_re.match(line)
        if m:
            current_symbol = m.group(1)
            continue
        if current_symbol and ASM_LINE_PATTERN.match(line) and RVV_MNEMONIC_PATTERN.search(line):
            symbol_counts[current_symbol] += 1
    return dict(symbol_counts)


def main():
    ap = argparse.ArgumentParser(description="Compare RVV instruction counts between two disassembly files.")
    ap.add_argument("file_no_rvv", type=Path, help="Assembly without Eigen RVV (e.g. asm-eigen-1/bench_common_app.full.asm)")
    ap.add_argument("file_with_rvv", type=Path, help="Assembly with Eigen RVV (e.g. asm-eigen-rvv/bench_common_app.full.asm)")
    ap.add_argument("--by-symbol", action="store_true", help="Print per-symbol RVV counts for both files")
    ap.add_argument("--top", type=int, default=0, help="Print top N symbols by RVV count (0 = off)")
    ap.add_argument("--diff-only", action="store_true", help="Print symbols that appear only in one file or have different counts")
    args = ap.parse_args()

    for p in (args.file_no_rvv, args.file_with_rvv):
        if not p.exists():
            print(f"Error: file not found: {p}")
            return 1

    total1, lines1, by_mnem1 = count_rvv_in_file(args.file_no_rvv)
    total2, lines2, by_mnem2 = count_rvv_in_file(args.file_with_rvv)

    # Eigen 归属统计（依赖 .full.asm 中的源码路径/符号行）
    t1, e1, _, _ = count_rvv_with_eigen_attribution(args.file_no_rvv)
    t2, e2, _, by_eigen2 = count_rvv_with_eigen_attribution(args.file_with_rvv)

    print("=" * 60)
    print("1. RVV 指令行数（全量，仅反汇编指令行中的 RVV 助记符）")
    print("=" * 60)
    print(f"  无 Eigen RVV:  {args.file_no_rvv}  ->  {total1} 行")
    print(f"  有 Eigen RVV:  {args.file_with_rvv}  ->  {total2} 行")
    print(f"  差值:          {total2 - total1:+d} 行")
    if total1 > 0:
        pct = (total2 - total1) / total1 * 100
        print(f"  相对变化:      {pct:+.1f}%")
    print()

    print("=" * 60)
    print("2. Eigen 归属 RVV（仅统计“当前源码路径或符号在 Eigen 内”的 RVV 指令行）")
    print("=" * 60)
    print(f"  无 Eigen RVV:  {args.file_no_rvv}  ->  {e1} 行  (全量 {t1})")
    print(f"  有 Eigen RVV:  {args.file_with_rvv}  ->  {e2} 行  (全量 {t2})")
    print(f"  差值:          {e2 - e1:+d} 行")
    if t2 > 0:
        print(f"  有 RVV 时 Eigen 占比:  {e2}/{t2} = {e2/t2*100:.1f}%")
    print()

    print("按助记符分类（有 Eigen RVV 的样本，全量）:")
    for mnem, cnt in sorted(by_mnem2.items(), key=lambda x: -x[1])[:15]:
        c1 = by_mnem1.get(mnem, 0)
        print(f"  {mnem:25s}  {cnt:5d}  (无 RVV 时: {c1})")
    if by_eigen2:
        print("\n按助记符分类（仅 Eigen 归属部分，有 Eigen RVV 的样本）:")
        for mnem, cnt in sorted(by_eigen2.items(), key=lambda x: -x[1])[:12]:
            print(f"  {mnem:25s}  {cnt:5d}")
    print()

    if args.by_symbol or args.top > 0 or args.diff_only:
        sym1 = count_rvv_by_symbol(args.file_no_rvv)
        sym2 = count_rvv_by_symbol(args.file_with_rvv)
        if args.by_symbol:
            print("按符号统计（有 Eigen RVV）:")
            for s, c in sorted(sym2.items(), key=lambda x: -x[1]):
                c1 = sym1.get(s, 0)
                print(f"  {c:5d}  {s[:70]}")
        if args.top > 0:
            print(f"\nTop {args.top} 符号（按 Eigen RVV 版本中的 RVV 行数）:")
            for s, c in sorted(sym2.items(), key=lambda x: -x[1])[: args.top]:
                c1 = sym1.get(s, 0)
                delta = c - c1
                print(f"  {c:5d}  ({delta:+d})  {s[:60]}")
        if args.diff_only:
            all_syms = set(sym1) | set(sym2)
            diffs = [(s, sym1.get(s, 0), sym2.get(s, 0)) for s in all_syms
                     if sym1.get(s, 0) != sym2.get(s, 0)]
            print("\n仅在有/无 Eigen RVV 间有差异的符号:")
            for s, c1, c2 in sorted(diffs, key=lambda x: -abs(x[2] - x[1]))[:30]:
                print(f"  {c1:5d} -> {c2:5d}  ({c2 - c1:+d})  {s[:55]}")

    print()
    if total2 > total1:
        print("结论（全量）: 有 Eigen RVV 的版本中 RVV 指令行更多，Eigen RVV 很可能已参与生成代码。")
    elif total2 == total1:
        print("结论（全量）: 两份文件中 RVV 行数相同，Eigen RVV 可能未在该 bench 路径上生效，或差异在库内。")
    else:
        print("结论（全量）: 无 Eigen RVV 的版本 RVV 行更多（可能因 -march 不同导致其他自动向量化差异）。")
    if e2 > e1:
        print(f"结论（Eigen 归属）: 有 Eigen RVV 时，归属到 Eigen 源码/符号的 RVV 行数为 {e2}（无时 {e1}），可认为 Eigen RVV 在该 bench 中生效。")
    elif e2 == 0 and e1 == 0:
        print("结论（Eigen 归属）: 两份文件中均无“Eigen 上下文”内的 RVV 行（或 .full.asm 无足够源码行信息）。")
    return 0


if __name__ == "__main__":
    exit(main())
