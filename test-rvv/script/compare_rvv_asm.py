#!/usr/bin/env python3
"""
Compare RVV instruction lines and mnemonic histograms between two RISC-V objdump listings.

By default: section 1 (total RVV lines) + mnemonic table (top N by counts in the RVV listing;
positional order is ``std`` asm then ``rvv`` asm; labels default to ``std`` / ``rvv``).

Optional --with-eigen: requires interleaved source lines from objdump -S -l (.full.asm);
counts RVV lines whose nearest source path or symbol is in Eigen headers / Eigen::.

Examples:
  python3 compare_rvv_asm.py bench_std.full.asm bench_rvv.full.asm
  python3 compare_rvv_asm.py a.full.asm b.full.asm --with-eigen
  python3 compare_rvv_asm.py a.asm b.asm --by-symbol --symbol-top 20
  python3 compare_rvv_asm.py a.asm b.asm --diff-only
  # Legacy: --top N is an alias for --symbol-top N
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


RVV_MNEMONIC_PATTERN = re.compile(
    r"\b("
    r"vsetvli?|"
    r"vle\d+\.v|vlse\d+\.v|vluxei\d+\.v|"
    r"vse\d+\.v|vsse\d+\.v|vsuxei\d+\.v|"
    r"vlseg\d+e\d+\.v|vsseg\d+e\d+\.v|"
    r"vlsseg\d+e\d+\.v|vssseg\d+e\d+\.v|"
    r"vfadd|vfsub|vfmul|vfmacc|vfmsub|vfmsac|vfnmsac|vfnmacc|"
    r"vfredosum|vfredmax|vfredmin|vfredusum|"
    r"vmul\.vv|vadd\.vv|vsub\.vv|"
    r"vmv\.v\.|vmv\.s\.|vmv\.v\.x|vmv\.s\.x|"
    r"vmerge|vcompress|vcpop|"
    r"vreinterpret|vfncvt|vfwcvt|"
    r"vrgather|vslideup|vslidedown|"
    r"vmul\.vx|vadd\.vx|vsub\.vx|"
    r"vfmv\.f\.s|vfmv\.s\.f|vfmv\.v\.f|"
    r"vmfeq|vmflt|vmfgt|vmfne|"
    r"vfirst"
    r")\b",
    re.IGNORECASE,
)

ASM_LINE_PATTERN = re.compile(r"^\s*[0-9a-f]+:\s+\S")
SOURCE_PATH_PATTERN = re.compile(r".*[/\\].+\.(h|hpp|cpp|cc):\d+")
SYMBOL_WITH_ADDR_RE = re.compile(r"^\s*[0-9a-f]+\s+<([^>]+)>:")
SYMBOL_DEMANGLED_RE = re.compile(r"^[^0-9\s].*::.*\(.*\)\s*:\s*$")


def _is_eigen_context(source_path: str | None, symbol: str | None) -> bool:
    if source_path and "eigen" in source_path.lower():
        return True
    if symbol and "Eigen::" in symbol:
        return True
    return False


def count_rvv_in_file(path: Path) -> tuple[int, dict[str, int]]:
    total = 0
    by_mnemonic: defaultdict[str, int] = defaultdict(int)
    for line in path.read_text(errors="replace").splitlines():
        if not ASM_LINE_PATTERN.match(line):
            continue
        matches = RVV_MNEMONIC_PATTERN.findall(line)
        if not matches:
            continue
        total += 1
        for m in matches:
            by_mnemonic[m] += 1
    return total, dict(by_mnemonic)


def count_rvv_with_eigen_attribution(path: Path) -> tuple[int, int, dict[str, int], dict[str, int]]:
    total = 0
    eigen_attributed = 0
    by_all: defaultdict[str, int] = defaultdict(int)
    by_eigen: defaultdict[str, int] = defaultdict(int)
    current_source_path: str | None = None
    current_symbol: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        if ASM_LINE_PATTERN.match(line):
            matches = RVV_MNEMONIC_PATTERN.findall(line)
            if matches:
                total += 1
                in_eigen = _is_eigen_context(current_source_path, current_symbol)
                if in_eigen:
                    eigen_attributed += 1
                for m in matches:
                    by_all[m] += 1
                    if in_eigen:
                        by_eigen[m] += 1
            continue
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
    return total, eigen_attributed, dict(by_all), dict(by_eigen)


def count_rvv_by_symbol(path: Path) -> dict[str, int]:
    symbol_re = re.compile(r"^\s*[0-9a-f]+\s+<([^>]+)>:")
    current_symbol: str | None = None
    symbol_counts: defaultdict[str, int] = defaultdict(int)
    for line in path.read_text(errors="replace").splitlines():
        m = symbol_re.match(line)
        if m:
            current_symbol = m.group(1)
            continue
        if (
            current_symbol
            and ASM_LINE_PATTERN.match(line)
            and RVV_MNEMONIC_PATTERN.search(line)
        ):
            symbol_counts[current_symbol] += 1
    return dict(symbol_counts)


def print_mnemonic_table(
    by_std: dict[str, int],
    by_rvv: dict[str, int],
    top_n: int,
    label_std: str,
    label_rvv: str,
) -> None:
    if top_n <= 0:
        return
    print(
        f"Mnemonics (sorted by {label_rvv}, top {top_n}; "
        f"parenthetical = same mnemonic in {label_std})"
    )
    for mnem, cnt_rvv in sorted(by_rvv.items(), key=lambda x: -x[1])[:top_n]:
        cnt_std = by_std.get(mnem, 0)
        print(f"  {mnem:30s}  {cnt_rvv:5d}  ({label_std}: {cnt_std})")
    only_std = set(by_std) - set(by_rvv)
    if only_std:
        extra = sorted(only_std, key=lambda m: -by_std[m])[:5]
        print()
        print(f"(Mnemonics only in {label_std}, up to 5)")
        for mnem in extra:
            print(f"  {mnem:30s}  {label_std}: {by_std[mnem]}")


def print_conclusions(
    total_std: int,
    total_rvv: int,
    eigen_std: int,
    eigen_rvv: int,
    label_std: str,
    label_rvv: str,
) -> None:
    print()
    if total_rvv > total_std:
        print(
            f"Summary (totals): {label_rvv} has more RVV lines "
            f"({total_rvv} vs {total_std}); builds may differ in vectorization or linked code."
        )
    elif total_rvv == total_std:
        print(
            f"Summary (totals): same RVV line count ({total_rvv}); "
            "check same binary/options if a difference was expected."
        )
    else:
        print(
            f"Summary (totals): {label_std} has more RVV lines "
            f"({total_std} vs {total_rvv}); toolchain or optimization may differ."
        )
    if eigen_rvv > eigen_std:
        print(
            f"Summary (Eigen): Eigen-context RVV lines — {label_rvv}: {eigen_rvv}, "
            f"{label_std}: {eigen_std}. Cross-check with source paths in the listing."
        )
    elif eigen_rvv == 0 and eigen_std == 0:
        print(
            "Summary (Eigen): no Eigen-context RVV lines in either file; "
            "use objdump -d -C -S -l if you need this breakdown."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare RVV instruction counts between two objdump disassembly files."
    )
    ap.add_argument(
        "asm_std",
        type=Path,
        metavar="ASM_STD",
        help="Non-RVV (or baseline) disassembly (.asm or .full.asm)",
    )
    ap.add_argument(
        "asm_rvv",
        type=Path,
        metavar="ASM_RVV",
        help="RVV (or comparison) disassembly",
    )
    ap.add_argument(
        "--label-std",
        default="std",
        metavar="TEXT",
        help="Output label for ASM_STD (default: std)",
    )
    ap.add_argument(
        "--label-rvv",
        default="rvv",
        metavar="TEXT",
        help="Output label for ASM_RVV (default: rvv)",
    )
    ap.add_argument(
        "--mnemonic-top",
        type=int,
        default=20,
        metavar="N",
        help="Print top N mnemonics by --label-rvv counts (0 = skip; default: 20)",
    )
    ap.add_argument(
        "--with-eigen",
        action="store_true",
        help="Add Eigen-attributed RVV counts (needs source-interleaved .full.asm)",
    )
    ap.add_argument(
        "--by-symbol",
        action="store_true",
        help="Print per-symbol RVV line counts for both files",
    )
    ap.add_argument(
        "--symbol-top",
        type=int,
        default=0,
        metavar="N",
        help="Print top N symbols by --label-rvv RVV count (0 = off unless --by-symbol)",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--diff-only",
        action="store_true",
        help="List symbols whose RVV line counts differ between files",
    )
    ap.add_argument(
        "--conclusions",
        action="store_true",
        help="Print heuristic summaries (off unless --with-eigen or this flag)",
    )
    ap.add_argument(
        "--no-conclusions",
        action="store_true",
        help="Suppress summaries even with --with-eigen",
    )
    args = ap.parse_args()
    if args.top is not None:
        args.symbol_top = args.top

    for p in (args.asm_std, args.asm_rvv):
        if not p.exists():
            print(f"Error: file not found: {p}")
            return 1

    lb_std, lb_rvv = args.label_std, args.label_rvv

    total_std, by_std = count_rvv_in_file(args.asm_std)
    total_rvv, by_rvv = count_rvv_in_file(args.asm_rvv)

    eigen_std = 0
    eigen_rvv = 0
    by_eigen_rvv: dict[str, int] = {}

    print("=" * 60)
    print("1. RVV instruction lines (disassembly lines containing RVV mnemonics)")
    print("=" * 60)
    print(f"  {lb_std}:  {args.asm_std}  ->  {total_std}")
    print(f"  {lb_rvv}:  {args.asm_rvv}  ->  {total_rvv}")
    print(f"  delta ({lb_rvv} - {lb_std}): {total_rvv - total_std:+d}")
    if total_std > 0:
        pct = (total_rvv - total_std) / total_std * 100
        print(f"  relative change: {pct:+.1f}%")
    print()

    if args.with_eigen:
        _t_std, eigen_std, _, _ = count_rvv_with_eigen_attribution(args.asm_std)
        _t_rvv, eigen_rvv, _, by_eigen_rvv = count_rvv_with_eigen_attribution(
            args.asm_rvv
        )
        print("=" * 60)
        print(
            "2. Eigen-attributed RVV lines "
            "(nearest source path or symbol contains Eigen / Eigen::)"
        )
        print("=" * 60)
        print(f"  {lb_std}:  {eigen_std}  (total RVV lines in file: {_t_std})")
        print(f"  {lb_rvv}:  {eigen_rvv}  (total RVV lines in file: {_t_rvv})")
        print(f"  delta ({lb_rvv} - {lb_std}): {eigen_rvv - eigen_std:+d}")
        if _t_rvv > 0:
            print(
                f"  Eigen share in {lb_rvv}: {eigen_rvv}/{_t_rvv} = "
                f"{eigen_rvv / _t_rvv * 100:.1f}%"
            )
        print()
        if by_eigen_rvv:
            print(f"Mnemonics (Eigen-attributed only, {lb_rvv}, top 12)")
            for mnem, cnt in sorted(by_eigen_rvv.items(), key=lambda x: -x[1])[:12]:
                print(f"  {mnem:25s}  {cnt:5d}")
            print()

    print_mnemonic_table(by_std, by_rvv, args.mnemonic_top, lb_std, lb_rvv)

    if args.by_symbol or args.symbol_top > 0 or args.diff_only:
        sym_std = count_rvv_by_symbol(args.asm_std)
        sym_rvv = count_rvv_by_symbol(args.asm_rvv)
        if args.by_symbol:
            print(f"\nPer-symbol RVV lines ({lb_rvv}, with {lb_std} in parentheses)")
            for s, c in sorted(sym_rvv.items(), key=lambda x: -x[1]):
                c0 = sym_std.get(s, 0)
                print(f"  {c:5d}  ({lb_std}: {c0})  {s[:70]}")
        if args.symbol_top > 0:
            print(f"\nTop {args.symbol_top} symbols by RVV lines in {lb_rvv}")
            for s, c in sorted(sym_rvv.items(), key=lambda x: -x[1])[
                : args.symbol_top
            ]:
                c0 = sym_std.get(s, 0)
                print(f"  {c:5d}  ({c - c0:+d} vs {lb_std})  {s[:60]}")
        if args.diff_only:
            all_syms = set(sym_std) | set(sym_rvv)
            diffs = [
                (s, sym_std.get(s, 0), sym_rvv.get(s, 0))
                for s in all_syms
                if sym_std.get(s, 0) != sym_rvv.get(s, 0)
            ]
            print("\nSymbols with differing RVV line counts (top 30 by |delta|)")
            for s, c_std, c_rvv in sorted(diffs, key=lambda x: -abs(x[2] - x[1]))[:30]:
                print(f"  {c_std:5d} -> {c_rvv:5d}  ({c_rvv - c_std:+d})  {s[:55]}")

    show_conclusions = not args.no_conclusions and (
        args.conclusions or args.with_eigen
    )
    if show_conclusions:
        if args.with_eigen:
            print_conclusions(
                total_std, total_rvv, eigen_std, eigen_rvv, lb_std, lb_rvv
            )
        else:
            print()
            if total_rvv > total_std:
                print(
                    f"Summary: {lb_rvv} has +{total_rvv - total_std} more RVV lines "
                    f"than {lb_std}."
                )
            elif total_rvv < total_std:
                print(
                    f"Summary: {lb_std} has +{total_std - total_rvv} more RVV lines "
                    f"than {lb_rvv}."
                )
            else:
                print("Summary: same RVV line count.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
