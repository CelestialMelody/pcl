#!/usr/bin/env python3
"""
对比两份 benchmark 日志（通常为 run_bench_std.log / run_bench_rvv.log），输出 Std vs RVV 汇总表。

解析格式：
  - auto（默认）：先按 PCL 2D 块解析；失败则退回 generic。
  - pcl_2d：要求「PCL 2D Module Benchmark」+ Image Size + Iterations + ms/iter 行。
  - generic：不强制标题；从全文提取 Iterations（及可选 Image Size）与所有「… : x ms / iter」行，
    适合板卡直跑、tee 路径不同、或将来类似格式的日志。行倍率默认与 pcl_2d 相同（见 --multiplier-rules）。

不支持：已排版成「Benchmark Item | Impl | …」表格、且无「… ms / iter」行的输出（例如部分 sac 程序只打印表格），
需另写解析或让程序同时打印一行一行的 ms/iter 结果。

依赖：仅标准库。推荐：test-rvv/2d/.venv/bin/python
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BAR_CHAR = "="
SEP_CHAR = "-"
W_IMPL = 6
W_AVG = 14
W_TOT = 14
W_SPD = 14

# Capture both `ms / iter` and `us / iter`, and normalize to ms in parser.
ROW_RE = re.compile(
    r"^(.*)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(u|m)s\s*/\s*iter\s*$"
)


def print_bar(ch: str, width: int) -> None:
    print(ch * width)


def parse_multiplier_rules(spec: str | None) -> list[tuple[str, int]]:
    """
    规则：'prefix:factor'，同一前缀行的循环次数 = base_iters * factor。
    例：'Set Operation:5' 表示名称以 'Set Operation:' 开头的行用 base_iters*5。
    多条用逗号分隔：'Set Operation:5,Foo:2'
    传 none / off 或空字符串：所有行均只用 base_iters（无额外倍率）。
    """
    if spec is None:
        return [("Set Operation:", 5)]
    s = spec.strip()
    if not s or s.lower() in ("none", "off"):
        return []
    rules: list[tuple[str, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"无效的 --multiplier-rules 片段（需要 prefix:factor）: {part!r}")
        pref, fac = part.rsplit(":", 1)
        pref = pref.strip()
        rules.append((pref, int(fac)))
    return rules if rules else [("Set Operation:", 5)]


def loop_multiplier_for_name(
    name: str, base_iters: int, rules: list[tuple[str, int]]
) -> int:
    if not rules:
        return base_iters
    for prefix, factor in rules:
        if name.startswith(prefix):
            return base_iters * factor
    return base_iters


def parse_run_command(text: str) -> tuple[str, str]:
    """从日志中推断 Device / VLEN 描述。"""
    joined = text
    if "qemu-riscv64" in joined:
        vm = re.search(r"vlen=(\d+)", joined)
        el = re.search(r"elen=(\d+)", joined)
        vlen = vm.group(1) if vm else "?"
        elen = el.group(1) if el else "?"
        return (
            "RISC-V RVV target (qemu, rv64gcv)",
            f"{vlen}-bit (vlen={vlen}, elen={elen})",
        )
    if re.search(r"\./build/.+bench", joined) and "qemu" not in joined:
        return ("Native / host (no qemu in log)", "n/a (host SIMD)")
    if re.search(r"/bench_2d_app|/pcl-test/.+bench", joined):
        return ("RISC-V board / device (binary path in log, no -cpu line)", "see SoC / ELF (log 无 vlen)")
    return ("Unknown (see full log)", "n/a")


def _collect_rows(lines: list[str]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in lines:
        m = ROW_RE.match(line.rstrip("\n"))
        if m:
            name = m.group(1).strip()
            value = float(m.group(2))
            unit = m.group(3)  # "u" or "m"
            value_ms = (value * 0.001) if unit == "u" else value
            rows.append((name, value_ms))
    return rows


def parse_pcl_2d_block(text: str) -> dict:
    """严格 PCL 2D：标题 + Image Size + Iterations + 首段 ms/iter 块。"""
    lines = text.splitlines()
    title = "PCL 2D Module Benchmark"
    image_w = image_h = None
    base_iters = None
    in_block = False
    rows: list[tuple[str, float]] = []

    title_re = re.compile(r"^\s*PCL 2D Module Benchmark\s*$")
    size_re = re.compile(r"^\s*Image Size:\s*(\d+)\s*x\s*(\d+)\s*$")
    iters_re = re.compile(r"^\s*Iterations:\s*(\d+)\s*$")

    for raw in lines:
        line = raw.rstrip("\n")
        if not in_block:
            if title_re.match(line):
                in_block = True
            continue

        if line.strip().startswith("=" * 20):
            if rows:
                break
            continue

        m_size = size_re.match(line)
        if m_size:
            image_w, image_h = int(m_size.group(1)), int(m_size.group(2))
            continue
        m_it = iters_re.match(line)
        if m_it:
            base_iters = int(m_it.group(1))
            continue

        m_row = ROW_RE.match(line)
        if m_row:
            name = m_row.group(1).strip()
            value = float(m_row.group(2))
            unit = m_row.group(3)  # "u" or "m"
            value_ms = (value * 0.001) if unit == "u" else value
            rows.append((name, value_ms))

    if base_iters is None or image_w is None:
        raise ValueError("非完整 PCL 2D 块（缺少 Image Size 或 Iterations）。")
    if not rows:
        raise ValueError("未找到任何 'ms / iter' 或 'us / iter' 结果行。")

    return {
        "format": "pcl_2d",
        "title": title,
        "image_w": image_w,
        "image_h": image_h,
        "base_iterations": base_iters,
        "rows": rows,
    }


def parse_generic_ms_per_iter(text: str) -> dict:
    """
    宽松解析：全文搜索 Iterations、可选 Image Size、按出现顺序收集 ms/iter 行。
    不要求「PCL 2D Module Benchmark」标题行。
    """
    m_it = re.search(r"Iterations:\s*(\d+)", text)
    if not m_it:
        raise ValueError("generic 模式需要日志中出现 'Iterations: N'，或用 --iterations 指定。")
    base_iters = int(m_it.group(1))

    m_size = re.search(r"Image Size:\s*(\d+)\s*x\s*(\d+)", text)
    image_w = int(m_size.group(1)) if m_size else None
    image_h = int(m_size.group(2)) if m_size else None

    rows = _collect_rows(text.splitlines())
    if not rows:
        raise ValueError("未找到任何 '… : x ms / iter' 或 '… : x us / iter' 行。")

    title_m = re.search(r"^\s*(PCL .+?)\s*$", text, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else "Benchmark (generic)"

    return {
        "format": "generic",
        "title": title,
        "image_w": image_w,
        "image_h": image_h,
        "base_iterations": base_iters,
        "rows": rows,
    }


def parse_one_log(text: str, fmt: str, iterations_override: int | None) -> dict:
    if fmt == "pcl_2d":
        d = parse_pcl_2d_block(text)
    elif fmt == "generic":
        d = parse_generic_ms_per_iter(text)
    else:
        try:
            d = parse_pcl_2d_block(text)
        except ValueError:
            d = parse_generic_ms_per_iter(text)

    if iterations_override is not None:
        d = dict(d)
        d["base_iterations"] = iterations_override

    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description="对比 Std / RVV 两份 benchmark 日志（ms/iter 行 + Iterations）并打印表格。"
    )
    ap.add_argument(
        "--std-log",
        type=Path,
        default=Path("output/run_bench_std.log"),
        help="Std 侧日志",
    )
    ap.add_argument(
        "--rvv-log",
        type=Path,
        default=Path("output/run_bench_rvv.log"),
        help="RVV 侧日志",
    )
    ap.add_argument(
        "--format",
        choices=("auto", "pcl_2d", "generic"),
        default="auto",
        help="auto：先试 PCL 2D 块，失败再用 generic；pcl_2d / generic 固定模式",
    )
    ap.add_argument(
        "--heading",
        default="",
        help="表格大标题；默认按解析到的 title 自动生成",
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="覆盖两侧日志中的 Iterations（仅当日志缺省或需强制统一时使用）",
    )
    ap.add_argument(
        "--multiplier-rules",
        default="Set Operation:5",
        help="逗号分隔 prefix:factor；名称以 prefix 开头的项 Total=avg*(base_iters*factor)。"
        " 传 none 关闭倍率规则（各行仅 base_iters）。",
    )
    ap.add_argument(
        "--device",
        default="",
        help="覆盖 [Benchmark Context] 中的 Device 行（默认从日志推断）",
    )
    ap.add_argument(
        "--vlen-desc",
        default="",
        help="覆盖 VLEN 行",
    )
    ap.add_argument(
        "--dataset",
        default="",
        help="覆盖 Dataset 行（默认：2D 用 Image Size；否则 generic 可手写说明）",
    )
    args = ap.parse_args()

    std_text = args.std_log.read_text(encoding="utf-8", errors="replace")
    rvv_text = args.rvv_log.read_text(encoding="utf-8", errors="replace")

    rules = parse_multiplier_rules(args.multiplier_rules)

    std_d = parse_one_log(std_text, args.format, args.iterations)
    rvv_d = parse_one_log(rvv_text, args.format, args.iterations)

    if std_d["base_iterations"] != rvv_d["base_iterations"]:
        print(
            f"[WARN] Std Iterations={std_d['base_iterations']} 与 RVV Iterations={rvv_d['base_iterations']} 不一致，仍按名称对齐。",
            file=sys.stderr,
        )
    iw, ih = std_d.get("image_w"), std_d.get("image_h")
    if (iw, ih) != (rvv_d.get("image_w"), rvv_d.get("image_h")) and iw is not None and ih is not None:
        print(
            "[WARN] 两侧 Image Size 不一致，上下文以 Std 为准。",
            file=sys.stderr,
        )

    base_iters = std_d["base_iterations"]
    rvv_map = dict(rvv_d["rows"])

    device, vlen_desc = parse_run_command(std_text + "\n" + rvv_text)
    if args.device.strip():
        device = args.device.strip()
    if args.vlen_desc.strip():
        vlen_desc = args.vlen_desc.strip()

    heading = args.heading.strip()
    if not heading:
        heading = f"{std_d['title']} (Std vs RVV)"

    names_ordered = [n for n, _ in std_d["rows"]]
    w_item = max(24, max(len(n) for n in names_ordered) + 1)
    total_width = w_item + W_IMPL + W_AVG + W_TOT + W_SPD + (4 * 3) + 1

    print_bar(BAR_CHAR, total_width)
    print(f" {heading}")
    print_bar(BAR_CHAR, total_width)
    print("[Benchmark Context]")
    print(f"  Device     : {device}")
    print(f"  VLEN       : {vlen_desc}")

    if args.dataset.strip():
        ds = args.dataset.strip()
    elif iw is not None and ih is not None:
        pixels = iw * ih
        ds = f"synthetic 2D image {iw} x {ih} ({pixels} points / pixels)"
    else:
        ds = "(未在日志中解析 Image Size；可用 --dataset 说明数据来源)"
    print(f"  Dataset    : {ds}")

    # 规则里可写 "Set Operation:5"，解析后前缀常为 "Set Operation"（无冒号），与 bench 名 "Set Operation: …" 仍用 startswith 匹配
    so_rule = next((f for f in rules if f[0].startswith("Set Operation")), None)
    if so_rule:
        setop_note = f"（Set Operation 类项：loop=base_iters×{so_rule[1]}）"
    elif rules:
        setop_note = "（按 --multiplier-rules 前缀匹配行放大 loop）"
    else:
        setop_note = "（各行 loop 均为 base_iters）"

    print(f"  Iterations : {base_iters} {setop_note}")
    print(f"  Std log    : {args.std_log}")
    print(f"  RVV log    : {args.rvv_log}")
    # print(f"  Parse mode : {std_d['format']} / {rvv_d['format']}")
    print()

    print_bar(SEP_CHAR, total_width)
    hdr = (
        f"{'Benchmark Item':<{w_item}} | "
        f"{'Impl':<{W_IMPL}} | "
        f"{'Avg Time':>{W_AVG}} | "
        f"{'Total Time':>{W_TOT}} | "
        f"{'Speedup':>{W_SPD}}"
    )
    print(hdr)
    sep = (
        "-" * (w_item + 1)
        + "|"
        + "-" * (W_IMPL + 2)
        + "|"
        + "-" * (W_AVG + 2)
        + "|"
        + "-" * (W_TOT + 2)
        + "|"
        + "-" * (W_SPD + 2)
    )
    print(sep)

    for name, std_avg in std_d["rows"]:
        mult = loop_multiplier_for_name(name, base_iters, rules)
        std_tot = std_avg * mult
        rvv_avg = rvv_map.get(name)
        if rvv_avg is None:
            print(f"# 跳过（RVV 日志无此项）: {name}", file=sys.stderr)
            continue
        rvv_tot = rvv_avg * mult
        speedup_rvv = (std_tot / rvv_tot) if rvv_tot > 0 else 0.0

        def row(item: str, impl: str, avg: float, tot: float, sp: float) -> None:
            avg_s = f"{avg:.4f} ms"
            tot_s = f"{tot:.4f} ms"
            spd_s = f"[ {sp:6.2f}x ]"
            print(
                f"{item:<{w_item}} | "
                f"{impl:<{W_IMPL}} | "
                f"{avg_s:>{W_AVG}} | "
                f"{tot_s:>{W_TOT}} | "
                f"{spd_s:>{W_SPD}}"
            )

        row(name, "Std", std_avg, std_tot, 1.00)
        row("", "RVV", rvv_avg, rvv_tot, speedup_rvv)
        print(sep)

    print_bar(BAR_CHAR, total_width)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as e:
        print(f"错误: {e}", file=sys.stderr)
        raise SystemExit(1)
