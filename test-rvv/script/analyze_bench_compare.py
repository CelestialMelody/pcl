#!/usr/bin/env python3
"""
对比两份 benchmark 日志（通常为 run_bench_std.log / run_bench_rvv.log），输出 Std vs RVV 汇总表。

位置：test-rvv/script/（各子目录 Makefile 可指向本文件。）

解析格式：
  - auto（默认）：先试 structured；失败再用 loose。
  - structured：日志中先出现 Image Size，其后在遇第一条计时行前出现 Iterations，再收集计时行直到下一段长 `====` 分隔；表头取 Image Size 上一行非分隔文字（若有）。
  - loose：全文第一个 Iterations、可选 Image Size、收集全文所有计时行。

Dataset 行推断优先级：
  1) --dataset
  2) 日志中 Dataset: / Workload:
  3) 解析到的 Image Size → 网格规模描述
  4) 日志中 Cloud size + Vector size（同一行）
  5) 否则提示在日志中打印 Dataset: 或使用 --dataset

Total Time：每行均为 Avg × Iterations（与日志中的 Iterations 一致）。若某 bench 内部还有额外内层循环，
应在程序中调整打印的「每 iter」含义，或在日志里用 Dataset:/Workload: 说明，本脚本不维护 per-模块倍率表。

依赖：仅标准库。
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

# 同时支持 ms / iter 与 us / iter，统一换算为 ms
ROW_RE = re.compile(
    r"^(.*)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(u|m)s\s*/\s*iter\s*$"
)

DATASET_LINE_RE = re.compile(
    r"^\s*(?:Dataset|Workload)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
CLOUD_VECTOR_RE = re.compile(
    r"Cloud\s+size:\s*(\d+)\s+Vector\s+size:\s*(\d+)", re.IGNORECASE
)

STRUCTURED_SIZE_RE = re.compile(r"^\s*Image Size:\s*(\d+)\s*x\s*(\d+)\s*$")
STRUCTURED_ITERS_RE = re.compile(r"^\s*Iterations:\s*(\d+)\s*$")


def _line_is_equals_separator(line: str) -> bool:
    s = line.strip()
    return len(s) >= 8 and all(c == "=" for c in s)


def print_bar(ch: str, width: int) -> None:
    print(ch * width)


def print_context_kv(key: str, value: str, key_width: int = 10) -> None:
    """
    打印 [Benchmark Context] 的一条键值项，支持多行 value。
    例：
      key="Dataset", value="line1\\nline2"
    输出：
      "  Dataset   : line1"
      "              line2"
    """
    prefix = f"  {key:<{key_width}}: "
    cont = " " * len(prefix)
    lines = value.splitlines() or [""]
    print(prefix + lines[0])
    for s in lines[1:]:
        print(cont + s)


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
    # 板卡上常见：绝对路径下的 bench_* 可执行文件
    if re.search(r"/bench_[A-Za-z0-9_./-]+|/pcl-test/.+bench", joined):
        return ("RISC-V board / device (binary path in log, no -cpu line)", "see SoC / ELF (log 无 vlen)")
    return ("Unknown (see full log)", "n/a")


def _row_value_ms(m: re.Match[str]) -> tuple[str, float]:
    name = m.group(1).strip()
    value = float(m.group(2))
    unit = m.group(3)
    value_ms = (value * 0.001) if unit == "u" else value
    return name, value_ms


def _collect_rows(lines: list[str]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in lines:
        m = ROW_RE.match(line.rstrip("\n"))
        if m:
            rows.append(_row_value_ms(m))
    return rows


def extract_iterations(text: str) -> int | None:
    """
    从日志中提取迭代次数。

    兼容：
      - structured 风格：`Iterations: N`（独占一行）
      - 常见缩写/变体：Iter / Iters / iteration / iterations / iter(s)
    """
    # 独占一行：Iterations: / iter: 等
    m = re.search(
        r"^\s*(?:iters?|iter(?:ation)?s?)\s*:\s*(\d+)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return int(m.group(1)) if m else None


def extract_dataset_line(text: str) -> str | None:
    m = DATASET_LINE_RE.search(text)
    return m.group(1).strip() if m else None


def extract_cloud_vector_sizes(text: str) -> tuple[int, int] | None:
    m = CLOUD_VECTOR_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def resolve_dataset_description(
    dataset_cli: str,
    std_text: str,
    rvv_text: str,
    std_d: dict,
    rvv_d: dict,
) -> str:
    """生成 [Benchmark Context] 中的 Dataset 一行（不含前缀）。"""
    if dataset_cli.strip():
        return dataset_cli.strip()

    ds_std = extract_dataset_line(std_text)
    ds_rvv = extract_dataset_line(rvv_text)
    if ds_std and ds_rvv and ds_std != ds_rvv:
        print(
            "[WARN] Std 与 RVV 日志中 Dataset/Workload 文案不一致，展示以 Std 为准。",
            file=sys.stderr,
        )
    if ds_std:
        return ds_std
    if ds_rvv:
        return ds_rvv

    iw, ih = std_d.get("image_w"), std_d.get("image_h")
    if iw is not None and ih is not None:
        pixels = iw * ih
        return (
            f"synthetic image grid {iw} x {ih} ({pixels} samples / pixels; from Image Size line)"
        )

    cv_std = extract_cloud_vector_sizes(std_text)
    cv_rvv = extract_cloud_vector_sizes(rvv_text)
    if cv_std and cv_rvv and cv_std != cv_rvv:
        print(
            "[WARN] Std 与 RVV 日志中 Cloud size / Vector size 不一致，展示以 Std 为准。",
            file=sys.stderr,
        )
    cv = cv_std or cv_rvv
    if cv:
        c, v = cv
        return f"workloads from log (cloud {c} points, vector {v} elements)"

    return (
        "未解析到 Image Size / Dataset\n"
        "请在 bench 输出中增加一行 `Dataset: ...` 说明数据，或使用 --dataset"
    )


def _infer_title_before(lines: list[str], image_line_idx: int) -> str:
    """取 Image Size 行之前、最近一条非空且非全 = 分隔线的内容作表头。"""
    for k in range(image_line_idx - 1, -1, -1):
        s = lines[k].strip()
        if not s or _line_is_equals_separator(lines[k]):
            continue
        return s
    return "Benchmark"


def parse_structured_header_block(text: str) -> dict:
    """
    仅按版式识别：某行 Image Size，且在其后、首条计时行前出现 Iterations；
    计时行从 Iterations 之后开始收集，遇「已有数据后的长 ==== 分隔」则结束。
    """
    lines = [raw.rstrip("\n") for raw in text.splitlines()]

    for i, line in enumerate(lines):
        m_size = STRUCTURED_SIZE_RE.match(line)
        if not m_size:
            continue
        image_w, image_h = int(m_size.group(1)), int(m_size.group(2))
        base_iters: int | None = None
        scan_from: int | None = None
        for j in range(i + 1, len(lines)):
            lj = lines[j]
            if ROW_RE.match(lj):
                break
            m_it = STRUCTURED_ITERS_RE.match(lj)
            if m_it:
                base_iters = int(m_it.group(1))
                scan_from = j + 1
                break
        if base_iters is None or scan_from is None:
            continue

        rows: list[tuple[str, float]] = []
        for k in range(scan_from, len(lines)):
            lk = lines[k]
            if _line_is_equals_separator(lk) and rows:
                break
            m_row = ROW_RE.match(lk)
            if m_row:
                rows.append(_row_value_ms(m_row))

        if rows:
            title = _infer_title_before(lines, i)
            return {
                "format": "structured",
                "title": title,
                "image_w": image_w,
                "image_h": image_h,
                "base_iterations": base_iters,
                "rows": rows,
            }

    raise ValueError(
        "未找到结构化块：需要按顺序出现 Image Size、Iterations，且之后有一段计时行"
    )


def parse_loose_ms_per_iter(text: str) -> dict:
    """
    宽松解析：全文搜索 Iterations、可选 Image Size、按出现顺序收集计时行。
    不要求结构化标题行。
    """
    base_iters = extract_iterations(text)

    m_size = re.search(r"Image Size:\s*(\d+)\s*x\s*(\d+)", text)
    image_w = int(m_size.group(1)) if m_size else None
    image_h = int(m_size.group(2)) if m_size else None

    rows = _collect_rows(text.splitlines())
    if not rows:
        raise ValueError("未找到任何 '… : x ms / iter' 或 '… : x us / iter' 行")

    title: str | None = None
    for pat in (
        re.compile(r"^\s*(PCL .+?)\s*$", re.MULTILINE),
        re.compile(r"^\s*(.+?Module Benchmark)\s*$", re.MULTILINE),
    ):
        title_m = pat.search(text)
        if title_m:
            title = title_m.group(1).strip()
            break
    if not title:
        title = "Benchmark (loose parse)"

    return {
        "format": "loose",
        "title": title,
        "image_w": image_w,
        "image_h": image_h,
        "base_iterations": base_iters,
        "rows": rows,
    }


def parse_one_log(text: str, fmt: str, iterations_override: int | None) -> dict:
    if fmt == "structured":
        d = parse_structured_header_block(text)
    elif fmt == "loose":
        d = parse_loose_ms_per_iter(text)
    else:
        try:
            d = parse_structured_header_block(text)
        except ValueError:
            d = parse_loose_ms_per_iter(text)

    if iterations_override is not None:
        d = dict(d)
        d["base_iterations"] = iterations_override
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description="对比 Std / RVV 两份 benchmark 日志（ms/iter 行 + Iterations）并打印表格"
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
        choices=("auto", "structured", "loose"),
        default="auto",
        help="auto：先试 structured，失败再用 loose。",
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
        help="覆盖 Dataset 行（否则按日志中的 Dataset:/Workload:/Image Size/Cloud+Vector 推断）",
    )
    args = ap.parse_args()

    std_text = args.std_log.read_text(encoding="utf-8", errors="replace")
    rvv_text = args.rvv_log.read_text(encoding="utf-8", errors="replace")

    std_d = parse_one_log(std_text, args.format, args.iterations)
    rvv_d = parse_one_log(rvv_text, args.format, args.iterations)

    if (
        std_d.get("base_iterations") is not None
        and rvv_d.get("base_iterations") is not None
        and std_d["base_iterations"] != rvv_d["base_iterations"]
    ):
        print(
            f"[WARN] Std Iterations={std_d['base_iterations']} 与 RVV Iterations={rvv_d['base_iterations']} 不一致，仍按名称对齐",
            file=sys.stderr,
        )
    iw, ih = std_d.get("image_w"), std_d.get("image_h")
    if (iw, ih) != (rvv_d.get("image_w"), rvv_d.get("image_h")) and iw is not None and ih is not None:
        print(
            "[WARN] 两侧 Image Size 不一致，上下文以 Std 为准",
            file=sys.stderr,
        )

    base_iters: int | None = std_d.get("base_iterations")
    if base_iters is None:
        base_iters = rvv_d.get("base_iterations")
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

    ds = resolve_dataset_description(
        args.dataset, std_text, rvv_text, std_d, rvv_d
    )

    print_bar(BAR_CHAR, total_width)
    print(f" {heading}")
    print_bar(BAR_CHAR, total_width)
    print("[Benchmark Context]")
    print_context_kv("Device", device)
    print_context_kv("VLEN", vlen_desc)
    print_context_kv("Dataset", ds)

    if base_iters is None:
        print_context_kv(
            "Iterations",
            "未解析到 Iter/Iterations\n"
            "Total Time 不计算（n/a）\n"
            "建议在 bench 输出中增加 Iterations: N/iter: N，或用 --iterations 覆盖",
        )
    else:
        print_context_kv("Iterations", f"{base_iters} （每行 Total = Avg × {base_iters}）")
    print_context_kv("Std log", str(args.std_log))
    print_context_kv("RVV log", str(args.rvv_log))
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
        std_tot = (std_avg * base_iters) if base_iters is not None else None
        rvv_avg = rvv_map.get(name)
        if rvv_avg is None:
            print(f"# 跳过（RVV 日志无此项）: {name}", file=sys.stderr)
            continue
        rvv_tot = (rvv_avg * base_iters) if base_iters is not None else None
        speedup_rvv = (std_avg / rvv_avg) if rvv_avg > 0 else 0.0

        def row(item: str, impl: str, avg: float, tot: float | None, sp: float) -> None:
            avg_s = f"{avg:.4f} ms"
            tot_s = f"{tot:.4f} ms" if tot is not None else "n/a"
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
