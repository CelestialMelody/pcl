#!/usr/bin/env python3
"""
Remez for atan(t) odd polynomial on [0, 1].

Target form (matching common.hpp):
  atan(t) ~= t * (a1 + a3*t^2 + a5*t^4 + ... + a(2m+1)*t^(2m))

So we solve minimax for:
  f(t) = atan(t),  P(t) = t * sum_{k=0..m} c_k * t^(2k)

Usage:
  uv venv .venv && uv pip install numpy
  .venv/bin/python parm_remez_atan.py --max-odd 11 --iterations 40
"""

import argparse
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _eval_poly_odd(ts, coeffs):
    """
    Evaluate t * (c0 + c1*t^2 + c2*t^4 + ...).
    coeffs[k] corresponds to t^(2k+1) coefficient.
    """
    t2 = ts * ts
    q = np.zeros_like(ts)
    for c in reversed(coeffs):
        q = q * t2 + c
    return ts * q


def remez_atan_odd_numpy(m, a=1e-8, b=1.0, max_iter=40, n_fine=20000):
    """
    Minimax odd polynomial for atan(t) on [a,b] using Remez exchange.

    m: number of q terms - 1
       max odd power = 2*m+1
    Returns (coeffs, max_abs_err)
      coeffs size = m+1, mapping:
        atan(t) ~= c0*t + c1*t^3 + ... + cm*t^(2m+1)
    """
    n = m
    n_ref = n + 2

    # Chebyshev initial nodes in [a, b]
    k = np.arange(1, n_ref + 1)
    x_cheb = np.cos((2 * k - 1) * np.pi / (2 * n_ref))
    ref = 0.5 * (a + b) + 0.5 * (b - a) * x_cheb
    ref = np.sort(ref)

    max_abs_err = None
    coeffs = np.zeros(n + 1, dtype=float)

    for _ in range(max_iter):
        # Solve at ref points:
        #   t_i * sum_{j=0..n} c_j * t_i^(2j) + (-1)^i E = atan(t_i)
        cols = []
        for j in range(n + 1):
            cols.append(ref ** (2 * j + 1))
        V = np.column_stack(cols)
        col_e = np.array([(-1) ** i for i in range(n_ref)], dtype=float)
        A = np.column_stack([V, col_e])
        rhs = np.arctan(ref)

        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break
        coeffs = sol[: n + 1]

        t_fine = np.linspace(a, b, n_fine)
        p_fine = _eval_poly_odd(t_fine, coeffs)
        e_fine = np.arctan(t_fine) - p_fine
        max_abs_err = float(np.max(np.abs(e_fine)))

        # local extrema candidates
        de = np.diff(e_fine)
        sign_changes = np.diff(np.sign(de)) != 0
        ext_idx = np.where(sign_changes)[0] + 1
        if len(ext_idx) == 0:
            ext_idx = np.array([np.argmax(np.abs(e_fine))])

        t_cand = np.concatenate([[a], t_fine[ext_idx], [b]])
        e_cand = np.arctan(t_cand) - _eval_poly_odd(t_cand, coeffs)
        order = np.argsort(t_cand)
        t_cand = t_cand[order]
        e_cand = e_cand[order]

        # pick alternating-sign points
        ref_new = []
        j = 0
        for i in range(n_ref):
            need = (-1) ** i
            while j < len(t_cand):
                s = np.sign(e_cand[j])
                if s == 0 or s == need:
                    ref_new.append(t_cand[j])
                    j += 1
                    break
                j += 1

        if len(ref_new) < n_ref:
            # fallback uniform nodes
            ref = np.linspace(a, b, n_ref)
        else:
            ref = np.array(sorted(ref_new[:n_ref]))

    return coeffs.tolist(), (max_abs_err if max_abs_err is not None else float("nan"))


def validate_on_signed_interval(coeffs, n=200000):
    """
    Validate odd polynomial on [-1,1], returns max abs/rel error for atan(x).
    """
    xs = np.linspace(-1.0, 1.0, n, dtype=float)
    ref = np.arctan(xs)
    appr = _eval_poly_odd(xs, np.array(coeffs, dtype=float))
    abs_err = np.abs(ref - appr)
    # avoid division blow-up near zero
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-12)
    return float(np.max(abs_err)), float(np.max(rel_err))


def print_coeffs(coeffs):
    """
    Print in common.hpp style:
      a1, a3, ..., a(2m+1)
    """
    for i, c in enumerate(coeffs):
        p = 2 * i + 1
        print(f"a{p} = {c:.16g}f;")


def main():
    ap = argparse.ArgumentParser(description="Remez odd polynomial coefficients for atan on [0,1]")
    ap.add_argument("--max-odd", type=int, default=11, help="Highest odd power (e.g. 11 => a1..a11)")
    ap.add_argument("--iterations", type=int, default=40, help="Max Remez iterations")
    ap.add_argument("--fine", type=int, default=20000, help="Fine-grid points for exchange step")
    args = ap.parse_args()

    if args.max_odd < 1 or (args.max_odd % 2 == 0):
        raise SystemExit("--max-odd must be positive odd integer, e.g. 11")
    if not HAS_NUMPY:
        raise SystemExit("numpy is required. Please install numpy in .venv.")

    m = (args.max_odd - 1) // 2
    coeffs, max_err_01 = remez_atan_odd_numpy(m=m, max_iter=args.iterations, n_fine=args.fine)
    max_abs_signed, max_rel_signed = validate_on_signed_interval(coeffs)

    print("Remez odd polynomial for atan(t) on [0,1]")
    print(f"atan(t) ~= t * (a1 + a3*t^2 + ... + a{args.max_odd}*t^{args.max_odd - 1})")
    print(f"max abs err on [0,1]:  {max_err_01:.16g}")
    print(f"max abs err on [-1,1]: {max_abs_signed:.16g}")
    print(f"max rel err on [-1,1]: {max_rel_signed:.16g}")
    print("")
    print_coeffs(coeffs)


if __name__ == "__main__":
    main()
