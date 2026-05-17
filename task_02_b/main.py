"""Порівняння методів Монте-Карло на трьох функціях з різними «характерами».

Кожна функція підібрана так, щоб підкреслити сильну сторону певного методу:
  A) eˣ  на [0, 1]            — control variate, antithetic
  B) exp(-50(x-0.5)²) на [0, 1] — importance sampling
  C) sin(x) на [0, 4]          — quasi-MC + stratified (плюс перевірка знаку)

Для кожної функції виводиться таблиця з: оцінка, |err|, std err,
VRF (variance reduction factor) = (σ_mv / σ_method)² — у скільки разів
дисперсія методу менша за дисперсію mean-value MC (baseline).
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as spi

from functions import ALL_CASES, TestCase
from monte_carlo_advanced import (
    antithetic_mc,
    control_variate_mc,
    importance_sampling_mc,
    mean_value_mc,
    quasi_mc,
    stratified_mc,
)


SEED = 42
N = 2 ** 14  # 16384 — степінь двійки потрібна для Sobol


def run_case(case: TestCase) -> None:
    quad_value, _ = spi.quad(case.f, case.a, case.b)
    print("=" * 82)
    print(case.title)
    print(f"  {case.description}")
    print("-" * 82)
    print(f"  Аналітичне : {case.analytical:.10f}")
    print(f"  scipy.quad : {quad_value:.10f}")
    print("=" * 82)
    print(f"{'Метод':<22} {'оцінка':>12} {'|err|':>10} {'std err':>10} {'VRF':>9}")
    print("-" * 82)

    # baseline
    rng = np.random.default_rng(SEED)
    mv_est, mv_se = mean_value_mc(case.f, case.a, case.b, N, rng=rng)
    print(f"{'1. Mean-value':<22} {mv_est:>12.6f} "
          f"{abs(mv_est - case.analytical):>10.6f} {mv_se:>10.6f} {'1.0x':>9}")

    def row(name: str, est: float | None, se: float | None) -> None:
        if est is None:
            print(f"{name:<22} {'—':>12} {'—':>10} {'—':>10} {'—':>9}  (не застосовується)")
            return
        vrf = (mv_se / se) ** 2 if se and se > 0 else float("inf")
        print(f"{name:<22} {est:>12.6f} "
              f"{abs(est - case.analytical):>10.6f} {se:>10.6f} {vrf:>8.1f}x")

    rng = np.random.default_rng(SEED)
    row("2. Antithetic", *antithetic_mc(case.f, case.a, case.b, N, rng=rng))

    rng = np.random.default_rng(SEED)
    row("3. Stratified (k=20)", *stratified_mc(case.f, case.a, case.b, N, k=20, rng=rng))

    if case.control_g is not None:
        rng = np.random.default_rng(SEED)
        row("4. Control variate", *control_variate_mc(
            case.f, case.control_g, case.control_g_integral,
            case.a, case.b, N, rng=rng))
    else:
        row("4. Control variate", None, None)

    if case.is_sampler is not None:
        rng = np.random.default_rng(SEED)
        row("5. Importance sampl.", *importance_sampling_mc(
            case.f, case.is_sampler, case.is_pdf, N, rng=rng))
    else:
        row("5. Importance sampl.", None, None)

    row("6. Quasi-MC (Sobol)", *quasi_mc(case.f, case.a, case.b, N, seed=SEED))
    print()


def main() -> None:
    print(f"# Параметри: N = {N:,} (= 2^14),  seed = {SEED}")
    print("VRF (variance reduction factor) — у скільки разів дисперсія методу менша за mean-value MC. \n"
          "VRF = 10 означає, що метод при N точок дає таку саму точність, як mean-value MC при 10N точок.")
    print()
    for case in ALL_CASES:
        run_case(case)


if __name__ == "__main__":
    main()
