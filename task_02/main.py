"""Порівняння методу Монте-Карло з аналітичним результатом та scipy.quad."""

from __future__ import annotations

import numpy as np
import scipy.integrate as spi

from monte_carlo import (
    integrate_hit_or_miss,
    integrate_hit_or_miss_signed,
    integrate_mean_value,
)


def f(x):
    return np.sin(x)


A = 0.0
B = 4.0
ANALYTICAL = float(np.cos(A) - np.cos(B))  # ∫ sin x dx = -cos x  ⇒  cos(A) − cos(B)

# межі по осі Y для bounding box (на дрібній сітці, незалежно від знаку f)
_grid = f(np.linspace(A, B, 10_001))
Y_MAX = float(np.max(_grid))               # для класичного hit-or-miss (тільки додатна частина)
Y_MIN_SIGNED = float(min(0.0, _grid.min()))  # для знакового hit-or-miss
Y_MAX_SIGNED = float(max(0.0, _grid.max()))


def main() -> None:
    quad_value, quad_err = spi.quad(f, A, B)
    print("=" * 86)
    print(f"Функція           : f(x) = sin(x)")
    print(f"Межі інтегрування : [{A}, {B}]  "
          f"(sin змінює знак у π ≈ 3.14, тому на цьому інтервалі f має від'ємну ділянку)")
    print("=" * 86)
    print(f"Аналітичне значення : {ANALYTICAL:.10f}  (= cos(0) − cos(4))")
    print(f"scipy.integrate.quad: {quad_value:.10f}  (±{quad_err:.2e})")
    print("-" * 86)
    print()

    seed = 42
    n_values = (1_000, 10_000, 100_000, 1_000_000)
    print(f"Монте-Карло (seed = {seed}) для різних N:")
    print("=" * 86)
    print(f"{'N':>10} | {'mean-value':>12} {'err':>9} | "
          f"{'hit-or-miss':>12} {'err':>9} | "
          f"{'hm-signed':>12} {'err':>9}")
    print("-" * 86)
    for n in n_values:
        rng = np.random.default_rng(seed=seed)
        mv, _ = integrate_mean_value(f, A, B, n, rng=rng)
        hm, _ = integrate_hit_or_miss(f, A, B, Y_MAX, n, rng=rng)
        hms, _ = integrate_hit_or_miss_signed(f, A, B, Y_MIN_SIGNED, Y_MAX_SIGNED, n, rng=rng)
        print(
            f"{n:>10,} | "
            f"{mv:>12.6f} {abs(mv - ANALYTICAL):>9.6f} | "
            f"{hm:>12.6f} {abs(hm - ANALYTICAL):>9.6f} | "
            f"{hms:>12.6f} {abs(hms - ANALYTICAL):>9.6f}"
        )
    print("=" * 86)
    print(
        "Зверніть увагу: класичний hit-or-miss збігається не до аналітичного значення,\n"
        "а до інтеграла лише додатної частини sin на [0, π] ≈ 2.0 — він «не бачить»\n"
        "від'ємну ділянку. Знаковий hit-or-miss і mean-value працюють коректно."
    )


if __name__ == "__main__":
    main()
