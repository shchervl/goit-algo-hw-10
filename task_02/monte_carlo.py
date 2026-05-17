"""Обчислення визначеного інтеграла методом Монте-Карло."""

from __future__ import annotations

from typing import Callable

import numpy as np


def integrate_mean_value(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Метод середнього значення: I ≈ (b - a) · mean(f(X)), X ~ U[a, b].

    Повертає (оцінку, стандартну похибку оцінки).
    """
    rng = rng or np.random.default_rng()
    xs = rng.uniform(a, b, size=n)
    fs = f(xs)
    estimate = (b - a) * fs.mean()
    std_err = (b - a) * fs.std(ddof=1) / np.sqrt(n)
    return float(estimate), float(std_err)


def integrate_hit_or_miss(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    y_max: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Метод «влучань»: частка точок під кривою у прямокутнику [a, b] × [0, y_max].

    Працює лише для f(x) ≥ 0 на [a, b]: від'ємні ділянки графіка
    не потрапляють у прямокутник і не враховуються.
    Повертає (оцінку, стандартну похибку).
    """
    rng = rng or np.random.default_rng()
    xs = rng.uniform(a, b, size=n)
    ys = rng.uniform(0, y_max, size=n)
    hits = ys <= f(xs)
    p = hits.mean()
    box_area = (b - a) * y_max
    estimate = box_area * p
    std_err = box_area * np.sqrt(p * (1 - p) / n)
    return float(estimate), float(std_err)


def integrate_hit_or_miss_signed(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    y_min: float,
    y_max: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Знаковий hit-or-miss: коректно обробляє f, що змінює знак на [a, b].

    Прямокутник [a, b] × [y_min, y_max] (де y_min ≤ 0 ≤ y_max).
    Точку рахуємо:
      +1, якщо 0 ≤ y ≤ f(x);
      −1, якщо f(x) ≤ y ≤ 0;
      0 в інших випадках («промах»).
    Інтеграл = площа_прямокутника · (знакова сума / N).
    """
    rng = rng or np.random.default_rng()
    xs = rng.uniform(a, b, size=n)
    ys = rng.uniform(y_min, y_max, size=n)
    fs = f(xs)
    pos = (ys >= 0) & (ys <= fs)
    neg = (ys < 0) & (ys >= fs)
    signed = pos.astype(np.int8) - neg.astype(np.int8)
    box_area = (b - a) * (y_max - y_min)
    estimate = box_area * signed.mean()
    std_err = box_area * signed.std(ddof=1) / np.sqrt(n)
    return float(estimate), float(std_err)
