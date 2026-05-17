"""Розширені методи Монте-Карло інтегрування з технікою зменшення дисперсії.

Усі функції повертають (estimate, std_error) — стандартну похибку оцінки,
що дозволяє безпосередньо порівнювати дисперсію між методами.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import qmc


def mean_value_mc(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Базовий sample-mean MC: I ≈ (b-a) · mean(f(X)), X ~ U[a, b].

    Синоніми в літературі: mean-value method, sample-mean method, crude MC.
    Слугує еталоном (baseline) для оцінки variance reduction factor (VRF).
    """
    rng = rng or np.random.default_rng()
    xs = rng.uniform(a, b, size=n)
    fs = f(xs)
    est = (b - a) * fs.mean()
    se = (b - a) * fs.std(ddof=1) / np.sqrt(n)
    return float(est), float(se)


def antithetic_mc(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Антитетичні змінні: пари (X, a+b-X) — від'ємно скорельовані.

    Для монотонної f дисперсія може впасти у кілька разів.
    """
    rng = rng or np.random.default_rng()
    half = n // 2
    u = rng.uniform(0.0, 1.0, size=half)
    xs1 = a + (b - a) * u
    xs2 = a + (b - a) * (1.0 - u)
    pair_means = 0.5 * (f(xs1) + f(xs2))
    est = (b - a) * pair_means.mean()
    se = (b - a) * pair_means.std(ddof=1) / np.sqrt(half)
    return float(est), float(se)


def stratified_mc(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    k: int = 20,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Стратифікований MC: [a, b] ділимо на k рівних страт, у кожній — n/k точок."""
    rng = rng or np.random.default_rng()
    per = n // k
    edges = np.linspace(a, b, k + 1)
    width = edges[1] - edges[0]

    total_est = 0.0
    total_var = 0.0
    for i in range(k):
        xs = rng.uniform(edges[i], edges[i + 1], size=per)
        fs = f(xs)
        total_est += width * fs.mean()
        total_var += (width ** 2) * fs.var(ddof=1) / per
    return float(total_est), float(np.sqrt(total_var))


def control_variate_mc(
    f: Callable[[np.ndarray], np.ndarray],
    g: Callable[[np.ndarray], np.ndarray],
    g_integral: float,
    a: float,
    b: float,
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Control variate: I_f = E[f] - c · (E[g] - I_g / (b-a)) · (b-a).

    g — функція з відомим інтегралом g_integral на [a, b].
    Оптимальне c = Cov(f, g) / Var(g) оцінюємо з вибірки.
    """
    rng = rng or np.random.default_rng()
    xs = rng.uniform(a, b, size=n)
    fs = f(xs)
    gs = g(xs)

    cov_fg = np.cov(fs, gs, ddof=1)[0, 1]
    var_g = gs.var(ddof=1)
    c = cov_fg / var_g

    mean_g_true = g_integral / (b - a)
    h = fs - c * (gs - mean_g_true)
    est = (b - a) * h.mean()
    se = (b - a) * h.std(ddof=1) / np.sqrt(n)
    return float(est), float(se)


def importance_sampling_mc(
    f: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    pdf: Callable[[np.ndarray], np.ndarray],
    n: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Importance sampling: X ~ p, I ≈ mean(f(X) / p(X)).

    sampler(n, rng) — генерує n точок з розподілу p.
    pdf(x) — повертає p(x).
    """
    rng = rng or np.random.default_rng()
    xs = sampler(n, rng)
    weights = f(xs) / pdf(xs)
    est = weights.mean()
    se = weights.std(ddof=1) / np.sqrt(n)
    return float(est), float(se)


def quasi_mc(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    seed: int | None = None,
) -> tuple[float, float]:
    """Quasi-MC за послідовністю Соболя (з рандомізованим скремблінгом).

    Похибка детермінована — std/√n тут лише орієнтовний індикатор,
    реальна збіжність QMC — O((log n)^d / n), швидша за √n.
    """
    sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
    u = sampler.random(n).ravel()
    xs = a + (b - a) * u
    fs = f(xs)
    est = (b - a) * fs.mean()
    se = (b - a) * fs.std(ddof=1) / np.sqrt(n)
    return float(est), float(se)
