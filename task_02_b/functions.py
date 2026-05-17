"""Тестові функції для демонстрації різних MC-технік.

Кожна функція має «природу» (монотонна, пікова, оссилююча, зі зміною знаку),
що добре відповідає сильним сторонам того чи іншого методу.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import erf
from scipy.stats import truncnorm


@dataclass
class TestCase:
    key: str
    title: str
    f: Callable[[np.ndarray], np.ndarray]
    a: float
    b: float
    analytical: float
    # control variate
    control_g: Optional[Callable[[np.ndarray], np.ndarray]] = None
    control_g_integral: Optional[float] = None
    # importance sampling
    is_sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None
    is_pdf: Optional[Callable[[np.ndarray], np.ndarray]] = None
    description: str = ""


# ───────────────────────── Case A: монотонна e^x ─────────────────────────
def _A_sample(n, rng):  # p(x) = (2/3)(1 + x) на [0, 1], інверсний CDF
    u = rng.uniform(0.0, 1.0, size=n)
    return -1.0 + np.sqrt(1.0 + 3.0 * u)


CASE_A = TestCase(
    key="A",
    title="A) f(x) = eˣ  на  [0, 1]  — монотонна, гладка, додатна",
    f=lambda x: np.exp(x),
    a=0.0,
    b=1.0,
    analytical=float(np.e - 1.0),
    # лінійна апроксимація — сильно скорельована з eˣ
    control_g=lambda x: 1.0 + x,
    control_g_integral=1.5,  # ∫₀¹ (1 + x) dx
    is_sampler=_A_sample,
    is_pdf=lambda x: (2.0 / 3.0) * (1.0 + x),
    description="Майже лінійна гладка функція."
)


# ─────────────────── Case B: гострий пік exp(-50(x-0.5)²) ───────────────────
# Усічений нормальний розподіл на [0, 1] із центром 0.5 та σ = 0.1
# точно повторює форму піка f(x) = exp(-50(x-0.5)²) (бо 50 = 1/(2·0.1²)).
# σ=0.15 ширша за «ідеальну» (0.1 дала б p ∝ f і нульову дисперсію — академічний випадок).
# Така помірна неузгодженість дає реалістичну, але все одно драматичну користь IS.
_B_TRUNC_A, _B_TRUNC_B, _B_LOC, _B_SCALE = (-0.5 / 0.11), (0.5 / 0.11), 0.5, 0.11


def _B_sample(n, rng):
    return truncnorm.rvs(
        a=_B_TRUNC_A, b=_B_TRUNC_B,
        loc=_B_LOC, scale=_B_SCALE,
        size=n, random_state=rng,
    )


def _B_pdf(x):
    return truncnorm.pdf(
        x, a=_B_TRUNC_A, b=_B_TRUNC_B,
        loc=_B_LOC, scale=_B_SCALE,
    )


CASE_B = TestCase(
    key="B",
    title="B) f(x) = exp(−50·(x−0.5)²)  на  [0, 1]  — гострий пік",
    f=lambda x: np.exp(-50.0 * (x - 0.5) ** 2),
    a=0.0,
    b=1.0,
    # ∫ exp(-50(x-0.5)²) dx = √(π/50) · erf(0.5·√50)
    analytical=float(np.sqrt(np.pi / 50.0) * erf(0.5 * np.sqrt(50.0))),
    # CV: лінійна g тут безпомічна (немає кореляції з симетричним піком).
    control_g=None,
    control_g_integral=None,
    # IS: «намет» (tent) з вершиною в 0.5 — добре наближає форму піка
    is_sampler=_B_sample,
    is_pdf=_B_pdf,
    description="Майже вся маса інтеграла зосереджена у вузькому околі x=0.5 "
    "(шир. ≈ 0.2). Crude MC «марнує» ~80% точок. Importance sampling з "
    "усіченим Гауссом N(0.5, 0.11) повторює форму піка → найбільший VRF.",
)


# ───────────────── Case C: sin(x) на [0, 4] — зі зміною знаку ─────────────────
CASE_C = TestCase(
    key="C",
    title="C) f(x) = sin(x)  на  [0, 4]  — осцилююча, міняє знак у x = π",
    f=lambda x: np.sin(x),
    a=0.0,
    b=4.0,
    analytical=float(np.cos(0.0) - np.cos(4.0)),  # ≈ 1.6536
    # CV: лінійна g(x) = x — частково скорельована з sin на цьому інтервалі
    control_g=lambda x: x,
    control_g_integral=8.0,  # ∫₀⁴ x dx = 16/2
    # IS немає природного «піка» — пропускаємо (поставимо None)
    is_sampler=None,
    is_pdf=None,
    description="Гладка, але змінює знак. Усі методи мають коректно "
    "обробляти знак. QMC і stratified шикарно працюють для гладких f.",
)


ALL_CASES: list[TestCase] = [CASE_A, CASE_B, CASE_C]
