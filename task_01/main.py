"""Демонстрація та порівняння алгоритмів видачі решти."""

from timeit import timeit

from cash_register import find_coins_greedy, find_min_coins


def demo() -> None:
    amount = 113
    print(f"Сума: {amount}")
    print(f"  greedy: {find_coins_greedy(amount)}")
    print(f"  dp    : {find_min_coins(amount)}")
    print()


def benchmark() -> None:
    print(f"{'amount':>12} | {'greedy, s':>12} | {'dp, s':>12} | {'dp / greedy':>14}")
    print("-" * 60)
    for amount in (100, 1_000, 10_000, 100_000, 1_000_000):
        repeats = 5 if amount <= 100_000 else 1
        t_greedy = timeit(lambda: find_coins_greedy(amount), number=repeats) / repeats
        t_dp = timeit(lambda: find_min_coins(amount), number=repeats) / repeats
        ratio = t_dp / t_greedy if t_greedy else float("inf")
        print(f"{amount:>12,} | {t_greedy:>12.6f} | {t_dp:>12.6f} | {ratio:>14,.1f}x")


if __name__ == "__main__":
    demo()
    benchmark()
