"""Видача решти: жадібний алгоритм та динамічне програмування."""

COINS = [50, 25, 10, 5, 2, 1]


def find_coins_greedy(amount: int, coins: list[int] = COINS) -> dict[int, int]:
    """Жадібний алгоритм: на кожному кроці беремо найбільший доступний номінал.
    Складність: O(k), де k — кількість номіналів (сортування одноразове).
    """
    result: dict[int, int] = {}
    for coin in sorted(coins, reverse=True):
        if amount <= 0:
            break
        count, amount = divmod(amount, coin)
        if count:
            result[coin] = count
    return result


def find_min_coins(amount: int, coins: list[int] = COINS) -> dict[int, int]:
    """Динамічне програмування: гарантовано мінімальна кількість монет.

    Складність: O(n * k) за часом і O(n) за пам'яттю,
    де n — сума, k — кількість номіналів.
    """
    if amount <= 0:
        return {}

    min_coins = [0] + [amount + 1] * amount
    coin_used = [0] * (amount + 1)

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and min_coins[i - coin] + 1 < min_coins[i]:
                min_coins[i] = min_coins[i - coin] + 1
                coin_used[i] = coin

    result: dict[int, int] = {}
    remaining = amount
    while remaining > 0:
        coin = coin_used[remaining]
        result[coin] = result.get(coin, 0) + 1
        remaining -= coin
    return dict(sorted(result.items()))
