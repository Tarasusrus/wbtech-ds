"""Синтетический маркетплейс — сквозные данные раздела «Статистика».

Генеральная совокупность — все заказы маркетплейса за месяц.
В реальной работе она скрыта: аналитик видит только выборку.
Здесь мы создаём её сами, поэтому знаем истину и можем проверить,
насколько хороши наши оценки.

Как устроены поля (это «закон природы» нашего мира):

- ``category`` — категория заказа (номинальный признак).
- ``check`` — сумма заказа, ₽. Внутри категории распределена
  **ровно логнормально**: ``ln(check) ~ N(mu, sigma)`` с параметрами
  из ``CATEGORIES``. Поэтому для каждой категории есть аналитическая
  истина: ``E[check] = exp(mu + sigma**2 / 2)``.
- ``device`` — с чего оформлен заказ: ``app`` или ``web``. Доля
  приложения зависит от категории — электронику чаще покупают
  из приложения. Отсюда ловушка нерепрезентативной выборки.
- ``region`` — регион доставки.
- ``delivery_days`` — срок доставки в днях (целое число, много повторов).
  Зависит от региона.
- ``rating`` — оценка заказа 1–5 (порядковый признак). Чем дольше
  доставка, тем ниже оценка.
- ``returned`` — был ли возврат (True/False). Вероятность возврата
  зависит от категории.

Поля появляются по мере надобности в уроках. У каждого поля свой поток
случайных чисел (``_streams``), поэтому новое поле не меняет уже
существующие при том же seed. Новый поток добавлять только в конец ``FIELDS``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Category:
    share: float  # доля заказов категории
    mu: float  # параметр ln(check) ~ N(mu, sigma)
    sigma: float
    p_app: float  # доля заказов из приложения
    p_return: float  # вероятность возврата


CATEGORIES: dict[str, Category] = {
    "Электроника": Category(share=0.15, mu=8.3, sigma=0.70, p_app=0.85, p_return=0.06),
    "Одежда": Category(share=0.35, mu=7.6, sigma=0.50, p_app=0.65, p_return=0.25),
    "Дом": Category(share=0.20, mu=7.4, sigma=0.60, p_app=0.55, p_return=0.08),
    "Красота": Category(share=0.20, mu=6.9, sigma=0.45, p_app=0.60, p_return=0.04),
    "Книги": Category(share=0.10, mu=6.5, sigma=0.40, p_app=0.40, p_return=0.03),
}

# регион: (доля заказов, базовый срок доставки в днях)
REGIONS: dict[str, tuple[float, int]] = {
    "Москва": (0.30, 1),
    "Санкт-Петербург": (0.15, 2),
    "Казань": (0.10, 3),
    "Самара": (0.10, 3),
    "Екатеринбург": (0.10, 3),
    "Краснодар": (0.15, 4),
    "Новосибирск": (0.10, 5),
}

N_ORDERS = 1_000_000
SEED = 42

# порядок задаёт номер потока: дописывать только в конец
FIELDS = ("category", "check", "device", "region", "delivery_days", "rating", "returned")


def _streams(seed: int) -> dict[str, np.random.Generator]:
    """Независимый генератор на каждое поле."""
    children = np.random.SeedSequence(seed).spawn(len(FIELDS))
    return {name: np.random.default_rng(ss) for name, ss in zip(FIELDS, children)}


def true_check_mean(category: str) -> float:
    """Аналитическое математическое ожидание суммы заказа в категории."""
    c = CATEGORIES[category]
    return float(np.exp(c.mu + c.sigma**2 / 2))


def make_population(n_orders: int = N_ORDERS, seed: int = SEED) -> pd.DataFrame:
    """Генеральная совокупность: все заказы за месяц."""
    rng = _streams(seed)

    cat_names = list(CATEGORIES)
    cats = [CATEGORIES[name] for name in cat_names]
    cat_idx = rng["category"].choice(len(cats), size=n_orders, p=[c.share for c in cats])

    mu = np.array([c.mu for c in cats])[cat_idx]
    sigma = np.array([c.sigma for c in cats])[cat_idx]
    check = np.exp(rng["check"].normal(mu, sigma)).round(2)

    p_app = np.array([c.p_app for c in cats])[cat_idx]
    device_code = (rng["device"].random(n_orders) >= p_app).astype(np.int8)  # 0 — app, 1 — web

    reg_names = list(REGIONS)
    reg_idx = rng["region"].choice(len(reg_names), size=n_orders, p=[REGIONS[r][0] for r in reg_names])
    base_days = np.array([REGIONS[r][1] for r in reg_names])[reg_idx]
    delivery_days = base_days + rng["delivery_days"].poisson(1.0, size=n_orders)

    score = 4.6 - 0.3 * (delivery_days - 2) + rng["rating"].normal(0, 0.9, size=n_orders)
    rating = np.clip(np.rint(score), 1, 5).astype(int)

    p_return = np.array([c.p_return for c in cats])[cat_idx]
    returned = rng["returned"].random(n_orders) < p_return

    return pd.DataFrame(
        {
            "order_id": np.arange(1, n_orders + 1),
            "category": pd.Categorical.from_codes(cat_idx, cat_names),
            "check": check,
            "device": pd.Categorical.from_codes(device_code, ["app", "web"]),
            "region": pd.Categorical.from_codes(reg_idx, reg_names),
            "delivery_days": delivery_days,
            "rating": rating,
            "returned": returned,
        }
    )
