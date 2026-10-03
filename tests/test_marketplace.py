import hashlib

import numpy as np
import pytest

from ds_course import marketplace as mp
from ds_course.marketplace import CATEGORIES, FIELDS, make_population, true_check_mean

# Эталон для seed=42, n=10_000. Падение значит, что существующие данные изменились:
# пороги проверок в тетрадках подобраны под эти данные. Новое поле добавлять
# отдельным потоком в конец FIELDS, а не менять старые.
GOLDEN = {
    "category": "e60ec63c82045445",
    "check": "1be110c60cc233b3",
    "device": "8fbd435897cfe4e0",
    "region": "de15fea9fbbdfcf9",
    "delivery_days": "72160bc4013f4f66",
    "rating": "0c9fc1caa5f860c5",
    "returned": "f18cfe02eea8dd3b",
}


def column_hash(series) -> str:
    if hasattr(series, "cat"):
        arr = series.cat.codes.to_numpy()
    elif series.name == "check":
        arr = np.rint(series.to_numpy() * 100).astype(np.int64)  # копейки: не зависим от последнего бита float
    else:
        arr = series.to_numpy()
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def small_pop():
    return make_population(n_orders=10_000)


@pytest.mark.parametrize("col", FIELDS)
def test_existing_columns_unchanged(small_pop, col):
    assert column_hash(small_pop[col]) == GOLDEN[col]


def test_new_stream_does_not_shift_existing(monkeypatch):
    before = mp._streams(42)
    monkeypatch.setattr(mp, "FIELDS", FIELDS + ("new_field",))
    after = mp._streams(42)
    for name in FIELDS:
        assert np.array_equal(before[name].random(100), after[name].random(100)), name


def test_reproducible():
    assert make_population(5_000).equals(make_population(5_000))


@pytest.mark.parametrize("category", list(CATEGORIES))
def test_check_matches_analytic_mean(category):
    pop = make_population()
    got = pop.loc[pop["category"] == category, "check"].mean()
    assert got == pytest.approx(true_check_mean(category), rel=0.01)
