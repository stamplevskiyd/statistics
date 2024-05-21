from functools import partial
from typing import Callable

import numpy as np
from scipy.integrate import quad


def normal_density(x, mu=0, sigma=1):
    """Плотность нормального распределения"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def p_interval(left: float, right: float, density_func: Callable) -> float:
    """Вероятность попасть в заданный интервал"""
    p, err = quad(func=density_func, a=left, b=right)
    return p


def create_intervals(left: float, right: float, segment_count: int = 100) -> np.ndarray:
    """Список правых границ интервала"""
    return np.linspace(left, right, segment_count)


def chi_square(data: np.ndarray, left: float, right: float, segment_count: int) -> float:
    """Посчитать значение Хи-квадрат"""
    # Вычисляем оценки параметров
    # TODO: тоже функции
    mu: float = np.mean(data)
    sigma_squared: float = np.var(data)
    density_func = partial(normal_density, mu=mu, sigma=np.sqrt(sigma_squared))

    # Разбиваем выборку на сегменты
    segments: np.ndarray = create_intervals(left, right, segment_count)

    # Вычисляем ню -- число объектов, реально попавших в интервал
    nu: np.ndarray = np.zeros(segment_count, dtype=int)
    for i in range(segment_count):
        for elem in data:
            if (i == 0 and elem < segments[i]) or (segments[i - 1] < elem < segments[i]):
                nu[i] += 1

    # Вычисляем pi -- вероятность попасть в интервал
    p: np.ndarray = np.zeros(segment_count, dtype=float)
    p[0] = p_interval(-np.inf, segments[0], density_func=density_func)
    for i in range(1, segment_count - 1):
        p[i] = p_interval(segments[i], segments[i + 1], density_func=density_func)

    return np.sum((nu - segment_count * p) ** 2 / segment_count * p)
