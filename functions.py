import numpy as np
from scipy.stats import t, norm

def t_confidence_interval(data, alpha=0.05):
    """
    Рассчёт t-доверительного интервала для среднего значения.

    Параметры
    ----------
    data : array-like
        Выборка значений (например, ежедневные средние чеки).
    alpha : float, optional (default=0.05)
        Уровень значимости (по умолчанию 0.05 для 95% доверительного интервала).

    Возвращает
    ----------
    mean : float
        Выборочное среднее.
    lower : float
        Нижняя граница доверительного интервала.
    upper : float
        Верхняя граница доверительного интервала.
    """
    n = len(data)
    x_mean = np.mean(data)
    x_std = np.std(data, ddof=1)
    df = n - 1
    lower, upper = t.interval(
        confidence=1-alpha,
        df=df,
        loc=x_mean,
        scale=x_std/np.sqrt(n)
    )
    return x_mean, lower, upper


def proportion_conf_interval(x_success, n_total, alpha=0.05):
    """
    Рассчёт доверительного интервала для пропорции (например, конверсии).

    Параметры
    ----------
    x_success : int
        Число успехов (например, число покупок).
    n_total : int
        Общее число наблюдений (например, число визитов).
    alpha : float, optional (default=0.05)
        Уровень значимости (по умолчанию 0.05 для 95% доверительного интервала).

    Возвращает
    ----------
    p_hat : float
        Выборочная пропорция (оценка конверсии).
    lower : float
        Нижняя граница доверительного интервала.
    upper : float
        Верхняя граница доверительного интервала.
    """
    p_hat = x_success / n_total
    z_crit = norm.ppf(1 - alpha/2)   # более читаемо, чем -ppf(...)
    se = np.sqrt(p_hat * (1 - p_hat) / n_total)
    lower = p_hat - z_crit * se
    upper = p_hat + z_crit * se
    return p_hat, lower, upper


def diff_proportion_conf_interval(x_p, n, gamma=0.95):
    """
    Рассчёт доверительного интервала для разницы двух пропорций (например, разница конверсий между группами).

    Параметры
    ----------
    x_p : list of float
        Список из двух значений пропорций (например, [p_A, p_B]).
    n : list of int
        Список из двух размеров выборок (например, [n_A, n_B]).
    gamma : float, optional (default=0.95)
        Уровень доверия (по умолчанию 0.95 для 95% доверительного интервала).

    Возвращает
    ----------
    diff : float
        Разница пропорций (второе значение минус первое, например B − A).
    lower : float
        Нижняя граница доверительного интервала.
    upper : float
        Верхняя граница доверительного интервала.
    """
    alpha = 1 - gamma
    diff = x_p[1] - x_p[0]  # p_B - p_A
    se = np.sqrt(x_p[0]*(1-x_p[0])/n[0] + x_p[1]*(1-x_p[1])/n[1])
    z_crit = norm.ppf(1 - alpha/2)
    lower_bound = diff - z_crit * se
    upper_bound = diff + z_crit * se
    return diff, lower_bound, upper_bound