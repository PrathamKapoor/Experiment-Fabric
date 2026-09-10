"""Probability distributions, implemented from scratch.

Fabric's core has no third-party dependencies, so it carries its own
special functions rather than importing SciPy.  The implementations are
the standard ones (Lentz's continued fractions for the incomplete beta,
series/continued-fraction split for the incomplete gamma) and are checked
against published critical-value tables in ``tests/test_distributions.py``
to roughly 1e-10.

If SciPy *is* installed, nothing here changes: Fabric deliberately keeps
one implementation so that a p-value computed on a laptop matches one
computed on a cluster node bit for bit.
"""

from __future__ import annotations

import math

_EPS = 3.0e-16
_FPMIN = 1e-300
_MAX_ITER = 500


# --------------------------------------------------------------------------
# normal
# --------------------------------------------------------------------------


def norm_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal probability density."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def norm_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal cumulative distribution, via the error function."""
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def norm_sf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal survival function ``P(X > x)``, accurate in the far tail."""
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * math.erfc(z)


def norm_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal quantile function (Acklam's rational approximation, refined).

    The refinement step (one Halley iteration against ``erfc``) takes the
    approximation from ~1e-9 to full double precision, which matters when
    a bootstrap BCa interval feeds the quantile back through the CDF.
    """
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("norm_ppf requires 0 <= p <= 1")

    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    err = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return mu + sigma * x


# --------------------------------------------------------------------------
# incomplete beta
# --------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# --------------------------------------------------------------------------
# incomplete gamma
# --------------------------------------------------------------------------


def _gser(a: float, x: float) -> float:
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(a, x)``."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


# --------------------------------------------------------------------------
# Student's t
# --------------------------------------------------------------------------


def t_cdf(t: float, df: float) -> float:
    """Cumulative distribution of Student's t."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_sf(t: float, df: float) -> float:
    """Survival function of Student's t."""
    return 1.0 - t_cdf(t, df)


def t_ppf(p: float, df: float) -> float:
    """Quantile function of Student's t, by bracketed bisection.

    Bisection is slower than a series inversion but is unconditionally
    stable across the tiny degrees of freedom that seed-level experiments
    actually produce (n = 3 is common, and asymptotic expansions are poor
    there).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("t_ppf requires 0 < p < 1")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
        if lo < -1e12:
            return -math.inf
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# F and chi-squared
# --------------------------------------------------------------------------


def f_cdf(f: float, df1: float, df2: float) -> float:
    """Cumulative distribution of the F ratio."""
    if f <= 0.0:
        return 0.0
    return betainc(df1 / 2.0, df2 / 2.0, df1 * f / (df1 * f + df2))


def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail probability of the F ratio (the ANOVA p-value)."""
    return 1.0 - f_cdf(f, df1, df2)


def chi2_cdf(x: float, df: float) -> float:
    """Cumulative distribution of chi-squared."""
    if x <= 0.0:
        return 0.0
    return gammainc(df / 2.0, x / 2.0)


def chi2_sf(x: float, df: float) -> float:
    """Upper-tail probability of chi-squared."""
    return 1.0 - chi2_cdf(x, df)
