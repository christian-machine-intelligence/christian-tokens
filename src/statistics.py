"""Statistical functions: confidence intervals, hypothesis tests, extrapolation."""

import math
from dataclasses import dataclass
from typing import Optional

from scipy import stats


@dataclass
class ProportionEstimate:
    """A proportion with confidence interval."""
    proportion: float
    ci_lower: float
    ci_upper: float
    n: int  # sample size
    successes: int


@dataclass
class ExtrapolatedEstimate:
    """An extrapolated token count with CI."""
    est_tokens: int
    ci_lower: int
    ci_upper: int
    proportion: ProportionEstimate
    total_tokens: int  # full dataset size


@dataclass
class ComparisonResult:
    """Result of comparing two proportions."""
    p1: float
    p2: float
    difference: float
    ratio: float  # p1 / p2
    z_stat: float
    p_value: float
    cohens_h: float
    significant: bool  # at alpha=0.05


def wilson_ci(
    successes: int, n: int, alpha: float = 0.05
) -> ProportionEstimate:
    """Wilson score interval for a proportion.

    Better than Wald for small proportions — avoids negative lower bounds.
    """
    if n == 0:
        return ProportionEstimate(0.0, 0.0, 0.0, 0, 0)

    p = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z * z

    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom

    return ProportionEstimate(
        proportion=p,
        ci_lower=max(0, center - spread),
        ci_upper=min(1, center + spread),
        n=n,
        successes=successes,
    )


def extrapolate(
    prop: ProportionEstimate, total_tokens: int
) -> ExtrapolatedEstimate:
    """Extrapolate a sample proportion to a full dataset token count."""
    return ExtrapolatedEstimate(
        est_tokens=round(prop.proportion * total_tokens),
        ci_lower=round(prop.ci_lower * total_tokens),
        ci_upper=round(prop.ci_upper * total_tokens),
        proportion=prop,
        total_tokens=total_tokens,
    )


def two_proportion_z_test(
    p1: float, n1: int, p2: float, n2: int
) -> ComparisonResult:
    """Two-proportion z-test comparing two categories."""
    if n1 == 0 or n2 == 0 or (p1 == 0 and p2 == 0):
        return ComparisonResult(
            p1=p1, p2=p2, difference=0, ratio=0,
            z_stat=0, p_value=1.0, cohens_h=0, significant=False,
        )

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return ComparisonResult(
            p1=p1, p2=p2, difference=p1 - p2,
            ratio=p1 / p2 if p2 > 0 else float("inf"),
            z_stat=0, p_value=1.0, cohens_h=0, significant=False,
        )

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Cohen's h effect size
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))

    ratio = p1 / p2 if p2 > 0 else float("inf")

    return ComparisonResult(
        p1=p1, p2=p2,
        difference=p1 - p2,
        ratio=ratio,
        z_stat=z,
        p_value=p_value,
        cohens_h=h,
        significant=p_value < 0.05,
    )


def format_tokens(n: int) -> str:
    """Format a token count with appropriate suffix."""
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.1f}T"
    elif n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
