"""Descriptive diagnostics for historical experiments; no independence assumed."""
import numpy as np


def boost_diagnostics(c4, c2, full):
    """Describe cumulative boost counters; c4 is a subset of c2.

    full is a separate recorded boost counter, not a population denominator.
    Only composition within c2 is normalized.
    """
    arrays = [np.asarray(x) for x in (c4, c2, full)]
    if any(x.ndim != 1 or not x.size or x.dtype.kind not in 'iu' for x in arrays):
        raise ValueError('Expected nonempty integer counter vectors')
    if len({x.shape for x in arrays}) != 1:
        raise ValueError('Counter shapes must match')
    increments = []
    for a in arrays:
        values = [int(x) for x in a]
        delta = [values[0]]+[b-a for a,b in zip(values,values[1:])]
        if any(x < 0 for x in values+delta):
            raise ValueError('Expected nonnegative cumulative counters')
        increments.append(np.asarray(delta,dtype=object))
    d4,d2,df = increments
    if any(a > b for a,b in zip(d4,d2)):
        raise ValueError('c4 must be included in c2 in each interval')
    partial = np.stack([d4,d2-d4]).astype(float)
    composition = np.full(partial.shape,np.nan)
    denominator = np.asarray(d2,dtype=float)
    np.divide(partial,denominator,out=composition,where=denominator>0)
    return dict(increments=np.stack(increments),partial_counts=partial,
                within_c2=composition)


def counter_diagnostics(c4, c2, full):
    """Convert nested cumulative coordinate counts into interval exposures.

    Counts are repeated parameter/gradient evaluations, not unique parameters.
    The first saved interval starts at zero; it need not be a full epoch.
    """
    arrays = [np.asarray(x) for x in (c4, c2, full)]
    if any(x.ndim != 1 or x.size == 0 for x in arrays):
        raise ValueError('Counters must be nonempty vectors')
    if len({x.shape for x in arrays}) != 1:
        raise ValueError('Counter shapes must match')
    if any(x.dtype.kind not in 'iu' for x in arrays):
        raise ValueError('Counters must be integers')
    # Python ints avoid signed/unsigned overflow before checking monotonicity.
    increments = []
    for a in arrays:
        values = [int(x) for x in a]
        if any(x < 0 for x in values):
            raise ValueError('Counts must be nonnegative')
        delta = [values[0]] + [b-a for a, b in zip(values, values[1:])]
        if any(x < 0 for x in delta):
            raise ValueError('Cumulative counts must be nondecreasing')
        increments.append(np.array(delta, dtype=object))
    d4, d2, dt = increments
    if any(a > b or b > t for a, b, t in zip(d4, d2, dt)):
        raise ValueError('Interval counts must obey c4 <= c2 <= full')
    if int(arrays[2][-1]) == 0:
        raise ValueError('At least one evaluation is required')
    bins = np.stack([d4, d2-d4, dt-d2]).astype(float)
    rates = np.full(bins.shape, np.nan)
    np.divide(bins, np.asarray(dt, dtype=float), out=rates, where=np.asarray(dt) > 0)
    cumulative_total = np.cumsum(np.asarray(dt, dtype=float))
    cumulative_rates = np.full(bins.shape, np.nan)
    np.divide(np.cumsum(bins, axis=1), cumulative_total, out=cumulative_rates,
              where=cumulative_total > 0)
    return dict(increments=np.stack(increments), bins=bins,
                rates=rates, cumulative_rates=cumulative_rates,
                pooled_rates=bins.sum(axis=1)/int(arrays[2][-1]))


def pooled_window(diag, start, stop):
    """Exposure-weighted rates in a half-open interval; empty exposure is missing."""
    bins = diag['bins'][:, start:stop].sum(axis=1)
    total = bins.sum()
    return bins/total if total else np.full(3, np.nan)


def threshold_bounds(f4, f2, divisors):
    """Identification bounds on the observed trajectory, without interpolation.

    f4 = P(D <= v/4), f2 = P(D <= v/2). These bounds do not predict retraining
    with another divisor. No assumption about a distribution within bins.
    """
    if not (0 <= f4 <= f2 <= 1):
        raise ValueError('Expected 0 <= f4 <= f2 <= 1')
    result = []
    for divisor in divisors:
        if not np.isfinite(divisor) or divisor < 1:
            raise ValueError('Floor divisor must be finite and >= 1')
        if divisor > 4:
            lo, hi = 0., f4
        elif divisor == 4:
            lo = hi = f4
        elif divisor > 2:
            lo, hi = f4, f2
        elif divisor == 2:
            lo = hi = f2
        else:
            lo, hi = f2, 1.
        result.append(dict(divisor=float(divisor),threshold=1./divisor,
                           lower=lo,upper=hi,identified=lo == hi))
    return result


def first_sustained(curve, target, width=5):
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 1 or not curve.size or not np.isfinite(curve).all():
        raise ValueError('Expected a finite, nonempty curve')
    if not np.isfinite(target) or not isinstance(width, int) or width < 1:
        raise ValueError('Expected a finite target and positive integer width')
    for i in range(len(curve)-width+1):
        if np.all(curve[i:i+width] <= target):
            return i+1
    return None


def curve_summary(curve, window=20):
    a = np.asarray(curve, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all():
        raise ValueError('Expected at least two finite observations')
    if not isinstance(window, int) or not 2 <= window <= a.size:
        raise ValueError('Window must be between 2 and curve length')
    late = a[-window:]
    auc = float(np.sum((a[1:]+a[:-1])/2))
    slope, intercept = np.polyfit(np.arange(window), late, 1)
    residual = late-(intercept+slope*np.arange(window))
    return dict(first=float(a[0]),last=float(a[-1]),minimum=float(a.min()),
                best_epoch=int(a.argmin()+1),auc=auc,normalized_auc=auc/(a.size-1),
                last_window_mean=float(late.mean()),
                temporal_sd=float(late.std(ddof=1)),late_slope=float(slope),
                detrended_rms=float(np.sqrt(np.mean(residual**2))),
                successive_change_rms=float(np.sqrt(np.mean(np.diff(late)**2))))
