"""Zero-lag low-pass filter for tracks with gaps, and smoothness measures (SPARC, LDLJ) of a speed profile."""
import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_runs(x, fs, fc, order=4, min_len=16):
    """Zero-lag Butterworth (fs, fc in Hz) on each non-NaN run of >= min_len samples; shorter runs, gaps unchanged."""
    b, a = butter(order, fc / (fs / 2))
    y = np.full_like(x, np.nan, dtype=float)
    ok = ~np.isnan(x)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_len:
            y[s:e] = filtfilt(b, a, x[s:e], padlen=min(3 * max(len(a), len(b)), e - s - 1))
        else:
            y[s:e] = x[s:e]
    return y


def sparc(speed, fs, pad=4, fc=10.0, amp_th=0.05):
    """Spectral arc length (Balasubramanian et al. 2015). More negative = less smooth."""
    n = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(n) * fs / n
    M = np.abs(np.fft.fft(speed, n))
    M = M / M.max()
    sel = f <= fc
    f, M = f[sel], M[sel]
    idx = np.flatnonzero(M >= amp_th)
    f, M = f[idx[0]:idx[-1] + 1], M[idx[0]:idx[-1] + 1]
    if len(f) < 2:
        return np.nan
    return -np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(M) ** 2))


def ldlj(speed, t):
    """Log dimensionless jerk from the speed profile (Balasubramanian et al. 2015)."""
    mt = t[-1] - t[0]
    vpk = speed.max()
    if mt <= 0 or vpk <= 0:
        return np.nan
    j = np.gradient(np.gradient(speed, t), t)
    return -np.log((mt ** 3 / vpk ** 2) * np.trapezoid(j ** 2, t))
