"""Camera model (DLT, VICON mm -> image px), keypoint filtering, track interpolation and 3D -> 2D affines."""
import numpy as np
from scipy.optimize import least_squares

CUTOFF_HZ = 6.0      # low-pass of the keypoint tracks, the same as for the VICON reference


def hampel(x, win=7, k=3.0):
    """Replace single-frame keypoint jumps (> k SDs from the running median, SD = 1.4826 MAD >= 2 px) with NaN."""
    import pandas as pd
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=3).median() * 1.4826
    out = x.copy()
    out[(s - med).abs().values > k * np.maximum(mad.values, 2.0)] = np.nan
    return out


def smooth(x, fps, on=True):
    """Hampel spike removal, then a zero-lag CUTOFF_HZ Butterworth per coordinate on the valid runs of x (T, D).
    A frame is dropped from every coordinate if any coordinate spikes."""
    if not on:
        return x
    from kinematics import lowpass_runs
    bad = np.zeros(len(x), bool)
    for d in range(x.shape[1]):
        bad |= np.isnan(hampel(x[:, d])) & ~np.isnan(x[:, d])
    x = x.copy()
    x[bad] = np.nan
    return np.stack([lowpass_runs(x[:, d], fps, CUTOFF_HZ) for d in range(x.shape[1])], 1)


def interp_track(t_src, x_src, t_dst, max_gap=0.1):
    """Linear interpolation of an (N, D) track with NaNs; NaN where the nearest valid source
    samples are more than max_gap s away."""
    out = np.full((len(t_dst), x_src.shape[1]), np.nan)
    ok = ~np.isnan(x_src).any(1)
    if ok.sum() < 2:
        return out
    ts, xs = t_src[ok], x_src[ok]
    for d in range(x_src.shape[1]):
        out[:, d] = np.interp(t_dst, ts, xs[:, d], left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ts, t_dst), 1, len(ts) - 1)
    gap = np.minimum(np.abs(t_dst - ts[i - 1]), np.abs(ts[i] - t_dst))
    out[gap > max_gap] = np.nan
    return out


def _norm(p):
    """Similarity that moves the points p (N, D) to zero mean and mean distance sqrt(D) (Hartley normalisation)."""
    c = p.mean(0)
    s = np.sqrt(p.shape[1]) / np.mean(np.linalg.norm(p - c, axis=1))
    T = np.eye(p.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return T


def dlt(X, x):
    """Normalised DLT: 3x4 P with x ~ P [X; 1]."""
    TX, Tx = _norm(X), _norm(x)
    Xh = (TX @ np.c_[X, np.ones(len(X))].T).T
    xh = (Tx @ np.c_[x, np.ones(len(x))].T).T
    A = []
    for (X1, X2, X3, X4), (u, v, w) in zip(Xh, xh):
        Xr = [X1, X2, X3, X4]
        A.append([0, 0, 0, 0] + [-w * a for a in Xr] + [v * a for a in Xr])
        A.append([w * a for a in Xr] + [0, 0, 0, 0] + [-u * a for a in Xr])
    P = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 4)
    P = np.linalg.inv(Tx) @ P @ TX
    return P / np.linalg.norm(P[2, :3])


def project(P, X):
    """(N, 3) VICON mm -> (N, 2) image px."""
    h = np.c_[X, np.ones(len(X))] @ P.T
    return h[:, :2] / h[:, 2:3]


def refine_P(P, X, x):
    """Least-squares refinement of P on the reprojection error (soft-L1 loss, scale 3 px)."""
    r = least_squares(lambda p: (project(p.reshape(3, 4), X) - x).ravel(), P.ravel(),
                      loss="soft_l1", f_scale=3.0)
    P = r.x.reshape(3, 4)
    return P / np.linalg.norm(P[2, :3])


def fit_robust(X, x, iters=200, thr=25.0, seed=0):
    """RANSAC over 8-point DLTs (inlier: reprojection error < thr px), then DLT and refine_P on the inliers."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i = rng.choice(len(X), 8, replace=False)
        try:
            P = dlt(X[i], x[i])
        except np.linalg.LinAlgError:
            continue
        inl = np.linalg.norm(project(P, X) - x, axis=1) < thr
        if best is None or inl.sum() > best[1].sum():
            best = (P, inl)
    P = refine_P(dlt(X[best[1]], x[best[1]]), X[best[1]], x[best[1]])
    return P


def px_per_mm(P, X):
    """Local image scale at 3D points X (N, 3): mean of the two singular values of the 2x3
    projection Jacobian, the px per mm in the plane facing the camera."""
    base = project(P, X)
    J = np.stack([project(P, X + e) - base for e in np.eye(3)], 2)      # (N, 2, 3)
    return np.linalg.svd(J, compute_uv=False).mean(1)


def fit_affine3(X, y):
    """3D -> 2D affine (4x2) by least squares, two passes trimming the worst 10 %."""
    A = np.c_[X, np.ones(len(X))]
    keep = np.ones(len(X), bool)
    for _ in range(2):
        M = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        r = np.linalg.norm(A @ M - y, axis=1)
        keep = r <= np.percentile(r, 90)
    return M


def apply_affine3(M, X):
    return np.c_[X, np.ones(len(X))] @ M
