"""Small datasets for reproducing the paper's experiments.

- `abalone_aging_split`: a simulated cohort observed one year apart, from the
  bundled UCI abalone data (Section VII of the paper).
- UCI time-series loaders (Air Quality, Appliances, Electricity) with an
  on-disk cache and same-day/higher-hour subsets for the transport
  characterization benchmark of Section IV-B.
- `synthetic_ts`: a controllable periodic multivariate time series enabling an
  analytic ground-truth relevance (equation (6)).
- `synthetic_domains`: two labeled domains with spurious (domain-shifted)
  features for the domain-alignment use case (Section V analog).

All loaders are small by construction: subsets are at most a few thousand
rows so the whole suite runs comfortably on a laptop.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import io
import os
import urllib.request
import zipfile

import numpy as np

__all__ = [
    "load_abalone",
    "abalone_aging_split",
    "ABALONE_FEATURES",
    "pkg_data_path",
    "cache_dir",
    "fetch_air_quality",
    "fetch_appliances",
    "fetch_electricity",
    "ts_shift",
    "synthetic_ts",
    "synthetic_domains",
    "standardize",
    "preprocess_series",
    "preprocess_tabular",
    "load_musk1",
    "load_wisconsin",
    "load_wine_quality",
    "SAMPLES_PER_HOUR",
]

ABALONE_FEATURES = [
    "Length",
    "Diameter",
    "Height",
    "WholeWeight",
    "ShuckedWeight",
    "VisceraWeight",
    "ShellWeight",
]

ABALONE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
ABALONE_MD5 = "a769fd0119787cac09158fe08971e480"

#: Minimum fraction of non-missing values for an Air Quality channel to be kept.
AIR_QUALITY_MIN_COVERAGE = 0.9

#: Samples per hour of each periodic dataset, so a delay given in hours (as in
#: Table III) can be converted to the sample offset ``ts_shift`` expects.
SAMPLES_PER_HOUR = {"air-quality": 1, "appliances": 6, "electricity": 4}


def pkg_data_path(name: str) -> str:
    """Absolute path of a small data file bundled inside the package."""
    return str(importlib.resources.files("wax") / "data" / name)


def cache_dir() -> str:
    """Directory for datasets downloaded at run time.

    Downloads must not be written inside the installed package, so they go to
    ``$WAX_CACHE`` if set and to ``~/.cache/wax`` otherwise.
    """
    path = os.environ.get("WAX_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "wax"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv_float(
    url_or_path: str,
    skiprows: int = 0,
    usecols: list[int] | None = None,
    max_rows: int | None = None,
    delimiters: tuple[str, ...] = (",", ";"),
) -> np.ndarray:
    """Read a numeric CSV into a float array, trying a set of delimiters.

    The semicolon-delimited UCI files (Air Quality, Electricity) are written in
    an Italian/Portuguese locale that uses the comma as the *decimal* separator,
    so commas are rewritten to points before parsing those.  Candidate parses
    are then scored by how many finite values they produce: ``np.genfromtxt``
    returns an all-NaN array for the wrong delimiter rather than raising, and
    silently losing whole columns that way is the exact failure this guards
    against.
    """
    if hasattr(url_or_path, "seek"):
        url_or_path.seek(0)
        raw = url_or_path.read()
    elif os.path.exists(url_or_path):
        with open(url_or_path, "rb") as f:
            raw = f.read()
    else:
        with urllib.request.urlopen(url_or_path) as f:
            raw = f.read()

    raw = raw.replace(b'"', b"")  # UCI Appliances quotes even its numeric fields
    best, best_finite, errors = None, -1, []
    for delim in delimiters:
        body = raw.replace(b",", b".") if delim == ";" else raw
        try:
            data = np.asarray(
                np.genfromtxt(
                    io.BytesIO(body),
                    skip_header=skiprows,
                    usecols=usecols,
                    delimiter=delim,
                    max_rows=max_rows,
                    comments="#",
                    dtype=float,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append((delim, exc))
            continue
        finite = int(np.isfinite(data).sum())
        if finite > best_finite:
            best, best_finite = data, finite
    if best is None:
        raise ValueError(f"could not parse CSV with any delimiter: {errors}")
    return best


def load_abalone(path: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the bundled abalone data.

    Returns ``(X, sex, rings)`` where ``X`` holds the seven continuous
    features in :data:`ABALONE_FEATURES`.
    """
    if path is None:
        path = pkg_data_path("abalone.data")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(ABALONE_URL, path)
    if _md5(path) != ABALONE_MD5:
        raise RuntimeError(f"abalone data at {path} failed its checksum")
    rows = []
    sex, rings = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            sex.append(parts[0])
            rows.append([float(v) for v in parts[1:8]])
            rings.append(int(parts[8]))
    X = np.asarray(rows, dtype=float)
    return X, np.asarray(sex), np.asarray(rings, dtype=int)


def abalone_aging_split(
    n: int = 250,
    seed: int = 0,
    ring_lo: int = 6,
    ring_hi: int = 15,
    path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """A simulated abalone cohort measured twice ~one year apart.

    Source abalones (rings in ``[ring_lo, ring_hi - 1]``) are matched to the
    most similar abalone one year older (rings + 1), so every row pair mimics
    the same individual observed twice.  Returns ``(X_source, X_target,
    feature_names)``.
    """
    X, _, rings = load_abalone(path)
    rng = np.random.default_rng(seed)
    source_pool = np.nonzero((rings >= ring_lo) & (rings <= ring_hi - 1))[0]
    chosen = rng.choice(source_pool, size=n, replace=False)
    target_idx = np.empty(n, dtype=int)
    for i, idx in enumerate(chosen):
        r = rings[idx]
        candidates = np.nonzero(rings == r + 1)[0]
        if len(candidates) == 0:
            candidates = np.nonzero(rings >= r + 1)[0]
        dist = np.linalg.norm(X[candidates] - X[idx], axis=1)
        target_idx[i] = candidates[int(np.argmin(dist))]
    return X[chosen], X[target_idx], list(ABALONE_FEATURES)


def _download_cached(url: str, cache_name: str, size_hint_mb: str = "small") -> str:
    target = os.path.join(cache_dir(), cache_name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    print(f"[datasets] downloading {url} ({size_hint_mb}) ...")
    urllib.request.urlretrieve(url, target)
    return target


def fetch_air_quality(
    cache_dir: str | None = None, interpolate: bool = False
) -> np.ndarray:
    """UCI Air Quality: hourly multisite pollutant readings (UCI 360).

    The 13 numeric channels are cut to those with at least
    ``AIR_QUALITY_MIN_COVERAGE`` non-missing values, which drops the four
    reference-analyser channels ``CO(GT)`` (6%), ``NMHC(GT)`` (10%),
    ``NOx(GT)`` (82%) and ``NO2(GT)`` (82%) and leaves the five metal-oxide
    sensors, ``C6H6(GT)`` and the three meteorological channels: d = 9, which
    is the width reported in Table III.  Remaining ``-200`` sentinels are left
    as ``nan``: Supplementary Note E *removes* samples with missing values for
    the Section IV-B benchmark rather than filling them, so interpolation is
    opt-in.  Returns a float array of shape (n_hours, 9).
    """
    zip_path = _download_cached(
        "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "AirQualityUCI.zip",
    )
    arr = _read_zip_csv(zip_path, "AirQualityUCI.csv", skiprows=1, delimiters=(";",))
    X = arr[:, 2:15]
    X = np.where(X == -200.0, np.nan, X)
    X = X[np.isfinite(X).any(axis=1)]  # drop the trailing all-empty rows
    keep = np.mean(np.isfinite(X), axis=0) >= AIR_QUALITY_MIN_COVERAGE
    X = X[:, keep]
    if not interpolate:
        return X
    for i in range(X.shape[1]):
        col = X[:, i]
        nan = np.isnan(col)
        if nan.any():
            idx = np.arange(len(col))
            sample = np.nonzero(~nan)[0]
            if sample.size == 0:
                col[:] = 0.0
            else:
                col[nan] = np.interp(idx[nan], idx[sample], col[sample])
    return X


def fetch_appliances(cache_dir: str | None = None) -> np.ndarray:
    """UCI Appliances energy prediction (UCI 374), 10-minute resolution.

    Keeps ``Appliances`` plus the 24 temperature, humidity and weather
    channels, dropping the ``date`` string, ``lights`` and the two synthetic
    random columns ``rv1``/``rv2``: d = 25, the width reported in Table III.
    Returns a float array (144 samples per day).
    """
    zip_path = _download_cached(
        "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
        "energydata.zip",
    )
    cols = [1] + list(range(3, 27))  # Appliances, T1..RH_9, T_out..Tdewpoint
    arr = _read_zip_csv(zip_path, "energydata_complete.csv", skiprows=1, delimiters=(",",))
    return arr[:, cols]


def fetch_electricity(n_cols: int = 7, cache_dir: str | None = None) -> np.ndarray:
    """UCI Electricity Load Diagrams 2011-2014 (UCI 321), 15-minute resolution.

    Reads the first ``n_cols`` client columns over the full 2011-2014 span
    (~1400 days), which is the d = 7 / N ~ 1200 setting of Table III.  Clients
    that are all-zero over the period (they enter the panel later) are skipped.
    The archive is ~260 MB and is cached on disk; set ``ELECTRICITY_SKIP`` in
    the environment to make this raise instead of downloading.
    """
    if "ELECTRICITY_SKIP" in os.environ:
        raise RuntimeError("electricity download skipped via ELECTRICITY_SKIP")
    zip_path = _download_cached(
        "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        "LD2011_2014.zip",
        size_hint_mb="~260 MB",
    )
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".txt", ".csv")))
        buf = io.BytesIO(zf.read(name))
        # read a wide slice, then keep the first n_cols clients that are active
        arr = _read_csv_float(
            buf, skiprows=1, usecols=list(range(1, 41)), delimiters=(";",)
        )
    active = np.nonzero(np.nan_to_num(arr).sum(axis=0) > 0)[0][:n_cols]
    return arr[:, active]


def _read_zip_csv(
    zip_path: str, name: str, skiprows: int = 0, delimiters: tuple[str, ...] = (",", ";")
) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        member = name if name in zf.namelist() else zf.namelist()[0]
        buf = io.BytesIO(zf.read(member))
    return _read_csv_float(buf, skiprows=skiprows, delimiters=delimiters)


def ts_shift(
    X: np.ndarray,
    period: int,
    t: int,
    dt: int,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset the time series into source/target as in Section IV-B.

    With hourly/daily-stamped rows, ``DS = {x_{t + k*period}}`` and
    ``DT = {x_{t + dt + k*period}}`` for full days ``k`` with both samples
    available.  Returns ``(X_source, X_target, ground_truth)`` where the
    ground truth (equation (6)) is the per-feature mean squared displaced
    sample difference.
    """
    rows = X.shape[0]
    if t + dt >= period:
        raise ValueError("t + dt must be < period")
    n_days = rows // period
    k = np.arange(n_days)
    idx_s = k * period + t
    idx_t = k * period + t + dt
    mask = idx_t < rows
    if valid is not None:
        # Supplementary Note E: when a sample is dropped its coupled match is
        # dropped too, so the ground-truth coupling stays complete.
        mask = mask & valid[np.clip(idx_s, 0, rows - 1)] & valid[np.clip(idx_t, 0, rows - 1)]
    Xs = X[idx_s[mask]]
    Xt = X[idx_t[mask]]
    truth = np.mean((Xt - Xs) ** 2, axis=0)
    return Xs, Xt, truth


def synthetic_ts(
    n_days: int = 240,
    period: int = 24,
    d: int = 8,
    t: int = 7,
    dt: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic periodic multivariate series with an analytic ground truth.

    A full ``n_days * period`` series is generated and then cut into source and
    target by :func:`ts_shift`, exactly as the real time series are, so the
    delay ``dt`` genuinely changes the transport phenomenon.  Each feature is
    assigned a type: ``mean`` (an hour-dependent level, i.e. a translation
    between the two hours), ``var`` (an hour-dependent noise scale at constant
    level - invisible to MeanShift) or ``none`` (stationary across hours).

    Note that source and target here are *different samples of the same
    process*, not a deterministic per-row transformation of one another, so the
    optimal coupling is not the identity and the recovered relevance is not
    trivially equal to the ground truth.
    """
    rng = np.random.default_rng(seed)
    kinds = np.resize(
        np.array(["mean", "var", "mean", "none", "var", "none", "mean", "var"]), d
    )
    n = n_days * period
    hour = np.arange(n) % period
    phase = 2 * np.pi * hour / period
    base = rng.uniform(-2.0, 2.0, d)
    amp = rng.uniform(0.2, 2.0, d)
    off = rng.uniform(0.0, 2 * np.pi, d)
    X = np.empty((n, d))
    for i in range(d):
        if kinds[i] == "mean":
            X[:, i] = base[i] + amp[i] * np.sin(phase + off[i]) + rng.normal(0, 0.15, n)
        elif kinds[i] == "var":
            scale = 0.15 + amp[i] * (1.0 + np.sin(phase + off[i]))
            X[:, i] = base[i] + rng.normal(0, 1.0, n) * scale
        else:
            X[:, i] = base[i] + rng.normal(0, 0.15, n)
    return ts_shift(X, period=period, t=t, dt=dt)


def synthetic_domains(
    n: int = 300, d: int = 15, n_informative: int = 5, n_spurious: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Two labeled domains with spurious (domain-shifted) features.

    ``n_informative`` features correlate with the binary label in both domains
    (domain-invariant).  ``n_spurious`` features shift strongly between the
    domains (a 'batch effect') and carry no label information.  Returns
    ``(X_source, y_source, X_target, y_target, meta)`` with
    ``meta['roles']`` marking every feature as ``informative``, ``spurious``
    or ``noise``.
    """
    rng = np.random.default_rng(seed)
    ys = rng.integers(0, 2, n)
    yt = rng.integers(0, 2, n)
    roles = ["informative"] * n_informative + ["spurious"] * n_spurious + \
        ["noise"] * (d - n_informative - n_spurious)
    Xs = np.zeros((n, d))
    Xt = np.zeros((n, d))
    for i, role in enumerate(roles):
        if role == "informative":
            strength = rng.uniform(0.8, 1.6)
            Xs[:, i] = (ys - 0.5) * strength + rng.normal(0, 0.4, n)
            Xt[:, i] = (yt - 0.5) * strength + rng.normal(0, 0.4, n)
        elif role == "spurious":
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            shift = rng.uniform(1.5, 3.0)
            Xs[:, i] = sgn * shift + rng.normal(0, 0.3, n)
            Xt[:, i] = -sgn * shift + rng.normal(0, 0.3, n)
        else:
            Xs[:, i] = rng.normal(0, 1, n)
            Xt[:, i] = rng.normal(0, 1, n)
    meta = {"roles": np.asarray(roles), "d": d, "n_informative": n_informative,
            "n_spurious": n_spurious}
    return Xs, ys, Xt, yt, meta

def standardize(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score both matrices using the pooled mean and standard deviation.

    The SRG of (5) compares Wasserstein distances across feature subsets and is
    therefore *not* scale invariant: without this step a feature that merely
    happens to be recorded in larger units dominates every attribution and
    every baseline agrees with every other one.  Constant columns are left
    alone rather than divided by zero.
    """
    Z = np.vstack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float)])
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    return (X - mu) / sd, (Y - mu) / sd


def _uncompress_z(raw: bytes) -> bytes:
    """Decompress Unix ``.Z`` (LZW) data using the system ``gzip``.

    ponytail: a subprocess instead of a pure-python LZW decoder - both GNU and
    BSD ``gzip`` read ``.Z``, which covers Linux and macOS.  Add the ``unlzw3``
    dependency if Windows support is ever needed.
    """
    import subprocess

    return subprocess.run(
        ["gzip", "-dc"], input=raw, capture_output=True, check=True
    ).stdout


def load_musk1() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UCI Musk (Version 1), d = 166: source = non-musk, target = musk.

    Under the Note E filter this yields N/M = 133/167, matching Table I exactly.
    """
    zip_path = _download_cached(
        "https://archive.ics.uci.edu/static/public/74/musk+version+1.zip", "musk1.zip"
    )
    with zipfile.ZipFile(zip_path) as zf:
        raw = _uncompress_z(zf.read("clean1.data.Z"))
    arr = _read_csv_float(io.BytesIO(raw), usecols=list(range(2, 169)), delimiters=(",",))
    X, y = arr[:, :-1], arr[:, -1].astype(int)
    return X[y == 0], X[y == 1], [f"f{i}" for i in range(X.shape[1])]


def load_wisconsin() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Breast Cancer Wisconsin (Diagnostic), d = 30: source = benign, target = malignant.

    Shipped inside scikit-learn, so this one needs no network access.
    """
    try:
        from sklearn.datasets import load_breast_cancer
    except ImportError as exc:
        raise ImportError("load_wisconsin needs scikit-learn: pip install 'wax-ot[repro]'") from exc

    ds = load_breast_cancer()
    X, y = ds.data, ds.target  # 1 = benign, 0 = malignant
    return X[y == 1], X[y == 0], list(ds.feature_names)


def load_wine_quality() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UCI Wine Quality, d = 12: source = white wines, target = red wines.

    Figure S1 of the supplement shows WaX explaining "the shift between red and
    white wine", with ``quality`` among the plotted features, so the label
    column is kept and d = 12 as in Table I.
    """
    zip_path = _download_cached(
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip", "winequality.zip"
    )
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        header = zf.read("winequality-white.csv").decode().splitlines()[0]
        for member in ("winequality-white.csv", "winequality-red.csv"):
            arr = _read_csv_float(io.BytesIO(zf.read(member)), skiprows=1, delimiters=(";",))
            out.append(arr[:, :12])  # `quality` is kept as a feature: d = 12
    names = [c.strip('"') for c in header.split(";")][:12]
    return out[0], out[1], names


def preprocess_series(
    X: np.ndarray, max_missing: float = 0.5, outlier_sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Supplementary Note E preprocessing for the Section IV-B time series.

    Drops channels with too many missing values, standardizes to zero mean and
    unit variance, and flags as invalid any sample that is still missing a value
    or that deviates by more than ``outlier_sigma`` standard deviations in any
    channel.  Returns ``(X_standardized, valid)``; the flags are returned rather
    than applied because :func:`ts_shift` has to drop a sample's coupled match
    along with the sample itself, which it can only do once the day structure is
    known.
    """
    X = np.asarray(X, dtype=float)
    keep = np.mean(np.isfinite(X), axis=0) >= (1.0 - max_missing)
    X = X[:, keep]
    finite = np.isfinite(X).all(axis=1)
    mu = np.nanmean(X[finite], axis=0)
    sd = np.nanstd(X[finite], axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    Z = (X - mu) / sd
    valid = finite & (np.abs(np.nan_to_num(Z, nan=np.inf)) <= outlier_sigma).all(axis=1)
    return Z, valid


def preprocess_tabular(
    X: np.ndarray,
    Y: np.ndarray,
    max_missing: float = 0.5,
    outlier_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Supplementary Note E preprocessing for the Section IV-A tabular datasets.

    In the order the note gives: drop features with a high proportion of missing
    values, drop duplicate instances, drop strong outliers (any feature more
    than ``outlier_sigma`` standard deviations from its mean, measured on the
    pooled source and target), then standardize to zero mean and unit variance.

    ponytail: the note's final nearest-neighbour imputation step is not
    implemented - none of the datasets this repository covers (Musk1, Wine,
    Wisconsin) has a missing value left at that point.  Add ``KNNImputer`` here
    if Crime or Mice are ever added.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    pooled = np.vstack([X, Y])
    keep = np.mean(np.isfinite(pooled), axis=0) >= (1.0 - max_missing)
    X, Y = X[:, keep], Y[:, keep]

    X = np.unique(X, axis=0)
    Y = np.unique(Y, axis=0)

    pooled = np.vstack([X, Y])
    mu = np.nanmean(pooled, axis=0)
    sd = np.nanstd(pooled, axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    def inlier(A: np.ndarray) -> np.ndarray:
        return (np.abs((A - mu) / sd) <= outlier_sigma).all(axis=1)

    X, Y = X[inlier(X)], Y[inlier(Y)]
    return standardize(X, Y)
