"""Load M4 as a multi-class TS-classification benchmark.

M4 is natively a forecasting benchmark with five frequency groups
(Daily / Weekly / Monthly / Quarterly / Yearly). Federated AutoML in
this project works on per-sample classification or regression, so we
reshape M4 into a classification task:

* each M4 series is one sample,
* label = frequency group index (0..4),
* every series is reduced to a fixed ``window_length`` by truncating
  or zero-padding from the tail.

Why by-frequency: this is a meaningful classification signal
(business cycles at different scales look different), is balanced by
construction, and has enough samples across groups to push the
combined dataset above the RAF activation threshold
(``BATCH_SIZE_FOR_FEDOT_WORKER = 1000``) without any synthetic
inflation.

Notes on loading: we do NOT go through
``datasetsforecast.m4.M4.load`` because that function ``pd.melt``'s
the wide per-group CSV into a (n_series × max_length)-row long
DataFrame. On Daily alone this is ~42 M rows / >1 GiB of peak
memory, which reliably OOMs on laptops. Instead we read the wide
CSV directly with ``pd.read_csv`` and iterate rows: each row is one
series with NaN padding at the tail. First run still triggers the
(one-time) download of the missing group CSVs via ``M4.download``.
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4
from sklearn.model_selection import train_test_split

from fedot_ind.tools.serialisation.path_lib import EXAMPLES_DATA_PATH


# Order is used as the integer label for each group.
M4_GROUPS = ('Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly')

_CACHE_DIR = os.path.join(EXAMPLES_DATA_PATH, 'm4', 'datasets')


def load_m4_classification(
        n_per_group: int = 400,
        window_length: int = 50,
        test_size: float = 0.3,
        random_state: int = 42,
        standardize: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray]]:
    """Assemble a multi-class classification task from M4 series.

    Args:
        n_per_group: how many series to sample per frequency group.
            Clamped down to the group's actual size (Weekly has only
            ~359 unique series). With the default ``400 x 5 = 2000``
            samples the train split (~1400 rows at ``test_size=0.3``)
            comfortably clears ``BATCH_SIZE_FOR_FEDOT_WORKER = 1000``.
        window_length: every series is reduced to this many points
            from its tail -- longer series are truncated, shorter
            ones are left-zero-padded.
        test_size: fraction of the combined pool held out for test
            via stratified split.
        random_state: seed for series sampling and for the split.
        standardize: z-score each series using its own mean / std.

    Returns:
        ``((X_train, y_train), (X_test, y_test))`` with
        ``X`` of shape ``(N, window_length)`` ``float32`` and ``y``
        of shape ``(N,)`` ``int64``, labels in ``0 .. len(M4_GROUPS)-1``.
    """
    X_all, y_all = [], []

    for label, group in enumerate(M4_GROUPS):
        values = _read_wide_group_csv(group, max_rows=n_per_group)

        for row in values:
            ts = row[~np.isnan(row)].astype(np.float32, copy=False)

            if standardize and ts.size > 1:
                std = float(ts.std()) or 1.0
                ts = (ts - float(ts.mean())) / std

            if ts.size < window_length:
                pad = np.zeros(window_length - ts.size, dtype=np.float32)
                ts = np.concatenate([pad, ts])
            else:
                ts = ts[-window_length:]

            X_all.append(ts)
            y_all.append(label)

    X = np.stack(X_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return (X_train, y_train), (X_test, y_test)


def _read_wide_group_csv(group: str, max_rows: int) -> np.ndarray:
    """Return at most ``max_rows`` series from an M4 group as a
    ``(n_series, max_length)`` ``float32`` matrix with NaN padding
    at the tail.

    Reads only the first ``max_rows`` rows of the wide CSV via
    :func:`pandas.read_csv` ``nrows`` -- the full Monthly / Quarterly
    / Yearly files are tens-of-thousands of series wide by thousands
    of timesteps, so ``pd.read_csv`` on the whole thing allocates
    1+ GiB of ``float64`` and OOMs on laptops. We only ever need
    ``n_per_group`` rows, so capping them at the reader level keeps
    peak memory proportional to ``n_per_group``.

    If the CSV isn't cached yet, triggers the one-time download of
    that group via :meth:`M4.download` (does not melt).
    """
    csv_path = os.path.join(_CACHE_DIR, f'{group}-train.csv')
    if not os.path.exists(csv_path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        M4.download(directory=EXAMPLES_DATA_PATH, group=group)

    # Wide M4 layout: first column is the series id (``V1`` after
    # pandas auto-naming), remaining ``V2..VN`` are the time steps.
    df = pd.read_csv(csv_path, nrows=max_rows, dtype={'V1': 'string'},
                     low_memory=False)
    id_col = df.columns[0]
    # Force the value columns into float32 after loading; they may come
    # back as object dtype if the first non-ID cell is ambiguous.
    values = df.drop(columns=[id_col]).to_numpy(dtype=np.float32)
    return values
