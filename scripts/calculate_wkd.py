"""Calculates and saves a one-sided exponentially decaying wind stress W_{kd} as in Austin and Barth (2002)."""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "tqdm",
#     "xarray[io,parallel,accel]",
# ]
# ///

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.integrate import simpson
from tqdm import tqdm as tq

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy import floating, int_
    from numpy.typing import NBitBase, NDArray

    T = floating[TypeVar("T", bound=NBitBase)]


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "../data/"

# dataset file names
WIND_FILE = Path("NDBC_46050/46050_wind_binned.nc")
WIND_SAVE_FILE = Path("NDBC_46050/46050_wind_binned_with_w5d_w8d.nc")


def ws_integrand(
    tp: NDArray[floating] | NDArray[int_],
    t: float,
    tau: NDArray[floating] | NDArray[int_],
    k: float,
    rho: float = 1025,
) -> NDArray[floating] | NDArray[int_]:
    """Integrand for computation of k-day exponentially weighted integral of wind stress. See Austin and Barth, 2002.

    Args:
        tp (array): integration variable, time
        t (scalar): upper limit of integration, time
        tau (array): wind stress array with same lenth as times tp
        k (scalar): relaxation timescale, same units as time
        rho (scalar, optional): Density of sea water. Defaults to 1000.

    Returns:
        array: integrand for use in scipy.integrate and computation of W_kd

    """
    return tau[: t + 1] / rho * np.exp((tp[: t + 1] - t) / k)


###############################
#        Wind Dataset         #
###############################

wind = xr.open_dataset(DATA_DIR / WIND_FILE, decode_timedelta=True)

# Interpolate small gaps (up to 3 days) in wind data for W5d and W8d calculation
wind = wind.interpolate_na(dim="time", max_gap=np.timedelta64(3, "D"), use_coordinate="time")

# Resample to daily mean
wind = wind.resample(time="1D").mean()

# calculate day number for use in integration for w5d and w8d
wind["day_num"] = (["time"], np.arange(len(wind.time)))

# compute w8d
avg_len = 8
fout = np.nan * np.zeros(len(wind["day_num"]))
for i, _f in enumerate(tq(fout, desc="Calculating W8d")):
    temp = ws_integrand(
        wind["day_num"].values[i - avg_len * 5 : i],
        wind["day_num"].values[i],
        wind["coare_y"].values[i - avg_len * 5 : i],
        avg_len,
        rho=1,
    )
    mask = ~np.isnan(temp)
    if temp.size == 0:
        continue
    if np.any(np.isnan(wind.coare_y[i - avg_len * 5 : i])):
        continue
    fout[i] = simpson(temp[mask], x=wind["day_num"].values[i - avg_len * 5 : i][mask]) / avg_len
wind["w8d"] = (["time"], fout)

# compute w5d
avg_len = 5
fout = np.nan * np.zeros(len(wind["day_num"]))
for i, _f in enumerate(tq(fout, desc="Calculating W5d")):
    temp = ws_integrand(
        wind["day_num"].values[i - avg_len * 5 : i],
        wind["day_num"].values[i],
        wind["coare_y"].values[i - avg_len * 5 : i],
        avg_len,
        rho=1,
    )
    mask = ~np.isnan(temp)
    if temp.size == 0:
        continue
    if np.any(np.isnan(wind.coare_y[i - avg_len * 5 : i])):
        continue
    fout[i] = simpson(temp[mask], x=wind["day_num"].values[i - avg_len * 5 : i][mask]) / avg_len
wind["w5d"] = (["time"], fout)

wind.attrs["created_by"] = "make_datasets.py"
wind.attrs["created_on"] = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")

wind.to_netcdf(
    DATA_DIR / WIND_SAVE_FILE,
)
