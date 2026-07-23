# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "scipy",
#     "xarray[accel,io,parallel]",
# ]
# ///
"""This script converts the NH10 ADCP velocity data from a .mat file to a NetCDF file."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import scipy.io as sio
import xarray as xr

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "../data/"
MAT_VELOCITY_FILE = DATA_DIR / "NH10_Mooring_Data/ADCP_NH10_1997_2023_V4.mat"

mat = sio.loadmat(MAT_VELOCITY_FILE, squeeze_me=True)
mat.pop("__header__")
mat.pop("__version__")
mat.pop("__globals__")

python_datetime = [
    datetime.fromordinal(int(t)) + timedelta(days=t % 1) - timedelta(days=366)
    for t in mat["time"]
]
times = [
    (np.datetime64(t) + np.timedelta64(30, "s"))
    .astype("datetime64[m]")
    .astype("datetime64[ns]")
    for t in python_datetime
]

ds = xr.Dataset(
    {
        "u": (["depth", "time"], mat["u"]),
        "v": (["depth", "time"], mat["v"]),
    },
    coords={
        "time": times,
        "depth": mat["depth"],
    },
)

ds.to_netcdf(
    DATA_DIR / "NH10_Mooring_Data/nh10_hourly_data_1997_2023_v4.nc",
)
