# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "xarray[accel,io,parallel]",
# ]
# ///

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy import signal as sig

if TYPE_CHECKING:
    from typing import TypeVar

    from numpy import double, float64, floating
    from numpy.typing import NBitBase, NDArray

    T = floating[TypeVar("T", bound=NBitBase)]


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "../data/"

# dataset file names
VELOCITY_FILE = DATA_DIR / "NH10_Mooring_Data/nh10_hourly_data_1997_2023_v4.nc"
VELOCITY_SAVE_FILE = (
    DATA_DIR
    / "NH10_Mooring_Data/nh10_hourly_data_1997_2023_rotated_filtered_streamwise_v4.nc"
)


def princax(
    u: NDArray[double] | xr.DataArray, v: NDArray[double] | xr.DataArray,
) -> tuple[double, double, double]:
    """Determine the principal axis of variance for the east and north velocities defined by u and v.

    Args:
        u (scalar or array): east velocity
        v (scalar or array): north velocity

    Returns:
        tuple of scalar: (theta, major, minor) - the angle of the principal axis CW from north,
            the variance along the major axis, and the variance along the minor axis

    """
    if isinstance(u, xr.DataArray):
        # convert to numpy array
        u = u.values
    if isinstance(v, xr.DataArray):
        v = v.values
    u = np.asarray(u)
    v = np.asarray(v)

    # only use finite values for covariance matrix
    ii = np.isfinite(u + v)
    uf = u[ii]
    vf = v[ii]

    # compute covariance matrix
    cov = np.cov(uf, vf)

    # calculate principal axis angle (ET, Equation 4.3.23b)
    # > 0 CCW from east axis, < 0 CW from east axis
    theta = 0.5 * np.rad2deg(np.arctan2(2.0 * cov[0, 1], (cov[0, 0] - cov[1, 1])))
    # switch to > 0 CW from north axis, < 0 CCW from north axis
    if theta >= 0:
        theta = 90 - theta
    elif theta < 0:
        theta = -(90 + theta)

    # calculate variance along major and minor axes (Equation 4.3.24)
    term1 = cov[0, 0] + cov[1, 1]
    term2 = ((cov[0, 0] - cov[1, 1]) ** 2 + 4 * (cov[0, 1] ** 2)) ** 0.5
    major = np.sqrt(0.5 * (term1 + term2))
    minor = np.sqrt(0.5 * (term1 - term2))

    return theta, major, minor


def rot(
    u: NDArray[T] | xr.DataArray,
    v: NDArray[T] | xr.DataArray,
    theta: float | double | floating,
) -> tuple[NDArray[T], NDArray[T]]:
    """Rotates a vector counter clockwise or a coordinate system clockwise.

    Designed to be used with theta output from princax(u, v).

    Args:
        u (scalar or array): x-component of vector
        v (scalar or array): y-component of vector
        theta (scalar): rotation angle (CCW > 0, CW < 0)

    Returns:
        tuple of scalar or array: (ur, vr) - x and y components of vector in rotated coordinate system

    """
    # convert to numpy array
    if isinstance(u, xr.DataArray):
        u = u.values
    if isinstance(v, xr.DataArray):
        v = v.values
    u = np.asarray(u)
    v = np.asarray(v)

    # rotate vector according to angle theta
    w = u + 1j * v
    ang = np.deg2rad(theta)
    wr = w * np.exp(1j * ang)
    ur = np.real(wr)
    vr = np.imag(wr)
    return ur, vr


velocity = xr.open_mfdataset(VELOCITY_FILE)
velocity = velocity.squeeze()
# rename for convienience, unless already renamed
with suppress(ValueError):
    velocity = velocity.rename(
        {
            "eastward_velocity": "u",
            "northward_velocity": "v",
        },
    )

# velocity["u_interp"] = velocity["u"].copy(deep=True)
# velocity["v_interp"] = velocity["v"].copy(deep=True)

# # Split out OOI times for custom bottom interpolation
# OOI_TIME = slice(np.datetime64("2015-04-01"), None)
# velocity_ooi = velocity.sel(time=OOI_TIME)
# # interpolate to set bottom velocities for OOI times
# velocity_ooi["u_interp"][-1] = np.full(velocity_ooi["u"].shape[-1], 0.01)  # set the last depth to value
# velocity_ooi["v_interp"][-1] = np.full(velocity_ooi["v"].shape[-1], 0)  # set the last depth to value
# velocity_ooi = velocity_ooi.interpolate_na(dim="depth", method="linear", max_gap=10)

# velocity["u_interp"].loc[{"time": OOI_TIME}] = velocity_ooi["u_interp"]
# velocity["v_interp"].loc[{"time": OOI_TIME}] = velocity_ooi["v_interp"]

# get filtering weights for 40 hour low pass filter - assumes 1 hour time step in data
wts: NDArray[float64] = sig.firwin(101, 1 / 40, window="lanczos", fs=1)

# compute cross-shore and along-shore velocities based on principal axis of variance
evel_filt: NDArray[float64] = sig.filtfilt(wts, 1, velocity["u"].values, axis=1)
nvel_filt: NDArray[float64] = sig.filtfilt(wts, 1, velocity["v"].values, axis=1)
theta, major, minor = princax(
    np.nanmean(evel_filt, axis=1), np.nanmean(nvel_filt, axis=1),
)
cs_vel, as_vel = rot(evel_filt, nvel_filt, theta)
velocity["u_filt"] = (["depth", "time"], evel_filt)
velocity["v_filt"] = (["depth", "time"], nvel_filt)
velocity["cs"] = (["depth", "time"], cs_vel)
velocity["cs"] -= velocity["cs"].mean(
    dim="depth", keep_attrs=True,
)  # remove depth average
velocity["as"] = (["depth", "time"], as_vel)

# compute cross-shore and along-shore velocities based on meandering along-shelf flow as in McCabe et al. (2015)
phi = np.arctan2(np.nanmean(as_vel, axis=0), np.nanmean(cs_vel, axis=0))
u_n = np.array(
    [
        -u * np.sin(p) + v * np.cos(p)
        for u, v, p in zip(cs_vel.T, as_vel.T, phi, strict=True)
    ],
).T

# use masked array for dot product to avoid NaN issues
u_n_m = np.ma.array(u_n, mask=np.isnan(u_n))
u_m = np.ma.array(cs_vel, mask=np.isnan(cs_vel))
u_p = np.ma.array(
    [
        (np.ma.dot(un, u) / np.ma.dot(u, u)) * u
        for u, un in zip(u_m.T, u_n_m.T, strict=True)
    ],
).T

velocity["cs_proj"] = (["depth", "time"], u_p)

velocity.attrs["created_by"] = "make_datasets.py"
velocity.attrs["created_on"] = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
velocity.attrs["description"] = (
    "Velocities from Stitch In Time dataset first filtered with a 33 hour low pass filtered using a Lanczos window."
    "Cross-shore and along-shore velocities calculated based on the principal axis of variance."
    "Velocities resampled to daily mean."
)
velocity.attrs["theta"] = theta

velocity.to_netcdf(VELOCITY_SAVE_FILE)
