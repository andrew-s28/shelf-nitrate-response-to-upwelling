# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "xarray[accel,io,parallel]",
# ]
# ///

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy import signal as sig

if TYPE_CHECKING:
    from numpy import double, int_
    from numpy.typing import NDArray

SCRIPT_DIR = Path().resolve()
DATA_DIR = SCRIPT_DIR / "../data/"

# dataset file names
VELOCITY_FILE = list(
    Path(DATA_DIR / "NH10_Mooring_Data").glob("nh10_hourly_data_1997_2021_part*.nc")
)
VELOCITY_SAVE_FILE = Path(
    "NH10_Mooring_Data/nh10_hourly_data_1997_2021_rotated_filtered.nc"
)


def princax(
    u: NDArray[double | int_] | xr.DataArray, v: NDArray[double | int_] | xr.DataArray
) -> tuple[double, double, double]:
    """
    Determines the principal axis of variance for the east and north velocities defined by u and v

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
    C = np.cov(uf, vf)

    # calculate principal axis angle (ET, Equation 4.3.23b)
    # > 0 CCW from east axis, < 0 CW from east axis
    theta = 0.5 * np.rad2deg(np.arctan2(2.0 * C[0, 1], (C[0, 0] - C[1, 1])))
    # switch to > 0 CW from north axis, < 0 CCW from north axis
    if theta >= 0:
        theta = 90 - theta
    elif theta < 0:
        theta = -(90 + theta)

    # calculate variance along major and minor axes (Equation 4.3.24)
    term1 = C[0, 0] + C[1, 1]
    term2 = ((C[0, 0] - C[1, 1]) ** 2 + 4 * (C[0, 1] ** 2)) ** 0.5
    major = np.sqrt(0.5 * (term1 + term2))
    minor = np.sqrt(0.5 * (term1 - term2))

    return theta, major, minor


def rot(
    u: NDArray[double | int_] | xr.DataArray,
    v: NDArray[double | int_] | xr.DataArray,
    theta: float | int | double | int_,
) -> tuple[NDArray[double | int_], NDArray[double | int_]]:
    """
    Rotates a vector counter clockwise or a coordinate system clockwise
    Designed to be used with theta output from princax(u, v)

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


velocity = xr.open_mfdataset(
    VELOCITY_FILE,
)
velocity = velocity.squeeze()
# rename for convienience
velocity = velocity.rename(
    {
        "eastward_velocity": "u",
        "northward_velocity": "v",
    }
)

# get filtering weights for 33 hour low pass filter - assumes 1 hour time step in data
wts = sig.firwin(120, 1 / 33, window="lanczos", fs=1)

# apply filter to east and north velocities
u_filt = sig.filtfilt(wts, 1, velocity["u"].values)
v_filt = sig.filtfilt(wts, 1, velocity["v"].values)
velocity["u_filt"] = (["depth", "time"], u_filt)
velocity["v_filt"] = (["depth", "time"], v_filt)

# compute cross-shore and along-shore velocities based on meandering along-shelf flow as in McCabe et al. (2015)
phi = np.arctan2(
    np.nanmean(velocity["v_filt"], axis=0), np.nanmean(velocity["u_filt"], axis=0)
)
rot_array = np.array([[[np.cos(p), np.sin(p)], [-np.sin(p), np.cos(p)]] for p in phi])
vel = np.einsum("ijk->jki", np.array([velocity["u_filt"], velocity["v_filt"]]))
ns = np.array([vt @ r for vd in vel for vt, r in zip(vd, rot_array)]).reshape(vel.shape)
n = ns[:, :, 0]
s = ns[:, :, 1]
uproj = np.array([np.sin(np.abs(phi)) * nd for nd in n])
velocity["cs"] = (["depth", "time"], uproj)

# compute cross-shore and along-shore velocities based on principal axis of variance
evel_filt = sig.filtfilt(wts, 1, velocity["u"].values)
nvel_filt = sig.filtfilt(wts, 1, velocity["v"].values)
theta, major, minor = princax(
    np.nanmean(evel_filt, axis=1), np.nanmean(nvel_filt, axis=1)
)
cs_vel, as_vel = rot(evel_filt, nvel_filt, theta)

velocity["cs_total"] = (["depth", "time"], cs_vel)
velocity["as"] = (["depth", "time"], as_vel)

# remove depth average from cross-shore velocity
velocity["cs"] = velocity["cs_total"] - velocity["cs_total"].mean(dim="depth")

velocity = velocity.resample(time="1D").mean()
velocity.attrs["created_by"] = "make_datasets.py"
velocity.attrs["created_on"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
velocity.attrs["description"] = (
    "Velocities from Stitch In Time dataset first filtered with a 33 hour low pass filtered using a Lanczos window."
    "Cross-shore and along-shore velocities calculated based on the principal axis of variance."
    "Velocities resampled to daily mean."
)
velocity.attrs["theta"] = theta

velocity.to_netcdf(DATA_DIR / VELOCITY_SAVE_FILE)
