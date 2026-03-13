# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
#     "xarray[accel,io,parallel]",
# ]
# ///

from __future__ import annotations

import warnings
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy import signal as sig

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "../data/"

# dataset file names
VELOCITY_FILE = DATA_DIR / "NH10_Mooring_Data/nh10_hourly_data_1997_2024_v5.nc"
VELOCITY_SAVE_FILE = DATA_DIR / "NH10_Mooring_Data/nh10_hourly_data_1997_2024_rotated_filtered_streamwise_v5.2.nc"

warnings.filterwarnings(
    "ignore",
    "Mean of empty slice",
    RuntimeWarning,
)


def extrapolate_bottom_velocity(
    velocity: xr.Dataset,
    max_depth: int = 70,
) -> xr.Dataset:
    """Extrapolate bottom velocity to zero using linear interpolation.

    Args:
        velocity (xr.Dataset): dataset containing east and north velocities
        max_depth (int): maximum depth for original data, default is 70 m

    Returns:
        xr.Dataset: dataset with bottom velocity linearly extrapolated to zero

    """
    velocity = velocity.where(velocity["depth"] <= max_depth)

    # we only want to set zeroes where there is data in the original
    # i.e., selecting for times when there is at least one valid velocity measurement in the original profile
    zeroes = xr.full_like(velocity.isel(depth=0), 0)
    zeroes = zeroes.where(~velocity.isnull().all(dim="depth"))

    # we also don't want to extrapolate if there isn't any valid data near the MAX_DEPTH we set earlier
    zeroes = zeroes.where(~velocity.isnull().sel(depth=max_depth, method="nearest"))

    # set the deepest value to zero
    velocity[{"depth": -1}] = zeroes

    # now use linear interpolation over a max gap of 20 m (2 m depth bins) to fill any remaining NaNs
    # at this point, only bottom values should be remaining NaN, so this will extrapolate only the bottom values
    velocity = velocity.interpolate_na(dim="depth", method="polynomial", order=1, limit=10)

    return velocity


def extrapolate_top_velocity(
    velocity: xr.Dataset,
    min_depth: int = 15,
) -> xr.Dataset:
    """Extrapolate top velocity using constant velocity extrapolation from the top-most depth.

    Args:
        velocity (xr.Dataset): dataset containing east and north velocities
        min_depth (int): minimum depth for original data, default is 15 m

    Returns:
        xr.Dataset (velocity) - dataset with top velocity extrapolated to constant value from the top-most depth

    """
    # now mask out the surface and depths
    velocity = velocity.where(velocity["depth"] >= min_depth)

    # bfill will backfill the surface values with the nearest valid value, only within 20 m (2 m depth bins)
    velocity = velocity.bfill(dim="depth", limit=10)

    return velocity


def princax(
    u: NDArray[floating] | xr.DataArray,
    v: NDArray[floating] | xr.DataArray,
) -> tuple[floating, floating, floating]:
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
    u: NDArray[floating] | xr.DataArray,
    v: NDArray[floating] | xr.DataArray,
    theta: float | floating,
) -> tuple[NDArray[floating], NDArray[floating]]:
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

# first interpolate any nan to fill gaps in the middle of the profiles
velocity = velocity.interpolate_na(dim="depth", method="linear", max_gap=10)

# now extrapolate top and bottom velocities
velocity = extrapolate_top_velocity(velocity)
velocity = extrapolate_bottom_velocity(velocity)

# resample to ensure hourly data for filtering, then transpose since it reorders coords for some reason
velocity = velocity.resample(time="1h").mean().transpose("depth", "time")

# save places where velocity is not null to reapply mask after filtering
mask = velocity["u"].notnull() | velocity["v"].notnull()

num_taps = 101  # window length
cutoff_freq = 1 / 33  # cutoff frequency in cycles per hour (for a 33 hour low pass filter)
wts = xr.DataArray(sig.firwin(num_taps, cutoff_freq, window="lanczos", fs=1), dims=["time_win"])

# filter east/north velocities with zero phase shift filter along time axis
velocity["u_filt"] = (
    # apply filter once in the forward direction
    velocity["u"]
    .fillna(0)
    .rolling(time=num_taps, center=True)
    .construct(time="time_win")
    .dot(wts)
    # apply filter again in the backward direction to achieve zero phase shift
    .isel(time=slice(None, None, -1))
    .rolling(time=num_taps, center=True)
    .construct(time="time_win")
    .dot(wts)
)
velocity["v_filt"] = (
    # apply filter once in the forward direction
    velocity["v"]
    .fillna(0)
    .rolling(time=num_taps, center=True)
    .construct(time="time_win")
    .dot(wts)
    # apply filter again in the backward direction to achieve zero phase shift
    .isel(time=slice(None, None, -1))
    .rolling(time=num_taps, center=True)
    .construct(time="time_win")
    .dot(wts)
)
velocity["u_filt"] = velocity["u_filt"].where(mask)
velocity["v_filt"] = velocity["v_filt"].where(mask)

# compute cross-shore and along-shore velocities based on principal axis of variance of depth mean velocities
theta, major, minor = princax(
    velocity["u_filt"].mean(dim="depth").values,
    velocity["v_filt"].mean(dim="depth").values,
)

# rotate into new coordinate system
cs_vel, as_vel = rot(velocity["u_filt"].values, velocity["v_filt"].values, theta)

velocity["u_cs"] = (velocity["u_filt"].dims, cs_vel)
velocity["u_as"] = (velocity["v_filt"].dims, as_vel)

# compute cross-shore and along-shore velocities based on meandering along-shelf flow as in McCabe et al. (2015)
# first find the time-varying angle of the strongest depth mean flow
phi = xr.ufuncs.arctan2(
    velocity["u_as"].mean(dim="depth"),
    velocity["u_cs"].mean(dim="depth"),
)

# now compute the velocity component normal to that flow (Eqn. 3 in McCabe et al. 2015)
u_n = -velocity["u_cs"] * xr.ufuncs.sin(phi) + velocity["u_as"] * xr.ufuncs.cos(phi)

# project normal component back into cross-shelf direction
velocity["u_proj"] = u_n * -xr.ufuncs.sin(phi)

# finally, resample to daily means
velocity = velocity.resample(time="1D").mean()

velocity.attrs["created_by"] = "make_velocity_datasets.py"
velocity.attrs["created_on"] = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
velocity.attrs["description"] = (
    "Velocities from Stitch In Time dataset first filtered with a 33 hour low pass filtered using a Lanczos window."
    "Cross-shore and along-shore velocities calculated based on the principal axis of variance."
    "Velocities resampled to daily mean."
)
velocity.attrs["theta"] = theta

velocity.to_netcdf(VELOCITY_SAVE_FILE)
