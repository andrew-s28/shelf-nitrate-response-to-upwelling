"""A script to download NDBC data, compute wind stress, rotate into principal axes, and daily average."""

import argparse
import io
import re
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import xarray as xr
from bs4 import BeautifulSoup
from numpy import floating
from numpy.typing import NDArray
from pycoare import coare_35 as c35
from scipy import signal
from tqdm import tqdm as tq


def search_for_float(pattern: str, string: str, not_found_msg: str = "Value not found.") -> float:
    """Search for a float value in a string using a regex pattern.

    Args:
        pattern (str): regex pattern to search for
        string (str): string to search within
        not_found_msg (str, optional): message to print if value not found. Defaults to "Value not found.".

    Returns:
        float or None: the found float value, or None if not found

    """
    m_var = re.search(pattern + r"(sea level|\d+\.?\d*)", string)
    if m_var:
        m_val = re.search(r"sea level|\d+\.?\d*", m_var.string[m_var.start() : m_var.end()])
        if m_val:
            if m_val.string[m_val.start() : m_val.end()] == "sea level":
                return 0.0
            return float(m_val.string[m_val.start() : m_val.end()])
    print(not_found_msg)
    return np.nan


def ndbc_heights(url: str) -> tuple[float, float, float, float, float, float, float]:
    """Obtain station metadata from NDBC site stations, since they don't include it in their metadata.

    Args:
        url (str): URL of NDBC station page

    Raises:
        RuntimeError: If an incorrect station number is put in or the page cannot be retrieved.

    Returns:
        tuple of scalar: site elevation, air temp height, anemometer height,
            barometer elevation, sea temp depth, water depth, and watch circle radius, all in meters.

    """
    with requests.session() as s:
        page = s.get(url).text
    soup = BeautifulSoup(page, "html.parser")

    if soup.title is None or "Station not found" in str(soup.title.string):
        msg = f"Could not retrieve station page at URL: {url}"
        raise RuntimeError(msg)

    station_metadata = soup.select_one("div#stn_metadata>p")
    station_metadata_stripped = station_metadata.text.strip() if station_metadata else None
    if station_metadata_stripped is not None:
        site_el = search_for_float(
            r"Site elevation: ",
            station_metadata_stripped,
            not_found_msg="No site elevation found.",
        )
        air_val = search_for_float(
            r"Air temp height: ",
            station_metadata_stripped,
            not_found_msg="No air temp height found.",
        )
        ane_val = search_for_float(
            r"Anemometer height: ",
            station_metadata_stripped,
            not_found_msg="No anemometer height found.",
        )
        bar_val = search_for_float(
            r"Barometer elevation: ",
            station_metadata_stripped,
            not_found_msg="No barometer height found.",
        )
        sea_val = search_for_float(
            r"Sea temp depth: ",
            station_metadata_stripped,
            not_found_msg="No sea temperature depth found.",
        )
        dep_val = search_for_float(r"Water depth: ", station_metadata_stripped, not_found_msg="No water depth found.")
        rad_val = (
            search_for_float(
                r"Watch circle radius: ",
                station_metadata_stripped,
                not_found_msg="No watch circle radius found.",
            )
            / 1.094  # convert from yards to meters
        )
    else:
        site_el, air_val, ane_val, bar_val, sea_val, dep_val, rad_val = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    # set to default of 10 if not changed in html search
    if np.isnan(site_el):
        print("No site elevation found.")
    if np.isnan(air_val):
        print("No air temp height found.")
    if np.isnan(ane_val):
        print("No anemometer height found.")
    if np.isnan(bar_val):
        print("No barometer height found.")
    if np.isnan(sea_val):
        print("No sea temperature depth found.")
    if np.isnan(dep_val):
        print("No water depth found.")
    if np.isnan(rad_val):
        print("No watch circle radius found.")

    return site_el, air_val, ane_val, bar_val, sea_val, dep_val, rad_val


def list_files(url: str, tag: str = r".*\.nc$") -> list[str]:
    r"""Create a list of the netCDF data files in the THREDDS catalog created by a request to the M2M system.

    Obtained from 2022 OOIFB workshop.

    Args:
        url (str): URL to a THREDDS catalog specific to a data request
        tag (regexp, optional): Regex pattern used to distinguish files of interest. Defaults to r'.*\\.nc$'.

    Returns:
        array: list of files in the catalog with the URL path set relative to the catalog

    """
    with requests.session() as s:
        page = s.get(url).text

    soup = BeautifulSoup(page, "html.parser")
    pattern = re.compile(tag)
    nc_files = [node.get("href") for node in soup.find_all("a", string=pattern)]  # ty:ignore[unresolved-attribute]

    nc_files = [re.sub(r"catalog\.html\?dataset=", r"", str(file)) for file in nc_files]
    return nc_files


def princax(u: NDArray[floating], v: NDArray[floating]) -> tuple[float, float, float]:
    """Determine the principal axis of variance for the east and north velocities defined by u and v.

    Args:
        u (scalar or array): east velocity
        v (scalar or array): north velocity

    Returns:
        tuple of scalar: (theta, major, minor) - the angle of the principal axis CW from north,
            the variance along the major axis, and the variance along the minor axis

    """
    u = np.array(u)
    v = np.array(v)

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


def rot(u: NDArray[floating], v: NDArray[floating], theta: float) -> tuple[NDArray[floating], NDArray[floating]]:
    """Rotate a vector counter clockwise or a coordinate system clockwise.

    Args:
        u (scalar or array): x-component of vector
        v (scalar or array): y-component of vector
        theta (scalar): rotation angle (CCW > 0, CW < 0)

    Returns:
        tuple of scalar or array: (ur, vr) - x and y components of vector in rotated coordinate system

    """
    w = u + 1j * v
    ang = np.deg2rad(theta)
    wr = w * np.exp(1j * ang)
    ur = np.real(wr)
    vr = np.imag(wr)
    return ur, vr


def uv_from_spddir(
    speed: NDArray[floating],
    direction: NDArray[floating],
    which: str = "from",
) -> tuple[NDArray[floating], NDArray[floating]]:
    """Compute east and west vectors of velocity vector.

    Args:
        speed (scalar or array): Velocity magnitude.
        direction (scalar or array): Direction of velocity, CW from true north. Behavior controlled by which.
        which ({"from", "to"}, default: "from"): Determines if direction defines the velocity coming "from" direction
            (common for wind) or going "to" direction (common for currents).

    Returns:
        tuple of scalar or array: (u, v) - east velocity "u" and north velocity "v"

    Raises:
        ValueError: If an invalid argument is given for which.

    """
    theta = np.array(direction)
    theta = np.deg2rad(theta)
    if which == "from":
        u = -speed * np.sin(theta)
        v = -speed * np.cos(theta)
    elif which == "to":
        u = speed * np.sin(theta)
        v = speed * np.cos(theta)
    else:
        msg = f"Invalid argument for 'which': {which}. Must be 'from' or 'to'."
        raise ValueError(msg)
    return (u, v)


def spddir_from_uv(
    u: NDArray[floating],
    v: NDArray[floating],
    which: Literal["from", "to"] = "from",
) -> tuple[NDArray[floating], NDArray[floating]]:
    """Compute speed and direction from east and north vectors of velocity vector.

    Args:
        u (scalar or array): East velocity.
        v (scalar or array): North velocity.
        which ({"from", "to"}, default: "from"): Determines if direction defines the velocity coming "from" direction
            (common for wind) or going "to" direction (common for currents).

    Returns:
        tuple of scalar or array: (speed, direction) - velocity magnitude and direction CW from true north

    Raises:
        ValueError: If an invalid argument is given for which.

    """
    speed = np.sqrt(u**2 + v**2)
    if which == "from":
        theta = np.arctan2(-u, -v)
    elif which == "to":
        theta = np.arctan2(u, v)
    else:
        msg = f"Invalid argument for 'which': {which}. Must be 'from' or 'to'."
        raise ValueError(msg)
    direction = np.rad2deg(theta) % 360  # convert to degrees and ensure 0-360 range
    return (speed, direction)


def relative_humidity_from_dewpoint(t: NDArray[floating], t_dew: NDArray[floating]) -> NDArray[floating]:
    """Relative humidity as a function of air temp. and dew point temp.

    Args:
        t (scalar or array): air temperature (degC)
        t_dew (scalar or array): dew point temperature (degC)

    Returns:
        scalar or array: relative humidity as a percent (0->100)

    """
    e = 610.94 * np.exp(17.625 * t_dew / (t_dew + 243.04))
    es = 610.94 * np.exp(17.625 * t / (t + 243.04))
    rh = e / es * 100
    return rh


# parse command line arguments
parser = argparse.ArgumentParser(
    description="Download NDBC data, compute wind stress, rotate into principal axes, and daily average.",
)
parser.add_argument("station", metavar="station", type=str, nargs=1, help="station ID to download data from")
parser.add_argument(
    "-d",
    "--directory",
    metavar="directory",
    type=str,
    nargs="?",
    help="directory to store output files",
    default="./output/",
)
parser.add_argument(
    "-n",
    "--name",
    metavar="name",
    type=str,
    nargs="?",
    help="file name for output file, do not include extension .nc",
    default=None,
)
parser.add_argument(
    "-p",
    "--princax",
    metavar="princax",
    nargs=1,
    type=float,
    help="angle (degrees CW from true north) to rotate wind data into principal axes.",
    default=None,
)
parser.add_argument(
    "-f",
    "--filter",
    metavar="filter",
    nargs=1,
    type=int,
    help="length of low-pass filter (in hours) to apply to wind data before computing principal axes. \
    Default is no filtering.",
    default=None,
)
args = parser.parse_args()
args = vars(args)
site = args["station"][0].lower()
out_path = args["directory"]
out_file = args["name"]

# get ndbc site metadata for instrument elevation
ndbc_site_url = "https://www.ndbc.noaa.gov/station_page.php?station=" + site
elev, zt, zu, zb, zt_sea, depth, radius = ndbc_heights(ndbc_site_url)
zt += elev
zu += elev
zb += elev

# get all available files
url = "https://dods.ndbc.noaa.gov/thredds/catalog/data/stdmet/" + site + "/catalog.html"
tag = r"[1-2][0-9][0-9][0-9].*\.nc$"
nc_files = list_files(url, tag)
file_url = "https://dods.ndbc.noaa.gov/thredds/fileServer/"
nc_url = [file_url + i + "#mode=bytes" for i in nc_files]
# https://dods.ndbc.noaa.gov/thredds/fileServer/data/stdmet/46050/46050h1998.nc
# load datasets
ds_orig = []
for f in tq(nc_url, desc="Downloading datasets"):
    r = requests.get(f, timeout=(3.05, 120))
    # ensure request worked
    if r.ok:
        ds_orig.append(xr.load_dataset(io.BytesIO(r.content), decode_timedelta=True))

ds: xr.Dataset = xr.concat(ds_orig, dim="time").drop_duplicates("time")

# Convert to single lat/lon point by averaging (all should be very close for the same station)
mean_lat = ds["latitude"].mean().values
mean_lon = ds["longitude"].mean().values
ds = ds.mean(["longitude", "latitude"])
ds = ds.assign_coords(latitude=mean_lat, longitude=mean_lon)

# Convert wave period to seconds to avoid issues with timedelta64
if "average_wpd" in ds.variables:
    ds["average_wpd"] = ds["average_wpd"] / np.timedelta64(1, "s")
    ds["average_wpd"] = ds["average_wpd"].astype("float32")
    ds["average_wpd"].attrs["units"] = "s"
if "dominant_wpd" in ds.variables:
    ds["dominant_wpd"] = ds["dominant_wpd"] / np.timedelta64(1, "s")
    ds["dominant_wpd"] = ds["dominant_wpd"].astype("float32")
    ds["dominant_wpd"].attrs["units"] = "s"

# Get u and v components of wind velocity and apply optional filtering
ds["wind_east"], ds["wind_north"] = uv_from_spddir(ds["wind_spd"], ds["wind_dir"])

# fill bad values with mean for COARE inputs
ds["rh"] = relative_humidity_from_dewpoint(ds.air_temperature, ds.dewpt_temperature)
bad_data_thresh = 500  # % or degC, obviously bad values
ds["rh_filled"] = ds["rh"].where(ds["rh"] < bad_data_thresh, ds["rh"].mean())
ds["air_temperature_filled"] = ds["air_temperature"].where(
    ds["air_temperature"] < bad_data_thresh,
    ds["air_temperature"].mean(),
)
ds["sea_surface_temperature_filled"] = ds["sea_surface_temperature"].where(
    ds["sea_surface_temperature"] < bad_data_thresh,
    ds["air_temperature"].mean(),
)

coare_mag = c35.tau(
    ds["wind_spd"].values,
    t=ds["air_temperature_filled"].values,
    rh=ds["rh_filled"].values,
    ts=ds["sea_surface_temperature_filled"].values,
    lat=ds["latitude"].values,
    zu=zu,
    zt=zt,
    zq=zt,
)
ds["coare_mag"] = (["time"], coare_mag)
ds["coare_east"], ds["coare_north"] = uv_from_spddir(ds["coare_mag"], ds["wind_dir"])

# drop filled variables
ds = ds.drop_vars(["rh_filled", "air_temperature_filled", "sea_surface_temperature_filled"])

if args["filter"] is not None:
    # get filtering weights for 33 hour low pass filter - assumes 1 hour time step in data
    wts = signal.firwin(101, 1 / args["filter"][0], window="lanczos", fs=1)
    ds["wind_east"] = (["time"], signal.filtfilt(wts, 1, ds["wind_east"].values, axis=0))
    ds["wind_north"] = (["time"], signal.filtfilt(wts, 1, ds["wind_north"].values, axis=0))
    ds["wind_spd"], ds["wind_dir"] = spddir_from_uv(ds["wind_east"], ds["wind_north"])
    ds["coare_east"] = (["time"], signal.filtfilt(wts, 1, ds["coare_east"].values, axis=0))
    ds["coare_north"] = (["time"], signal.filtfilt(wts, 1, ds["coare_north"].values, axis=0))
    ds["coare_mag"], ds["wind_dir"] = spddir_from_uv(ds["coare_east"], ds["coare_north"])

# rotate wind velocity and wind stress into principal axis
if args["princax"] is not None:
    theta = args["princax"][0]
else:
    theta, _, _ = princax(ds["wind_east"], ds["wind_north"])

ds["cs"], ds["as"] = rot(ds["wind_east"], ds["wind_north"], theta)
ds["coare_x"], ds["coare_y"] = rot(ds["coare_east"], ds["coare_north"], theta)

# add metadata
ds["wind_east"].attrs = {"comment": "Eastwards wind velocity", "units": "m/s"}
ds["wind_north"].attrs = {"comment": "Northwards wind velocity", "units": "m/s"}
ds["coare_mag"].attrs = {"comment": "Magnitude of wind stress computed by COARE v3.5", "units": "N/m^2"}
ds["coare_east"].attrs = {"comment": "Eastwards wind stress computed by COARE v3.5", "units": "N/m^2"}
ds["coare_north"].attrs = {"comment": "Northwards wind stress computed by COARE v3.5", "units": "N/m^2"}
ds["rh"].attrs = {"comment": "Relative humidity computed from air and dewpoint temperature", "units": "%"}
ds["cs"].attrs = {"comment": "Cross-shelf component of wind velocity computed by principal axis", "units": "m/s"}
ds["as"].attrs = {"comment": "Along-shelf component of wind velocity computed by principal axis", "units": "m/s"}
ds["coare_x"].attrs = {
    "comment": "Cross-shelf component of wind stress computed by principal axis",
    "units": "m/s",
}
ds["coare_y"].attrs = {
    "comment": "Along-shelf component of wind stress computed by principal axis",
    "units": "m/s",
}
ds.attrs = {
    "Site Elevation (m)": f"{elev:.02f}",
    "Air temp height (m)": f"{zt:.02f}",
    "Anemometer height (m)": f"{zu:.02f}",
    "Barometer height (m)": f"{zb:.02f}",
    "Sea temp depth (m)": f"{zt_sea:.02f}",
    "Water depth (m)": f"{depth:.02f}",
    "Watch radius (m)": f"{radius:02f}",
    "Principal axis (deg CW of true north)": f"{theta:.02f}",
}


# setup output folders
if not Path(out_path).exists():
    Path(out_path).mkdir()
if not Path(out_path, "raw").exists():
    Path(out_path, "raw").mkdir()
out_file = site + "_wind_binned.nc" if out_file is None else out_file + ".nc"

# save output files
ds.to_netcdf(Path(out_path, out_file))
for i, d in enumerate(iterable=ds_orig):
    d.to_netcdf(Path(out_path, "raw", nc_files[i][-13:-3] + ".nc"))
