"""Process OOI nitrate profiler data from the inner and mid shelf sites.

Designed to work with nitrate datasets that have been downloaded using ooi-profiler-nitrate-retriever.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "xarray[accel,io,parallel]",
# ]
# ///

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "../data/"
GEBCO_PATH = list(Path(DATA_DIR / "GEBCO/").glob("*.nc"))

NHL_LAT = 44.66
D_MIN_MIDSHELF = -18.5
D_MAX_MIDSHELF = -7
D_MIN_INNERSHELF = -7
D_MAX_INNERSHELF = 0

# dataset file names
INNER_SHELF_NITRATE_FILE = Path(
    "CE01ISSP/CE01ISSP_nitrate_binned_baseline_subtracted_2014-04-17_2023-09-17.nc",
)
INNER_SHELF_NITRATE_SAVE_FILE = Path(
    "CE01ISSP/CE01ISSP_nitrate_binned_baseline_subtracted_2014-04-17_2023-09-17_with_dndt_resampled_v2.nc",
)
MID_SHELF_NITRATE_FILE = Path(
    "CE02SHSP/CE02SHSP_nitrate_binned_baseline_subtracted_2015-03-18_2024-07-14.nc",
)
MID_SHELF_NITRATE_SAVE_FILE = Path(
    "CE02SHSP/CE02SHSP_nitrate_binned_baseline_subtracted_2015-03-18_2024-07-14_with_dndt_resampled_v2.nc",
)


def extrapolate_bottom_nitrate(
    nitrate: xr.Dataset,
    limit: int = 10,
) -> xr.Dataset:
    """Extrapolate bottom nitrate using constant nitrate extrapolation from the bottom-most depth.

    Args:
        nitrate (xr.Dataset): dataset containing nitrate data
        limit (int): maximum number of depth bins to extrapolate, default is 10 indices

    Returns:
        xr.Dataset: dataset with bottom nitrate extrapolated to constant value from the bottom-most depth

    """
    # bfill will backfill the surface values with the nearest valid value, only within 20 m (2 m depth bins)
    nitrate = nitrate.ffill(dim="depth", limit=limit)

    return nitrate


def extrapolate_top_nitrate(
    nitrate: xr.Dataset,
    limit: int = 10,
) -> xr.Dataset:
    """Extrapolate top nitrate using constant nitrate extrapolation from the top-most depth.

    Args:
        nitrate (xr.Dataset): dataset containing nitrate data
        limit (int): maximum number of depth bins to extrapolate, default is 10 indices

    Returns:
        xr.Dataset (nitrate) - dataset with top nitrate extrapolated to constant value from the top-most depth

    """
    # bfill will backfill the surface values with the nearest valid value, only within 20 m (2 m depth bins)
    nitrate = nitrate.bfill(dim="depth", limit=limit)

    return nitrate


def calculate_tendency_based_on_scaling_profiles(
    nitrate: xr.Dataset,
    d_min: float,
    d_max: float,
) -> xr.DataArray:
    """Calculate the volume of water in a 1 m wide control volume extending cross-shelf from d_min to d_max.

    Args:
        nitrate (xr.Dataset): a dataset containing a 'dndt' variable, with 'time' and 'depth' dimensions
        d_min (float): the distance from shore defining the outer boundary of the control volume
        d_max (float): the distance from shore defining the inner boundary of the control volume

    Returns:
        xr.DataArray: a dataarray containing the tendency of nitrate in the specified control volume

    """
    # load bathymetry and interpolate to the NHL latitude set above
    bathymetry = (
        xr.open_mfdataset(GEBCO_PATH)
        .interp(lat=NHL_LAT)
        .interp(
            {"lon": np.linspace(-130, -120, int(1e6))},  # interpolate to a very high resolution covering the shelf
        )
    )

    # identify the coast by finding the longitude where the elevation is closest to zero
    coast = bathymetry.isel({"lon": np.nanargmin(np.abs(bathymetry.elevation.values))})

    # distance along line of constant latitude is given by dlon * R * cos(lat), with R=6371 km (radius of Earth)
    bathymetry["distance_from_shore"] = (
        xr.ufuncs.deg2rad(bathymetry["lon"] - coast["lon"]) * 6371 * xr.ufuncs.cos(xr.ufuncs.deg2rad(NHL_LAT))
    )

    # use distance from shore as the new x coordinate instead of longitude
    bathymetry = bathymetry.swap_dims({"lon": "distance_from_shore"})

    # interpolate to 10 m resolution between d_min and d_max
    elevation = xr.ufuncs.abs(bathymetry["elevation"].interp(distance_from_shore=np.arange(d_min, d_max, 0.01)))

    # volume = 1 m * 0.01 km (cross-shelf step size) * 1000 m/km (convert to meters) * depth (elevation in dataarray)
    volume = (0.01 * 1000 * elevation).sum()

    tendency = nitrate["dndt"].mean(dim="depth") * volume

    return tendency


def calculate_dndt(nitrate: xr.Dataset) -> xr.DataArray:
    """Calculate the dndt (change in nitrate concentration over time) for the given nitrate profiler dataset.

    Args:
        nitrate (xarray.Dataset): The dataset containing nitrate data.

    Returns:
        xarray.Dataset: The dataset with dndt calculated.

    """
    # depth varying dn/dt
    dndt = nitrate["nitrate"].differentiate(
        "time",
        datetime_unit="s",
    )

    return dndt


def make_nitrate_dataset(
    nitrate: xr.Dataset,
    location: Literal["midshelf", "innershelf"],
    limit: int = 10,
) -> xr.Dataset:
    """Process the raw nitrate dataset to fill in missing values and calculate dndt.

     This function will:
        - Resample to daily mean
        - Interpolate gaps in the middle of profiles
        - Extrapolate missing data at the top and bottom of profiles,
            up to a limit of 5 points (5 m) at inner shelf and 10 points (10 m) at mid shelf
        - Calculate dndt and depth averaged nitrate

    Args:
        nitrate (xr.Dataset): dataset containing nitrate data
        location ("midshelf" or "innershelf"): site of nitrate data
        limit (int): number of depth bins to extrapolate

    Returns:
        xr.Dataset: processed dataset with dndt calculated and missing values filled in

    """
    # resample to daily mean
    nitrate = nitrate.resample(time="1D").mean()

    # interpolate gaps in the middle of profiles
    nitrate = nitrate.interpolate_na(dim="time", method="polynomial", order=1, limit=2).interpolate_na(
        dim="depth",
        method="polynomial",
        order=1,
        limit=limit,
    )

    # extrapolate missing data, up to a limit of 5 points (5 m) at inner shelf and 10 points (10 m) at mid shelf
    nitrate = extrapolate_bottom_nitrate(nitrate, limit=limit)
    nitrate = extrapolate_top_nitrate(nitrate, limit=limit)

    # calculate dndt and depth averaged nitrate
    nitrate["dndt"] = calculate_dndt(nitrate)

    if location == "midshelf":
        nitrate["dndt_volume_integrated"] = calculate_tendency_based_on_scaling_profiles(
            nitrate,
            D_MIN_MIDSHELF,
            D_MAX_MIDSHELF,
        )
    elif location == "innershelf":
        nitrate["dndt_volume_integrated"] = calculate_tendency_based_on_scaling_profiles(
            nitrate,
            D_MIN_INNERSHELF,
            D_MAX_INNERSHELF,
        )

    return nitrate


###############################
# Inner Shelf Nitrate Dataset #
###############################

inner_nitrate = xr.open_dataset(
    DATA_DIR / INNER_SHELF_NITRATE_FILE,
)

# rename for convienience
inner_nitrate = inner_nitrate.rename(
    {"salinity_corrected_nitrate": "nitrate", "sigma_theta": "density"},
)

inner_nitrate = make_nitrate_dataset(inner_nitrate, "innershelf", limit=10)

# add metadata that includes deployment numbers in dataset and creation script and time
deployments = np.unique(inner_nitrate.deployment.values)[~np.isnan(np.unique(inner_nitrate.deployment.values))]
inner_nitrate.attrs["deployment"] = deployments
inner_nitrate.attrs["created_by"] = "make_nitrate_datasets.py"
inner_nitrate.attrs["created_on"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

# save the dataset to a netcdf file
inner_nitrate.to_netcdf(
    DATA_DIR / INNER_SHELF_NITRATE_SAVE_FILE,
)

#############################
# Mid Shelf Nitrate Dataset #
#############################

midshelf_nitrate = xr.open_dataset(
    DATA_DIR / MID_SHELF_NITRATE_FILE,
)

# rename for convienience
midshelf_nitrate = midshelf_nitrate.rename(
    {"salinity_corrected_nitrate": "nitrate", "sigma_theta": "density"},
)

midshelf_nitrate = make_nitrate_dataset(midshelf_nitrate, "midshelf", limit=20)

# add metadata that includes deployment numbers in dataset and creation script and time
deployments = np.unique(midshelf_nitrate.deployment.values)[~np.isnan(np.unique(midshelf_nitrate.deployment.values))]
midshelf_nitrate.attrs["deployment"] = deployments
midshelf_nitrate.attrs["created_by"] = "make_nitrate_datasets.py"
midshelf_nitrate.attrs["created_on"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

# save the dataset to a netcdf file
midshelf_nitrate.to_netcdf(
    DATA_DIR / MID_SHELF_NITRATE_SAVE_FILE,
)
