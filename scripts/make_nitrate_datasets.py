"""Process OOI nitrate profiler data from the inner and mid shelf sites.

Designed to work with nitrate datasets that have been downloaded using ooi-profiler-nitrate-retriever.
"""

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "xarray[io]",
# ]
# ///

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "../data/"

# dataset file names
INNER_SHELF_NITRATE_FILE = Path(
    "CE01ISSP/CE01ISSP_nitrate_binned_baseline_subtracted_2014-04-17_2025-07-26.nc",
)
INNER_SHELF_NITRATE_SAVE_FILE = Path(
    "CE01ISSP/CE01ISSP_nitrate_binned_baseline_subtracted_2014-04-17_2025-07-26_with_dndt_resampled.nc",
)
MID_SHELF_NITRATE_FILE = Path(
    "CE02SHSP/CE02SHSP_nitrate_binned_baseline_subtracted_2015-03-18_2024-09-15.nc",
)
MID_SHELF_NITRATE_SAVE_FILE = Path(
    "CE02SHSP/CE02SHSP_nitrate_binned_baseline_subtracted_2015-03-18_2024-09-15_with_dndt_resampled.nc",
)


def calculate_dndt(ds: xr.Dataset) -> xr.Dataset:
    """Calculate the dndt (change in nitrate concentration over time) for the given nitrate profiler dataset.

    Args:
        ds (xarray.Dataset): The dataset containing nitrate data.

    Returns:
        xarray.Dataset: The dataset with dndt calculated.

    """
    # Calculate depth-integrated nitrate
    ds["depth_integrated_nitrate"] = (
        ["time"],
        xr.apply_ufunc(
            lambda x, y: np.array(
                [
                    np.trapezoid(yi[~np.isnan(yi)], x[~np.isnan(yi)]) if len(yi[~np.isnan(yi)]) > 0 else np.nan
                    for yi in y
                ],
            ),
            ds["depth"].values,
            ds["nitrate"].values,
        ),
    )
    # Resample to daily mean
    ds = ds.resample(time="1D").mean()
    # Calculate dndt
    ds["dndt"] = ds["depth_integrated_nitrate"].differentiate("time", datetime_unit="s")
    return ds


###############################
# Inner Shelf Nitrate Dataset #
###############################

nitrate = xr.open_dataset(
    DATA_DIR / INNER_SHELF_NITRATE_FILE,
)

# rename for convienience
nitrate = nitrate.rename(
    {"salinity_corrected_nitrate": "nitrate", "sigma_theta": "density"},
)

# calculate dndt and depth averaged nitrate
nitrate = calculate_dndt(nitrate)

# drop time steps with no nitrate data
nitrate = nitrate.dropna("time", how="all", subset=["nitrate"])

# add metadata that includes deployment numbers in dataset and creation script and time
deployments = np.unique(nitrate.deployment.values)[~np.isnan(np.unique(nitrate.deployment.values))]
nitrate.attrs["deployment"] = deployments
nitrate.attrs["created_by"] = "make_datasets.py"
nitrate.attrs["created_on"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

# save the dataset to a netcdf file
nitrate.to_netcdf(
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

# calculate dndt and depth averaged nitrate
midshelf_nitrate = calculate_dndt(midshelf_nitrate)

# drop time steps with no nitrate data
midshelf_nitrate = midshelf_nitrate.dropna("time", how="all", subset=["nitrate"])

# add metadata that includes deployment numbers in dataset and creation script and time
deployments = np.unique(midshelf_nitrate.deployment.values)[~np.isnan(np.unique(midshelf_nitrate.deployment.values))]
midshelf_nitrate.attrs["deployment"] = deployments
midshelf_nitrate.attrs["created_by"] = "make_datasets.py"
midshelf_nitrate.attrs["created_on"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

# save the dataset to a netcdf file
midshelf_nitrate.to_netcdf(
    DATA_DIR / MID_SHELF_NITRATE_SAVE_FILE,
)
