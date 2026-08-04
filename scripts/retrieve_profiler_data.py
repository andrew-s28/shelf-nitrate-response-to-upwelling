# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "bs4",
#     "flox",
#     "gsw",
#     "numpy",
#     "pandas",
#     "requests",
#     "tqdm",
#     "xarray[accel,io,parallel]",
# ]
# ///
"""Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save."""

# Python Standard Library
import argparse
import io
import re
import warnings
from datetime import UTC, datetime
from itertools import starmap
from pathlib import Path
from typing import Any, cast

import gsw
import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup
from flox.xarray import xarray_reduce
from tqdm import tqdm as tq

SCRIPT_DIR = Path(__file__).parent.resolve()


def _filter_dates(nc_files: list, start_date: str, end_date: str) -> list:
    date_regex = r"[0-9]+T[0-9]+\.[0-9]+-[0-9]+T[0-9]+\.[0-9]+"
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").astimezone(UTC)
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").astimezone(UTC)
    # get start and end dates of each deployment from the file names
    date_search = [re.search(date_regex, f) for f in nc_files]
    if not all(date_search):
        msg = f"Some files do not have valid date formats in their names. Files: {[f for f, d in zip(nc_files, date_search, strict=True) if not d]}"
        raise ValueError(msg)
    nc_files_start_dates = [
        datetime.strptime(d.group(0).split("-")[0], "%Y%m%dT%H%M%S.%f").astimezone(
            UTC,
        )
        for d in date_search
        if d is not None
    ]
    nc_files_end_dates = [
        datetime.strptime(d.group(0).split("-")[1], "%Y%m%dT%H%M%S.%f").astimezone(
            UTC,
        )
        for d in date_search
        if d is not None
    ]
    # filter by start and end date, including any data files that end after the start date or start before the end date
    nc_files = [
        f
        for i, f in enumerate(nc_files)
        if nc_files_end_dates[i] >= start_date_dt and nc_files_start_dates[i] <= end_date_dt
    ]
    return nc_files


def list_files(url: str, start_date: str, end_date: str, tag: str = r".*\.nc$") -> list[str]:
    r"""Create a list of the netCDF data files in the THREDDS catalog created by a request to the M2M system.

    Obtained from 2022 OOIFB workshop.

    Args:
        url (str): URL to a THREDDS catalog specific to a data request
        start_date (str): Start date for the data range
        end_date (str): End date for the data range
        tag (regexp, optional): Regex pattern used to distinguish files of interest. Defaults to r'.*\\.nc$'.

    Returns:
        array: list of files in the catalog with the URL path set relative to the catalog

    """
    with requests.session() as s:
        page = s.get(url).text

    soup = BeautifulSoup(page, "html.parser")
    pattern = re.compile(tag)
    nc_files = [node.get("href") for node in soup.find_all("a", string=pattern)]  # ty:ignore[unresolved-attribute]
    nc_files = [re.sub(r"catalog.html\?dataset=", "", file) for file in nc_files]
    nc_files = _filter_dates(nc_files, start_date, end_date)
    return nc_files


def _drop_unused_vars(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.drop_vars(
        [
            "nutnr_nitrogen_in_nitrate_qc_executed",
            "nutnr_spectrum_average",
            "nutnr_fit_base_2",
            "nutnr_fit_base_1",
            "year",
            "salinity_corrected_nitrate_qc_results",
            "sea_water_pressure_qc_executed",
            "nutnr_current_main",
            "sea_water_practical_salinity_qc_executed",
            "nitrate_concentration_qc_results",
            "humidity",
            "voltage_main",
            "sea_water_pressure_qc_results",
            "spectral_channels",
            "temp_spectrometer",
            "temp_lamp",
            "day_of_year",
            "nutnr_nitrogen_in_nitrate",
            "nitrate_concentration",
            "nutnr_absorbance_at_254_nm",
            "nutnr_absorbance_at_350_nm",
            "temp_interior",
            "nutnr_bromide_trace",
            "sea_water_temperature_qc_results",
            "nutnr_integration_time_factor",
            "lamp_time",
            "nitrate_concentration_qc_executed",
            "time_of_sample",
            "nutnr_nitrogen_in_nitrate_qc_results",
            "nitrate_concentration_qartod_results",
            "sea_water_practical_salinity_qc_results",
            "sea_water_temperature_qc_executed",
            "nutnr_voltage_int",
            "salinity_corrected_nitrate_qc_executed",
            "salinity_corrected_nitrate_qartod_results",
            "suspect_timestamp",
            "aux_fitting_1",
            "nutnr_fit_rmse",
            "dark_val",
            "aux_fitting_2",
            "voltage_lamp",
        ],
    ).drop_vars("wavelength")
    return ds


def _comput_tsrho(ds: xr.Dataset) -> xr.Dataset:
    # renames and computes variables
    ds = ds.swap_dims({"obs": "time"})

    # assign coords for lat and lon
    ds = ds.assign_coords({"latitude": ds.attrs["geospatial_lat_min"]})
    ds = ds.assign_coords({"longitude": ds.attrs["geospatial_lon_min"]})

    ds = ds.drop_vars(["sea_water_temperature", "sea_water_practical_salinity", "sea_water_pressure"])
    # use ctdfp_* variables instead
    ds = ds.rename(
        {
            "ctdpf_j_cspp_instrument_recovered-sea_water_temperature": "temperature",
            "ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity": "practical_salinity",
            "nutnr_dark_value_used_for_fit": "dark_val",
            "int_ctd_pressure": "pressure",
        },
    )
    ds["absolute_salinity"] = gsw.conversions.SA_from_SP(
        SP=ds["practical_salinity"],
        p=ds["pressure"],
        lon=ds["lon"],
        lat=ds["lat"],
    )
    ds["conservative_temperature"] = gsw.conversions.CT_from_t(
        SA=ds["absolute_salinity"],
        t=ds["temperature"],
        p=ds["pressure"],
    )
    ds["sigma_theta"] = gsw.density.sigma0(SA=ds["absolute_salinity"], CT=ds["conservative_temperature"])
    return ds


def _nutnr_qc(
    spectral_channels: np.ndarray,
    dark_val: np.ndarray,
    salinity: np.ndarray,
    time: np.ndarray,
    qartod: np.ndarray,
    min_salinity: float,
    rmse_lim: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    # does the actual QC as vectorized numpy operations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", message="invalid value encountered in log10")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

        # wavelength range for nitrate calculation
        wavelengths = np.arange(190, 394.01, 0.8)[30:]
        # get counts by spectral channels
        spectral_channels = (spectral_channels.T - dark_val).T[:, 30:]
        # mask where True means good data and False means bad data
        mask = np.full(time.shape, True, dtype=bool)  # noqa:FBT003

        # initialize array to hold slope values for each spectrum
        a_all = np.empty(len(time))
        a_all[:] = np.nan

        data_removed_by_step = {
            "bad_qartod": 0,
            "bad_salinity": 0,
            "bad_spectra": 0,
            "bad_fit": 0,
            "bad_slope": 0,
        }

        mask[qartod == 4] = False
        data_removed_by_step["bad_qartod"] = np.count_nonzero(~mask)

        # mask observations with low salinity values
        mask[(salinity <= min_salinity) & (mask)] = False
        data_removed_by_step["bad_salinity"] = np.count_nonzero(~mask) - data_removed_by_step["bad_qartod"]

        # mask spectra with non-finite or negative values
        mask[(np.any(~np.isfinite(spectral_channels), axis=1)) & (mask)] = False
        mask[(np.any(spectral_channels <= 0, axis=1)) & (mask)] = False
        data_removed_by_step["bad_spectra"] = (
            np.count_nonzero(~mask) - data_removed_by_step["bad_qartod"] - data_removed_by_step["bad_salinity"]
        )

        # fit spectra to linear model between wavelengths ~360 nm to 380 nm and remove bad fits
        wl_fit = wavelengths[217 - 30 : 241 - 30]
        spec_fit = np.log10(spectral_channels[:, 217 - 30 : 241 - 30])
        rmse_all = np.empty(len(spec_fit))
        rmse_all[:] = np.nan
        for i, sp in enumerate(spec_fit):
            # safety check to ensure that all non-finite values have been removed in step 2
            if np.any(~np.isfinite(sp)):
                if mask[i]:
                    print(
                        "Why is this happening? This shouldn't be happening.",
                    )  # any non-finite values should have been removed in step 2
                    mask[i] = False
                    data_removed_by_step["bad_spectra"] += 1
            else:
                # remove spectra with high RMSE from linear fit in wavelength range
                a, _ssres, _, _ = np.linalg.lstsq(np.vstack([np.ones(wl_fit.shape), wl_fit]).T, sp, rcond=None)
                residuals = sp - wl_fit * a[1] - a[0]
                rmse = np.sqrt(np.sum(residuals**2) / (residuals.size - 2))
                rmse_all[i] = rmse
                if rmse > rmse_lim:
                    if mask[i]:
                        mask[i] = False
                        data_removed_by_step["bad_fit"] += 1
                else:
                    a_all[i] = a[1]

        # remove spectra with slope far from the mean slope
        a_all[~mask] = np.nan
        mask[(np.abs(a_all - np.nanmean(a_all)) >= 3 * np.nanstd(a_all)) & (mask)] = False

        data_removed_by_step["bad_slope"] = (
            np.count_nonzero(~mask)
            - data_removed_by_step["bad_qartod"]
            - data_removed_by_step["bad_salinity"]
            - data_removed_by_step["bad_spectra"]
            - data_removed_by_step["bad_fit"]
        )

    return mask, a_all, data_removed_by_step


def _nutnr_qc_wrapper(
    ds: xr.Dataset,
    min_salinity: float,
    rmse_lim: float = 0.02,
) -> tuple[xr.Dataset | None, int, dict[str, int]]:
    """Remove bad fits in OOI nutnr datasets.

    Args:
        ds (Dataset): OOI nutnr dataset
        min_salinity (float): Minimum salinity for the dataset
        rmse_lim (int, optional): Maximum RMSE for fit to be kept. Defaults to 0.02.

    Returns:
        tuple: (Dataset with bad fits removed, number of bad fits removed, dictionary of counts of data removed by each step)

    """
    # provide numpy arrays to apply QC
    mask, _a_all, data_removed_by_step = _nutnr_qc(
        ds.spectral_channels.values,
        ds.dark_val.values,
        ds.practical_salinity.values,
        ds.time.values,
        ds.salinity_corrected_nitrate_qartod_results.values,
        min_salinity,
        rmse_lim=rmse_lim,
    )
    # mask the data
    if np.all(~mask):
        return None, np.count_nonzero(mask), data_removed_by_step

    ds = ds.where(xr.DataArray(mask, coords={"time": ds.time.values}), drop=True)
    return ds, np.count_nonzero(mask), data_removed_by_step


def split_profiles(ds: xr.Dataset) -> list[xr.Dataset]:
    """Split the data set into individual profiles, where each profile is a collection of data from a single deployment and profile sequence.

    The resulting data sets are returned in a list.

    Args:
        ds (xr.Dataset): OOI profiler dataset

    Returns:
        list[xr.Dataset]: List of individual profiles

    """
    # split the data into profiles, assuming at least 120 seconds between profiles
    dt = ds.where(ds["time"].diff("time") > np.timedelta64(120, "s"), drop=True).get_index("time")

    # process each profile, adding the results to a list of profiles
    profiles = []
    jback = np.timedelta64(30, "s")  # 30 second jump back to avoid collecting data from the following profile
    for i, d in enumerate(dt):
        # pull out the profile
        if i == 0:
            profile = ds.sel(time=slice(ds["time"].values[0], d - jback))
        else:
            profile = ds.sel(time=slice(dt[i - 1], d - jback))

        # add the profile to the list
        profiles.append(profile)

    # grab the last profile and append it to the list
    profile = ds.sel(time=slice(d, ds["time"].values[-1]))
    profiles.append(profile)
    return profiles


def profiler_binning(
    d: xr.Dataset,
    z: np.ndarray,
    z_lab: str = "depth",
    t_lab: str = "time",
    offset: float = 0.5,
) -> xr.Dataset:
    """Bins a profiler time series into daily bins and depth bins.

    Removes any non-numeric data types, including any time types, outside of the coordinates.

    Args:
        d (xr.dataset): OOI profiler dataset
        z (array): edges of depth/pressure bins
        z_lab (str, optional): name of depth/pressure in dataset. Defaults to 'depth'.
        t_lab (str, optional): name of time in dataset. Defaults to 'time'.
        offset (float, optional): Distance between SUNA V2 and CTD on profiler (positive when CTD is higher).
            Defaults to 0.5.

    Returns:
        xr.dataset: binned dataset

    """
    types = [d[i].dtype for i in d]
    var_names = list(d.keys())
    exclude = []
    for i, t in enumerate(types):
        if not (np.issubdtype(t, np.number)):
            exclude.append(var_names[i])
    d = d.drop_vars(exclude)

    # handle nitrate offset
    nitr = d["salinity_corrected_nitrate"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # handle nitrate data with offset
        nitr_out: xr.DataArray = xarray_reduce(
            nitr,
            d[t_lab],
            d[z_lab] + offset,
            func="nanmean",
            expected_groups=(None, z - offset),
            isbin=[False, True],
            method="map-reduce",
            skipna=True,
        )
        # handle ctd data without offset
        rest_out: xr.Dataset = xarray_reduce(
            d,
            d[t_lab],
            d[z_lab],
            func="nanmean",
            expected_groups=(None, z),
            isbin=[False, True],
            method="map-reduce",
            skipna=True,
        )

    # manually re-assign depth bins in nitrate to match those in the rest of the variables
    # we want the nitrate concentration to be binned into bins a half meter shallower than the rest
    # but once we bin, we then want to treat these depth bins as if they were in the CTD depth bins
    nitr_out["depth_bins"] = rest_out["depth_bins"]
    out = rest_out.copy()
    out["salinity_corrected_nitrate"] = nitr_out

    depth = np.array([x.mid for x in out.depth_bins.values])
    out[z_lab] = ([z_lab + "_bins"], depth)
    out = out.swap_dims({z_lab + "_bins": z_lab})
    out = out.drop_vars([z_lab + "_bins"])

    return out


def _is_list_of_strings_or_string(lst: Any) -> bool:  # noqa: ANN401
    if isinstance(lst, str):
        return True
    return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


def _parse_args() -> tuple[list[str], str, str]:
    parser = argparse.ArgumentParser(
        description="Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save.",
    )
    parser.add_argument(
        "site",
        metavar="site",
        type=str,
        nargs="+",
        help="Site to download from, either CE01ISSP or CE02SHSP",
    )
    parser.add_argument(
        "-s",
        "--start",
        metavar="start",
        type=str,
        nargs="?",
        default="2010-01-01",
        help="Start date in YYYY-MM-DD format (default: 2010-01-01)",
    )
    parser.add_argument(
        "-e",
        "--end",
        metavar="end",
        type=str,
        nargs="?",
        default=datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        help=f"End date in YYYY-MM-DD format (default: {datetime.now(tz=UTC).strftime('%Y-%m-%d')})",
    )
    args = parser.parse_args()
    args = vars(args)
    sites = args["site"]
    start_date = args["start"]
    end_date = args["end"]
    sites = [site.upper() for site in sites]

    if not (
        _is_list_of_strings_or_string(sites)
        and _is_list_of_strings_or_string(start_date)
        and _is_list_of_strings_or_string(end_date)
    ):
        msg = f"Invalid argument types: sites={type(sites)}, start_date={type(start_date)}, end_date={type(end_date)}. All arguments must be strings."
        raise ValueError(msg)

    # we've now validated types, so safe to cast
    sites = cast("list[str]", sites)
    start_date = cast("str", start_date)
    end_date = cast("str", end_date)

    return sites, start_date, end_date


def _make_dirs(out_path: str, site: str) -> Path:
    path = Path(out_path)
    path.mkdir(parents=True, exist_ok=True)
    site_path = path / site
    site_path.mkdir(parents=True, exist_ok=True)
    (site_path / "raw").mkdir(parents=True, exist_ok=True)
    return site_path


def _get_datasets(site: str, start_date: str, end_date: str) -> list[xr.Dataset]:
    if site == "CE01ISSP":
        # setup defaults to use in subsequent data queries
        refdes = site + "-SP001-06-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    elif site == "CE02SHSP":
        refdes = site + "-SP001-05-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    elif site == "CE06ISSP":
        refdes = site + "-SP001-06-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    elif site == "CE07SHSP":
        refdes = site + "-SP001-05-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    else:
        msg = f"Invalid site selection: {site}. Choose one or more of CE01ISSP (Oregon inshore), CE02SHSP (Oregon midshelf), CE06ISSP (Washington inshore), or CE07SHSP (Washington shelf)."
        raise ValueError(msg)

    # construct the OOI Gold Copy THREDDS catalog URL for this data set
    base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
    url = base_url + f"{refdes}-{method}-{stream}" + "/catalog.html"
    tag = r"NUTNRJ000.*.nc$"  # setup regex for files we want
    nc_files = list_files(url, start_date, end_date, tag)
    base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
    nc_url = [base_url + i + "#mode=bytes" for i in nc_files]  # create urls for download

    # load datasets
    ds = []
    for i, f in enumerate(tq(nc_url, desc="Downloading datasets")):
        r = requests.get(f, timeout=(3.05, 120))
        if r.ok:
            ds.append(xr.open_dataset(io.BytesIO(r.content)))
            ds[i].load()

    return ds


def _add_global_attrs(ds: xr.Dataset, site: str, baseline_subtraced: bool) -> xr.Dataset:  # noqa:FBT001
    """Add global attributes to the dataset following CF metadata conventions.

    Args:
        ds: The dataset to which global attributes will be added.
        site: The site name.
        baseline_subtraced: A boolean indicating whether baseline subtraction was applied.

    Returns:
        xr.Dataset: The dataset with added global attributes.

    """
    summary = "Binned, quality-controlled nitrate and co-located CTD data from OOI Endurance Array profilers. Quality control process includes a per-spectra analysis of the nitrate fit, removing data with non-finite values, low salinity, bad fits, and outlier slopes. Binning process splits data into individual profiles and bins the data into 1-meter depth bins centered on the mean time of the profile."
    summary += (
        "Baseline subtraction is applied to the nitrate data to align with bottle sample data."
        if baseline_subtraced
        else ""
    )
    ds.attrs["title"] = f"OOI Endurance Array {site} Profiler Nitrate Data"
    ds.attrs["summary"] = summary
    ds.attrs["date_created"] = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    ds.attrs["date_modified"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    ds.attrs["date_issued"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    ds.attrs["history"] = f"Created: {datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%S')}"
    ds.attrs["keywords"] = (
        "OSU, OOI, Endurance Array, nitrate, profiler, quality control, binned, temperature, density, salinity"
    )
    ds.attrs["reference"] = (
        "A.L. Scherer, M.R. Fewings, 2026. Binned and Quality Controlled Nitrate Data from NSF Ocean Observatories Initiative Endurance Array Profilers on the Newport Hydrographic Line. In Prep."
    )
    ds.attrs["conventions"] = "CF-1.6"
    ds.attrs["standard_name_vocabulary"] = "CF Standard Name Table v94"
    ds.attrs["references"] = f"https://doi.org/10.58046/OOI-{site}"
    ds.attrs["license"] = "Creative Commons Attribution 4.0 International"
    ds.attrs["creator_name"] = "NSF Ocean Observatories Initiative (OOI) Endurance Array"
    ds.attrs["creator_url"] = "https://oceanobservatories.org/"
    ds.attrs["publisher_name"] = "Andrew Scherer"
    ds.attrs["publisher_email"] = "scherand@oregonstate.edu"
    ds.attrs["publisher_url"] = "sourcedepth.com"
    ds.attrs["publisher_institution"] = "College of Earth, Ocean, and Atmospheric Sciences, Oregon State University"
    ds.attrs["contributor_name"] = "Andrew Scherer"
    ds.attrs["contributor_email"] = "scherand@oregonstate.edu"
    ds.attrs["contributor_role"] = "processor"
    ds.attrs["contributor_role_vocabulary"] = "https://vocab.nerc.ac.uk/collection/G04/current/"
    ds.attrs["geospatial_lat_min"] = str(float(ds["latitude"].min()))
    ds.attrs["geospatial_lat_max"] = str(float(ds["latitude"].max()))
    ds.attrs["geospatial_lat_units"] = "degrees_north"
    ds.attrs["geospatial_lon_min"] = str(float(ds["longitude"].min()))
    ds.attrs["geospatial_lon_max"] = str(float(ds["longitude"].max()))
    ds.attrs["geospatial_lon_units"] = "degrees_east"
    ds.attrs["geospatial_vertical_min"] = float(ds.depth.min())
    ds.attrs["geospatial_vertical_max"] = float(ds.depth.max())
    ds.attrs["geospatial_vertical_units"] = "meter"
    ds.attrs["geospatial_vertical_positive"] = "down"
    ds.attrs["geospatial_vertical_resolution"] = "1 meter"
    ds.attrs["time_coverage_start"] = ds.time.min().dt.strftime("%Y-%m-%dT%H:%M:%S").item()
    ds.attrs["time_coverage_end"] = ds.time.max().dt.strftime("%Y-%m-%dT%H:%M:%S").item()
    ds.attrs["time_coverage_duration"] = (ds.time.max() - ds.time.min()).astype("timedelta64[D]").astype(str).item()
    ds.attrs["time_coverage_resolution"] = "1 day"
    return ds


def _add_var_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Add variable attributes to the dataset following CF metadata conventions.

    Args:
        ds: The dataset to which variable attributes will be added.

    Returns:
        xr.Dataset: The dataset with added variable attributes.

    """
    var_attrs = {
        "time": {
            "long_name": "Time",
            "standard_name": "time",
        },
        "latitude": {
            "long_name": "Latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
        },
        "longitude": {
            "long_name": "Longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
        },
        "depth": {
            "long_name": "Depth",
            "standard_name": "depth",
            "units": "meters",
            "positive": "down",
        },
        "pressure": {
            "long_name": "Seawater Pressure",
            "standard_name": "sea_water_pressure",
            "units": "dbar",
        },
        "temperature": {
            "long_name": "Seawater Temperature",
            "standard_name": "sea_water_temperature",
            "units": "degrees_Celsius",
        },
        "practical_salinity": {
            "long_name": "Seawater Practical Salinity",
            "standard_name": "sea_water_practical_salinity",
            "units": "1",
        },
        "conservative_temperature": {
            "long_name": "Seawater Conservative Temperature",
            "standard_name": "sea_water_conservative_temperature",
            "units": "degrees_Celsius",
        },
        "absolute_salinity": {
            "long_name": "Seawater Absolute Salinity",
            "standard_name": "sea_water_absolute_salinity",
            "units": "g/kg",
        },
        "sigma_theta": {
            "long_name": "Seawater Potential Density Anomaly",
            "standard_name": "sea_water_sigma_theta",
            "units": "kg/m^3",
            "reference_pressure": 0,
        },
        "nitrate": {
            "long_name": "Nitrate Concentration",
            "standard_name": "mole_concentration_of_nitrate_in_sea_water",
            "units": "mmol/m^3",
        },
    }
    for var, attrs in var_attrs.items():
        ds[var].attrs.update(attrs)
    return ds


if __name__ == "__main__":
    # parse command line arguments
    sites, start_date, end_date = _parse_args()
    out_path_top = str(Path(SCRIPT_DIR) / "../data")

    for site in sites:
        # setup output folders
        out_path = _make_dirs(out_path_top, site)

        # get datasets
        ds_list = _get_datasets(site, start_date, end_date)
        ds_orig = ds_list.copy()

        for i, d in enumerate(ds_list):
            # compute derived variables
            ds_list[i] = _comput_tsrho(d)
            # save metadata to include in final dataset

        # QC nitrate data and remove short datasets
        ds_qced, deployment_numbers, total_data, good_data, data_removed_by_step, removed = [], [], [], [], [], []
        for _i, d in enumerate(tq(ds_list, desc="QC'ing datasets")):
            deployment_numbers.append(np.unique(d.deployment)[0])
            total_data.append(d.time.size)
            ds_i, good_data_i, data_removed_by_step_i = _nutnr_qc_wrapper(d, min_salinity=25)
            if ds_i is not None:
                ds_qced.append(ds_i)
                removed.append(False)
            data_removed_by_step.append(data_removed_by_step_i)
            good_data.append(good_data_i)

        for i, rem in enumerate(removed):
            if rem:
                print(f"Deployment {deployment_numbers[i]} removed due to lack of data after QC.")

        qc_results = pd.DataFrame(
            {
                "deployment": deployment_numbers,
                "total_data": total_data,
                "good_data": good_data,
                "bad_data": np.array(total_data) - np.array(good_data),
                "bad_qartod": [d["bad_qartod"] for d in data_removed_by_step],
                "bad_salinity": [d["bad_salinity"] for d in data_removed_by_step],
                "bad_spectra": [d["bad_spectra"] for d in data_removed_by_step],
                "bad_fit": [d["bad_fit"] for d in data_removed_by_step],
                "bad_slope": [d["bad_slope"] for d in data_removed_by_step],
            },
        )

        qc_results.to_csv(Path(out_path) / f"{site}_qc_results.csv")

        # find minimum and maximum depth bins over all nitrate data
        sur = np.min(xr.concat(ds_qced, dim="time")["depth"])
        bot = np.max(xr.concat(ds_qced, dim="time")["depth"])

        # setup depth/pressure bins
        step = 1
        sur = np.floor(sur) - step / 2
        bot = np.ceil(bot) + step / 2
        # pressure_grid is centers of bins
        pressure_grid = np.arange(sur + step / 2, bot + step, step)
        # pressure_bins is edges of bins
        pressure_bins = np.nan * np.empty(len(pressure_grid) + 1)
        pressure_bins[0] = pressure_grid[0] - step / 2
        pressure_bins[-1] = pressure_grid[-1] + step / 2
        for i in range(len(pressure_bins) - 2):
            pressure_bins[i + 1] = np.average([pressure_grid[i], pressure_grid[i + 1]])

        # bin datasets into new list
        deployment_profiles = []
        times = []
        for d in tq(ds_qced, desc="Splitting and binning profiles"):
            profiles = []
            profile_times = []
            # split dataset into individual profiles and bin each profile, saving mean of profile along time dimension and the time of the profile
            for profile in split_profiles(d):
                profile_binned = profiler_binning(profile, pressure_bins, offset=0.5)
                profiles.append(profile_binned.mean(dim="time"))
                profile_times.append(profile_binned.time.mean().values)
            deployment_profiles.append(profiles)
            times.append(xr.DataArray(profile_times, dims="time"))
        deployments = list(starmap(xr.concat, zip(deployment_profiles, times, strict=True)))

        remove_dep_nums = []
        # remove datasets with less than 10 profiles data
        for i, d in enumerate(deployments):
            if d.time.size < 10:
                remove_dep_nums.append(deployment_numbers[i])
                print(
                    f"Deployment {d['deployment'].mean().values} removed due to having less than 10 profiles after QC and binning.",
                )
        deployments = [d for i, d in enumerate(deployments) if deployment_numbers[i] not in remove_dep_nums]

        # concatenate binned datasets
        ds_bin = xr.concat(deployments, dim="time")
        ds_bin = ds_bin.drop_duplicates("time", keep="first")
        ds_bin = ds_bin.where(~np.isinf(ds_bin["salinity_corrected_nitrate"]))
        # now safe to drop unused variables including QC variables
        ds_bin = _drop_unused_vars(ds_bin)

        if site == "CE01ISSP":
            ds_bin = ds_bin.where(ds_bin["deployment"] != 20, drop=True)

        if site == "CE01ISSP":
            # custom baseline subtractions for deployments at inshore, based on nitrate-density relationship and some overlapping bottle samples
            for i, dep in enumerate(deployments):
                d = np.unique(dep["deployment"])[0]
                # custom baseline subtractions for deployments at inshore, based on nitrate-density relationship and some overlapping bottle samples
                baseline_subtraction = {
                    1: 9,
                    2: -2,
                    4: -1,
                    5: 1,
                    6: 10,
                    10: -2,
                    14: -2,
                    18: -1,
                    19: 3,
                    20: 3,
                    21: 1,
                    22: -1,
                }
                deployments[i]["salinity_corrected_nitrate"] = dep[
                    "salinity_corrected_nitrate"
                ] - baseline_subtraction.get(d, 0)
        elif site == "CE06ISSP":
            for i, dep in enumerate(deployments):
                d = np.unique(dep["deployment"])[0]
                baseline_subtraction = {
                    1: 5,
                    4: 2,
                    5: 2,
                    6: 1,
                    17: 3,
                }
                deployments[i]["salinity_corrected_nitrate"] = dep[
                    "salinity_corrected_nitrate"
                ] - baseline_subtraction.get(d, 0)

        # at the midshelf sites, adjust median surface nitrate to be near zero, but only if the minimum value is less than -3 mmol/m^3 as the expected accuracy of the SUNA is +/- 3 mmol/m^3
        # for some deployments that have many high nitrate events at the surface (consistent upwelling?), the median is too large
        # in these cases, we adjust the baseline to ensure that the minimum value is at least -3 mmol/m^3
        # deployments which don't have this issue are not adjusted to avoid the opposite problem
        elif site in {"CE02SHSP", "CE07SHSP"}:
            for i, dep in enumerate(deployments):
                d = np.unique(dep["deployment"])[0]
                baseline = dep.isel(depth=slice(0, 5))["salinity_corrected_nitrate"].median()
                if (dep["salinity_corrected_nitrate"] - baseline).min() < -3:
                    baseline += (dep["salinity_corrected_nitrate"] - baseline).min() + 3
                deployments[i]["salinity_corrected_nitrate"] = dep["salinity_corrected_nitrate"] - baseline

        ds_bin_baseline_subtracted = xr.concat(deployments, dim="time")
        ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.drop_duplicates("time", keep="first")
        ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.where(
            ~np.isinf(ds_bin_baseline_subtracted["salinity_corrected_nitrate"]),
        )
        # now safe to drop unused variables including QC variables
        ds_bin_baseline_subtracted = _drop_unused_vars(ds_bin_baseline_subtracted)

        if site == "CE01ISSP":
            ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.where(
                ds_bin_baseline_subtracted["deployment"] != 20,
                drop=True,
            )
        elif site == "CE06ISSP":
            ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.where(
                ds_bin_baseline_subtracted["deployment"] != 15,
                drop=True,
            )

        # some final renaming and adding attributes
        ds_bin = ds_bin.rename({"salinity_corrected_nitrate": "nitrate"})
        ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.rename({"salinity_corrected_nitrate": "nitrate"})
        ds_bin = ds_bin.drop_attrs()
        ds_bin_baseline_subtracted = ds_bin_baseline_subtracted.drop_attrs()
        ds_bin = _add_global_attrs(ds_bin, site, baseline_subtraced=False)
        ds_bin_baseline_subtracted = _add_global_attrs(ds_bin_baseline_subtracted, site, baseline_subtraced=True)
        ds_bin = _add_var_attrs(ds_bin)
        ds_bin_baseline_subtracted = _add_var_attrs(ds_bin_baseline_subtracted)

        print("Saving datasets...", end="", flush=True)

        # save start and end dates for naming output files
        save_start_date = ds_bin.time.min().values.astype("datetime64[D]").astype(str)
        save_end_date = ds_bin.time.max().values.astype("datetime64[D]").astype(str)

        # save binned data
        out_file = site + "_nitrate_binned_qced_" + save_start_date + "_" + save_end_date + ".nc"
        ds_bin.to_netcdf(Path(out_path) / out_file)
        ds_bin.close()

        # save binned and baseline subtracted data
        out_file = site + "_nitrate_binned_qced_baseline_subtracted_" + save_start_date + "_" + save_end_date + ".nc"
        ds_bin_baseline_subtracted.to_netcdf(Path(out_path) / out_file)
        ds_bin_baseline_subtracted.close()

        # save raw datasets
        for d in ds_orig:
            start_date_dep = d.time.min().values.astype("datetime64[D]").astype(str)
            end_date_dep = d.time.max().values.astype("datetime64[D]").astype(str)
            d.to_netcdf(
                Path(out_path) / f"raw/{site}_dep{d.deployment[0]:02.0f}_{start_date_dep}_{end_date_dep}.nc",
                mode="w",
            )

        print(f"done! {site} data saved to {out_path}.")
