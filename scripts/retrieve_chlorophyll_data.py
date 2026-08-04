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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import requests
import xarray as xr
from bs4 import BeautifulSoup
from tqdm import tqdm as tq

SCRIPT_DIR = Path(__file__).parent.resolve()


def _filter_dates(nc_files: list, start_date: str, end_date: str) -> list:
    date_regex = r"[0-9]+T[0-9]+(?:\.[0-9]+)?-[0-9]+T[0-9]+(?:\.[0-9]+)?"
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").astimezone(UTC)
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").astimezone(UTC)
    # get start and end dates of each deployment from the file names
    date_search = [re.search(date_regex, f) for f in nc_files]
    if not all(date_search):
        msg = f"Some files do not have valid date formats in their names. Files: {[f for f, d in zip(nc_files, date_search, strict=True) if not d]}"
        raise ValueError(msg)
    nc_files_start_dates = [
        datetime.strptime(re.sub(r"\.\d+", "", d.group(0).split("-")[0]), "%Y%m%dT%H%M%S").astimezone(UTC)
        for d in date_search
        if d is not None
    ]
    nc_files_end_dates = [
        datetime.strptime(re.sub(r"\.\d+", "", d.group(0).split("-")[1]), "%Y%m%dT%H%M%S").astimezone(UTC)
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
            # for fluorometer data
            "suspect_timestamp",
            "fluorometric_cdom",
            "fluorometric_cdom_qc_executed",
            "fluorometric_cdom_qc_results",
            "fluorometric_cdom_qartod_executed",
            "fluorometric_cdom_qartod_results",
            "total_volume_scattering_coefficient",
            "total_volume_scattering_coefficient_qc_executed",
            "total_volume_scattering_coefficient_qc_results",
            "total_volume_scattering_coefficient_qartod_executed",
            "total_volume_scattering_coefficient_qartod_results",
            "optical_backscatter",
            "optical_backscatter_qc_executed",
            "optical_backscatter_qc_results",
            "optical_backscatter_qartod_executed",
            "optical_backscatter_qartod_results",
            "fluorometric_chlorophyll_a_qc_executed",
            "fluorometric_chlorophyll_a_qc_results",
            "fluorometric_chlorophyll_a_qartod_executed",
            "fluorometric_chlorophyll_a_qartod_results",
            "raw_signal_chl",
            "raw_signal_cdom",
            "raw_signal_beta",
            "raw_internal_temp",
            "measurement_wavelength_chl",
            "measurement_wavelength_cdom",
            "measurement_wavelength_beta",
            "seawater_scattering_coefficient",
        ],
        errors="ignore",
    )
    return ds


def _is_list_of_strings_or_string(lst: Any) -> bool:
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
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    site_path = path / site
    if not site_path.exists():
        site_path.mkdir(parents=True, exist_ok=True)
    if not (path / "raw").exists():
        (path / "raw").mkdir(parents=True, exist_ok=True)
    return site_path


def _get_datasets(refdes: str, method: str, stream: str, start_date: str, end_date: str, tag: str) -> list[xr.Dataset]:
    # construct the OOI Gold Copy THREDDS catalog URL for this data set
    base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
    url = base_url + f"{refdes}-{method}-{stream}" + "/catalog.html"
    nc_files = list_files(url, start_date, end_date, tag)
    base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/"
    nc_url = [base_url + i + "#mode=bytes" for i in nc_files]  # create urls for download

    # load datasets
    ds = []
    for i, f in enumerate(tq(nc_url, desc="Downloading datasets")):
        r = requests.get(f, timeout=(3.05, 120))
        if r.ok:
            ds.append(xr.open_dataset(io.BytesIO(r.content)))
            ds[i] = ds[i].swap_dims({"obs": "time"})  # rename obs dimension to time
            ds[i] = ds[i].drop_vars(["obs"])
            ds[i] = ds[i].resample(time="1D").mean(dim="time", skipna=True)  # resample to daily mean
            ds[i].load()

    return ds


def _get_chl_datasets(site: str, start_date: str, end_date: str) -> list[xr.Dataset]:
    if site == "CE01ISSM":
        # setup defaults to use in subsequent data queries
        fluorometer_refdes = site + "-RID16-02-FLORTD000"
        fluorometer_method = "recovered_host"
        fluorometer_stream = "flort_sample"
        fluorometer_tag = r"FLORTD000.*.nc$"
        # acs_refdes = site + "-RID16-01-OPTAAD000"
        # acs_method = "recovered_host"
        # acs_stream = "optaa_dj_dcl_instrument_recovered"
        # acs_tag = r"OPTAAD000.*.nc$"
    else:
        msg = f"Invalid site selection: {site}. Choose one or more of CE01ISSM (Oregon inshore)."
        raise ValueError(msg)

    fluorometer_datasets = _get_datasets(
        fluorometer_refdes,
        fluorometer_method,
        fluorometer_stream,
        start_date,
        end_date,
        fluorometer_tag,
    )
    fluorometer_datasets = [
        ds.where(ds["fluorometric_chlorophyll_a_qartod_results"] != 4, drop=True) for ds in fluorometer_datasets
    ]  # drop any data flagged as bad by QARTOD
    # acs_datasets = _get_datasets(acs_refdes, acs_method, acs_stream, start_date, end_date, acs_tag)

    return fluorometer_datasets


def _add_global_attrs(ds: xr.Dataset, site: str) -> xr.Dataset:
    """Add global attributes to the dataset following CF metadata conventions.

    Args:
        ds: The dataset to which global attributes will be added.
        site: The site name.

    Returns:
        xr.Dataset: The dataset with added global attributes.

    """
    summary = "Chlorophyll data from 7 m OOI Endurance Array mooring near-surface instrument frame. Chlorophyll data come are taken from the fluorometer."
    ds.attrs["title"] = f"OOI Endurance Array {site} Mooring Chlorophyll Data"
    ds.attrs["summary"] = summary
    ds.attrs["date_created"] = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    ds.attrs["date_modified"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    ds.attrs["date_issued"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    ds.attrs["history"] = f"Created: {datetime.now(tz=UTC).strftime('%Y-%m-%dT%H:%M:%S')}"
    ds.attrs["keywords"] = "OSU, OOI, Endurance Array, chlorophyll, mooring"
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
        "chlorophyll_a": {
            "long_name": "Seawater Chlorophyll",
            "standard_name": "sea_water_chlorophyll",
            "units": "mg/m^3",
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
        ds_flort = _get_chl_datasets(site, start_date, end_date)

        # concatenate datasets
        ds_concat = xr.concat(ds_flort, dim="time")

        # assign coords for lat and lon
        ds_concat = ds_concat.assign_coords({"latitude": ds_concat.attrs["geospatial_lat_min"]})
        ds_concat = ds_concat.assign_coords({"longitude": ds_concat.attrs["geospatial_lon_min"]})

        ds_concat = ds_concat.rename(
            {
                "fluorometric_chlorophyll_a": "chlorophyll_a",
                "sea_water_temperature": "temperature",
                "sea_water_practical_salinity": "practical_salinity",
            },
        )

        ds_concat = _drop_unused_vars(ds_concat)
        ds_concat = _add_global_attrs(ds_concat, site)

        ds_concat = ds_concat.sortby("time").resample(time="1D").mean(dim="time")

        print("Saving datasets...", end="", flush=True)

        # save start and end dates for naming output files
        save_start_date = ds_concat.time.min().values.astype("datetime64[D]").astype(str)
        save_end_date = ds_concat.time.max().values.astype("datetime64[D]").astype(str)

        # save binned data
        out_file = site + "_chlorophyll_" + save_start_date + "_" + save_end_date + ".nc"
        ds_concat.to_netcdf(Path(out_path) / out_file)
        ds_concat.close()

        print(f"done! {site} data saved to {out_path}.")
