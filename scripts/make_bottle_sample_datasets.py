"""Create bottle sample datasets from ship data and gridded CTD data."""

from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "../data"


def _make_ce01_bottle_samples() -> None:
    # Load the OOI Endurance Array ship data
    file_list = Path(DATA_DIR / "ship/ea_ship_data/").glob("*.csv")
    df_list = []
    col_names = [
        "Station",
        "Start Time [UTC]",
        "Start Latitude [degrees]",
        "Start Longitude [degrees]",
        "CTD Pressure [db]",
        "CTD Salinity 1 [psu]",
        "CTD Salinity 2 [psu]",
        "CTD Temperature 1 [deg C]",
        "CTD Temperature 2 [deg C]",
        "Discrete Nitrate [uM]",
    ]
    for f in file_list:
        df_temp = pd.read_csv(DATA_DIR / "ship/ea_ship_data/" / f, usecols=col_names)
        rows = df_temp.loc[df_temp["Station"] == "CE01"]
        df_list.append(rows)

    inner_ooi_crse = pd.concat(df_list)
    inner_ooi_crse = inner_ooi_crse.rename(
        columns={
            "Start Time [UTC]": "time",
            "Start Latitude [degrees]": "lat",
            "Start Longitude [degrees]": "lon",
            "CTD Pressure [db]": "pressure",
            "CTD Salinity 1 [psu]": "sal1",
            "CTD Salinity 2 [psu]": "sal2",
            "CTD Temperature 1 [deg C]": "temp1",
            "CTD Temperature 2 [deg C]": "temp2",
            "Discrete Nitrate [uM]": "nitrate",
        },
    )
    inner_ooi_crse["time"] = pd.to_datetime(inner_ooi_crse["time"])

    # convert to xarray - need unique indices for time and pressure
    inner_ooi_crse = inner_ooi_crse.iloc[
        np.unique(inner_ooi_crse.set_index(["time", "pressure"]).index, return_index=True)[1]
    ]
    inner_ooi_crse = inner_ooi_crse.set_index(["time", "pressure"])
    inner_ooi_crse = inner_ooi_crse.to_xarray()
    inner_ooi_crse["time"] = pd.DatetimeIndex(inner_ooi_crse["time"].values)

    # remove some bad data
    inner_ooi_crse = inner_ooi_crse.where(inner_ooi_crse["nitrate"] > -100, drop=True)

    # calculate potential density anomaly
    inner_ooi_crse["pot_density_anom"] = gsw.density.sigma0(
        inner_ooi_crse["sal1"],
        gsw.conversions.CT_from_t(
            inner_ooi_crse["sal1"],
            inner_ooi_crse["temp1"],
            inner_ooi_crse["pressure"],
        ),
    )

    # save bottle samples data
    inner_ooi_crse.to_netcdf(
        DATA_DIR / "ship/CE01ISSP_bottle_samples.nc",
    )


def _make_nh01_nh03_bottle_samples() -> None:
    # ctd data from NH, nhl_crse_ctd = nh line cruise ctd data
    inner_nhl_crse_ctd = xr.open_dataset(
        DATA_DIR / "NHL_Gridded/newport_hydrographic_line_gridded_sections.nc",
    )
    # fix some xr stuff, assign coords, remove unnecessary dims, etc
    inner_nhl_crse_ctd = inner_nhl_crse_ctd.assign_coords(
        date=inner_nhl_crse_ctd["time"].astype("datetime64[D]"),
    )
    inner_nhl_crse_ctd = inner_nhl_crse_ctd.swap_dims({"time": "date"})
    inner_nhl_crse_ctd["longitude"] = inner_nhl_crse_ctd["longitude"] - 360
    inner_nhl_crse_ctd = inner_nhl_crse_ctd.squeeze()

    # nitrate data from NH line cruises, nhl_crse_nit = nh line cruise nitrate data
    inner_nhl_crse_nit = pd.read_csv(DATA_DIR / "ship/nhl_ship_data/NHL_Cruise_Nutrients_All_NCC.csv")

    inner_nhl_crse_nit["Sample Date"] = pd.to_datetime(
        inner_nhl_crse_nit["Sample Date"].to_numpy(),
        unit="ns",
    )

    nitr_tbin, pres_tbin, time_tbin, long_tbin, stat_tbin = [], [], [], [], []
    for _i, t in enumerate(inner_nhl_crse_ctd["date"].to_numpy()):
        # find places where nitrate data lines up with ctd data
        # need exact date matches - don't want nearest neighbor for this step
        time_mask = np.where(inner_nhl_crse_nit["Sample Date"] == t)
        for n, d, s, p, lon in zip(
            inner_nhl_crse_nit["no3"].to_numpy()[time_mask],
            inner_nhl_crse_nit["Sample Date"].to_numpy()[time_mask],
            inner_nhl_crse_nit["Station"].to_numpy()[time_mask],
            inner_nhl_crse_nit["pressure"].to_numpy()[time_mask],
            inner_nhl_crse_nit["Longitude"].to_numpy()[time_mask],
            strict=False,
        ):
            nitr_tbin.append(n)
            pres_tbin.append(p)
            time_tbin.append(d)
            long_tbin.append(lon)
            stat_tbin.append(s)

    # make numpy arrays
    nitr_tbin, pres_tbin, time_tbin, long_tbin, stat_tbin = (
        np.array(nitr_tbin),
        np.array(pres_tbin),
        np.array(time_tbin),
        np.array(long_tbin),
        np.array(stat_tbin),
    )

    # make dataarrays for vectorized indexing, doesn't work with np arrays for some reason
    pres_targ = xr.DataArray(pres_tbin, dims="date")
    time_targ = xr.DataArray(time_tbin, dims="date")
    long_targ = xr.DataArray(long_tbin, dims="date")

    # bin the ctd data using the arrays from the bottle sample binning, using nearest binning method
    # e.g., if pressure=0 from bottle, then pressure=1 from ship since this is closest.
    inner_nhl_crse = inner_nhl_crse_ctd.sel(
        date=time_targ,
        pressure=pres_targ,
        longitude=long_targ,
        method="nearest",
    )

    # add nitrate and station
    inner_nhl_crse["nitrate"] = (("date"), nitr_tbin)
    inner_nhl_crse["station"] = (("date"), stat_tbin)

    # view created xarray
    inner_nhl_crse = inner_nhl_crse.where(
        (inner_nhl_crse.station == "NH01") | (inner_nhl_crse.station == "NH03"),
        drop=True,
    )
    inner_nhl_crse = inner_nhl_crse.swap_dims({"date": "time"})

    # save bottle samples data
    inner_nhl_crse.to_netcdf(
        DATA_DIR / "ship/NH01_NH03_bottle_samples.nc",
    )


def _make_ce02_bottle_samples() -> None:
    file_list = Path(DATA_DIR / "ship/ea_ship_data/").glob("*.csv")
    df_list = []
    col_names = [
        "Station",
        "Start Time [UTC]",
        "Start Latitude [degrees]",
        "Start Longitude [degrees]",
        "CTD Pressure [db]",
        "CTD Salinity 1 [psu]",
        "CTD Salinity 2 [psu]",
        "CTD Temperature 1 [deg C]",
        "CTD Temperature 2 [deg C]",
        "Discrete Nitrate [uM]",
    ]
    for f in file_list:
        df_temp = pd.read_csv(DATA_DIR / "ship/ea_ship_data/" / f, usecols=col_names)
        rows = df_temp.loc[df_temp["Station"] == "CE02"]
        df_list.append(rows)

    mid_shelf_ooi_crse = pd.concat(df_list)
    mid_shelf_ooi_crse = mid_shelf_ooi_crse.rename(
        columns={
            "Start Time [UTC]": "time",
            "Start Latitude [degrees]": "lat",
            "Start Longitude [degrees]": "lon",
            "CTD Pressure [db]": "pressure",
            "CTD Salinity 1 [psu]": "sal1",
            "CTD Salinity 2 [psu]": "sal2",
            "CTD Temperature 1 [deg C]": "temp1",
            "CTD Temperature 2 [deg C]": "temp2",
            "Discrete Nitrate [uM]": "nitrate",
        },
    )
    mid_shelf_ooi_crse["time"] = pd.to_datetime(mid_shelf_ooi_crse["time"])
    # ooi_cruse = mid_shelf_ooi_crse.set_index(["time", "pressure"])
    mid_shelf_ooi_crse = mid_shelf_ooi_crse.iloc[
        np.unique(
            mid_shelf_ooi_crse.set_index(["time", "pressure"]).index,
            return_index=True,
        )[1]
    ]
    mid_shelf_ooi_crse = mid_shelf_ooi_crse.set_index(["time", "pressure"])
    mid_shelf_ooi_crse = mid_shelf_ooi_crse.to_xarray()
    mid_shelf_ooi_crse["time"] = pd.DatetimeIndex(mid_shelf_ooi_crse["time"].values)
    mid_shelf_ooi_crse = mid_shelf_ooi_crse.where(
        mid_shelf_ooi_crse.nitrate > -100,
        drop=True,
    )
    mid_shelf_ooi_crse["pot_density_anom"] = gsw.density.sigma0(
        mid_shelf_ooi_crse["sal1"],
        gsw.conversions.CT_from_t(
            mid_shelf_ooi_crse["sal1"],
            mid_shelf_ooi_crse["temp1"],
            mid_shelf_ooi_crse["pressure"],
        ),
    )

    # save bottle samples data
    mid_shelf_ooi_crse.to_netcdf(
        DATA_DIR / "ship/CE02SHSP_bottle_samples.nc",
    )


def _make_nh10_bottle_samples() -> None:
    # ctd data from NH, nhl_crse_ctd = nh line cruise ctd data
    mid_nhl_crse_ctd = xr.open_dataset(
        DATA_DIR / "NHL_Gridded/newport_hydrographic_line_gridded_sections.nc",
    )
    # fix some xr stuff, assign coords, remove unnecessary dims, etc
    mid_nhl_crse_ctd = mid_nhl_crse_ctd.assign_coords(
        date=mid_nhl_crse_ctd["time"].astype("datetime64[D]"),
    )
    mid_nhl_crse_ctd = mid_nhl_crse_ctd.swap_dims({"time": "date"})
    mid_nhl_crse_ctd["longitude"] = mid_nhl_crse_ctd["longitude"] - 360
    mid_nhl_crse_ctd = mid_nhl_crse_ctd.squeeze()

    # nitrate data from NH line cruises, nhl_crse_nit = nh line cruise nitrate data
    mid_nhl_crse_nit = pd.read_csv(DATA_DIR / "ship/nhl_ship_data/NHL_Cruise_Nutrients_All_NCC.csv")

    mid_nhl_crse_nit["Sample Date"] = pd.to_datetime(
        mid_nhl_crse_nit["Sample Date"].to_numpy(),
        unit="ns",
    )

    nitr_tbin, pres_tbin, time_tbin, long_tbin, stat_tbin = [], [], [], [], []
    for _i, t in enumerate(mid_nhl_crse_ctd["date"].to_numpy()):
        # find places where nitrate data lines up with ctd data
        # need exact date matches - don't want nearest neighbor for this step
        time_mask = np.where(mid_nhl_crse_nit["Sample Date"] == t)
        for n, d, s, p, lon in zip(
            mid_nhl_crse_nit["no3"].to_numpy()[time_mask],
            mid_nhl_crse_nit["Sample Date"].to_numpy()[time_mask],
            mid_nhl_crse_nit["Station"].to_numpy()[time_mask],
            mid_nhl_crse_nit["pressure"].to_numpy()[time_mask],
            mid_nhl_crse_nit["Longitude"].to_numpy()[time_mask],
            strict=False,
        ):
            nitr_tbin.append(n)
            pres_tbin.append(p)
            time_tbin.append(d)
            long_tbin.append(lon)
            stat_tbin.append(s)

    # make numpy arrays
    nitr_tbin, pres_tbin, time_tbin, long_tbin, stat_tbin = (
        np.array(nitr_tbin),
        np.array(pres_tbin),
        np.array(time_tbin),
        np.array(long_tbin),
        np.array(stat_tbin),
    )

    # make dataarrays for vectorized indexing, doesn't work with np arrays for some reason
    pres_targ = xr.DataArray(pres_tbin, dims="date")
    time_targ = xr.DataArray(time_tbin, dims="date")
    long_targ = xr.DataArray(long_tbin, dims="date")

    # bin the ctd data using the arrays from the bottle sample binning, using nearest binning method
    # e.g., if pressure=0 from bottle, then pressure=1 from ship since this is closest.
    mid_nhl_crse = mid_nhl_crse_ctd.sel(
        date=time_targ,
        pressure=pres_targ,
        longitude=long_targ,
        method="nearest",
    )

    # add nitrate and station
    mid_nhl_crse["nitrate"] = (("date"), nitr_tbin)
    mid_nhl_crse["station"] = (("date"), stat_tbin)

    # view created xarray
    mid_nhl_crse = mid_nhl_crse.where(mid_nhl_crse.station == "NH10", drop=True)
    mid_nhl_crse = mid_nhl_crse.swap_dims({"date": "time"})

    # save bottle sample data
    mid_nhl_crse.to_netcdf(
        DATA_DIR / "ship/NH10_bottle_samples.nc",
    )


if __name__ == "__main__":
    _make_ce01_bottle_samples()
    _make_ce02_bottle_samples()
    _make_nh01_nh03_bottle_samples()
    _make_nh10_bottle_samples()
