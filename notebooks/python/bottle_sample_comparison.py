# ---
# jupyter:
#   jupytext:
#     formats: ipynb,python//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: nitrate-upwelling (3.12.9)
#     language: python
#     name: python3
# ---

# %%
import calendar
import string
from pathlib import Path
from typing import cast

import cmocean.cm as cmo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import xarray as xr
from matplotlib import colormaps as cmaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import distributions


# %% [markdown]
# ## Dataset Loading

# %%
# For plotting months with colormap
colors = cmaps["viridis"](np.linspace(0, 1, 6))

FIG_SAVE_FMT = "png"

# %%
NOTEBOOK_DIR = Path().resolve()
DATA_DIR = NOTEBOOK_DIR / "../data"
FIGURES_DIR = NOTEBOOK_DIR / "../figures"
INNER_NITRATE_PATH = (
    DATA_DIR / "CE01ISSP/CE01ISSP_nitrate_binned_qced_baseline_subtracted_2014-04-17_2025-07-26_with_dndt_resampled.nc"
)
MIDSHELF_NITRATE_PATH = (
    DATA_DIR / "CE02SHSP/CE02SHSP_nitrate_binned_qced_baseline_subtracted_2015-03-18_2024-09-15_with_dndt_resampled.nc"
)
WIND_PATH = DATA_DIR / "NDBC_46050/46050_wind_binned_with_w5d_w8d.nc"
VEL_PATH = DATA_DIR / "NH10_ADCP_Data/nh10_hourly_data_1997_2024_rotated_filtered_streamwise_v5.2.nc"

# %%
inner_nitrate = xr.open_dataset(INNER_NITRATE_PATH)
midshelf_nitrate = xr.open_dataset(MIDSHELF_NITRATE_PATH)
velocity = xr.open_dataset(VEL_PATH)

# make sure nan times are removed for bottle sample comparison
inner_nitrate = inner_nitrate.dropna(dim="time", how="all", subset=["nitrate"])
midshelf_nitrate = midshelf_nitrate.dropna(dim="time", how="all", subset=["nitrate"])

inner_ooi_crse = xr.open_dataset(DATA_DIR / "ship/CE01ISSP_bottle_samples.nc")
midshelf_ooi_crse = xr.open_dataset(DATA_DIR / "ship/CE02SHSP_bottle_samples.nc")
inner_nhl_crse = xr.open_dataset(DATA_DIR / "ship/NH01_NH03_bottle_samples.nc")
midshelf_nhl_crse = xr.open_dataset(DATA_DIR / "ship/NH10_bottle_samples.nc")


# %%
def get_overlapping_bottles(
    profiler: xr.Dataset,
    ship: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Find the profiler data that overlaps with the bottle samples within one day.

    Args:
        profiler (xr.Dataset): The profiler dataset, containing a "time" coordinate.
        ship (xr.Dataset): The ship dataset, containing a "time" coordinate and a "nitrate" variable.

    Returns:
        tuple[xr.Dataset, xr.Dataset]: A tuple containing two datasets:
            the first is the ship dataset with only the overlapping bottle samples
            and the second is the profiler dataset with only the overlapping data.

    """
    # find closest profiler data to OOI bottle samples
    profiler = profiler.sel(time=ship.time, method="nearest")
    ship_overlapping_bottles_slices: list[xr.Dataset] = []
    ship_notoverlapping_bottles_slices: list[xr.Dataset] = []
    prof_overlapping_bottles_slices: list[xr.Dataset] = []
    prof_notoverlapping_bottles_slices: list[xr.Dataset] = []
    for i, (t1, t2) in enumerate(zip(profiler.time.to_numpy(), ship.time.to_numpy(), strict=False)):
        # only consider profiler data within 1 day of OOI bottle samples
        if np.abs((t1 - t2) / np.timedelta64(1, "D")) < 1:
            ship_overlapping_bottles_slices.append(ship.isel(time=i))
            prof_overlapping_bottles_slices.append(profiler.isel(time=i))
        else:
            ship_notoverlapping_bottles_slices.append(ship.isel(time=i))
            prof_notoverlapping_bottles_slices.append(profiler.isel(time=i))
    print(
        f"{len(ship_overlapping_bottles_slices)} out of "
        f"{len(ship_overlapping_bottles_slices) + len(ship_notoverlapping_bottles_slices)} "
        "bottle samples overlap with profiler deployments within one day.",
    )
    print(
        f"{len(ship_notoverlapping_bottles_slices)} out of "
        f"{len(ship_overlapping_bottles_slices) + len(ship_notoverlapping_bottles_slices)} "
        "bottle samples do not overlap with profiler deployments within one day.",
    )

    # create combined datasets for overlapping and non-overlapping samples
    ship_overlapping_bottles = xr.concat(ship_overlapping_bottles_slices, dim="time")
    prof_overlapping_bottles = xr.concat(prof_overlapping_bottles_slices, dim="time")
    return ship_overlapping_bottles, prof_overlapping_bottles


# %% [markdown]
# ## Inner Shelf Nitrate Bottle Sample Comparison

# %%
ooi_overlapping_bottles, prof_overlapping_bottles = get_overlapping_bottles(
    profiler=inner_nitrate,
    ship=inner_ooi_crse,
)

# %%
plt.plot(
    ooi_overlapping_bottles.isel(time=2).nitrate,
    -ooi_overlapping_bottles.isel(time=2).temp1,
    "o",
)
plt.plot(
    prof_overlapping_bottles.isel(time=2).nitrate,
    -prof_overlapping_bottles.isel(time=2).temperature,
    "X",
)

# %%
nhl_overlapping_bottles, prof_overlapping_bottles = get_overlapping_bottles(
    profiler=inner_nitrate,
    ship=inner_nhl_crse,
)

# %%
temp = nhl_overlapping_bottles.isel(time=0).date
for i in range(len(nhl_overlapping_bottles.time)):
    if temp != nhl_overlapping_bottles.isel(time=i).date:
        plt.figure()
    plt.plot(
        nhl_overlapping_bottles.isel(time=i).nitrate,
        nhl_overlapping_bottles.isel(time=i).temperature,
        "o",
    )
    plt.plot(
        prof_overlapping_bottles.isel(time=i).nitrate,
        prof_overlapping_bottles.isel(time=i).temperature,
        "X",
    )
    plt.annotate(
        f"{nhl_overlapping_bottles.isel(time=i).date.to_numpy()}\n{np.unique(prof_overlapping_bottles.isel(time=i).deployment.to_numpy())}",
        (0.9, 0.9),
        xycoords="axes fraction",
    )
    temp = nhl_overlapping_bottles.isel(time=i).date

# %% [markdown]
# ## Mid Shelf Nitrate Bottle Sample Comparison

# %%
ooi_overlapping_bottles, prof_overlapping_bottles = get_overlapping_bottles(
    profiler=midshelf_nitrate,
    ship=midshelf_ooi_crse,
)

# %%
nhl_overlapping_bottles, prof_overlapping_bottles = get_overlapping_bottles(
    profiler=midshelf_nitrate,
    ship=midshelf_nhl_crse,
)

# %%
plt.plot(
    ooi_overlapping_bottles.isel(time=-1).temp1,
    ooi_overlapping_bottles.isel(time=-1).pressure,
    "o",
)
plt.plot(
    prof_overlapping_bottles.isel(time=-1).temperature,
    prof_overlapping_bottles.isel(time=-1).depth,
    "X",
)

# %%
temp = ooi_overlapping_bottles.isel(time=0).time
for i in range(len(ooi_overlapping_bottles.time)):
    plt.figure()
    plt.plot(
        ooi_overlapping_bottles.isel(time=i).nitrate,
        ooi_overlapping_bottles.isel(time=i).pressure,
        "o",
    )
    plt.plot(
        prof_overlapping_bottles.isel(time=i).nitrate,
        prof_overlapping_bottles.isel(time=i).depth,
        "X",
    )
    plt.annotate(
        f"{ooi_overlapping_bottles.isel(time=i).time.to_numpy()}\n{np.unique(prof_overlapping_bottles.isel(time=i).deployment.to_numpy())}",
        (0.9, 0.9),
        xycoords="axes fraction",
    )
    temp = ooi_overlapping_bottles.isel(time=i).time

# %%
nhl_overlapping_bottles, prof_overlapping_bottles = get_overlapping_bottles(
    profiler=midshelf_nitrate,
    ship=midshelf_nhl_crse,
)

# %% [markdown]
# ## Nitrate Density Relationship and Comparison with Bottle Samples

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 4), sharey=True)
# ax0.plot(midshelf_nitrate.density, midshelf_nitrate.nitrate ,'.', color='#004488', label='Profiler', rasterized=True)
# ax0.plot(nhl_crse.potential_density, nhl_crse.nitrate, 'o', color='#DDAA33', label='NHL')
ax0.plot(
    inner_nitrate["density"],
    inner_nitrate["nitrate"],
    ".",
    color="#004488",
    label="Profiler",
)
ax0.plot(
    inner_nhl_crse["potential_density"],
    inner_nhl_crse["nitrate"],
    "o",
    color="#DDAA33",
    label="NHL",
)
ax0.plot(
    inner_ooi_crse["pot_density_anom"],
    inner_ooi_crse["nitrate"],
    "X",
    color="#BB5566",
    label="OOI",
)
ax0.set_xlim(21, 27)

handles, labels = ax0.get_legend_handles_labels()
by_label = dict(
    zip(labels, handles, strict=False),
)  # dicts can't have duplicate keys, avoids duplicate legend entries
ax0.legend(by_label.values(), by_label.keys())
print(
    f"Mean N below 25.8 sigma: {np.nanmean(inner_nitrate.where(inner_nitrate['density'] < 25.8)['nitrate'])}",
)
print(
    f"Mean N above 25.8 sigma: {np.nanmean(inner_nitrate.where(inner_nitrate['density'] > 25.8)['nitrate'])}",
)

ax1.plot(
    midshelf_nitrate["density"],
    midshelf_nitrate["nitrate"],
    ".",
    color="#004488",
    label="Profiler",
    rasterized=True,
)
ax1.plot(
    midshelf_nhl_crse["potential_density"],
    midshelf_nhl_crse["nitrate"],
    "o",
    color="#DDAA33",
    label="NHL",
)
ax1.plot(
    midshelf_ooi_crse["pot_density_anom"],
    midshelf_ooi_crse["nitrate"],
    "X",
    color="#BB5566",
    label="OOI",
)
ax1.set_xlim(21, 27)
# ax1.set_xlabel('Density Anomaly ($\mathsf{kg \; m^{-3}}$)')
# ax1.set_ylabel('Nitrate Conc. [$\mathsf{\mu M}$]')
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(
    zip(labels, handles, strict=False),
)  # dicts can't have duplicate keys, avoids duplicate legend entries
ax1.legend(by_label.values(), by_label.keys())
fig.supxlabel("Potential Density Anomaly ($\\mathsf{kg \\; m^{-3}}$)", y=-0.02)
fig.supylabel("Nitrate Conc. [$\\mathsf{\\mu M}$]", x=0.02)

ax0.text(
    0.95,
    0.05,
    "(a)",
    transform=ax0.transAxes,
    fontsize=10,
    ha="right",
    va="bottom",
)
ax1.text(
    0.95,
    0.05,
    "(b)",
    transform=ax1.transAxes,
    fontsize=10,
    ha="right",
    va="bottom",
)

print(
    f"Mean N below 25.8 sigma: {np.nanmean(midshelf_nitrate.where(midshelf_nitrate['density'] < 25.8)['nitrate'])}",
)
print(
    f"Mean N above 25.8 sigma: {np.nanmean(midshelf_nitrate.where(midshelf_nitrate['density'] > 25.8)['nitrate'])}",
)

plt.savefig(
    FIGURES_DIR / f"manuscript/{FIG_SAVE_FMT}/nitrate-density.{FIG_SAVE_FMT}",
    format=FIG_SAVE_FMT,
    bbox_inches="tight",
    dpi=600,
)

# %% [markdown]
# ## Nitrate density relationship colored by spiciness
#
# Minor side quest to see if deviations from mean nitrate-density relationship are related to spiciness

# %%
import gsw

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
# ax0.plot(midshelf_nitrate.density, midshelf_nitrate.nitrate ,'.', color='#004488', label='Profiler', rasterized=True)
# ax0.plot(nhl_crse.potential_density, nhl_crse.nitrate, 'o', color='#DDAA33', label='NHL')
ax0.scatter(
    inner_nitrate["density"],
    inner_nitrate["nitrate"],
    c=gsw.spiciness0(inner_nitrate["absolute_salinity"], inner_nitrate["conservative_temperature"]),
    label="Profiler",
    alpha=0.7,
    vmin=-4,
    vmax=4,
    cmap=cmo.balance,
)
sca = ax1.scatter(
    midshelf_nitrate["density"],
    midshelf_nitrate["nitrate"],
    c=gsw.spiciness0(midshelf_nitrate["absolute_salinity"], midshelf_nitrate["conservative_temperature"]),
    label="Profiler",
    alpha=0.7,
    vmin=-4,
    vmax=4,
    cmap=cmo.balance,
)
plt.colorbar(sca, label="Spiciness ($\\mathsf{kg \\; m^{-3}}$)", ax=[ax0, ax1], orientation="vertical", pad=0.05)
fig.supxlabel("Potential Density Anomaly ($\\mathsf{kg \\; m^{-3}}$)", y=-0.02)
fig.supylabel("Nitrate Conc. ($\\mathsf{mmol \\; m^{-3}}$)", x=0.02)


# %% [markdown]
# It does seem like the

# %% [markdown]
# ## Density by Deployment Times

# %%
GLOBEC_TIME = slice(np.datetime64("1997-01-01"), np.datetime64("2004-12-31"))
NANOOS_TIME = slice(np.datetime64("2006-07-01"), np.datetime64("2014-09-30"))
OOI_TIME = slice(np.datetime64("2015-04-01"), None)

# %%
nhl_ctd = xr.open_dataset(
    DATA_DIR / "NHL_Gridded/newport_hydrographic_line_gridded_sections.nc",
)
nhl_ctd["longitude"] = nhl_ctd["longitude"] - 360
nhl_ctd = nhl_ctd.sel(longitude=-124.3).squeeze().dropna("pressure", how="all")

# %%
nhl_ctd_globec = nhl_ctd.sel(time=GLOBEC_TIME)
nhl_ctd_nanoos = nhl_ctd.sel(time=NANOOS_TIME)
nhl_ctd_ooi = nhl_ctd.sel(time=OOI_TIME)

# %%
plt.violinplot(
    [
        # nhl_ctd_globec["potential_density"].dropna("time").to_numpy().flatten(),
        nhl_ctd_nanoos["potential_density"].dropna("time").to_numpy().flatten(),
        nhl_ctd_ooi["potential_density"].dropna("time").to_numpy().flatten(),
    ],
    positions=[1, 2],
    # showmeans=True,
    showmedians=True,
    # quantiles=[0.25, 0.5, 0.75],
)
# plt.ylim(22, 28)
x_tick_labels = [
    # f"$\\mathbf{{GLOBEC}}$\n{GLOBEC_TIME.start}\nto\n{GLOBEC_TIME.stop}",
    f"$\\mathbf{{NANOOS}}$\n{NANOOS_TIME.start}\nto\n{NANOOS_TIME.stop}",
    f"$\\mathbf{{OOI}}$\n{OOI_TIME.start}\nto\n{'Present' if OOI_TIME.stop is None else OOI_TIME.stop}",
]
plt.gca().set_xticks([1, 2], x_tick_labels)
plt.ylabel("Potential Density Anomaly ($\\mathsf{kg \\; m^{-3}}$)")


# %%
def compare_nhl_densities(ds1: xr.Dataset, ds2: xr.Dataset) -> None:
    """Compare the potential density distributions of two datasets using a t-test.

    Args:
        ds1 (xr.Dataset): The first dataset, containing a "potential_density" variable and a "pressure" coordinate.
        ds2 (xr.Dataset): The second dataset, containing a "potential_density" variable and a "pressure" coordinate.

    """
    pressure = np.intersect1d(ds1["pressure"], ds2["pressure"])
    for p in pressure:
        ds1_p = ds1.sel(pressure=p, method="nearest").dropna("time")
        ds2_p = ds2.sel(pressure=p, method="nearest").dropna("time")
        if len(ds1_p.time) > 5 and len(ds2_p.time) > 5:
            ds1_p_stats = sm.stats.DescrStatsW(
                ds1_p["potential_density"].to_numpy().flatten(),
            )
            ds2_p_stats = sm.stats.DescrStatsW(
                ds2_p["potential_density"].to_numpy().flatten(),
            )
            ds1_ds2 = sm.stats.CompareMeans(ds1_p_stats, ds2_p_stats)
            if ds1_ds2.ttest_ind(alternative="two-sided")[1] < 0.05:
                print(f"Pressure: {p} dbar")
                print(ds1_ds2.ttest_ind(alternative="two-sided"))


# %%
nhl_ctd_nanoos_summer = nhl_ctd_nanoos.sel(
    time=nhl_ctd_nanoos["time.month"].isin([5, 6, 7, 8, 9]),
)
nhl_ctd_ooi_summer = nhl_ctd_ooi.sel(
    time=nhl_ctd_ooi["time.month"].isin([5, 6, 7, 8, 9]),
)

# %%
compare_nhl_densities(nhl_ctd_nanoos, nhl_ctd_ooi)


# %%
def generate_harmonics(t: np.ndarray, harmonics: int, f: float = 1 / 365.2422) -> np.ndarray:
    """Generate a matrix of sine and cosine harmonics for regression.

    Args:
        t (np.ndarray): 1D array of time values (e.g., day of year).
        harmonics (int): Number of harmonics to generate (e.g., 3 for annual, semiannual, and triannual).
        f (float): base frequency (default is 1 cycle per year).

    Returns:
        (xr.Dataset) A 2D array for input to sm.OLS

    """
    exog = np.full((harmonics * 2, len(t)), np.nan)
    for i in range(harmonics * 2):
        if i % 2 == 0:
            exog[i] = np.sin((i // 2 + 1) * 2 * np.pi * f * t)
        else:
            exog[i] = np.cos((i // 2 + 1) * 2 * np.pi * f * t)
    exog = exog.T
    exog = sm.add_constant(exog)
    return exog


def calc_climatology(
    t: np.ndarray,
    y: np.ndarray,
    harmonics: int,
    f: float = 1 / 365.2422,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the climatology of a time series using OLS regression with harmonic terms.

    Args:
        t (np.ndarray): 1D array of time values (e.g., day of year).
        y (np.ndarray): 1D array of the variable to fit (e.g., nitrate concentration).
        harmonics (np.ndarray): Number of harmonics to include in the regression.
        f (np.ndarray): base frequency for the harmonics (default is 1 cycle per year).

    Returns:
        (tuple[np.ndarray, np.ndarray]) containing the fitted climatology values and the regression results object.

    """
    exog = generate_harmonics(t, harmonics=harmonics, f=f)
    endog = y
    mod = sm.OLS(endog, exog)
    res = mod.fit()
    return res.params @ exog.T, res


def calc_climatology_by_depth(da: xr.DataArray, harmonics: int) -> xr.Dataset:
    """Calculate the climatology of a 2-D DataArray with `time` and `depth` dimensions.

    Args:
        da (xr.DataArray): DataArray with 2-D data with `time` and `depth` dimensions.
        harmonics (np.ndarray): Number of harmonics to include in the regression.

    Returns:
        (xr.Dataset) containing the fitted climatology values, standard deviation of the fit,
            and confidence interval of the fit for each depth level.

    """
    pressure = da["pressure"]
    fit_list = []
    for p in pressure:
        ds_p = da.sel(pressure=p).dropna("time")
        t_fit = np.arange(1, 366)
        exog = generate_harmonics(t_fit, harmonics=harmonics)
        if len(ds_p.time) > 50:
            ds_p = ds_p.groupby("time.dayofyear").mean()
            _, res = calc_climatology(
                ds_p["dayofyear"].to_numpy(),
                ds_p.to_numpy(),
                harmonics=harmonics,
            )
            fit_ds = xr.Dataset(
                {
                    "fit": (
                        ("dayofyear"),
                        res.params @ generate_harmonics(t_fit, harmonics=harmonics).T,
                    ),
                    "fit_std": (
                        ("dayofyear"),
                        np.sqrt(
                            res.mse_resid
                            * np.einsum(
                                "ij,jk,ki->i",
                                exog,
                                res.normalized_cov_params,
                                exog.T,
                            ),
                        ),
                    ),
                    "fit_ci": (
                        ("dayofyear"),
                        distributions.t.ppf(0.91, t_fit.size - exog.shape[1])
                        * np.sqrt(
                            res.mse_resid
                            * np.einsum(
                                "ij,jk,ki->i",
                                exog,
                                res.normalized_cov_params,
                                exog.T,
                            ),
                        ),
                    ),
                },
                coords={"dayofyear": t_fit},
            )
            fit_list.append(fit_ds)
        else:
            fit_ds = xr.Dataset(
                {
                    "fit": (("dayofyear"), np.full(365, np.nan)),
                    "fit_std": (("dayofyear"), np.full(365, np.nan)),
                    "fit_ci": (("dayofyear"), np.full(365, np.nan)),
                },
                coords={"dayofyear": t_fit},
            )
            fit_list.append(fit_ds)
    fit_ds = xr.concat(fit_list, dim="pressure")
    fit_ds = fit_ds.assign_coords({"pressure": pressure})

    return fit_ds


# %%
def add_colorbar(ax, mappable, label: str):  # noqa: ANN001, D103, ANN201
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar = plt.colorbar(mappable, cax=ax_cb, extend="both")
    cbar.set_label(label, fontsize=10)


nhl_clima_nanoos = calc_climatology_by_depth(
    nhl_ctd_nanoos["potential_density"],
    harmonics=3,
)
nhl_clima_ooi = calc_climatology_by_depth(nhl_ctd_ooi["potential_density"], harmonics=3)

fig, (ax0, ax1, ax2) = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(6, 8),
    sharex=True,
    sharey=True,
)
pcm0 = ax0.pcolormesh(
    nhl_clima_nanoos["dayofyear"] - 1,
    -nhl_clima_nanoos["pressure"],
    nhl_clima_nanoos["fit"],
    shading="auto",
    cmap=cmo.dense,
    vmin=24,
    vmax=27,
)
add_colorbar(ax0, pcm0, label="$\\mathsf{kg \\; m^{-3}}$")
pcm1 = ax1.pcolormesh(
    nhl_clima_ooi["dayofyear"] - 1,
    -nhl_clima_ooi["pressure"],
    nhl_clima_ooi["fit"],
    shading="auto",
    cmap=cmo.dense,
    vmin=24,
    vmax=27,
)
add_colorbar(ax1, pcm1, label="$\\mathsf{kg \\; m^{-3}}$")
pcm2 = ax2.pcolormesh(
    nhl_clima_ooi["dayofyear"] - 1,
    -nhl_clima_ooi["pressure"],
    nhl_clima_ooi["fit"] - nhl_clima_nanoos["fit"],
    shading="auto",
    cmap=cmo.balance,
    vmin=-0.5,
    vmax=0.5,
)
ax2.contour(
    nhl_clima_ooi["dayofyear"] - 1,
    -nhl_clima_ooi["pressure"],
    nhl_clima_ooi["fit"] - nhl_clima_nanoos["fit"],
    levels=[0],
    colors="k",
    linewidths=0.5,
)
add_colorbar(ax2, pcm2, label="$\\mathsf{kg \\; m^{-3}}$")

# Calculate the lower and upper bounds for both climatologies at each (pressure, dayofyear)
ooi_lower = nhl_clima_ooi["fit"] - nhl_clima_ooi["fit_ci"]
ooi_upper = nhl_clima_ooi["fit"] + nhl_clima_ooi["fit_ci"]
nanoos_lower = nhl_clima_nanoos["fit"] - nhl_clima_nanoos["fit_ci"]
nanoos_upper = nhl_clima_nanoos["fit"] + nhl_clima_nanoos["fit_ci"]

# Intervals intersect if the lower bound of one is less than the upper bound of the other and vice versa
intersect = (ooi_lower <= nanoos_upper) & (nanoos_lower <= ooi_upper)

ax2.contourf(
    intersect["dayofyear"],
    -intersect["pressure"],
    intersect.where(~intersect),  # only fill where there's no intersection
    hatches=["xx"],
    alpha=0,
    levels=[0, 1],
)
ax2.contour(
    intersect["dayofyear"],
    -intersect["pressure"],
    intersect,
    levels=[0],
    linestyles="--",
    colors="k",
    linewidths=2,
)

ax0.set_title("(a) NHL Potential Density Climatology: NANOOS (2006-2014)")
ax1.set_title("(b) NHL Potential Density Climatology: OOI (2015-Present)")
ax2.set_title("(c) Difference: OOI - NANOOS")

fig.supylabel("Depth (m)")
fig.supxlabel("Day of Year")

locator = mdates.MonthLocator()
month_fmt = mdates.DateFormatter("%b")
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(month_fmt)

# %% [markdown]
# Discussion on the overlapping of two confidence intervals with approximately equal variance and size: https://stats.stackexchange.com/questions/18215/relation-between-confidence-interval-and-testing-statistical-hypothesis-for-t-te/18259#18259

# %% [markdown]
# ## Comparing bottle samples with monthly median nitrate profiles

# %%
MINIMUM_N_STAR = 5
midshelf_nitrate_monthly = xr.Dataset(
    {
        "mean": midshelf_nitrate.groupby("time.month").mean(dim="time", skipna=True)["nitrate"],
        "std": midshelf_nitrate.groupby("time.month").std(dim="time", skipna=True)["nitrate"],
        "count": midshelf_nitrate.groupby("time.month").count(dim="time")["nitrate"],
    },
)
midshelf_nitrate_monthly["ci"] = midshelf_nitrate_monthly["std"] / np.sqrt(5) * distributions.t(5 - 1).isf(0.025)
midshelf_nitrate_monthly = midshelf_nitrate_monthly.where(
    midshelf_nitrate_monthly["count"] / 7 >= MINIMUM_N_STAR,  # N* at least ~5
)

inner_nitrate_monthly = xr.Dataset(
    {
        "mean": inner_nitrate.groupby("time.month").mean(dim="time", skipna=True)["nitrate"],
        "std": inner_nitrate.groupby("time.month").std(dim="time", skipna=True)["nitrate"],
        "count": inner_nitrate.groupby("time.month").count(dim="time")["nitrate"],
    },
)
inner_nitrate_monthly["ci"] = inner_nitrate_monthly["std"] / np.sqrt(5) * distributions.t(5 - 1).isf(0.025)
inner_nitrate_monthly = inner_nitrate_monthly.where(
    inner_nitrate_monthly["count"] / 7 >= MINIMUM_N_STAR,  # N* at least ~5
)

# %%
midshelf_ooi_crse_stacked = midshelf_ooi_crse["nitrate"].sel(time=midshelf_ooi_crse["time.month"] == 7).stack(z=[...])


# %%
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8, 6))

for i, m in enumerate(range(4, 10)):
    monthly_data = midshelf_nitrate_monthly.sel(month=m)
    monthly_ooi_crse_data = midshelf_ooi_crse["nitrate"].sel(time=midshelf_ooi_crse["time.month"] == m).stack(z=[...])
    ax = cast("plt.Axes", axs.flatten()[i])
    ax.plot(
        monthly_data["mean"],
        -monthly_data["depth"],
        color=colors[i],
    )
    ax.scatter(
        midshelf_nhl_crse.sel(time=midshelf_nhl_crse["time.month"] == m)["nitrate"],
        -midshelf_nhl_crse.sel(time=midshelf_nhl_crse["time.month"] == m)["pressure"],
        edgecolors=colors[i],
        facecolors="white",
        linewidths=3,
    )
    ax.scatter(
        monthly_ooi_crse_data,
        -monthly_ooi_crse_data["pressure"],
        marker="X",
        edgecolors=colors[i],
        facecolors="white",
        linewidths=3,
    )
    ax.fill_betweenx(
        -monthly_data["depth"],
        monthly_data["mean"] - monthly_data["ci"],
        monthly_data["mean"] + monthly_data["ci"],
        ls="None",
        edgecolor="None",
        facecolor=colors[i],
        alpha=0.5,
    )
    ax.annotate(
        f"{calendar.month_name[m]}\nN*$\\approx${np.ceil(monthly_data['count'].mean().to_numpy() / 7):.0f}",
        xy=(0.95, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        ha="right",
        verticalalignment="top",
    )
    ax.annotate(
        f"({string.ascii_lowercase[i]})",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
    )
    ax.set_xlim(0, 40)
    ax.set_ylim(-80, 0)
    ax.minorticks_off()

fig.supxlabel("Nitrate Concentration [$\\mathsf{m mol \\; m^{-3}}$]", y=0.03)
fig.supylabel("z [$\\mathsf{{m}}$]", x=0.03)

plt.savefig(
    FIGURES_DIR / f"manuscript/{FIG_SAVE_FMT}/midshelf-mean-profiles-with-bottle-samples.{FIG_SAVE_FMT}",
    format=FIG_SAVE_FMT,
    bbox_inches="tight",
    dpi=1200,
)
