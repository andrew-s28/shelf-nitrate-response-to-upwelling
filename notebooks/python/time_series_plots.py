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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import calendar
import string
from pathlib import Path
from typing import cast

import cmocean.cm as cmo
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm
from matplotlib.figure import Figure
from scipy.stats import distributions
from tqdm import tqdm

# %%
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
CHL_PATH = DATA_DIR / "CE01ISSM/CE01ISSM_chlorophyll_2014-04-17_2025-09-22.nc"

VELOCITY_VARIABLE = "cs_proj"

# %%
inner_nitrate = xr.open_dataset(INNER_NITRATE_PATH)
midshelf_nitrate = xr.open_dataset(MIDSHELF_NITRATE_PATH)
wind = xr.open_dataset(WIND_PATH, decode_timedelta=True)
velocity = xr.open_dataset(VEL_PATH)
inner_shelf_chlorophyll = xr.open_dataset(CHL_PATH)
inner_shelf_chlorophyll = inner_shelf_chlorophyll.sortby("time").resample(time="1D").mean()


# %% [markdown]
# ## Nitrate and Wind Time Series


# %%
def _plot_wind_time_series(
    ax: plt.Axes,
    wind: xr.Dataset,
    year: int,
) -> None:
    wind = wind.where(wind["time.year"] == year, drop=True)

    ax.plot(wind.time, wind.coare_y, linestyle="-", linewidth=1, color="#4477AA", label="Stress", zorder=1.8)
    ax.plot(wind.time, wind.w5d, linestyle="-", linewidth=2, color="#EE6677", label=r"$\mathrm{W_{5d}}$", zorder=1.8)
    ax.axhline(0, ls="-", lw=2, color="black", zorder=1.5)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlim(np.datetime64(f"{year}-04-01"), np.datetime64(f"{year}-09-30"))
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.text(
        0.02,
        0.05,
        "Upwelling\nFavorable",
        bbox={"facecolor": "white", "alpha": 1, "edgecolor": "white", "boxstyle": "round,pad=0."},
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        zorder=2,
    )
    ax.text(
        0.02,
        0.95,
        "Downwelling\nFavorable",
        bbox={"facecolor": "white", "alpha": 1, "edgecolor": "white", "boxstyle": "round,pad=0."},
        transform=ax.transAxes,
        va="top",
        ha="left",
        zorder=2,
    )
    ax.set_ylabel("Along-shelf wind stress\n[$\\mathsf{N\\;m^{-2}}$]")
    ax.legend(ncols=2, loc="upper right", frameon=True, framealpha=1, columnspacing=1)


def _plot_velocity_time_series(
    ax: plt.Axes,
    velocity: xr.Dataset,
    year: int,
) -> None:
    velocity = velocity.where(velocity["time.year"] == year, drop=True)

    ax.axhline(0, zorder=1.5, color="k")
    ax.plot(
        velocity.time,
        velocity.sel(depth=30)["u_proj"],
        linestyle="-",
        linewidth=1,
        label="30 m",
        color="#4477AA",
    )
    ax.plot(
        velocity.time,
        velocity.sel(depth=70)["u_proj"],
        linestyle="-",
        linewidth=2,
        label="70 m",
        color="#EE6677",
    )
    ax.set_ylabel("Cross-shelf\nVelocity [$\\mathsf{m \\; s^{-1}}$]")
    ax.legend(
        ncols=2,
        loc="upper right",
        frameon=True,
        framealpha=1,
        columnspacing=1,
    )
    ax.text(
        0.02,
        0.05,
        "Offshore",
        bbox={"facecolor": "white", "alpha": 1, "edgecolor": "white", "boxstyle": "round,pad=0."},
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        zorder=2,
    )
    ax.text(
        0.02,
        0.95,
        "Onshore",
        bbox={"facecolor": "white", "alpha": 1, "edgecolor": "white", "boxstyle": "round,pad=0."},
        transform=ax.transAxes,
        va="top",
        ha="left",
        zorder=2,
    )


def _plot_nitrate_time_series(
    fig: Figure,
    ax: plt.Axes,
    nitrate: xr.Dataset,
    year: int,
    location: str,
) -> None:
    # Filter to year and max depth
    nitrate = nitrate.where(nitrate["time.year"] == year)
    # Use negative depths for plotting
    nitrate["depth"] = -nitrate["depth"]
    nitrate = nitrate.sortby("depth")

    # Get unique deployments
    deployments = np.unique(nitrate["deployment"])
    deployments = deployments[~np.isnan(deployments)]

    # Setup colormap and discrete colors
    deep_r = cmo.tools.crop_by_percent(
        cmo.deep_r,  # pyright: ignore[reportAttributeAccessIssue]; reversed so deeper is darker
        20,
        which="max",
        N=len(nitrate["depth"]),
    )
    colors = deep_r(np.linspace(0, 1, len(nitrate["depth"])))

    # Plot each depth and deployment
    black_line_depths = np.arange(-80, 1, 10)
    for i, depth in enumerate(nitrate["depth"].to_numpy()):
        ax.plot(
            nitrate.sel(depth=depth).time,
            nitrate.sel(depth=depth).nitrate,
            color=colors[i],
            lw=0.5,
        )
        if depth in black_line_depths:
            ax.plot(
                nitrate.sel(depth=depth).time,
                nitrate.sel(depth=depth).nitrate,
                color="black",
                lw=2,
            )

    # Format x-axis
    dformatter = mdates.DateFormatter("%b")
    ax.xaxis.set_major_formatter(dformatter)

    # Format y-axis
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    if location.lower() == "midshelf":
        ax.set_ylabel("Midshelf nitrate conc.\n[$\\mathsf{mmol \\; m^{-3}}$]")
    elif location.lower() == "inner":
        ax.set_ylabel("Inshore nitrate conc.\n[$\\mathsf{mmol \\; m^{-3}}$]")
    else:
        ax.set_ylabel("Nitrate conc.\n[$\\mathsf{mmol \\; m^{-3}}$]")

    # Add colorbar
    cax = ax.inset_axes([1.02, 0, 0.02, 1])
    scm = cm.ScalarMappable(cmap=deep_r, norm=mcolors.Normalize(vmin=nitrate["depth"].min(), vmax=0))
    cbar = fig.colorbar(scm, cax=cax, fraction=0.01)
    cbar.ax.tick_params(labelsize=10, pad=5)
    cbar.ax.minorticks_off()
    cbar.set_label("z [m]", labelpad=10, rotation=270)


def _plot_chlorophyll_time_series(
    ax: plt.Axes,
    chlorophyll: xr.Dataset,
    year: int,
) -> None:
    chlorophyll = chlorophyll.where(chlorophyll["time.year"] == year, drop=True)

    ax.plot(
        chlorophyll["time"],
        chlorophyll["chlorophyll_a"],
        linestyle="-",
        linewidth=2,
        color="#228833",
    )
    ax.set_ylabel("Estimated chlorophyll\n[$\\mathsf{mg \\; m^{-3}}$]")
    ax.set_ylim(0, 25)


def plot_wind_velocity_nitrate_time_series(
    wind: xr.Dataset,
    velocity: xr.Dataset,
    inner_nitrate: xr.Dataset,
    midshelf_nitrate: xr.Dataset,
    year: int,
    *,  # make save a keyword-only argument
    save: bool = False,
) -> None:
    """Plot wind, velocity, and nitrate time series for a given year.

    Args:
        wind (xr.Dataset): xarray Dataset containing wind data.
        velocity (xr.Dataset): xarray Dataset containing ADCP velocity data.
        inner_nitrate (xr.Dataset): xarray Dataset containing inner shelf nitrate data.
        midshelf_nitrate (xr.Dataset): xarray Dataset containing midshelf nitrate data.
        year (int): Year to plot.
        save (bool, optional): Whether to save the figure. Defaults to False.
        max_nitrate_conc (float, optional): Maximum nitrate concentration to plot. Defaults to np.inf.
        max_inner_shelf_depth (float, optional): Maximum depth for inner shelf nitrate. Defaults to np.inf.
        max_midshelf_depth (float, optional): Maximum depth for midshelf nitrate. Defaults to np.inf.

    """
    # Don't show plot if saving
    if save:
        plt.ioff()
    fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True, layout="constrained")

    axs = cast("list[plt.Axes]", list(axs))
    fig.align_ylabels(axs)

    # Plot each time series
    _plot_wind_time_series(axs[0], wind, year)
    _plot_velocity_time_series(axs[1], velocity, year)
    _plot_nitrate_time_series(fig, axs[2], inner_nitrate, year, "inner")
    _plot_nitrate_time_series(fig, axs[3], midshelf_nitrate, year, "midshelf")
    _plot_chlorophyll_time_series(axs[4], inner_shelf_chlorophyll, year)

    # Add panel labels and turn off minor ticks
    for i, ax in enumerate(axs):
        ax.minorticks_off()
        ax.annotate(
            f"({string.ascii_lowercase[i]})",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 1, "edgecolor": "white", "boxstyle": "round,pad=0."},
        )

    if save:
        plt.savefig(
            FIGURES_DIR / f"manuscript/si/{FIG_SAVE_FMT}/wind-nitrate-time-series-{year}.{FIG_SAVE_FMT}",
            format=FIG_SAVE_FMT,
            bbox_inches="tight",
            dpi=600,
        )
    # Manually show plot if not saving
    if not save:
        plt.show()
    # Restore interactive mode for future plots
    plt.ion()
    plt.close()


# %%
for i in tqdm(range(2014, 2025)):
    plot_wind_velocity_nitrate_time_series(
        wind,
        velocity,
        inner_nitrate,
        midshelf_nitrate,
        i,
        save=True,
    )

# %%
midshelf_depth_mean_monthly = (
    midshelf_nitrate["nitrate"].mean(dim="depth").groupby("time.month").mean(dim="time").to_dataset()
)
midshelf_depth_mean_monthly["nitrate_std"] = (
    midshelf_nitrate["nitrate"].mean(dim="depth").groupby("time.month").std(dim="time")
)
midshelf_depth_mean_monthly["nitrate_count"] = (
    midshelf_nitrate["nitrate"].mean(dim="depth").groupby("time.month").count(dim="time")
)
midshelf_depth_mean_monthly["nitrate_ci"] = (
    midshelf_depth_mean_monthly["nitrate_std"] * distributions.t(10 - 1).isf(0.025) / np.sqrt(10)
)

inner_depth_mean_monthly = (
    inner_nitrate["nitrate"].mean(dim="depth").groupby("time.month").mean(dim="time").to_dataset()
)
inner_depth_mean_monthly["nitrate_std"] = (
    inner_nitrate["nitrate"].mean(dim="depth").groupby("time.month").std(dim="time")
)
inner_depth_mean_monthly["nitrate_count"] = (
    inner_nitrate["nitrate"].mean(dim="depth").groupby("time.month").count(dim="time")
)
inner_depth_mean_monthly["nitrate_ci"] = (
    inner_depth_mean_monthly["nitrate_std"] * distributions.t(10 - 1).isf(0.025) / np.sqrt(10)
)

# %%
fig, axs = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

axs.plot(
    inner_depth_mean_monthly["month"],
    inner_depth_mean_monthly["nitrate"],
    color="#004488",
    label="Inner-shelf",
)

axs.fill_between(
    inner_depth_mean_monthly["month"],
    inner_depth_mean_monthly["nitrate"] - inner_depth_mean_monthly["nitrate_ci"],
    inner_depth_mean_monthly["nitrate"] + inner_depth_mean_monthly["nitrate_ci"],
    alpha=0.5,
    color="#004488",
)

axs.plot(
    midshelf_depth_mean_monthly["month"],
    midshelf_depth_mean_monthly["nitrate"],
    color="#BB5566",
    label="Mid-shelf",
)

axs.fill_between(
    midshelf_depth_mean_monthly["month"],
    midshelf_depth_mean_monthly["nitrate"] - midshelf_depth_mean_monthly["nitrate_ci"],
    midshelf_depth_mean_monthly["nitrate"] + midshelf_depth_mean_monthly["nitrate_ci"],
    alpha=0.5,
    color="#BB5566",
)

axs.set_xlim(4, 10)
axs.set_ylim(0, 30)
axs.minorticks_off()
axs.legend(loc="lower right")
axs.xaxis.set_major_formatter(lambda x, pos: calendar.month_abbr[int(x)])  # noqa: ARG005
axs.set_ylabel("Depth mean nitrate conc. [$\\mathsf{mmol \\; m^{-3}}$]")

plt.savefig(
    FIGURES_DIR / f"manuscript/{FIG_SAVE_FMT}/monthly-depth-mean-nitrate.{FIG_SAVE_FMT}",
    format=FIG_SAVE_FMT,
    bbox_inches="tight",
    dpi=600,
)
