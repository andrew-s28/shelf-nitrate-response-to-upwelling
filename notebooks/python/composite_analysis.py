# ---
# jupyter:
#   jupytext:
#     formats: ipynb,python//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import calendar
import string
from pathlib import Path

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import distributions
from tqdm import tqdm

# %%
# colormap for plotting
cmap = cmo.tools.crop_by_percent(cmo.balance_i, 30, which="both")  # type: ignore
cmap = cmap.from_list("cmap", cmap(np.linspace(0, 1, 11)), 11)

# %%
NOTEBOOK_DIR = Path().resolve()
DATA_DIR = NOTEBOOK_DIR / "../data"
FIGURES_DIR = NOTEBOOK_DIR / "../figures"
INNER_NITRATE_PATH = (
    DATA_DIR
    / "CE01ISSP/CE01ISSP_nitrate_binned_baseline_subtracted_2014-04-17_2023-09-17_with_dndt_resampled.nc"
)
MIDSHELF_NITRATE_PATH = (
    DATA_DIR
    / "CE02SHSP/CE02SHSP_nitrate_binned_baseline_subtracted_2015-03-18_2024-07-14_with_dndt_resampled.nc"
)
WIND_PATH = DATA_DIR / "NDBC_46050/46050_wind_binned_with_w5d_w8d.nc"
VEL_PATH = (
    DATA_DIR
    / "NH10_Mooring_Data/nh10_hourly_data_1997_2021_rotated_filtered_streamwise.nc"
)
OLD_VEL_PATH = list(
    Path(DATA_DIR / "NH10_Mooring_Data/").glob("nh10_hourly_data_1997_2021_part*.nc")
)
NEW_VEL_PATH = DATA_DIR / "NH10_Mooring_Data/ADCP_NH10_1997_2024_V5.nc"

VELOCITY_VARIABLE = "cs"

# %%
inner_nitrate = xr.open_dataset(INNER_NITRATE_PATH)
midshelf_nitrate = xr.open_dataset(MIDSHELF_NITRATE_PATH)
wind = xr.open_dataset(WIND_PATH, decode_timedelta=True)
velocity = xr.open_dataset(VEL_PATH)

# bit of a lazy way to use the cs_proj variable, since the notebook is set up for cs
if VELOCITY_VARIABLE == "cs_proj":
    velocity = velocity.drop_vars("cs").rename({"cs_proj": "cs"})

# resample midshelf nitrate to fill some of the gaps for composite calclulations
midshelf_nitrate = midshelf_nitrate.resample(time="1D").mean()

# interpolate velocity depths to match 1 meter bins in midshelf nitrate
velocity = velocity.interp(depth=midshelf_nitrate.depth)

# %% [markdown]
# ## Comparing New and Old Velocity Datasets

# %%
new_velocity = xr.open_dataset(NEW_VEL_PATH)
old_velocity = xr.open_mfdataset(OLD_VEL_PATH)

# %%
times = old_velocity["time"]

# %%
plt.plot(
    new_velocity.sel(time=times, method="nearest").u,
    old_velocity.squeeze().eastward_velocity,
    ".",
)
plt.gca().set_aspect("equal")

# %% [markdown]
# ## Examine Rotation on Velocity Dataset

# %%
plt.figure()
plt.plot(
    velocity["u_filt"].mean(dim="depth"),
    velocity["v_filt"].mean(dim="depth"),
    "o",
    mec="blue",
    mfc="blue",
)
# plt.figure()
plt.gca().set_aspect("equal")
plt.plot(
    velocity["cs_total"].mean(dim="depth"),
    velocity["as"].mean(dim="depth"),
    "o",
    mec="black",
    mfc="black",
)
plt.gca().set_aspect("equal")

# %% [markdown]
# ## Computing Composites Based on Wind Stress

# %%
days = np.arange(-5, 6)
composite_wind_events = []
for i, (t1, t2) in enumerate(
    zip(tqdm(wind.time[:-1], desc="Finding Wind Stress Events"), wind.time[1:])
):
    wind_t1 = wind.sel({"time": t1})
    wind_t2 = wind.sel({"time": t2})
    # find times when wind switches from above 0.03 to below -0.05
    if (wind_t2.coare_y < -0.05) & (wind_t1.coare_y > -0.03):
        wind_slice = wind.sel(time=slice(t2, t2 + np.timedelta64(5, "D")))
        # only include events that have upwelling favorable winds for at least 5 days after initial change
        if np.all(wind_slice.coare_y < 0):
            composite_wind_events.append(
                wind.sel(
                    time=slice(t2 - np.timedelta64(5, "D"), t2 + np.timedelta64(5, "D"))
                )
            )
            # drop unnecessary variables
            composite_wind_events[-1] = composite_wind_events[-1].drop_vars(
                ["dominant_wpd", "average_wpd"]
            )
# select only composites with the full amount of time points (11)
composite_wind_events = [c for c in composite_wind_events if len(c.time) == len(days)]
composite_vel_events = [
    velocity.where(velocity.time.isin(cw.time), drop=True)
    for cw in composite_wind_events
]
composite_vel_events = [cv for cv in composite_vel_events if cv.time.size == len(days)]
# do the same for midshelf nitrate, but note that with the resampling a lot of NaN values are still included here
composite_midshelf_nitrate_events = [
    midshelf_nitrate.where(midshelf_nitrate["time"].isin(cw.time), drop=True)
    for cw in composite_wind_events
]
composite_midshelf_nitrate_events = [
    cmn for cmn in composite_midshelf_nitrate_events if cmn.time.size == len(days)
]

# deal with overlapping events
# if the time between events is less than 5 days, combine them
composite_times = [c.time[5].values for c in composite_wind_events]
for i, (t1, t2) in enumerate(zip(composite_times[:-1], composite_times[1:])):
    if t2 - t1 < np.timedelta64(5, "D"):
        composite_wind_events[i] = composite_wind_events[i].sel(
            time=slice(None, t2 - np.timedelta64(1, "D"))
        )
composite_times = [c.time[5].values for c in composite_vel_events]
for i, (t1, t2) in enumerate(zip(composite_times[:-1], composite_times[1:])):
    if t2 - t1 < np.timedelta64(5, "D"):
        composite_vel_events[i] = composite_vel_events[i].sel(
            time=slice(None, t2 - np.timedelta64(1, "D"))
        )
composite_times = [c.time[5].values for c in composite_midshelf_nitrate_events]
for i, (t1, t2) in enumerate(zip(composite_times[:-1], composite_times[1:])):
    if t2 - t1 < np.timedelta64(5, "D"):
        composite_midshelf_nitrate_events[i] = composite_midshelf_nitrate_events[i].sel(
            time=slice(None, t2 - np.timedelta64(1, "D"))
        )

# %%
# different because the velocity data does not span as far back as wind data
n = len(composite_wind_events)
print(f"Number of composite wind events: {n}")
n = len(composite_vel_events)
print(f"Number of composite velocity events: {n}")
n = len(composite_midshelf_nitrate_events)
print(f"Number of composite midshelf nitrate events: {n}")


# %%
def composite(
    events: list[xr.Dataset],
    var: str,
    composite_days: np.ndarray,
    monthly: bool = False,
) -> xr.Dataset:
    """Takes a list of events and computes monthly or annual composites.

    Events are a list of datasets, each of which contains one event, over a event length defined by composite_days.

    :param events: event data
    :type events: list[xr.Dataset]
    :param var: variable to composite
    :type var: str
    :param composite_days: days for each event
    :type composite_days: np.array
    :param monthly: computes composite as a function of month, defaults to False
    :type monthly: bool, optional
    :return: dataset containing mean, std, count, and confidence interval for composite
    :rtype: xr.Dataset
    """
    composite_length = composite_days.size
    ds_list = np.empty(composite_length, dtype=xr.Dataset)
    # for each day in the composite length, combine each event to get the mean, std, and number of data points (count)
    for i in range(composite_length):
        composite_data = [
            d.isel(time=i) for d in events if len(d.time) == composite_length
        ]
        composite_data = xr.concat(composite_data, dim="time")
        if monthly:
            composite_mean = composite_data.groupby("time.month").mean(
                dim="time", skipna=True
            )[var]
            composite_std = composite_data.groupby("time.month").std(
                dim="time", skipna=True
            )[var]
            composite_count = composite_data.groupby("time.month").count(dim="time")[
                var
            ]
            ds_list[i] = xr.Dataset(
                {
                    "mean": composite_mean,
                    "std": composite_std,
                    "count": composite_count,
                },
                coords={"time": composite_days[i]},
            )
        else:
            composite_mean = composite_data.mean(dim="time", skipna=True)[var]
            composite_std = composite_data.std(dim="time", skipna=True)[var]
            composite_count = composite_data.count(dim="time")[var]
            ds_list[i] = xr.Dataset(
                {
                    "mean": composite_mean,
                    "std": composite_std,
                    "count": composite_count,
                },
            )
    ds = xr.concat(ds_list, dim="time")
    ds["ci"] = (
        ds["std"] / np.sqrt(ds["count"]) * distributions.t(ds["count"] - 1).isf(0.025)
    )
    return ds


# %%
composite_stress = composite(composite_wind_events, "coare_y", days)

fig, axs = plt.subplots()

axs.plot(days, composite_stress["mean"], color="black", label="Median")
axs.fill_between(
    days,
    composite_stress["mean"] - composite_stress["ci"],
    composite_stress["mean"] + composite_stress["ci"],
    ls="None",
    edgecolor="None",
    facecolor="black",
    alpha=0.5,
)

axs.set_xticks(np.arange(-5, 6))
axs.minorticks_off()
axs.set_xlabel("Days from Beginning of Event")
axs.set_ylabel("Wind Stress [$\\mathsf{N} \\; \\mathsf{m^{-2}}$]")

# %%
composite_stress_monthly = composite(
    composite_wind_events, "coare_y", days, monthly=True
)

# %%
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7, 4))
axs = axs.flatten()
for i, m in enumerate(composite_stress_monthly["month"].sel(month=slice(4, 9))):
    data = composite_stress_monthly.sel(month=m)
    axs[i].axhline(0, color="black", lw=1.5, ls="--")
    axs[i].plot(days, data["mean"], c="k")
    axs[i].fill_between(
        days,
        data["mean"] - data["ci"],
        data["mean"] + data["ci"],
        ls="None",
        edgecolor="None",
        facecolor="black",
        alpha=0.5,
    )
    axs[i].minorticks_off()
    axs[i].annotate(
        f"{calendar.month_abbr[m.values]} N$^*\\approx${composite_stress_monthly['count'].sel(month=m).mean(dim='time'):.0f}",
        xy=(0.95, 0.9),
        xycoords="axes fraction",
        va="top",
        ha="right",
    )
    axs[i].annotate(
        f"({string.ascii_lowercase[i]})",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        va="bottom",
        ha="left",
    )
# axs[0].set_ylabel('Wind Stress [$\mathsf{N} \; \mathsf{m^{-2}}$]')
fig.supylabel("Wind Stress [$\\mathsf{N} \\; \\mathsf{m^{-2}}$]")
fig.supxlabel("Days from Beginning of Upwelling Event")
plt.savefig(FIGURES_DIR / "manuscript/composite_wind_stress.pdf", format="pdf")

# %%
composite_vel_monthly_cs = composite(composite_vel_events, "cs", days, monthly=True)

# %%
fig, axs = plt.subplots(nrows=6, ncols=11, sharex=True, sharey=True, figsize=(12, 10))
for i, d in enumerate(composite_vel_monthly_cs["month"].sel(month=slice(4, 9))):
    for j, day in enumerate(days):
        data = composite_vel_monthly_cs.sel(month=d, time=day)
        axs[i][j].axvline(0, color="black")
        axs[i][j].plot(data["mean"], -data["depth"], c=cmap(j / 11))
        axs[i][j].fill_betweenx(
            -data["depth"],
            data["mean"] - data["ci"],
            data["mean"] + data["ci"],
            ls="None",
            edgecolor="None",
            facecolor=cmap(j / 11),
            alpha=0.5,
        )
        # axs[i][j].plot(data['mean'] + data['ci'], -data['depth'], '--', c=cmap(j/11))
        # axs[i][j].plot(data['mean'] - data['ci'], -data['depth'], '--', c=cmap(j/11))
        axs[i][j].set_xlim([-0.1, 0.1])
        if j == 0:
            axs[i][j].set_ylabel(
                calendar.month_abbr[data["month"].values] + f" (N={np.nanmean(n):.0f})"
            )
        if i == 0:
            axs[i][j].set_title(f"{day} days")
fig.supxlabel("Velocity [$\\mathsf{m \\; s^{-1}}$]")
fig.supylabel("Depth [$\\mathsf{m}$]")
fig.suptitle("Cross-shelf velocity")

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(5, 3))
composite_vel_monthly_cs_slice = composite_vel_monthly_cs.sel(
    time=slice(-3, 3, 3), month=slice(4, 9)
)


for i, v in enumerate(composite_vel_monthly_cs_slice["time"]):
    for j, m in enumerate(composite_vel_monthly_cs_slice.sel(time=v)["month"]):
        data = composite_vel_monthly_cs_slice.sel(time=v, month=m)
        axs[i].plot(
            data["mean"],
            -data["depth"],
            label=calendar.month_abbr[m.values],
            c=cmap(j / 6),
        )
        axs[i].minorticks_off()
        axs[i].set_ylim([-83, 2])

axs[0].annotate("-5 days\n(a)", xy=(0.10, 0.05), xycoords="axes fraction", fontsize=10)
axs[1].annotate("0 days\n(b)", xy=(0.10, 0.05), xycoords="axes fraction", fontsize=10)
axs[2].annotate("+5 days\n(c)", xy=(0.10, 0.05), xycoords="axes fraction", fontsize=10)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.9, 0.5))

# axs[0].annotate('-5 days', xy=(0.95, 0.9), xycoords='axes fraction', fontsize=10, ha='right')
# axs[1].annotate('0 days', xy=(0.95, 0.9), xycoords='axes fraction', fontsize=10, ha='right')
# axs[2].annotate('+5 days', xy=(0.95, 0.9), xycoords='axes fraction', fontsize=10, ha='right')

fig.supxlabel("Cross-shelf Velocity [$\\mathsf{m \\; s^{-1}}$]", y=-0.07)
fig.supylabel("Depth [$\\mathsf{m}$]", x=-0.02)
plt.savefig(
    FIGURES_DIR / "manuscript/composite_cs_velocity.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
composite_vel_monthly_as = composite(composite_vel_events, "as", days, monthly=True)

# %%
fig, axs = plt.subplots(nrows=6, ncols=11, sharex=True, sharey=True, figsize=(12, 10))
for i, d in enumerate(composite_vel_monthly_as["month"].sel(month=slice(4, 9))):
    for j, day in enumerate(days):
        data = composite_vel_monthly_as.sel(month=d, time=day)
        axs[i][j].axvline(0, color="black")
        axs[i][j].plot(data["mean"], -data["depth"], c=cmap(j / 11))
        axs[i][j].fill_betweenx(
            -data["depth"],
            data["mean"] - data["ci"],
            data["mean"] + data["ci"],
            ls="None",
            edgecolor="None",
            facecolor=cmap(j / 11),
            alpha=0.5,
        )
        # axs[i][j].plot(data['mean'] + data['ci'], -data['depth'], '--', c=cmap(j/11))
        # axs[i][j].plot(data['mean'] - data['ci'], -data['depth'], '--', c=cmap(j/11))
        # axs[i][j].set_xlim([-0.1, 0.1])
        if j == 0:
            axs[i][j].set_ylabel(
                calendar.month_abbr[data["month"].values] + f" (N={np.nanmean(n):.0f})"
            )
        if i == 0:
            axs[i][j].set_title(f"{day} days")
fig.supxlabel("Velocity [$\\mathsf{m \\; s^{-1}}$]")
fig.supylabel("Depth [$\\mathsf{m}$]")
fig.suptitle("Along-shelf velocity")

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(5, 3))
composite_vel_monthly_as_slice = composite_vel_monthly_as.sel(
    time=slice(-5, 5, 5), month=slice(4, 9)
)

for i, v in enumerate(composite_vel_monthly_as_slice["time"]):
    for j, m in enumerate(composite_vel_monthly_as_slice.sel(time=v)["month"]):
        data = composite_vel_monthly_as_slice.sel(time=v, month=m)
        axs[i].plot(
            data["mean"],
            -data["depth"],
            label=calendar.month_abbr[m.values],
            c=cmap(j / 6),
        )
        axs[i].minorticks_off()
        axs[i].set_ylim([-83, 2])

axs[0].annotate("-5 days\n(a)", xy=(0.10, 0.05), xycoords="axes fraction", fontsize=10)
axs[1].annotate("0 days\n(b)", xy=(0.10, 0.05), xycoords="axes fraction", fontsize=10)
axs[2].annotate("+5 days\n(c)", xy=(0.10, 0.05), xycoords="axes fraction", fontsize=10)
# axs[0].annotate('-5 days', xy=(0.95, 0.9), xycoords='axes fraction', fontsize=10, ha='right')
# axs[1].annotate('0 days', xy=(0.95, 0.9), xycoords='axes fraction', fontsize=10, ha='right')
# axs[2].annotate('+5 days', xy=(0.95, 0.9), xycoords='axes fraction', fontsize=10, ha='right')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.9, 0.5))

fig.supxlabel("Along-shelf Velocity [$\\mathsf{m \\; s^{-1}}$]", y=-0.07)
fig.supylabel("Depth [$\\mathsf{m}$]", x=-0.02)
plt.savefig(
    FIGURES_DIR / "manuscript/composite_as_velocity.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
midshelf_nitrate_med = midshelf_nitrate.where(midshelf_nitrate.depth < 79).median(
    dim="time"
)
midshelf_nitrate_std = midshelf_nitrate.where(midshelf_nitrate.depth < 79).std(
    dim="time"
)
plt.plot(
    midshelf_nitrate_med.nitrate,
    -midshelf_nitrate_med.depth,
    color="black",
    label="Median",
)
plt.fill_betweenx(
    -midshelf_nitrate_med.depth,
    midshelf_nitrate_med.nitrate - midshelf_nitrate_std.nitrate,
    midshelf_nitrate_med.nitrate + midshelf_nitrate_std.nitrate,
    ls="None",
    edgecolor="None",
    facecolor="black",
    alpha=0.5,
)
plt.plot(
    midshelf_nitrate_med.nitrate - midshelf_nitrate_std.nitrate,
    -midshelf_nitrate_med.depth,
    "--",
    color="black",
    label=r"Median $\pm$ 1$\sigma$",
)
plt.plot(
    midshelf_nitrate_med.nitrate + midshelf_nitrate_std.nitrate,
    -midshelf_nitrate_med.depth,
    "--",
    color="black",
)
plt.xlabel(r"Nitrate Concentration [$\mu \mathsf{M}$]")
plt.ylabel(r"Depth [$\mathsf{m}$]")
plt.legend()

# %%
midshelf_nitrate_monthly = xr.Dataset(
    {
        "mean": midshelf_nitrate.groupby("time.month").mean(dim="time", skipna=True)[
            "nitrate"
        ],
        "std": midshelf_nitrate.groupby("time.month").std(dim="time", skipna=True)[
            "nitrate"
        ],
        "count": midshelf_nitrate.groupby("time.month").count(dim="time")["nitrate"],
    }
)
midshelf_nitrate_monthly["ci"] = (
    midshelf_nitrate_monthly["std"] / np.sqrt(5) * distributions.t(5 - 1).isf(0.025)
)

# %%
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7, 4))
axs = axs.flatten()
for i, m in enumerate(range(4, 10)):
    monthly_data = midshelf_nitrate_monthly.isel(month=m)
    # monthly_data = monthly_data.where(monthly_data["count"] > 5, drop=True)
    axs[i].plot(monthly_data["mean"], -monthly_data["depth"], color="black")
    axs[i].fill_betweenx(
        -monthly_data["depth"],
        monthly_data["mean"] - monthly_data["ci"],
        monthly_data["mean"] + monthly_data["ci"],
        ls="None",
        edgecolor="None",
        facecolor="black",
        alpha=0.5,
    )
    # axs[i].plot(monthly_data['mean'] + monthly_data['ci'], -monthly_data['depth'], color='black', ls='--', label=f'NEff$\\approx${np.ceil(monthly_data['count'].mean()/7)}')
    # axs[i].plot(monthly_data['mean'] + monthly_data['ci'], -monthly_data['depth'], color='black', ls='--')
    axs[i].annotate(
        f"({string.ascii_lowercase[i]})\n{calendar.month_abbr[m]}\nN$^*\\approx${np.ceil(monthly_data['count'].mean().values / 7)}",
        xy=(0.075, 0.1),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    # axs[i].set_title(calendar.month_abbr[m.values])
    axs[i].set_xlim(0, 40)
    axs[i].set_ylim(-83, 2)
    axs[i].minorticks_off()

    # ax.legend()
fig.supxlabel("Nitrate Concentration [$\\mathsf{m mol \\; m^{-3}}$]", y=-0.03)
fig.supylabel("Depth [$\\mathsf{m}$]")
plt.savefig(
    FIGURES_DIR / "manuscript/monthly_midshelf_nitrate.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
composite_midshelf_nitrate_flux_monthly = (
    midshelf_nitrate_monthly * composite_vel_monthly_cs
)
midshelf_velocity_nitrate_cov = xr.cov(
    midshelf_nitrate["nitrate"], velocity["cs"], ["time"]
)
composite_midshelf_nitrate_flux_monthly["std"] = np.sqrt(
    (midshelf_nitrate_monthly["mean"] * composite_vel_monthly_cs["std"]) ** 2
    + (composite_vel_monthly_cs["mean"] * midshelf_nitrate_monthly["std"]) ** 2
    + 2
    * midshelf_nitrate_monthly["mean"]
    * composite_vel_monthly_cs["mean"]
    * midshelf_velocity_nitrate_cov
)
composite_midshelf_nitrate_flux_monthly["count"] = composite_vel_monthly_cs["count"]
composite_midshelf_nitrate_flux_monthly = (
    composite_midshelf_nitrate_flux_monthly.transpose(
        *composite_vel_monthly_cs["count"].dims
    )
)
composite_midshelf_nitrate_flux_monthly["ci"] = (
    composite_midshelf_nitrate_flux_monthly["std"]
    / np.sqrt(composite_midshelf_nitrate_flux_monthly["count"])
    * distributions.t(composite_midshelf_nitrate_flux_monthly["count"] - 1).isf(0.025)
)

# %%
fig, axs = plt.subplots(nrows=6, ncols=11, sharex=True, sharey=True, figsize=(12, 10))
for i, d in enumerate(
    composite_midshelf_nitrate_flux_monthly["month"].sel(month=slice(4, 9))
):
    for j, day in enumerate(days):
        data = composite_midshelf_nitrate_flux_monthly.sel(month=d, time=day)
        axs[i][j].axvline(0, color="black")
        axs[i][j].plot(data["mean"], -data["depth"], c=cmap(j / 11))
        axs[i][j].fill_betweenx(
            -data["depth"],
            data["mean"] - data["ci"],
            data["mean"] + data["ci"],
            ls="None",
            edgecolor="None",
            facecolor=cmap(j / 11),
            alpha=0.5,
        )
        # axs[i][j].plot(data['mean'] + data['ci'], -data['depth'], '--', c=cmap(j/11))
        # axs[i][j].plot(data['mean'] - data['ci'], -data['depth'], '--', c=cmap(j/11))
        # axs[i][j].set_xlim([-0.1, 0.1])
        if j == 0:
            axs[i][j].set_ylabel(
                calendar.month_abbr[data["month"].values] + f" (N={np.nanmean(n):.0f})"
            )
        if i == 0:
            axs[i][j].set_title(f"{day} days")
fig.supxlabel("Velocity [$\\mathsf{m \\; s^{-1}}$]")
fig.supylabel("Depth [$\\mathsf{m}$]")
fig.suptitle("Cross-shelf Nitrate Flux")

# %%
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(5, 3))
composite_midshelf_nitrate_flux_monthly_slice = (
    composite_midshelf_nitrate_flux_monthly.sel(time=slice(-5, 5, 5), month=slice(4, 9))
)

for i, v in enumerate(composite_midshelf_nitrate_flux_monthly_slice["time"]):
    for j, m in enumerate(
        composite_midshelf_nitrate_flux_monthly_slice.sel(time=v)["month"]
    ):
        data = composite_midshelf_nitrate_flux_monthly_slice.sel(time=v, month=m)
        axs[i].plot(
            data["mean"],
            -data["depth"],
            label=calendar.month_abbr[m.values],
            c=cmap(j / 6),
        )
        axs[i].minorticks_off()
        axs[i].set_xlim([-1, 1.5])
        axs[i].set_ylim([-83, 2])

axs[0].annotate(
    "-5 days\n(a)",
    xy=(0.9, 0.95),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
    va="top",
)
axs[1].annotate(
    "0 days\n(b)",
    xy=(0.9, 0.95),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
    va="top",
)
axs[2].annotate(
    "+5 days\n(c)",
    xy=(0.9, 0.95),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
    va="top",
)
# axs[0].annotate('-5 days', xy=(0.50, 0.02), xycoords='axes fraction', fontsize=10, ha='center')
# axs[1].annotate('0 days', xy=(0.50, 0.02), xycoords='axes fraction', fontsize=10, ha='center')
# axs[2].annotate('+5 days', xy=(0.50, 0.02), xycoords='axes fraction', fontsize=10, ha='center')
""
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.9, 0.5))

fig.supxlabel(
    "Cross-shelf Nitrate Flux [$\\mathsf{m mol \\; m^{-1} \\; s^{-1}}$]", y=-0.07
)
fig.supylabel("Depth [$\\mathsf{m}$]", x=-0.02)
plt.savefig(
    FIGURES_DIR / "manuscript/composite_cs_nflux.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%
flux_60_80 = np.full(
    (
        len(composite_midshelf_nitrate_flux_monthly["month"]),
        len(composite_midshelf_nitrate_flux_monthly["time"]),
    ),
    np.nan,
)
flux_20_60 = np.full(
    (
        len(composite_midshelf_nitrate_flux_monthly["month"]),
        len(composite_midshelf_nitrate_flux_monthly["time"]),
    ),
    np.nan,
)
flux_full = np.full(
    (
        len(composite_midshelf_nitrate_flux_monthly["month"]),
        len(composite_midshelf_nitrate_flux_monthly["time"]),
    ),
    np.nan,
)
for i, m in enumerate(composite_midshelf_nitrate_flux_monthly["month"]):
    for j, t in enumerate(composite_midshelf_nitrate_flux_monthly["time"]):
        # 60 m to 80 m flux
        data = composite_midshelf_nitrate_flux_monthly.sel(
            month=m, time=t, depth=slice(60, 80)
        )
        mask = ~np.isnan(data["mean"])
        data = data.isel(depth=mask)
        if len(data["depth"]) > 0:
            flux_60_80[i, j] = np.trapezoid(data["mean"], data["depth"])

        # 20 m to 40 m flux
        data = composite_midshelf_nitrate_flux_monthly.sel(
            month=m, time=t, depth=slice(20, 60)
        )
        mask = ~np.isnan(data["mean"])
        data = data.isel(depth=mask)
        if len(data["depth"]) > 0:
            flux_20_60[i, j] = np.trapezoid(data["mean"], data["depth"])

        # full flux
        data = composite_midshelf_nitrate_flux_monthly.sel(month=m, time=t)
        mask = ~np.isnan(data["mean"])
        data = data.isel(depth=mask)
        if len(data["depth"]) > 0:
            flux_full[i, j] = np.trapezoid(data["mean"], data["depth"])

# %%
# c = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']
# cmap = cmo.tools.crop_by_percent(cmo.thermal, 30, which='both')
# cmap = cmap.from_list('cmap', cmap(np.linspace(0, 1, 11)), 11)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 3))
for i, m in enumerate(
    composite_midshelf_nitrate_flux_monthly.sel(month=slice(4, 9))["month"]
):
    # data = composite_midshelf_nitrate_flux_monthly.sel(month=m)
    axs[0].plot(
        days, flux_60_80[m - 1], color=cmap(i / 6), label=calendar.month_abbr[m.values]
    )
    axs[1].plot(
        days, flux_20_60[m - 1], color=cmap(i / 6), label=calendar.month_abbr[m.values]
    )

axs[0].minorticks_off()
axs[1].minorticks_off()
axs[0].axhline(0, color="black")
axs[1].axhline(0, color="black")
axs[1].legend(loc="upper center", ncol=3)
fig.supxlabel("Days from Beginning of Upwelling Event", y=-0.07)
fig.supylabel("Nitrate Flux [$\\mathsf{mmol \\; m^{-1} \\; s^{-1}}$]", x=-0.02)
axs[0].annotate("(a)", xy=(0.88, 0.05), xycoords="axes fraction", fontsize=10)
axs[1].annotate("(b)", xy=(0.88, 0.05), xycoords="axes fraction", fontsize=10)

# reset cmap
cmap = cmo.tools.crop_by_percent(cmo.balance, 30, which="both")  # type: ignore
cmap = cmap.from_list("cmap", cmap(np.linspace(0, 1, 11)), 11)
plt.savefig(
    FIGURES_DIR / "manuscript/layer_cs_nflux.pdf",
    format="pdf",
    bbox_inches="tight",
)
