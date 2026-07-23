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
import re
from pathlib import Path

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cycler import cycler
from flox.xarray import xarray_reduce
from matplotlib import gridspec
from scipy.stats import distributions
from tqdm import tqdm

# %%
NOTEBOOK_DIR = Path().resolve()
DATA_DIR = NOTEBOOK_DIR / "../data"
FIGURES_DIR = NOTEBOOK_DIR / "../figures"
BROADSCALE_FILE = DATA_DIR / "broadscale/Nitrate_AllProjects.csv"


# %%
def bin_profiles(d, z):
    types = [d[i].dtype for i in d]
    var_names = list(d.keys())
    exclude = []
    for i, t in enumerate(types):
        if not (np.issubdtype(t, np.number)):
            exclude.append(var_names[i])
    d = d.drop_vars(exclude)
    out = xarray_reduce(
        d,
        d["time"],
        d["depth"],
        func="nanmean",
        expected_groups=(None, z),
        isbin=[False, True],
        method="map-reduce",
        skipna=True,
    )
    depth = np.array([x.mid for x in out.depth_bins.values])
    out["depth"] = (["depth" + "_bins"], depth)
    out = out.swap_dims({"depth" + "_bins": "depth"})
    return out


# %%
broadscale = pd.read_csv(
    BROADSCALE_FILE,
    header=0,
    index_col=False,
    names=[
        "id",
        "code",
        "date",
        "transect",
        "station",
        "lat",
        "lon",
        "depth",
        "no2",
        "no3",
        "no2+no3",
        "comments",
    ],
)
broadscale = broadscale.loc[broadscale.lon < 0]
nhl_lat = broadscale.loc[broadscale["transect"] == "Newport Hydrographic"].lat.median()
broadscale = broadscale.loc[(np.abs(broadscale["lat"] - nhl_lat) < 5)]
broadscale["time"] = pd.to_datetime(broadscale.date)
broadscale.set_index("time", inplace=True)
broadscale = broadscale.to_xarray()
plt.plot(broadscale.lon, broadscale.lat, ".")
plt.axhline(nhl_lat)
broadscale["transect_abbr"] = xr.DataArray(
    xr.apply_ufunc(
        lambda d: [re.split("(\\d+)", str(s))[0] for s in d],
        broadscale.station.values,
    ),
    dims="time",
)

# %%
broadscale.where(broadscale.depth < 100).no3.max()

# %%
transect_station = np.array(
    [broadscale.transect_abbr.values, broadscale.transect.values],
).T.astype(str)
transect_station = np.unique(transect_station, axis=0)
exclude_transect = transect_station[transect_station[:, 1] == "nan"].T[0]
transect_station = transect_station[(transect_station[:, 1] != "nan") & (transect_station[:, 0] != "nan")]
transect_station = dict(transect_station)
transect_lats = dict(
    zip(
        transect_station.keys(),
        np.array(
            [
                broadscale.where(broadscale.transect_abbr == ta, drop=True).lat.median().values
                for ta in transect_station.keys()
            ],
        ),
        strict=False,
    ),
)

# %%
stations = np.array(
    [t for t in dict(zip(broadscale.station.values, broadscale.transect.values, strict=False)).keys()],
)
nhl_lat = broadscale.where(broadscale.transect == "Newport Hydrographic", drop=True).lat.median().data
t = "Newport Hydrographic"
broadscale_binned = []
depth = np.concatenate(
    [np.arange(0, 300, 10), np.arange(300, 500, 50), np.arange(500, 4000, 100)],
)
for s in tqdm(stations):
    if s != "nan" and re.split("(\\d+)", s)[0] not in exclude_transect:
        transect = transect_station[re.split("(\\d+)", s)[0]]
        transect_lat = transect_lats[re.split("(\\d+)", s)[0]]
        lat = broadscale.where(broadscale.transect == transect, drop=True).lat.median().values
        temp = broadscale.where(broadscale.station == s, drop=True)
        temp = bin_profiles(temp, depth)
        lon = temp.lon.median().values
        temp = temp.expand_dims({"station": [s]}).drop_vars("lat")
        temp["lat"] = (["station"], [lat])
        temp["lon"] = (["station"], [lon])
        temp = temp.where(np.abs(temp.lat - transect_lat) < 0.025, drop=True)
        broadscale_binned.append(temp)

broadscale_binned = xr.concat(broadscale_binned, dim="station")

# %%
len(
    xr.apply_ufunc(
        lambda d: [re.split(r"(\d+)", str(s))[0] for s in d],
        broadscale_binned.station.values,
    ),
)

# %%
broadscale_binned["transect_abbr"] = xr.DataArray(
    xr.apply_ufunc(
        lambda d: [re.split(r"(\d+)", str(s))[0] for s in d],
        broadscale_binned.station.values,
    ),
    dims="station",
)

# %%
lats = []
for t in transect_station.values():
    lat = broadscale.where(broadscale.transect == t, drop=True).lat.median().values
    lats.append(lat)
lats = np.array(lats)
transect_station = dict(
    np.array(list(transect_station.items()))[np.argsort(lats)][::-1],
)
lats = np.sort(lats)[::-1]
lats

# %%
plt.plot(broadscale_binned.lon, broadscale_binned.lat, ".")

# %%
broadscale_binned.sel(
    station=[s.startswith("NH") for s in broadscale_binned.station.values],
)


# %%
def reshape_plots(fig, axs, r, c):
    gs = gridspec.GridSpec(r, c, fig)
    for i, (ax, g) in enumerate(zip(axs, gs, strict=False)):
        ax.set_subplotspec(g)


# %%
cc = cycler(marker=["o", "X", "+", "*", "o", "X", "+", "*", "o", "X", "+"]) + cycler(
    color=[
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
    ],
)
cc = cycler(
    color=[
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
        "#DDDDDD",
        "#000000",
    ],
) * cycler(marker=["o"])
cc = list(cc)

cmap = cmo.tools.crop_by_percent(cmo.balance_i, 20, which="both")
lats_reldiff = 0.5 * (lats - nhl_lat) / np.abs((lats - nhl_lat)[-1]) + 0.5

min_count = 3
station_miles = np.array([1, 3, 5, 10, 15, 20, 25, 30, 35, 45, 50])
fig, axs = plt.subplots(1, len(station_miles) + 1, figsize=(15, 8))
handles, labels = [], []
for i, mile in enumerate(station_miles):
    # axs[i].set_prop_cycle(cc)
    axs[i].set_title(f"{mile} miles")
    temp = broadscale_binned.station.values

    for stat in broadscale_binned.station.values:
        temp = broadscale_binned.sel(station=stat)
        if int(re.split("(\\d+)", stat)[1]) == mile:
            broadscale_depth = temp.depth.values
            broadscale_mean = temp.no3.mean(dim="time").values
            broadscale_std = temp.no3.std(dim="time").values
            broadscale_count = temp.no3.count(dim="time").values
            broadscale_depth = broadscale_depth[broadscale_count > min_count]
            broadscale_mean = broadscale_mean[broadscale_count > min_count]
            broadscale_std = broadscale_std[broadscale_count > min_count]
            broadscale_count = broadscale_count[broadscale_count > min_count]
            broadscale_ci = np.array(
                [
                    std / np.sqrt(n) * distributions.t(n - 1).isf(0.025)
                    for std, n in zip(broadscale_std, broadscale_count, strict=False)
                ],
            )
            cc_idx = list(transect_station.keys()).index(re.split("(\\d+)", stat)[0])
            if len(broadscale_mean) > 0:
                axs[i].plot(
                    broadscale_mean,
                    -broadscale_depth,
                    marker="o",
                    color=cmap(lats_reldiff)[cc_idx],
                    label=transect_station[re.split("(\\d+)", stat)[0]],
                )
                axs[i].plot(
                    np.stack(
                        [
                            broadscale_mean - broadscale_ci,
                            broadscale_mean + broadscale_ci,
                        ],
                    ),
                    np.stack([-broadscale_depth, -broadscale_depth]),
                    marker="|",
                    color=cmap(lats_reldiff)[cc_idx],
                )
                # axs[i].plot(broadscale_count, -broadscale_depth, marker='o', color=cmap(lats_reldiff[cc_idx]))
                axs[i].set_xlim(0, 50)
    hand, lab = axs[i].get_legend_handles_labels()
    handles.append(hand)
    labels.append(lab)

axs[-1].set_axis_off()
reshape_plots(fig, axs, 2, 6)
handles = [hj for hi in handles for hj in hi]
labels = [lj for li in labels for lj in li]
labels, handles = np.array(
    [(hand, lab) for hand, lab in dict(zip(labels, handles, strict=False)).items()],
).T
handles = handles[
    np.argsort(
        np.array(
            [
                transect_lats[list(transect_station.keys())[list(transect_station.values()).index(lab)]]
                for lab in labels
            ],
        ),
    )[::-1]
]
labels = labels[
    np.argsort(
        np.array(
            [
                transect_lats[list(transect_station.keys())[list(transect_station.values()).index(lab)]]
                for lab in labels
            ],
        ),
    )[::-1]
]
axs[-1].legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5))
[ax.set_ylim(-200, 0) for ax in axs]

# %%
cc = cycler(marker=["o", "X", "+", "*", "o", "X", "+", "*", "o", "X", "+"]) + cycler(
    color=[
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#BBBBBB",
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
    ],
)
cc = cycler(
    color=[
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
        "#AA4499",
        "#DDDDDD",
        "#000000",
    ],
) * cycler(marker=["o"])
cc = list(cc)

cmap = cmo.tools.crop_by_percent(cmo.balance_i, 20, which="both")
lats_reldiff = 0.5 * (lats - nhl_lat) / np.abs((lats - nhl_lat)[-1]) + 0.5

min_count = 3
station_miles = np.array([1, 3, 5, 10])
fig, axs = plt.subplots(1, len(station_miles) + 1, figsize=(15, 8))
handles, labels = [], []
for i, mile in enumerate(station_miles):
    # axs[i].set_prop_cycle(cc)
    axs[i].set_title(f"{mile} miles")
    temp = broadscale_binned.station.values

    for stat in broadscale_binned.station.values:
        temp = broadscale_binned.sel(station=stat)
        if int(re.split("(\\d+)", stat)[1]) == mile:
            broadscale_depth = temp.depth.values
            broadscale_mean = temp.no3.mean(dim="time").values
            broadscale_std = temp.no3.std(dim="time").values
            broadscale_count = temp.no3.count(dim="time").values
            broadscale_depth = broadscale_depth[broadscale_count > min_count]
            broadscale_mean = broadscale_mean[broadscale_count > min_count]
            broadscale_std = broadscale_std[broadscale_count > min_count]
            broadscale_count = broadscale_count[broadscale_count > min_count]
            broadscale_ci = np.array(
                [
                    std / np.sqrt(n) * distributions.t(n - 1).isf(0.025)
                    for std, n in zip(broadscale_std, broadscale_count, strict=False)
                ],
            )
            cc_idx = list(transect_station.keys()).index(re.split("(\\d+)", stat)[0])
            if len(broadscale_mean) > 0:
                axs[i].plot(
                    broadscale_mean,
                    -broadscale_depth,
                    marker="o",
                    color=cmap(lats_reldiff)[cc_idx],
                    label=transect_station[re.split("(\\d+)", stat)[0]],
                )
                axs[i].plot(
                    np.stack(
                        [
                            broadscale_mean - broadscale_ci,
                            broadscale_mean + broadscale_ci,
                        ],
                    ),
                    np.stack([-broadscale_depth, -broadscale_depth]),
                    marker="|",
                    color=cmap(lats_reldiff)[cc_idx],
                )
                # axs[i].plot(broadscale_count, -broadscale_depth, marker='o', color=cmap(lats_reldiff[cc_idx]))
                axs[i].set_xlim(0, 50)
    hand, lab = axs[i].get_legend_handles_labels()
    handles.append(hand)
    labels.append(lab)

axs[-1].set_axis_off()
reshape_plots(fig, axs, 2, 6)
handles = [hj for hi in handles for hj in hi]
labels = [lj for li in labels for lj in li]
labels, handles = np.array(
    [(hand, lab) for hand, lab in dict(zip(labels, handles, strict=False)).items()],
).T
handles = handles[
    np.argsort(
        np.array(
            [
                transect_lats[list(transect_station.keys())[list(transect_station.values()).index(lab)]]
                for lab in labels
            ],
        ),
    )[::-1]
]
labels = labels[
    np.argsort(
        np.array(
            [
                transect_lats[list(transect_station.keys())[list(transect_station.values()).index(lab)]]
                for lab in labels
            ],
        ),
    )[::-1]
]
axs[-1].legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5))
[ax.set_ylim(-200, 0) for ax in axs]

# %%
(
    broadscale_binned.sel(
        station=[s.startswith("GH") for s in broadscale_binned.station.values],
    ).lat.mean()
    - broadscale_binned.sel(
        station=[s.startswith("NH") for s in broadscale_binned.station.values],
    ).lat.mean()
)

# %%
(
    broadscale_binned.sel(
        station=[s.startswith("NH") for s in broadscale_binned.station.values],
    ).lat.mean()
    - broadscale_binned.sel(
        station=[s.startswith("TH") for s in broadscale_binned.station.values],
    ).lat.mean()
)

# %%
for (stat, trans), lat_rel in zip(transect_station.items(), lats_reldiff, strict=False):
    if trans in labels:
        temp = broadscale_binned.where(
            broadscale_binned.transect_abbr == stat,
            drop=True,
        )
        plt.plot(temp.lon, temp.lat, "-o", color=cmap(lat_rel), label=trans)
