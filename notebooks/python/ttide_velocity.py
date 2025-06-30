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
import os
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import xarray as xr
from IPython.display import HTML
from matplotlib.figure import Figure
from numpy.typing import NDArray

import ttide as tt

HTML("""
    <style>
    .cell-output-ipywidget-background{background: transparent !important;}
    .widget-label{color: white !important;}
    </style>
""")

# %%
NOTEBOOK_DIR = Path().resolve()
DATA_DIR = NOTEBOOK_DIR / "../data"
FIGURES_DIR = NOTEBOOK_DIR / "../figures"
VEL_PATH_V1 = (
    DATA_DIR
    / "NH10_Mooring_Data/nh10_hourly_data_1997_2021_rotated_filtered_streamwise.nc"
)
VEL_PATH_V5 = (
    DATA_DIR
    / "NH10_Mooring_Data/nh10_hourly_data_1997_2021_rotated_filtered_streamwise_v5.nc"
)

GLOBEC_TIME = slice(np.datetime64("1997-01-01"), np.datetime64("2004-12-31"))
NANOOS_TIME = slice(np.datetime64("2006-07-01"), np.datetime64("2014-09-30"))
OOI_TIME = slice(np.datetime64("2015-04-01"), None)


# %%
def fit_ttide(u: xr.DataArray, v: xr.DataArray) -> dict:
    """Fit a T-Tide model to the given u and v components."""
    # u and v should have the same time dimension
    seconds_since_epoch = (u.time[0].values - np.datetime64(0, "s")) / np.timedelta64(
        1, "s"
    )
    stime = datetime.fromtimestamp(seconds_since_epoch, tz=timezone.utc)

    # don't need to print the output of t_tide, so redirect stdout to devnull
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull):
            # just save the dict output of t_tide
            out = tt.t_tide(
                u[:].values + v[:].values * 1j,
                lat=44.64,
                dt=1,
                stime=stime,
            )

    # squeeze the output arrays to remove any singleton dimensions
    out = {k: v.squeeze() for k, v in out.items() if isinstance(v, np.ndarray)}

    # out is a dict with keys 'nobs', 'ngood', 'dt', 'xin', 'xout', 'xres', 'xingd', 'xoutgd', 'xresgd', 'isComplex', 'ray', 'nodcor', 'z0', 'dz0', 'fu', 'nameu', 'tidecon', 'snr', 'synth', 'lat', 'ltype', 'stime'
    return out


def fit_ttide_from_ds(
    ds: xr.Dataset,
) -> xr.Dataset:
    ds_list = []
    for depth in ds["depth"]:
        if np.all(np.isnan(ds["u"].sel(depth=depth))) or np.all(
            np.isnan(ds["v"].sel(depth=depth))
        ):
            continue
        out = fit_ttide(
            ds["u"].sel(depth=depth),
            ds["v"].sel(depth=depth),
        )
        out["nameu"] = [name.astype(str).strip() for name in out["nameu"]]
        out_ds = xr.Dataset(
            {
                "uin": (["time"], out["xin"].real),
                "vin": (["time"], out["xin"].imag),
                "uout": (["time"], out["xout"].real),
                "vout": (["time"], out["xout"].imag),
                "ures": (["time"], out["xres"].real),
                "vres": (["time"], out["xres"].imag),
                "fu": (["constituent"], out["fu"]),
                "snr": (["constituent"], out["snr"]),
                "major": (["constituent"], out["tidecon"][:, 0]),
                "emajor": (["constituent"], out["tidecon"][:, 1]),
                "minor": (["constituent"], out["tidecon"][:, 2]),
                "eminor": (["constituent"], out["tidecon"][:, 3]),
                "inc": (["constituent"], out["tidecon"][:, 4]),
                "einc": (["constituent"], out["tidecon"][:, 5]),
                "phase": (["constituent"], out["tidecon"][:, 6]),
                "ephase": (["constituent"], out["tidecon"][:, 7]),
            },
            coords={
                "time": ds.time,
                "constituent": out["nameu"],
            },
        )
        out_ds = out_ds.expand_dims({"depth": [depth.values.astype(int)]}, axis=0)
        ds_list.append(out_ds)

    tide_out = xr.concat(ds_list, dim="depth")
    return tide_out


def plot_tidal_constituents(
    tide_ds: xr.Dataset,
    constituent: str,
    fig: Figure | None = None,
    snr: bool = True,
    **kwargs,
) -> tuple[Figure, NDArray]:
    """Plot the tidal constituents from a T-Tide output dataset."""
    if fig is None:
        if snr:
            fig, axs = plt.subplots(1, 5, figsize=(12, 6))
        else:
            fig, axs = plt.subplots(1, 4, figsize=(10, 6))
    else:
        axs = fig.get_axes()
    axs[0].plot(tide_ds["major"], -tide_ds["depth"], label=constituent, **kwargs)
    axs[0].fill_betweenx(
        -tide_ds["depth"],
        tide_ds["major"] - tide_ds["emajor"],
        tide_ds["major"] + tide_ds["emajor"],
        alpha=0.3,
    )
    axs[1].plot(tide_ds["minor"], -tide_ds["depth"], label=constituent, **kwargs)
    axs[1].fill_betweenx(
        -tide_ds["depth"],
        tide_ds["minor"] - tide_ds["eminor"],
        tide_ds["minor"] + tide_ds["eminor"],
        alpha=0.3,
    )
    axs[2].plot(tide_ds["inc"], -tide_ds["depth"], label=constituent, **kwargs)
    axs[2].fill_betweenx(
        -tide_ds["depth"],
        tide_ds["inc"] - tide_ds["einc"],
        tide_ds["inc"] + tide_ds["einc"],
        alpha=0.3,
    )
    axs[3].plot(tide_ds["phase"], -tide_ds["depth"], label=constituent, **kwargs)
    axs[3].fill_betweenx(
        -tide_ds["depth"],
        tide_ds["phase"] - tide_ds["ephase"],
        tide_ds["phase"] + tide_ds["ephase"],
        alpha=0.3,
    )
    axs[0].set_xlabel("Major Axis [$\\mathsf{cm \\; s^{-1}}$]")
    axs[1].set_xlabel("Minor Axis [$\\mathsf{cm \\; s^{-1}}$]")
    axs[2].set_xlabel("Inclination [degrees]")
    axs[3].set_xlabel("Phase [degrees]")
    if snr:
        axs[4].plot(tide_ds["snr"], -tide_ds["depth"], label=constituent, **kwargs)
        axs[4].set_xscale("log")
        axs[4].axvline(2, color="k", linestyle="--")
        axs[4].set_xlabel("Signal-to-Noise Ratio")
    return fig, axs


# %%
velocity_v1 = xr.open_dataset(VEL_PATH_V1).resample(time="1h").mean()
velocity_v5 = xr.open_dataset(VEL_PATH_V5).resample(time="1h").mean()

velocity_v1_nanoos = velocity_v1.sel(time=NANOOS_TIME)
velocity_v5_nanoos = velocity_v5.sel(time=NANOOS_TIME)
velocity_v1_ooi = velocity_v1.sel(time=OOI_TIME)
velocity_v5_ooi = velocity_v5.sel(time=OOI_TIME)

# %%
tide_v1_nanoos = fit_ttide_from_ds(velocity_v1_nanoos)
tide_v5_nanoos = fit_ttide_from_ds(velocity_v5_nanoos)
tide_v1_ooi = fit_ttide_from_ds(velocity_v1_ooi)
tide_v5_ooi = fit_ttide_from_ds(velocity_v5_ooi)


# %%
class LegendTitle:
    def __init__(self, text_props=None, width=None) -> None:
        self.text_props = text_props or {}
        self.width = width or None
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(
            x0,
            y0,
            f"\\underline{{\\textbf{{{orig_handle}}}}}",
            usetex=True,
            **self.text_props,
        )
        handlebox.add_artist(title)
        if self.width is not None:
            handlebox.width = self.width
        return title


# %%
fig, axs = plt.subplots(1, 5, figsize=(12, 6), sharey=True)
constituents = ["K1", "O1", "M2"]
for const in constituents:
    tide = tide_v1_nanoos.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig)

[ax.set_prop_cycle(None) for ax in axs]  # reset color cycle for next plot
for const in constituents:
    tide = tide_v5_nanoos.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig, ls="--")
handles, labels = axs[0].get_legend_handles_labels()
proxy = mpatches.FancyBboxPatch(
    xy=(0, 0), width=0, height=0, visible=False, mutation_aspect=0
)
handles.append("NANOOS v1")
labels.append("")
handles.append("NANOOS v5")
labels.append("")
order = [6, 0, 1, 2, 7, 3, 4, 5]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
leg = fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.95, 0.5),
    handler_map={str: LegendTitle(text_props={"fontsize": 10}, width=55)},
)
fig.suptitle("NANOOS v1 vs. v5 Selected Tidal Constituents", fontsize=12, y=0.95)
plt.savefig(
    FIGURES_DIR / "ttide_velocity_nanoos_v1_v5.png", dpi=300, bbox_inches="tight"
)

# %%
fig, axs = plt.subplots(1, 5, figsize=(12, 6), sharey=True)
constituents = ["K1", "O1", "M2"]
for const in constituents:
    tide = tide_v1_ooi.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig)

[ax.set_prop_cycle(None) for ax in axs]  # reset color cycle for next plot
for const in constituents:
    tide = tide_v5_ooi.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig, ls="--")
handles, labels = axs[0].get_legend_handles_labels()
proxy = mpatches.FancyBboxPatch(
    xy=(0, 0), width=0, height=0, visible=False, mutation_aspect=0
)
handles.append("OOI v1")
labels.append("")
handles.append("OOI v5")
labels.append("")
order = [6, 0, 1, 2, 7, 3, 4, 5]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
leg = fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.95, 0.5),
    handler_map={str: LegendTitle(text_props={"fontsize": 10}, width=55)},
)
fig.suptitle("OOI v1 vs. v5 Selected Tidal Constituents", fontsize=12, y=0.95)
plt.savefig(FIGURES_DIR / "ttide_velocity_ooi_v1_v5.png", dpi=300, bbox_inches="tight")

# %%
fig, axs = plt.subplots(1, 5, figsize=(12, 6), sharey=True)
constituents = ["K1", "O1", "M2"]
for const in constituents:
    tide = tide_v1_nanoos.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig)

[ax.set_prop_cycle(None) for ax in axs]  # reset color cycle for next plot
for const in constituents:
    tide = tide_v1_ooi.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig, ls="--")
handles, labels = axs[0].get_legend_handles_labels()
proxy = mpatches.FancyBboxPatch(
    xy=(0, 0), width=0, height=0, visible=False, mutation_aspect=0
)
handles.append("NANOOS v1")
labels.append("")
handles.append("OOI v1")
labels.append("")
order = [6, 0, 1, 2, 7, 3, 4, 5]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
leg = fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.95, 0.5),
    handler_map={str: LegendTitle(text_props={"fontsize": 10}, width=55)},
)
fig.suptitle("OOI vs. NANOOS v1 Selected Tidal Constituents", fontsize=12, y=0.95)
plt.savefig(
    FIGURES_DIR / "ttide_velocity_nanoos_ooi_v1.png", dpi=300, bbox_inches="tight"
)

# %%
fig, axs = plt.subplots(1, 5, figsize=(12, 6), sharey=True)
constituents = ["K1", "O1", "M2"]
for const in constituents:
    tide = tide_v5_nanoos.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig)

[ax.set_prop_cycle(None) for ax in axs]  # reset color cycle for next plot
for const in constituents:
    tide = tide_v5_ooi.sel(constituent=const)
    fig, axs = plot_tidal_constituents(tide, const, fig, ls="--")
handles, labels = axs[0].get_legend_handles_labels()
proxy = mpatches.FancyBboxPatch(
    xy=(0, 0), width=0, height=0, visible=False, mutation_aspect=0
)
handles.append("NANOOS v5")
labels.append("")
handles.append("OOI v5")
labels.append("")
order = [6, 0, 1, 2, 7, 3, 4, 5]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
leg = fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.95, 0.5),
    handler_map={str: LegendTitle(text_props={"fontsize": 10}, width=55)},
)
fig.suptitle("OOI vs. NANOOS v5 Selected Tidal Constituents", fontsize=12, y=0.95)
plt.savefig(
    FIGURES_DIR / "ttide_velocity_nanoos_ooi_v5.png", dpi=300, bbox_inches="tight"
)

# %%
