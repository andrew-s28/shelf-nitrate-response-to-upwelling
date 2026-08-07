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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import distributions

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
CHL_PATH = DATA_DIR / "CE01ISSM/CE01ISSM_chlorophyll_2014-04-17_2025-09-22.nc"

# %%
inner_nitrate = xr.open_dataset(INNER_NITRATE_PATH)
midshelf_nitrate = xr.open_dataset(MIDSHELF_NITRATE_PATH)
wind = xr.open_dataset(WIND_PATH, decode_timedelta=True)

estimated_chloro = xr.open_dataset(CHL_PATH)["chlorophyll_a"]
estimated_chloro = estimated_chloro.sortby("time").resample(time="1D").mean()
estimated_chloro = estimated_chloro.where(estimated_chloro["time"].dt.month.isin([4, 5, 6, 7, 8, 9]), drop=True)

# %%
alpha = 0.05
n_eff = len(estimated_chloro.dropna("time")) / 11
rho_crit = float(
    np.sqrt(
        distributions.f.isf(alpha, 1, n_eff - 2) / (n_eff - 2 + distributions.f.isf(alpha, 1, n_eff - 2)),
    ),
)

# %% [markdown]
# ## Chlorophyll-Nitrate

# %%
e_c, i_n, c_y = xr.align(
    estimated_chloro,
    inner_nitrate.nitrate.mean(dim="depth"),
    wind["w5d"],
)


# %%
f"Correlation coefficient between chlorophyll and wind: {xr.corr(e_c, c_y).values:.2f}"

# %%
e_c, i_n, c_y = xr.align(
    estimated_chloro,
    inner_nitrate.nitrate.mean(dim="depth"),
    wind["w5d"],
)

fig, ax0 = plt.subplots(1, 1, figsize=(4, 4))

ax1 = ax0.twinx()

ax0.scatter(
    c_y,
    i_n,
    color="#004488",
)
ax1.scatter(c_y, e_c, color="#BB5566", marker="X")

ax1.axvline(0, color="black", ls="--")
ax0.set_xlim(-0.15, 0.1)
ax0.set_ylim(0, 40)
ax1.set_ylim(0, 25)
ax0.set_xlabel("$\\mathsf{W_{5d}}$ [$\\mathsf{N \\; m^{-2}}$]")
ax0.set_ylabel("Depth Mean Nitrate Conc. [$\\mathsf{mmol \\; m^{-3}}$]")
ax1.set_ylabel(
    "7 meter Chlorophyll Conc. [$\\mathsf{mg \\; m^{-3}}$]",
    rotation=270,
    labelpad=20,
)

ax0.tick_params(axis="y", colors="#004488")
ax1.tick_params(axis="y", colors="#BB5566")
ax0.yaxis.label.set_color("#004488")
ax1.yaxis.label.set_color("#BB5566")

ax1.annotate(
    "Nutrient\nLimited",
    xy=(0.85, 0.3),
    xycoords="axes fraction",
    ha="center",
    fontsize=14,
    color="black",
    zorder=10,
)
ax1.annotate(
    "Advection\nLimited",
    xy=(0.17, 0.3),
    xycoords="axes fraction",
    ha="center",
    color="black",
    fontsize=14,
    zorder=10,
)
