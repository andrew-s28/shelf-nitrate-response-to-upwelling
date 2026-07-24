# Shelf Nitrate Response to Upwelling

This is the analysis code, datasets, and figures used for a manuscript, "Shelf Nitrate Response to Upwelling". 

This repository is based off of a [scientific data analysis template repository](https://github.com/andrew-s28/analysis-template).

## Rebuilding Datasets

The repository includes any datasets that cannot be easily downloaded from a public repository. For downloading the public data, follow these instructions:

1) Run the profiler data retriever:

```python
uv run scripts/retrieve_profiler_data.py CE01ISSP CE02SHSP
```

2) Run the wind data retriever:

```python
uv run scripts/ndbc_downloader.py 46050
```

3) Run the data processors:

```python
uv run make_nitrate_datasets.py
uv run make_velocity_datasets.py
uv run make_wind_datasets.py
uv run make_bottle_sample_datasets.py
```

1) You should now be able to run [all of the analysis notebooks](./notebooks/) to reproduce the figures from the associated manuscript!
