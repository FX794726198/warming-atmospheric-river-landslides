# Warming weakens a thermal constraint on atmospheric-river landslides

Code and a minimal data package for the manuscript

> Feng, X., Du, J., Chai, B. & Lombardo, L. *Warming weakens a thermal constraint on atmospheric-river landslides.* (submitted to *Nature Communications*).

The package redraws every figure of the paper and its Supplementary Information from the data in `data/`, gives the supplementary tables and Supplementary Data 1, and holds the scripts that produced the data from the public sources, including the set-up of the WRF experiments.

## Quick start

```bash
pip install -r requirements.txt
./make_figures.sh                 # writes fig/ (about 30 s); FIGOUT=/some/dir ./make_figures.sh writes elsewhere
```

`make_figures.sh` runs every script in `fig_code/` in turn. Supplementary Figs. S2–S9 and S12 are assembled from panels that the scripts first write to `fig/parts/` (`compose_si.py`). The output matches the figures of the submitted manuscript.

## Layout

| Folder | Content |
|---|---|
| `fig_code/` | One script per figure (see the table below). `arcommon.py` and `figcommon.py` hold shared helpers, `nature.mplstyle` the plot style. |
| `fig/` | The figures as PNG (and PDF for the main figures, Supplementary Figs. S1 and S11). |
| `data/` | The minimal data behind the figures (102 MB; see below). |
| `tables/` | Supplementary Tables S1–S14 as Markdown, with their captions. |
| `supplementary_data/` | Supplementary Data 1: the 231 regional landslide events and the records removed as duplicates. |
| `scripts/` | Analysis scripts that build `data/` from the public sources (see below). |
| `wrf/` | WRF and WPS namelists, the pseudo-global-warming scripts, run and post-processing scripts, gauge evaluation, applied warming and run provenance (see `wrf/README.md`). |

## Figures

| Figure | Script | Main inputs |
|---|---|---|
| Fig. 1 | `fig_code/fig1.py` | `data/derived/g1/` (AR share of extreme-precipitation days per cell, events and regional events), `data/derived/v6/chance_inventory.csv`, `data/inputs/arcat/clim_daily.npz`, `data/derived/a2/lsm.npy` |
| Fig. 2 | `fig_code/fig2.py` | `data/inputs/cases/` (IVT, precipitation and landslides of six events), `data/inputs/arcat/ar_case_masks.npz`, `data/derived/v6/events_fp48.parquet`, `data/derived/v6/logit_v6.csv` |
| Fig. 3 | `fig_code/fig3.py` | `data/derived/n3/` (temperature curve and bootstrap), `data/proc/rev/brake_robust.csv` (categorical models), `data/derived/n7/` (phase of AR extreme precipitation over mountains, 1941–2024), `data/proc/obs_bins.csv` |
| Fig. 4 | `fig_code/fig4.py` | `data/inputs/wrf/post/` (WRF control and warmer runs), `data/proc/cluster/WRF_ELEVBANDS.csv`, `WRF_HOURLY.csv`, `data/proc/rsl_all_runs.csv`, `data/proc/core_stats_local.csv`, `data/derived/n7/levels_summary.csv` |
| Fig. 5 | `fig_code/fig5.py` | `data/inputs/wrf/post/`, `data/inputs/dem/*_steep.npz`, `data/derived/n7/levels_summary.csv`, `data/proc/cluster/final/WRF_MODELS_NOR_2011.csv`, `data/proc/rev/cc_rates.csv` |
| Supplementary Fig. S1 | `fig_code/figS01_mechanism.py` (with `_labels` and `_zoom`) | schematic, no data |
| Supplementary Figs. S2, S4 | `fig_code/panels_obs1.py`, `panels_obs2.py`, then `compose_si.py` | `data/derived/g1/`, `data/derived/s1/` |
| Supplementary Figs. S3, S5–S8 | `fig_code/panels_supp.py`, then `compose_si.py` | `data/derived/s1/`, `data/derived/g1/trend_extreme_ar.csv` |
| Supplementary Fig. S9 | `fig_code/panels_obs4.py`, then `compose_si.py` | `data/derived/s1/episodes_extreme_cold.parquet`, `data/derived/g1/ar_ex_tclass.npz`, `events_t2.parquet` |
| Supplementary Fig. S10 | `fig_code/figS10_phase.py` | `data/derived/n7/` |
| Supplementary Fig. S11 | `fig_code/figS11_xsec.py` | `data/proc/cluster/WRF_XSEC_*.npz` |
| Supplementary Fig. S12 | `fig_code/figS12_wrf_full.py`, then `compose_si.py` | `data/inputs/wrf/post/` |

## Data

All files are derived from public data (sources in the Data availability statement of the paper). Folder names under `data/derived/` follow the prefixes of the scripts that wrote them.

| Folder | Content |
|---|---|
| `data/event_table/inventory_events.csv` | Every event of the five inventories that ended by 2022 (15,743 rows): identifier, inventory, number of landslides and cells, first and last day (`d0`, `d1`, days since 1 January 1979), centre latitude and longitude, share of cell-days under an AR (`ar_frac`; an AR event when ≥ 0.5), peak 2-day precipitation at the centre relative to the local 99th percentile (`pk`), accumulated extreme-precipitation footprint in cells (`fp`), largest simultaneous extreme-precipitation area in cells and km² (`fp_sim_cells`, `fp_sim_km2`), accumulated area in km² (`fp_uni_km2`), duration in days (`dur`) and whether the event enters the pooled statistics (`pooled`). |
| `data/derived/g1/` | Landslide events and regional landslide events (`events*.parquet`, `outbreaks_all.parquet`, `extra_outbreaks.parquet`), connected extreme-precipitation objects, AR share of extreme-precipitation cell-days by cell and month, ERA5 static fields on the 1° grid. |
| `data/derived/v6/` | Simultaneous extreme-precipitation area of each event, logistic models and chance AR shares. |
| `data/derived/s1/` | Cold-season AR episodes at landslide cells (`episodes_min.parquet`, all episodes with the variables used in the figures; `episodes_extreme_cold.parquet`, episodes with 2-day precipitation at or above the local 99th percentile, all variables), clustering rules, rain-matched comparisons, yields and land-surface models. |
| `data/derived/n3/` | Odds of a landslide by 2-day mean 2-m temperature within cells (the thermal constraint; files are named `brake_*`) and bootstrap. |
| `data/derived/n6/`, `n7/` | Phase of AR extreme precipitation over mountains (ERA5 0.25°, 1941–2024) by cell, region and year; WRF run summaries at four warming levels. |
| `data/inputs/` | Case fields of six events (`cases/`), AR masks of these events (`arcat/`), steep-slope fractions from the Copernicus GLO-90 DEM (`dem/`), shaded relief for the maps (`ne_relief/`, Natural Earth 1:50m shaded relief) and the post-processed WRF fields of the five storms (`wrf/post/`, event totals and 48-h maxima on the 3 km grid, IVT on the 9 km grid). |
| `data/proc/` | Statistics computed for the figures: categorical temperature models (`rev/brake_robust.csv`), Clausius–Clapeyron reference areas (`rev/cc_rates.csv`), rain share by elevation band, hourly core series and cross-sections of the WRF storms (`cluster/`), the single-model runs for western Norway (`cluster/final/`). |

Raw inputs are not redistributed: the AR catalogue, ERA5, IBTrACS, CMIP6, GHS-POP, GRIP4 and the landslide inventories can be obtained from their providers, and the full WRF output (several TB) is available on request.

## Analysis scripts

The scripts in `scripts/` were run on a workstation (observational analysis) and on the DelftBlue cluster (ERA5 0.25° phase analysis and WRF). They read and write under `work/` relative to the repository root and `/scratch/$USER/` on the cluster; adapt these paths to your set-up.

| Folder | Scripts and purpose |
|---|---|
| `scripts/landslides_ar/` | `ar_extract.py`, `ar_daily.py`, `extract_era5.py`, `extract_t2m.py`, `ivt_clim.py`: convert the AR catalogue and ERA5 to daily 1° arrays (lat −70 … 75, lon 0 … 359). `catalogs.py`, `cells.py`, `ar_match.py`: read the inventories and match landslides with ARs and TCs. `g3_events.py`, `g3_cluster.py`: join landslides into events and test the joining rules. `g5_p99.py`, `g5_footprint.py`, `v6_fp48.py`, `g5b_logit.py`: extreme-precipitation thresholds, footprints, simultaneous areas and logistic models. `g6_outbreaks.py`, `v6_regions.py`, `v6_perm.py`, `s2_driver_sensitivity.py`, `r1_catalog_check.py`: regional landslide events, chance shares and sensitivity to the attribution rules and the AR catalogue. `g4_episodes.py`, `g4_yield.py`, `w2_ar_yield_drivers.py`, `w3_surface.py`, `g2_analyze.py`, `g2_rainmatch.py`, `g7_trend.py`, `g7b_fpshare.py`, `g8_cellshare.py`, `g8b_cellshare_month.py`, `s1_supp_analyses.py`: AR episodes, rain-matched comparisons, trends and supplementary analyses. `g9_cases.py`, `g9b_cases.py`, `f1_event_points.py`: case studies and event points for the maps. |
| `scripts/thermal_constraint/` | `n3_hier_brake.py`, `n3b_brake_curve.py`, `n3c_cellfe.py`, `n3d_brake_boot.py`: odds of a landslide by temperature within cells (spline, categorical and bootstrap). `brake_robust.py`: categorical models with cell and month fixed effects, duration and year trend (Fig. 3c, Supplementary Table S7). `lrr_equiv.py`: precipitation change with the same odds ratio. `obs_bins.py`, `core_stats.py`: change of the phase by initial snow share and summary numbers. |
| `scripts/snow_phase/` | `arco_daily.py`, `n4_arco_daily.py`: daily ERA5 0.25° precipitation and snowfall from the analysis-ready ERA5 archive. `snow_counts.py`, `snow_regions.py`, `n1_aggregate.py`, `n1_snowshare.py`, `n5_snow_trend.py`: phase of AR extreme precipitation over mountains, its trends and regional series. |
| `scripts/wrf_analysis/` | `n2_dem_slope.py`: steep-slope fractions on the 3 km grids. `elev_bands.py`: rain share by elevation and rain–snow line. `xsec_numbers.py`: numbers of the cross-sections. `cc_area.py`: Clausius–Clapeyron reference areas and growth per degree. |
| `scripts/tables/` | `v6_tables.py`, `wrf_numbers.py`, `tab_wrfphase.py`, `tab_levels.py`, `tab_models.py`: write the supplementary tables from the outputs. |

## Tables

| Table | File |
|---|---|
| Supplementary Table S1 | `tables/TableS01_inventories.md` |
| Supplementary Table S2 | `tables/TableS02_event_inventories.md` |
| Supplementary Table S3 | `tables/TableS03_chance_ar_shares.md` |
| Supplementary Table S4 | `tables/TableS04_regional_events_by_region.md` |
| Supplementary Table S5 | `tables/TableS05_attribution_rules.md` |
| Supplementary Table S6 | `tables/TableS06_odds_ratios.md` |
| Supplementary Table S7 | `tables/TableS07_thermal_constraint.md` |
| Supplementary Table S8 | `tables/TableS08_phase_by_region.md` |
| Supplementary Table S9 | `tables/TableS09_wrf_experiments.md` |
| Supplementary Table S10 | `tables/TableS10_wrf_results.md` |
| Supplementary Table S11 | `tables/TableS11_wrf_phase.md` |
| Supplementary Table S12 | `tables/TableS12_wrf_warming_levels.md` |
| Supplementary Table S13 | `tables/TableS13_wrf_single_models.md` |
| Supplementary Table S14 | `tables/TableS14_wrf_gauges.md` |

## Software

Python 3.13 (cartopy downloads Natural Earth coastlines on first use) with numpy 2.5, pandas 3.0, scipy 1.18, statsmodels 0.15, matplotlib 3.11, cartopy 0.26, xarray, netCDF4, pyarrow and Pillow (`requirements.txt`). The data scripts also use gcsfs, zarr, cdsapi and rasterio. WRF 4.6.1 and WPS 4.6.0.

## Licence and citation

Code: MIT licence (`LICENSE`). Data and figures: CC BY 4.0 (`LICENSE-DATA`); third-party data keep their original licences. A versioned copy will be archived on Zenodo with a DOI on acceptance. Please cite the paper when using this material.

## Contact

Juan Du (dujuan@cug.edu.cn) and Luigi Lombardo (l.lombardo@utwente.nl).
