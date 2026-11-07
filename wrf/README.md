# WRF pseudo-global-warming experiments

Five cold-season AR storms were simulated with WRF 4.6.1 (WPS 4.6.0) in a control run with ERA5 boundary conditions and in warmer runs with the CMIP6 change added (pseudo-global warming). All times are UTC.

| Event | Run start (12 h spin-up) | Analysis window |
|---|---|---|
| BC_2021, British Columbia | 2021-11-12 00 | 2021-11-12 12 to 2021-11-16 12 |
| LIG_2014, Liguria and Piedmont | 2014-11-08 12 | 2014-11-09 00 to 2014-11-17 00 |
| NOR_2011, western Norway | 2011-12-23 12 | 2011-12-24 00 to 2011-12-28 00 |
| ORE_1996, Oregon | 1996-11-15 12 | 1996-11-16 00 to 1996-11-20 00 |
| RSB_2024, Rio Grande do Sul | 2024-04-26 12 | 2024-04-27 00 to 2024-05-03 00 |

## Set-up

- Two-way nested domains (Lambert): d01 9 km, 390 × 330 cells (390 × 390 for RSB_2024); d02 3 km, 300 × 300 cells centred on the landslides.
- 55 levels up to 50 hPa (hybrid coordinate), time steps 45 s (d01) and 15 s (d02); runs that stopped early on a CFL error were continued from the newest restart with 36 s and 12 s (`run_wrf.sh`).
- Physics: Thompson microphysics, RRTMG radiation (9 min), YSU boundary layer with the revised MM5 surface layer, Noah-MP land surface (4 soil layers, MODIS land use), Kain–Fritsch cumulus on d01 only, explicit convection on d02. Greenhouse gases from the CAMtr files (`camtr_pgw.py`).
- Initial and boundary conditions: ERA5, 0.25°, 3-hourly, 37 pressure levels (`get_era5.py`, or `arco_pl.py` from the analysis-ready ERA5 archive). SST is fixed at its initial value (`sst_update = 0`).
- Templates: `scripts/namelist.input.tmpl`, `scripts/iofields_d01.txt`, `scripts/iofields_d02.txt`. `namelists/` holds the WPS namelists of the five events and examples of the WRF namelists generated from the template. Set `geog_data_path` to your WPS static data.

## Warming

- `cmip6_delta.py`: monthly change (2071–2100 minus 1985–2014) of air temperature on pressure levels, 2-m and skin temperature under SSP5-8.5, mean of ten CMIP6 models (ACCESS-CM2, CNRM-CM6-1, EC-Earth3, GFDL-ESM4, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, MRI-ESM2-0, NorESM2-MM, UKESM1-0-LL; first member), from the Pangeo CMIP6 store. Weights by month follow the days of each run (`events.py`).
- `pgw_metem.py`: adds the change to the WPS met_em files: air temperature at all levels, skin, sea-surface and soil temperature; relative humidity held fixed (so specific humidity rises); winds unchanged; geopotential adjusted hydrostatically. Below-ground CMIP6 values are filled from the level above; skin and soil temperatures at snow-covered points that were below 0 °C are not raised above 0 °C; skin temperature over sea ice is capped at 271.35 K.
- Intermediate levels multiply the change by 0.25, 0.5 and 0.75; the single-model runs for western Norway use one model's change instead of the mean. `levels_table.py` summarizes these runs. `pgw_deltas/` gives the applied change averaged over each domain.

## Workflow (Slurm)

1. `corners.py`, `geo_event.sh`, `make_wps.py`, `wps_event.sh` (and `redo_d02.sh` for the 3 km nest): WPS for each event.
2. `cmip6_delta.py`, then `pgw_metem.py` for the warmer runs.
3. `setup_run.sh`, `real_batch.sh`, `submit_runs.sh` / `submit_segs.sh` → `run_wrf.sh` (set `ACCOUNT`, `WPS_DIR`, `WPS_ENV` and `WPS_GEOG` for your cluster).
4. `post_job.sh`, `post_all.sh` → `post_event.py`, `post_case.py`: event totals, 48-h maxima, phase, runoff and IVT; `make_light.py` writes small per-event copies of the outputs for analysis off the cluster.
5. `eval_stations.py` (GHCN-Daily gauges, `eval/get_ghcn.sh`), `eval_obstime_shift.py` and `eval_chirps.py` (CHIRPS for Rio Grande do Sul; INMET download script in `eval/inmet/`): control runs against observations (`eval/summary.csv`, Supplementary Table S14).
6. `provenance.py` → `provenance.md`: the job segments, restart times, time steps and output fingerprints behind every run.

The full model output (several TB) is not included and is available on request.
