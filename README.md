# Southeast moisture loop

Hourly MIMIC-TPW2 total precipitable water over the Southeast US, Gulf, Bahamas and western
Atlantic, with a zoomed Florida view, RAP analysis overlays, an HRRR forecast tail, GOES-19
lightning, observed-sounding checks and a PW anomaly view. Static site on GitHub Pages,
rebuilt twice an hour by GitHub Actions.

## Setup (browser only)

1. Create a public repo and upload `pipeline/`, `web/`, `requirements.txt`, `README.md`.
2. Add file → Create new file → `.github/workflows/build.yml`, paste the workflow.
3. Settings → Actions → General → Workflow permissions: **Read and write**.
4. Actions → `build-loop` → **Run workflow**. The first run backfills everything (15–25 min).
5. Settings → Pages → Deploy from a branch → `gh-pages` / `/ (root)`.

Check every data source first (Colab or laptop):

    pip install -r requirements.txt
    python -m pipeline.probe

## What's on the page

| Feature | Source | Notes |
|---|---|---|
| Shaded PW | MIMIC-TPW2 (UW-CIMSS) | 0.25° hourly composite, regridded to 0.1° |
| RAP PW / MIMIC − RAP | RAP 13 km f00 (AWS) | Difference shows model moisture bias |
| Anomaly | MIMIC rolling mean | 30-day 00/12Z mean at cold start, then ~30-day EMA |
| Model overlays (9) | RAP f00, per domain | q·V, MFC, streamlines, 500 hPa, theta-e, MLCAPE, PW contours |
| Forecast tail | HRRR wrfprs (AWS) | 12 h after the newest MIMIC frame, same overlays, amber timeline |
| Lightning | GOES-19 GLM L2 LCFA (AWS) | Flashes per 0.1° cell in the hour ending at each frame |
| Confidence hatch | MIMIC timeAwayGrid | Hatches where the nearest real pass is > 2/3/6 h away |
| PW threshold | browser | Adjustable contour on the shaded field, in GIFs too |
| Point history | browser + IEM RAOB | Click the map: MIMIC, RAP, HRRR and XMR sounding PW |
| GIF export | browser (gifenc) | Current loop with all layers and a labeled footer |

Keyboard: Space play/pause, arrows step, Z domain, L lightning, T threshold.
The URL hash keeps the full view (domain, field, overlay, layers, probe point).

## Pipeline

| File | Role |
|---|---|
| `pipeline/mimic.py` | MIMIC listing and subset; returns TPW and hours-from-nearest-pass |
| `pipeline/rap.py` | RAP/HRRR .idx byte-range fetch, Lambert wind rotation, regrid, q·V / theta-e / MFC |
| `pipeline/render.py` | Overlay PNGs per domain (and `_fc` copies for HRRR), coastlines |
| `pipeline/glm.py` | GLM listing, parallel download, flash binning |
| `pipeline/soundings.py` | IEM RAOB CSV → integrated PW |
| `pipeline/clim.py` | Rolling PW mean for the anomaly view |
| `pipeline/build.py` | Orchestrator: incremental, prunes to 48 h, writes `data/manifest.json` |
| `pipeline/probe.py` | Live check of every source |

Site layout on gh-pages: `data/{tpw,age,rappw,glm}/YYYYMMDDHH.bin` (uint8 grids),
`data/fcpw/`, `data/{ovl,ovl_fl}[_fc]/<overlay>/`, `data/clim/mean.f32`, `data/manifest.json`.

## Knobs (`pipeline/config.py`)

`DOMAINS`, `HOURS_BACK`, `FC_HOURS`, `HRRR_LEVELS`, `GLM_SAT`/`GLM_BUCKET`, `SOUNDINGS`,
`CLIM_DAYS`, `CLIM_EMA_HOURS`, `MARKERS`, `RAP_SOURCE`.

## Caveats

- MIMIC-TPW2 is experimental. SSEC notes Metop retrievals are still missing after the July 2026
  feed change, and the newest ~3 h can exaggerate over-land TPW near coasts (the confidence hatch
  shows where).
- HRRR's southern edge is near 21°N, so the forecast tail is blank over Cuba and the far south of
  the Southeast domain.
- The anomaly is relative to a rolling 30-day mean, not a long-term climatology.
- The keepalive re-enables the workflow daily; GitHub's 60-day rule is the reason it exists.
