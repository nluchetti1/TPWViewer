# Southeast moisture loop

Hourly MIMIC-TPW2 total precipitable water over the Southeast US, Gulf, Bahamas and western
Atlantic (17–37°N, 97–67°W), with toggleable RAP 13 km analysis overlays. Static site on
GitHub Pages, rebuilt hourly by GitHub Actions.

## Setup

1. Create the repo and push these files to `main`.
2. Settings → Actions → General → Workflow permissions: **Read and write**.
3. Actions tab → `build-loop` → **Run workflow** once. The first run backfills 48 hours
   (about 5–8 minutes).
4. Settings → Pages → Source: **Deploy from a branch**, branch `gh-pages`, folder `/ (root)`.

Before the first run you can check every data source from Colab or a laptop:

    pip install -r requirements.txt
    python -m pipeline.probe

## How it works

| Piece | What it does |
|---|---|
| `pipeline/mimic.py` | Lists `bin.ssec.wisc.edu/pub/mtpw2/data/YYYYMM/`, pulls `compYYYYMMDD.HH0000.nc`, interpolates `tpwGrid` to a 0.1° grid over the domain. |
| `pipeline/rap.py` | Pulls only the needed RAP f00 records (byte ranges from the AWS `.idx`, or the NOMADS grib filter), rotates grid-relative winds to earth-relative, regrids to the same 0.1° grid, computes q·V, theta-e and layer MFC. |
| `pipeline/render.py` | Draws each overlay as a transparent 1500×1000 PNG covering the identical extent, so layers stack pixel-for-pixel. |
| `pipeline/build.py` | Incremental: keeps existing frames, re-pulls the newest 6 h of MIMIC (SSEC re-issues late analyses), prunes past 48 h, writes `data/manifest.json`. |
| `web/index.html` | The viewer. MIMIC and RAP PW arrive as raw uint8 grids (60 kB per hour) and are colored in the browser, which is what makes the palette switch, mm/in toggle, MIMIC − RAP difference and hover readout instant. |

Grid encoding: `code = round(mm × 3)`, 255 = missing, row 0 = 37°N, column 0 = 97°W.

## Adding an overlay

1. Write a `draw_xxx(f, path)` in `render.py` using `_fig()` / `_save()` so the extent stays locked.
2. Add an entry to `OVERLAYS` (id, group, name, legend, default opacity).
3. Add it to the `jobs` dict in `render_all`.

The dropdown builds itself from the manifest. Existing hours backfill automatically on the next
run, because the build re-renders any hour missing any overlay.

## Knobs (`pipeline/config.py`)

- `W, E, S, N`: domain. Keep the 3:2 ratio or change `IMG_W/IMG_H` to match.
- `HOURS_BACK`: loop length kept on the site (the page offers 12/24/48 h).
- `RAP_SOURCE`: `"aws"` or `"nomads"`.
- `MFC_LAYER`: pressure layer for integrated moisture flux convergence.
- `MARKERS`: labeled points on the map.

## Caveats

- MIMIC-TPW2 is experimental and not operational. SSEC noted Metop retrievals are still missing
  after the July 2026 feed change, so expect holes.
- SSEC warns the newest ~3 hours can exaggerate over-land TPW near coasts (forward-advected only).
- MFC uses a 2-point Gaussian pre-smooth on the 0.1° grid; the shape is meaningful, but treat
  the magnitudes as qualitative.
