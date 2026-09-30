"""Shared configuration for the MIMIC-TPW Southeast loop."""
import datetime as dt

# Domain edges (degrees). Equirectangular, Florida roughly centered.
W, E, S, N = -97.0, -67.0, 17.0, 37.0

# Target grid spacing for the browser-rendered fields (MIMIC TPW, RAP PW).
DX = 0.1
NX = round((E - W) / DX)   # 300
NY = round((N - S) / DX)   # 200

# Cell-center coordinates: lon ascending W->E, lat descending N->S
# (row 0 = northern edge, matches canvas row order).
def target_lons():
    import numpy as np
    return W + (np.arange(NX) + 0.5) * DX

def target_lats():
    import numpy as np
    return N - (np.arange(NY) + 0.5) * DX

# uint8 encoding for gridded fields: code = round(mm * SCALE), 255 = missing.
SCALE = 3.0
MISSING = 255

# Overlay PNG size. 50 px/deg in both axes -> exact 3:2 equirectangular frame.
DPI = 100
IMG_W = 1500
IMG_H = 1000

# Loop window and refresh behavior.
HOURS_BACK = 48          # frames kept on the site
MIMIC_REFRESH_HOURS = 6  # newest N hours re-downloaded every run (SSEC re-issues late analyses)

# Data sources.
MIMIC_BASE = "https://bin.ssec.wisc.edu/pub/mtpw2/data"
RAP_SOURCE = "aws"       # "aws" (noaa-rap-pds + .idx byte ranges) or "nomads" (grib filter)
RAP_AWS = "https://noaa-rap-pds.s3.amazonaws.com"
RAP_NOMADS_FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_rap.pl"

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) mimic-se-loop/1.0 (+github actions)"

# Isobaric levels pulled from RAP.
RAP_LEVELS = [925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 500]
MFC_LAYER = (925, 700)   # layer-integrated moisture flux convergence

# Display domains. Base fields (MIMIC, RAP PW) are always stored on the Southeast 0.1° grid
# and cropped in the browser; overlays are rendered separately per domain so vector spacing,
# barb density and line detail suit the zoom. Keep every extent at a 3:2 lon:lat ratio so
# the frame stays equirectangular at IMG_W x IMG_H.
DOMAINS = [
    {"id": "se", "name": "Southeast", "extent": [W, E, S, N], "dir": "ovl", "coast": "coast.png",
     "grid": 5, "vec_step": 10, "barb_step": 15, "stream_density": 1.4, "ref_deg": 1.0,
     "counties": False},
    {"id": "fl", "name": "Florida", "extent": [-88.5, -76.5, 23.5, 31.5], "dir": "ovl_fl",
     "coast": "coast_fl.png", "grid": 2, "vec_step": 4, "barb_step": 6, "stream_density": 1.7,
     "ref_deg": 0.4, "counties": True},
]

# ---- HRRR short-range forecast tail (appended after the newest MIMIC frame)
HRRR_AWS = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
HRRR_LEVELS = [925, 850, 700, 500]   # enough for q·V, theta-e, 925-700 MFC and 500 hPa
FC_HOURS = 12                        # forecast frames after the last observed frame
HRRR_STRIDE = 3                      # thin the 3 km grid to ~9 km before regridding to 0.1°

# ---- GOES-East GLM flash density (flashes in the hour ending at each frame time)
GLM_BUCKET = "https://noaa-goes19.s3.amazonaws.com"
GLM_SAT = "G19"
GLM_MAX_HOURS_PER_RUN = 12           # backfill cap so a cold start does not stall one run
GLM_WORKERS = 16

# ---- Observed soundings (Iowa Environmental Mesonet RAOB archive, CSV)
RAOB_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
SOUNDINGS = [
    {"id": "KXMR", "name": "XMR", "lat": 28.48, "lon": -80.55},
]

# ---- PW climatology for the anomaly view
CLIM_DAYS = 30                       # cold-start backfill: 00Z and 12Z for this many days
CLIM_EMA_HOURS = 720                 # then an hourly exponential moving average (~30 days)

# ---- MIMIC confidence: hours from the nearest actual microwave pass
AGE_SCALE = 10.0                     # uint8 code = round(hours * 10), 255 = missing

# Map markers drawn by the web page (also written to the manifest).
MARKERS = [
    {"name": "Cape Canaveral", "lat": 28.49, "lon": -80.58},
]


def utc_now_hour():
    now = dt.datetime.now(dt.timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0)


def key(t: dt.datetime) -> str:
    return t.strftime("%Y%m%d%H")
