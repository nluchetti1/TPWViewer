"""MIMIC-TPW2 access: list hourly composites on the SSEC server, subset to the domain.

Files live at  {MIMIC_BASE}/YYYYMM/compYYYYMMDD.HHMMSS.nc  (optionally .gz).
Variables used: tpwGrid (mm), latArr, lonArr.
"""
import datetime as dt
import gzip
import logging
import os
import re
import tempfile

import numpy as np
import xarray as xr

from . import config as C

log = logging.getLogger("mimic")
_NAME = re.compile(r'href="(comp(\d{8}\.\d{6})\.nc(?:\.gz)?)"')


def list_recent(session, hours):
    """Return {valid_time: url} for composites within the last `hours` hours."""
    now = C.utc_now_hour()
    start = now - dt.timedelta(hours=hours)
    months = sorted({start.strftime("%Y%m"), now.strftime("%Y%m")})
    found = {}
    for ym in months:
        url = f"{C.MIMIC_BASE}/{ym}/"
        r = session.get(url, timeout=60)
        if r.status_code == 404:
            log.warning("MIMIC month directory missing: %s", url)
            continue
        r.raise_for_status()
        for name, stamp in _NAME.findall(r.text):
            t = dt.datetime.strptime(stamp, "%Y%m%d.%H%M%S").replace(tzinfo=dt.timezone.utc)
            if t.minute == 0 and start <= t <= now:
                # prefer the uncompressed name if both exist
                if t not in found or found[t].endswith(".gz"):
                    found[t] = url + name
    log.info("MIMIC: %d composites in the last %d h", len(found), hours)
    return found


def fetch_subset(session, url):
    """Download one composite and interpolate TPW (mm) to the target grid (NY, NX)."""
    r = session.get(url, timeout=180)
    r.raise_for_status()
    raw = gzip.decompress(r.content) if url.endswith(".gz") else r.content
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
            tpw = ds["tpwGrid"]
            units = str(tpw.attrs.get("units", "mm")).lower()
            lat = np.asarray(ds["latArr"].values, dtype="float64")
            lon = np.asarray(ds["lonArr"].values, dtype="float64")
            vals = np.asarray(tpw.values, dtype="float32")
    finally:
        os.unlink(path)

    if vals.shape != (lat.size, lon.size):
        raise ValueError(f"unexpected tpwGrid shape {vals.shape} vs lat {lat.size} lon {lon.size}")

    da = xr.DataArray(vals, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    if float(np.nanmax(lon)) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
    da = da.sortby("lat").sortby("lon")
    da = da.where((da >= 0) & (da < 150))
    if units.startswith("cm"):
        da = da * 10.0

    sub = da.sel(lat=slice(C.S - 1, C.N + 1), lon=slice(C.W - 1, C.E + 1))
    out = sub.interp(lat=C.target_lats(), lon=C.target_lons(), method="linear").values
    return out.astype("float32")
