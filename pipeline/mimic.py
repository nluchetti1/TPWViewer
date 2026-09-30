"""MIMIC-TPW2 access: list hourly composites on the SSEC server, subset to the domain.

Files live at  {MIMIC_BASE}/YYYYMM/compYYYYMMDD.HHMMSS.nc  (optionally .gz).
Variables used: tpwGrid (mm), latArr, lonArr, and timeAwayGridPrior / timeAwayGridSubseq
(time from the nearest real microwave pass before / after the analysis time).
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


def list_range(session, start, end, whole_hours=True):
    """Return {valid_time: url} for composites with start <= t <= end."""
    months = []
    m = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while m <= end:
        months.append(m.strftime("%Y%m"))
        m = (m + dt.timedelta(days=32)).replace(day=1)
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
            if (not whole_hours or t.minute == 0) and start <= t <= end:
                if t not in found or found[t].endswith(".gz"):
                    found[t] = url + name
    return found


def list_recent(session, hours):
    now = C.utc_now_hour()
    found = list_range(session, now - dt.timedelta(hours=hours), now)
    log.info("MIMIC: %d composites in the last %d h", len(found), hours)
    return found


def _to_hours(da):
    v = da.values
    if np.issubdtype(v.dtype, np.timedelta64):
        return np.abs(v / np.timedelta64(1, "h")).astype("float32")
    units = str(da.attrs.get("units", "hours")).lower()
    f = {"minutes": 1 / 60, "minute": 1 / 60, "seconds": 1 / 3600, "second": 1 / 3600}.get(units, 1.0)
    return np.abs(v.astype("float32") * f)


def fetch_subset(session, url):
    """Download one composite; return (tpw_mm, age_hours) on the target grid (NY, NX).

    age_hours is the time to the nearest actual microwave pass (min of prior/subsequent),
    or None if the file lacks those fields.
    """
    r = session.get(url, timeout=180)
    r.raise_for_status()
    raw = gzip.decompress(r.content) if url.endswith(".gz") else r.content
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(raw)
        path = tmp.name
    try:
        with xr.open_dataset(path, decode_times=False, decode_timedelta=True) as ds:
            tpw = ds["tpwGrid"]
            units = str(tpw.attrs.get("units", "mm")).lower()
            lat = np.asarray(ds["latArr"].values, dtype="float64")
            lon = np.asarray(ds["lonArr"].values, dtype="float64")
            vals = np.asarray(tpw.values, dtype="float32")
            ages = [_to_hours(ds[v]) for v in ("timeAwayGridPrior", "timeAwayGridSubseq") if v in ds]
    finally:
        os.unlink(path)

    if vals.shape != (lat.size, lon.size):
        raise ValueError(f"unexpected tpwGrid shape {vals.shape} vs lat {lat.size} lon {lon.size}")

    def regrid(arr, lo=None, hi=None):
        da = xr.DataArray(arr, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
        if float(np.nanmax(lon)) > 180:
            da = da.assign_coords(lon=((da.lon + 180) % 360) - 180)
        da = da.sortby("lat").sortby("lon")
        if lo is not None:
            da = da.where((da >= lo) & (da < hi))
        sub = da.sel(lat=slice(C.S - 1, C.N + 1), lon=slice(C.W - 1, C.E + 1))
        return sub.interp(lat=C.target_lats(), lon=C.target_lons(), method="linear").values.astype("float32")

    out = regrid(vals, 0, 150)
    if units.startswith("cm"):
        out = out * 10.0
    age = None
    if ages:
        with np.errstate(all="ignore"):
            a = np.nanmin(np.stack(ages), axis=0) if len(ages) > 1 else ages[0]
        age = regrid(a, 0, 1000)
    return out, age
