"""GOES-East GLM flash density: flashes per 0.1° cell in the hour ending at each frame time.

Files: {GLM_BUCKET}/GLM-L2-LCFA/YYYY/DDD/HH/OR_GLM-L2-LCFA_{SAT}_s...nc, ~20 s cadence,
~180 per hour. Listed with S3 ListObjectsV2, downloaded in parallel, parsed sequentially
(netCDF4/HDF5 is not thread-safe).
"""
import datetime as dt
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from . import config as C

log = logging.getLogger("glm")
_KEY = re.compile(r"<Key>([^<]+)</Key>")


def _list_hour(session, hour_start):
    prefix = f"GLM-L2-LCFA/{hour_start:%Y}/{hour_start:%j}/{hour_start:%H}/"
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = session.get(C.GLM_BUCKET + "/", params=params, timeout=60)
        r.raise_for_status()
        keys += [k for k in _KEY.findall(r.text) if f"_{C.GLM_SAT}_" in k]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return keys


def _download(session, key):
    try:
        r = session.get(f"{C.GLM_BUCKET}/{key}", timeout=60)
        r.raise_for_status()
        return r.content
    except Exception as exc:  # noqa: BLE001
        log.debug("GLM download failed %s: %s", key, exc)
        return None


def _flashes(blob):
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(blob)
        path = tmp.name
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            lat = np.asarray(ds["flash_lat"].values, dtype="float64")
            lon = np.asarray(ds["flash_lon"].values, dtype="float64")
            if "flash_quality_flag" in ds:
                good = np.asarray(ds["flash_quality_flag"].values) == 0
                lat, lon = lat[good], lon[good]
        return lat, lon
    except Exception as exc:  # noqa: BLE001
        log.debug("GLM parse failed: %s", exc)
        return np.empty(0), np.empty(0)
    finally:
        os.unlink(path)


def density_for(session, valid):
    """Return (counts[NY, NX] float32, n_files) for flashes in [valid-1h, valid)."""
    start = valid - dt.timedelta(hours=1)
    keys = _list_hour(session, start)
    if not keys:
        return None, 0
    with ThreadPoolExecutor(max_workers=C.GLM_WORKERS) as pool:
        blobs = list(pool.map(lambda k: _download(session, k), keys))
    lon_edges = C.W + np.arange(C.NX + 1) * C.DX
    lat_edges = C.S + np.arange(C.NY + 1) * C.DX
    counts = np.zeros((C.NY, C.NX), dtype="float32")
    for b in blobs:
        if not b:
            continue
        lat, lon = _flashes(b)
        if lat.size:
            h, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges])
            counts += h[::-1, :].astype("float32")      # south-first -> north-first rows
    return counts, sum(1 for b in blobs if b)
