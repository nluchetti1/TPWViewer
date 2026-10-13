"""GFS and ECMWF precipitable-water tails (PW only) for long-range looping.

GFS 0.25°:   {GFS_AWS}/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF  + wgrib2 .idx  (PWAT)
ECMWF 0.25°: {ROOT}/YYYYMMDD/HHz/ifs/0p25/oper/YYYYMMDDHH0000-{step}h-oper-fc.grib2 + JSON .index (tcwv)
Only the one PW record is byte-ranged out of each file.
"""
import datetime as dt
import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from . import config as C

log = logging.getLogger("longrange")
SOURCES = ("gfs", "ecmwf")
DIRS = {"gfs": "fc_gfs", "ecmwf": "fc_ecmwf"}


# ---------------------------------------------------------------- URLs and record lookup

def gfs_url(c, fh):
    return f"{C.GFS_AWS}/gfs.{c:%Y%m%d}/{c:%H}/atmos/gfs.t{c:%H}z.pgrb2.0p25.f{fh:03d}"


def ec_url(root, c, step):
    return f"{root}/{c:%Y%m%d}/{c:%H}z/ifs/0p25/oper/{c:%Y%m%d%H}0000-{step}h-oper-fc.grib2"


def _range_gfs(session, c, fh):
    base = gfs_url(c, fh)
    r = session.get(base + ".idx", timeout=60)
    if r.status_code != 200:
        return None, None
    lines = [ln.split(":") for ln in r.text.strip().splitlines()]
    for i, parts in enumerate(lines):
        if len(parts) > 4 and parts[3] == "PWAT" and parts[4].startswith("entire atmosphere"):
            start = int(parts[1])
            end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
            return base, (start, end)
    return None, None


def _range_ec(session, c, step):
    for root in C.ECMWF_ROOTS:
        base = ec_url(root, c, step)
        try:
            r = session.get(base[:-len(".grib2")] + ".index", timeout=60)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        for line in r.text.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("param") == "tcwv":
                off, ln = int(rec["_offset"]), int(rec["_length"])
                return base, (off, off + ln - 1)
    return None, None


def available(session, src, c, step):
    try:
        if src == "gfs":
            return session.head(gfs_url(c, step) + ".idx", timeout=30).status_code == 200
        for root in C.ECMWF_ROOTS:
            u = ec_url(root, c, step)[:-len(".grib2")] + ".index"
            if session.head(u, timeout=30).status_code == 200:
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


# ---------------------------------------------------------------- download + decode

def fetch_pw_bytes(session, src, c, step):
    base, rng = (_range_gfs if src == "gfs" else _range_ec)(session, c, step)
    if not base:
        return None
    s, e = rng
    r = session.get(base, headers={"Range": f"bytes={s}-{'' if e is None else e}"}, timeout=120)
    r.raise_for_status()
    return r.content


def decode_pw(grib_bytes):
    """Global 0.25° PW GRIB record -> target grid (NY, NX) in mm."""
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        path = tmp.name
    try:
        with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            name = next(iter(ds.data_vars))
            da = ds[name].load()
        da = da.rename({"latitude": "lat", "longitude": "lon"})
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")
        sub = da.sel(lat=slice(C.S - 1, C.N + 1), lon=slice(C.W - 1, C.E + 1))
        return sub.interp(lat=C.target_lats(), lon=C.target_lons()).values.astype("float32")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- tail builder

def _cycles(src, now):
    base = now.replace(hour=(now.hour // 6) * 6)
    for back in range(0, 7):
        c = base - dt.timedelta(hours=6 * back)
        if src == "ecmwf" and c.hour not in (0, 12):     # only 00/12Z run to 10 days
            continue
        yield c


def build(session, site, src, last_obs, write_bin):
    data = os.path.join(site, "data")
    meta_path = os.path.join(data, "fc", f"{src}.json")
    old = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh)
    steps_all = C.GFS_STEPS if src == "gfs" else C.ECMWF_STEPS

    cycle = None
    for c in _cycles(src, C.utc_now_hour()):
        if available(session, src, c, steps_all[-1]):
            cycle = c
            break
    if cycle is None:
        log.warning("%s: no complete cycle found; keeping previous tail", src.upper())
        return _trim(old, last_obs)

    def frames_for(c):
        out = []
        for st in steps_all:
            v = c + dt.timedelta(hours=st)
            if v > last_obs:
                out.append((v, st))
        return out

    wanted = frames_for(cycle)
    d = os.path.join(data, DIRS[src])
    if old and old.get("cycle") == C.key(cycle):
        meta = _trim(old, last_obs)
        log.info("%s tail unchanged (%sZ cycle, %d frames)", src.upper(), C.key(cycle), len(meta["frames"]))
        return meta

    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)

    def grab(item):
        v, st = item
        try:
            return v, st, fetch_pw_bytes(session, src, cycle, st)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s f%03d failed: %s", src, st, exc)
            return v, st, None

    with ThreadPoolExecutor(max_workers=C.LONG_WORKERS) as pool:
        got = list(pool.map(grab, wanted))
    frames = []
    for v, st, blob in got:
        if not blob:
            continue
        try:
            pw = decode_pw(blob)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s f%03d decode failed: %s", src, st, exc)
            continue
        k = C.key(v)
        write_bin(os.path.join(d, f"{k}.bin"), pw)
        frames.append({"t": v.strftime("%Y-%m-%dT%H:%MZ"), "key": k, "fh": st, "pw": True})
    meta = {"cycle": C.key(cycle), "frames": frames}
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    log.info("%s %sZ: %d/%d PW frames", src.upper(), C.key(cycle), len(frames), len(wanted))
    return meta


def _trim(meta, last_obs):
    if not meta:
        return None
    lo = last_obs.strftime("%Y-%m-%dT%H:%MZ")
    return {**meta, "frames": [f for f in meta["frames"] if f["t"] > lo]}
