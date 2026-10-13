"""GFS and ECMWF long-range tails: PW plus the same model overlays as the RAP/HRRR frames.

GFS 0.25°:   {GFS_AWS}/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF  + wgrib2 .idx
             PWAT, CAPE 90-0 mb, HGT/TMP/RH/UGRD/VGRD on LR_LEVELS
ECMWF 0.25°: {ROOT}/YYYYMMDD/HHz/ifs/0p25/oper/YYYYMMDDHH0000-{step}h-oper-fc.grib2 + JSON .index
             tcwv, mucape, gh/t/r/u/v on LR_LEVELS
Only the wanted records are byte-ranged out of each file. Both grids are regular lat/lon with
earth-relative winds, so fields are interpolated straight onto the 0.1° target grid.
"""
import datetime as dt
import json
import logging
import os
import shutil
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr

from . import config as C
from . import rap, render

log = logging.getLogger("longrange")
SOURCES = ("gfs", "ecmwf")
DIRS = {"gfs": "fc_gfs", "ecmwf": "fc_ecmwf"}
EC_PL = {"gh", "t", "r", "u", "v"}


# ---------------------------------------------------------------- URLs

def gfs_url(c, fh):
    return f"{C.GFS_AWS}/gfs.{c:%Y%m%d}/{c:%H}/atmos/gfs.t{c:%H}z.pgrb2.0p25.f{fh:03d}"


def ec_url(root, c, step):
    return f"{root}/{c:%Y%m%d}/{c:%H}z/ifs/0p25/oper/{c:%Y%m%d%H}0000-{step}h-oper-fc.grib2"


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


# ---------------------------------------------------------------- record selection + download

def _fetch_gfs(session, c, fh, full):
    base = gfs_url(c, fh)
    if full:
        return rap.fetch_idx_records(session, base, C.LR_LEVELS)
    return rap.fetch_idx_records(session, base, [])          # PWAT (and CAPE) only


def _fetch_ec(session, c, step, full):
    for root in C.ECMWF_ROOTS:
        base = ec_url(root, c, step)
        try:
            r = session.get(base[:-len(".grib2")] + ".index", timeout=60)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code != 200:
            continue
        ranges = []
        for line in r.text.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            p = rec.get("param")
            want = p == "tcwv" or (full and (
                p == "mucape" or
                (p in EC_PL and rec.get("levtype") == "pl" and int(rec.get("levelist", 0) or 0) in C.LR_LEVELS)))
            if want:
                off, ln = int(rec["_offset"]), int(rec["_length"])
                ranges.append([off, off + ln - 1])
        if ranges:
            return rap.fetch_ranges(session, base, ranges)
    return None


def fetch_bytes(session, src, c, step, full=True):
    return (_fetch_gfs if src == "gfs" else _fetch_ec)(session, c, step, full)


# ---------------------------------------------------------------- decode

def _to_target(da):
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon").sortby("lat")
    sub = da.sel(lat=slice(C.S - 1, C.N + 1), lon=slice(C.W - 1, C.E + 1))
    return sub.interp(lat=C.target_lats(), lon=C.target_lons()).values.astype("float32")


def decode(grib_bytes):
    """GRIB bytes -> field dict shaped like rap.decode(): u/v/t/rh/gh per level, pw, cape."""
    import cfgrib
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        path = tmp.name
    f = {"u": {}, "v": {}, "t": {}, "rh": {}, "gh": {}, "pw": None, "cape": None}
    names = {"u": "u", "v": "v", "t": "t", "r": "rh", "gh": "gh"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dsets = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
        for ds in dsets:
            ds = ds.load()
            for name, da in ds.data_vars.items():
                if "isobaricInhPa" in da.coords and name in names:
                    levs = np.atleast_1d(da["isobaricInhPa"].values)
                    for p in levs:
                        lvl = da.sel(isobaricInhPa=p) if "isobaricInhPa" in da.dims else da
                        f[names[name]][int(p)] = _to_target(lvl)
                elif name in ("pwat", "tcwv"):
                    f["pw"] = _to_target(da)
                elif name in ("cape", "mucape") and f["cape"] is None:
                    f["cape"] = _to_target(da.isel({da.dims[0]: 0}) if da.ndim == 3 else da)
        return f
    finally:
        os.unlink(path)


def decode_pw(grib_bytes):
    return decode(grib_bytes)["pw"]


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

    # Same cycle and overlays already made (or not wanted): just drop frames now in the past.
    have_ovl = bool(old and old.get("overlays"))
    if old and old.get("cycle") == C.key(cycle) and (have_ovl or not C.LR_OVERLAYS):
        meta = _trim(old, last_obs)
        log.info("%s tail unchanged (%sZ cycle, %d frames)", src.upper(), C.key(cycle), len(meta["frames"]))
        return meta

    wanted = [(cycle + dt.timedelta(hours=st), st) for st in steps_all
              if cycle + dt.timedelta(hours=st) > last_obs]
    d = os.path.join(data, DIRS[src])
    shutil.rmtree(d, ignore_errors=True)
    for dom in C.DOMAINS:
        shutil.rmtree(os.path.join(data, dom["dir"] + f"_fc_{src}"), ignore_errors=True)
    os.makedirs(d, exist_ok=True)

    full = C.LR_OVERLAYS
    todo = [(dm, o["id"]) for dm in C.DOMAINS for o in render.OVERLAYS] if full else []

    def grab(item):
        v, st = item
        try:
            return v, st, fetch_bytes(session, src, cycle, st, full)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s f%03d failed: %s", src, st, exc)
            return v, st, None

    frames = []
    # download in parallel batches, decode/render sequentially (matplotlib is not thread-safe)
    batch = C.LONG_WORKERS * 2
    with ThreadPoolExecutor(max_workers=C.LONG_WORKERS) as pool:
        for i in range(0, len(wanted), batch):
            for v, st, blob in pool.map(grab, wanted[i:i + batch]):
                if not blob:
                    continue
                try:
                    fld = decode(blob)
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s f%03d decode failed: %s", src, st, exc)
                    continue
                if fld["pw"] is None:
                    continue
                k = C.key(v)
                write_bin(os.path.join(d, f"{k}.bin"), fld["pw"])
                done = render.render_all(fld, k, site, todo, fc=src) if todo else []
                ovl = {dm["id"]: [o for (di, o) in done if di == dm["id"]] for dm in C.DOMAINS}
                frames.append({"t": v.strftime("%Y-%m-%dT%H:%MZ"), "key": k, "fh": st, "pw": True, "ovl": ovl})
    meta = {"cycle": C.key(cycle), "frames": frames, "overlays": full,
            "cape": "MUCAPE" if src == "ecmwf" else "90-0 hPa MLCAPE"}
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    n_ovl = sum(len(v) for fr in frames for v in fr["ovl"].values())
    log.info("%s %sZ: %d/%d frames, %d overlay images", src.upper(), C.key(cycle), len(frames), len(wanted), n_ovl)
    return meta


def _trim(meta, last_obs):
    if not meta:
        return None
    lo = last_obs.strftime("%Y-%m-%dT%H:%MZ")
    return {**meta, "frames": [f for f in meta["frames"] if f["t"] > lo]}
