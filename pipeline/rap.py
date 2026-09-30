"""RAP 13-km analysis (f00) and HRRR forecast access, plus derived fields on the target grid.

Source A (default): AWS Open Data bucket noaa-rap-pds, byte-range reads driven by the .idx.
Source B: NOMADS grib filter (filter_rap.pl) with a subregion.
Winds on grid 130 (Lambert conformal) are grid-relative and are rotated to earth-relative
before regridding.
"""
import logging
import os
import re
import tempfile

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

from . import config as C

log = logging.getLogger("rap")

ISO_VARS = ("HGT", "TMP", "RH", "UGRD", "VGRD")
R_EARTH = 6.371e6
_trapz = getattr(np, "trapezoid", None) or np.trapz
G = 9.80665


# ---------------------------------------------------------------- download

def _file_name(t):
    return f"rap.t{t:%H}z.awp130pgrbf00.grib2"


def _wanted(var, lev, levels):
    if var in ISO_VARS:
        m = re.fullmatch(r"(\d+) mb", lev)
        return bool(m) and int(m.group(1)) in levels
    if var == "PWAT":
        return lev.startswith("entire atmosphere")
    if var == "CAPE":
        return lev == "90-0 mb above ground"
    return False


def _fetch_aws(session, t):
    return fetch_idx_records(session, f"{C.RAP_AWS}/rap.{t:%Y%m%d}/{_file_name(t)}", C.RAP_LEVELS)


def hrrr_url(cycle, fh):
    return f"{C.HRRR_AWS}/hrrr.{cycle:%Y%m%d}/conus/hrrr.t{cycle:%H}z.wrfprsf{fh:02d}.grib2"


def hrrr_available(session, cycle, fh):
    try:
        r = session.head(hrrr_url(cycle, fh) + ".idx", timeout=30)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def fetch_hrrr(session, cycle, fh):
    try:
        return fetch_idx_records(session, hrrr_url(cycle, fh), C.HRRR_LEVELS)
    except Exception as exc:  # noqa: BLE001
        log.warning("HRRR %s f%02d fetch failed: %s", C.key(cycle), fh, exc)
        return None


def fetch_idx_records(session, base, levels):
    """Byte-range the wanted records out of a GRIB2 file using its wgrib2-style .idx."""
    r = session.get(base + ".idx", timeout=60)
    if r.status_code in (403, 404):
        return None
    r.raise_for_status()
    lines = [ln.split(":") for ln in r.text.strip().splitlines()]
    ranges = []
    for i, parts in enumerate(lines):
        if len(parts) < 5 or not _wanted(parts[3], parts[4], levels):
            continue
        start = int(parts[1])
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        ranges.append([start, end])
    if not ranges:
        raise RuntimeError(f"no matching RAP records in {base}.idx")

    # merge ranges separated by small gaps to cut the request count
    ranges.sort(key=lambda x: x[0])
    merged = [ranges[0][:]]
    for s, e in ranges[1:]:
        last = merged[-1]
        if last[1] is not None and s - last[1] <= 2_000_000:
            last[1] = e if (e is None or e > last[1]) else last[1]
        else:
            merged.append([s, e])

    chunks = []
    for s, e in merged:
        hdr = {"Range": f"bytes={s}-{'' if e is None else e}"}
        rr = session.get(base, headers=hdr, timeout=120)
        rr.raise_for_status()
        chunks.append((s, rr.content))

    # keep only the wanted records out of the merged chunks
    out = bytearray()
    for s, e in ranges:
        for cs, blob in chunks:
            if cs <= s and (e is None or e < cs + len(blob)):
                out += blob[s - cs: None if e is None else e - cs + 1]
                break
    return bytes(out)


def _fetch_nomads(session, t):
    params = {
        "dir": f"/rap.{t:%Y%m%d}",
        "file": _file_name(t),
        "subregion": "",
        "leftlon": C.W - 3, "rightlon": C.E + 3,
        "toplat": C.N + 3, "bottomlat": C.S - 3,
        "lev_entire_atmosphere_(considered_as_a_single_layer)": "on",
        "lev_90-0_mb_above_ground": "on",
        "var_PWAT": "on", "var_CAPE": "on",
    }
    for v in ISO_VARS:
        params[f"var_{v}"] = "on"
    for p in C.RAP_LEVELS:
        params[f"lev_{p}_mb"] = "on"
    r = session.get(C.RAP_NOMADS_FILTER, params=params, timeout=180)
    if r.status_code == 404 or not r.content.startswith(b"GRIB"):
        return None
    r.raise_for_status()
    return r.content


def fetch(session, t):
    """Return GRIB2 bytes for the RAP analysis valid at t, or None if unavailable."""
    fn = _fetch_aws if C.RAP_SOURCE == "aws" else _fetch_nomads
    try:
        return fn(session, t)
    except Exception as exc:  # noqa: BLE001
        log.warning("RAP %s fetch failed (%s): %s", C.key(t), C.RAP_SOURCE, exc)
        return None


# ---------------------------------------------------------------- decode + regrid

def _open(path, **keys):
    return xr.open_dataset(path, engine="cfgrib",
                           backend_kwargs={"filter_by_keys": keys, "indexpath": ""})


class _Regridder:
    """Barycentric (linear) interpolation from the Lambert grid to the target lat/lon grid."""

    def __init__(self, lat2d, lon2d):
        lon2d = ((lon2d + 180.0) % 360.0) - 180.0
        box = ((lon2d > C.W - 2) & (lon2d < C.E + 2) & (lat2d > C.S - 2) & (lat2d < C.N + 2))
        self.idx = np.flatnonzero(box.ravel())
        pts = np.column_stack([lon2d.ravel()[self.idx], lat2d.ravel()[self.idx]])
        tri = Delaunay(pts)
        glon, glat = np.meshgrid(C.target_lons(), C.target_lats())
        xi = np.column_stack([glon.ravel(), glat.ravel()])
        simp = tri.find_simplex(xi)
        T = tri.transform[simp]
        b = np.einsum("ijk,ik->ij", T[:, :2], xi - T[:, 2])
        self.w = np.column_stack([b, 1.0 - b.sum(axis=1)])
        self.verts = tri.simplices[simp]
        self.outside = simp < 0

    def __call__(self, field2d):
        v = np.asarray(field2d, dtype="float64").ravel()[self.idx]
        out = (v[self.verts] * self.w).sum(axis=1)
        out[self.outside] = np.nan
        return out.reshape(C.NY, C.NX).astype("float32")


def _rotate(u, v, lon2d, attrs):
    """Grid-relative -> earth-relative winds for a Lambert conformal grid."""
    if str(attrs.get("GRIB_gridType", "lambert")) != "lambert":
        return u, v
    lov = float(attrs.get("GRIB_LoVInDegrees", 265.0))
    latin1 = float(attrs.get("GRIB_Latin1InDegrees", 25.0))
    latin2 = float(attrs.get("GRIB_Latin2InDegrees", latin1))
    if abs(latin1 - latin2) < 1e-6:
        cone = np.sin(np.deg2rad(latin1))
    else:
        l1, l2 = np.deg2rad(latin1), np.deg2rad(latin2)
        cone = (np.log(np.cos(l1)) - np.log(np.cos(l2))) / (
            np.log(np.tan(np.pi / 4 + l2 / 2)) - np.log(np.tan(np.pi / 4 + l1 / 2)))
    dlon = ((lon2d - lov + 180.0) % 360.0) - 180.0
    a = np.deg2rad(cone * dlon)
    ca, sa = np.cos(a), np.sin(a)
    return ca * u + sa * v, -sa * u + ca * v


def decode(grib_bytes, stride=1):
    """Decode RAP/HRRR GRIB2 bytes into regridded base fields on the target grid.

    stride > 1 thins the native grid first (HRRR 3 km -> ~9 km) to keep triangulation cheap.
    """
    sl = (slice(None, None, stride), slice(None, None, stride))
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib_bytes)
        path = tmp.name
    try:
        iso = _open(path, typeOfLevel="isobaricInhPa").load()
        lat2d = iso["latitude"].values[sl]
        lon2d = iso["longitude"].values[sl]
        rg = _Regridder(lat2d, lon2d)
        levs = [int(p) for p in iso["isobaricInhPa"].values]

        f = {"u": {}, "v": {}, "t": {}, "rh": {}, "gh": {}}
        for p in levs:
            lvl = iso.sel(isobaricInhPa=p)
            ue, ve = _rotate(lvl["u"].values[sl], lvl["v"].values[sl], lon2d, iso["u"].attrs)
            f["u"][p], f["v"][p] = rg(ue), rg(ve)
            f["t"][p] = rg(lvl["t"].values[sl])
            f["rh"][p] = rg(lvl["r"].values[sl])
            f["gh"][p] = rg(lvl["gh"].values[sl])

        f["pw"] = None
        try:
            f["pw"] = rg(_open(path, shortName="pwat")["pwat"].values[sl])
        except Exception as exc:  # noqa: BLE001
            log.warning("RAP PWAT missing: %s", exc)
        f["cape"] = None
        try:
            cp = _open(path, shortName="cape", typeOfLevel="pressureFromGroundLayer")["cape"]
            if cp.ndim == 3:        # several layers came through; keep the shallowest (90-0 hPa)
                cp = cp.isel({cp.dims[0]: 0})
            f["cape"] = rg(cp.values[sl])
        except Exception as exc:  # noqa: BLE001
            log.warning("RAP MLCAPE missing: %s", exc)
        return f
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- derived fields

def _es(tc):
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def vapor(t_k, rh, p_hpa):
    """Return (vapor pressure hPa, specific humidity g/kg, mixing ratio g/kg)."""
    e = np.clip(rh, 0.5, 100.0) / 100.0 * _es(t_k - 273.15)
    q = 1000.0 * 0.622 * e / (p_hpa - 0.378 * e)
    r = 622.0 * e / (p_hpa - e)
    return e, q, r


def theta_e(t_k, rh, p_hpa):
    """Bolton (1980) equivalent potential temperature (K)."""
    e, _, r = vapor(t_k, rh, p_hpa)
    ln = np.log(e / 6.112)
    td = 243.5 * ln / (17.67 - ln) + 273.15
    tl = 1.0 / (1.0 / (td - 56.0) + np.log(t_k / td) / 800.0) + 56.0
    return (t_k * (1000.0 / p_hpa) ** (0.2854 * (1 - 0.00028 * r))
            * np.exp((3.376 / tl - 0.00254) * r * (1 + 0.00081 * r)))


def moisture_flux(f, p):
    _, q, _ = vapor(f["t"][p], f["rh"][p], float(p))
    return q * f["u"][p], q * f["v"][p]          # g kg-1 m s-1


def _divergence(fx, fy):
    lam = np.deg2rad(C.target_lons())
    phi = np.deg2rad(C.target_lats())
    cos = np.cos(phi)[:, None]
    dfx = np.gradient(fx, lam, axis=1)
    dfy = np.gradient(fy * cos, phi, axis=0)
    return (dfx + dfy) / (R_EARTH * cos)


def mfc_layer(f, top=None, bottom=None):
    """Layer-integrated moisture flux convergence (mm/day), positive = convergence."""
    bottom, top = C.MFC_LAYER if top is None else (bottom, top)
    levs = sorted(p for p in f["u"] if top <= p <= bottom)
    if len(levs) < 2:
        return None
    divs = []
    mask = np.zeros((C.NY, C.NX), bool)
    for p in levs:
        _, q, _ = vapor(f["t"][p], f["rh"][p], float(p))
        fx, fy = q / 1000.0 * f["u"][p], q / 1000.0 * f["v"][p]   # kg kg-1 m s-1
        mask |= ~np.isfinite(fx) | ~np.isfinite(fy)
        fx = gaussian_filter(np.nan_to_num(fx), 2.0)
        fy = gaussian_filter(np.nan_to_num(fy), 2.0)
        divs.append(_divergence(fx, fy))
    p_pa = np.array(levs, dtype="float64") * 100.0
    integral = _trapz(np.stack(divs), x=p_pa, axis=0) / G   # kg m-2 s-1
    mfc = -integral * 86400.0
    mfc[mask] = np.nan
    return mfc.astype("float32")
