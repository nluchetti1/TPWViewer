"""GOES-East ABI imagery: visible (band 2) and clean IR (band 13) as per-domain JPEG frames.

Source: {GLM_BUCKET}/ABI-L2-CMIPC/YYYY/DDD/HH/OR_ABI-L2-CMIPC-M6Cbb_G19_s....nc  (CONUS sector,
5-min cadence). For each frame hour the first scan of that hour (~HH:01) is used. Pixels are
mapped from the geostationary fixed grid to each domain's equirectangular frame (nearest
neighbour) with the GOES-R PUG navigation equations, so the images stack exactly with the
overlay PNGs.
"""
import datetime as dt
import logging
import math
import os
import re
import tempfile

import numpy as np
from PIL import Image

from . import config as C

log = logging.getLogger("goes")
_KEY = re.compile(r"<Key>([^<]+)</Key>")
_NAME = re.compile(r"-M\dC(\d\d)_G\d\d_s(\d{14})")
NODATA = (11, 31, 51)
_cache = {}


# ---------------------------------------------------------------- listing

def list_hour(session, hour_start):
    prefix = f"{C.ABI_PRODUCT}/{hour_start:%Y}/{hour_start:%j}/{hour_start:%H}/"
    r = session.get(C.GLM_BUCKET + "/", params={"list-type": "2", "prefix": prefix}, timeout=60)
    r.raise_for_status()
    return _KEY.findall(r.text)


def pick(keys, band):
    best = None
    for k in keys:
        m = _NAME.search(k)
        if m and int(m.group(1)) == band and (best is None or m.group(2) < best[0]):
            best = (m.group(2), k)
    return best[1] if best else None


# ---------------------------------------------------------------- solar elevation

def sun_elevation(t, lat, lon):
    doy = t.timetuple().tm_yday
    g = 2 * math.pi / 365.0 * (doy - 1 + (t.hour - 12) / 24.0)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = t.hour * 60 + t.minute + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1.0, min(1.0, cz))))


# ---------------------------------------------------------------- navigation

def scan_angles(lat_deg, lon_deg, proj):
    """Geodetic lat/lon (deg) -> ABI fixed-grid scan angles x, y (rad) and a visibility mask."""
    r_eq, r_pol, H, lon0 = proj
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    phi_c = np.arctan((r_pol ** 2 / r_eq ** 2) * np.tan(lat))
    e2 = (r_eq ** 2 - r_pol ** 2) / r_eq ** 2
    r_c = r_pol / np.sqrt(1 - e2 * np.cos(phi_c) ** 2)
    sx = H - r_c * np.cos(phi_c) * np.cos(lon - lon0)
    sy = -r_c * np.cos(phi_c) * np.sin(lon - lon0)
    sz = r_c * np.sin(phi_c)
    visible = H * (H - sx) > sy ** 2 + (r_eq ** 2 / r_pol ** 2) * sz ** 2
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return x, y, visible


def _index_map(dom, xs, ys, proj):
    """Row/col indices into the ABI grid for every pixel of the domain frame (cached)."""
    ck = (dom["id"], len(xs), len(ys), round(float(xs[0]), 7), round(float(ys[0]), 7), proj)
    if ck in _cache:
        return _cache[ck]
    w, e, s, n = dom["extent"]
    lons = w + (np.arange(C.IMG_W) + 0.5) * (e - w) / C.IMG_W
    lats = n - (np.arange(C.IMG_H) + 0.5) * (n - s) / C.IMG_H
    LON, LAT = np.meshgrid(lons, lats)
    x, y, vis = scan_angles(LAT, LON, proj)
    dx = (xs[-1] - xs[0]) / (len(xs) - 1)          # mean spacing: robust to packing round-off
    dy = (ys[-1] - ys[0]) / (len(ys) - 1)
    col = np.rint((x - xs[0]) / dx).astype(np.int64)
    row = np.rint((y - ys[0]) / dy).astype(np.int64)
    ok = vis & (col >= 0) & (col < len(xs)) & (row >= 0) & (row < len(ys))
    _cache[ck] = (row, col, ok)
    return _cache[ck]


# ---------------------------------------------------------------- colour

def _lut(stops, lo, hi):
    vals = np.linspace(lo, hi, 256)
    xs = [v for v, _ in stops]
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in stops], float)
    return np.stack([np.interp(vals, xs, rgb[:, k]) for k in range(3)], axis=1).astype(np.uint8)


IR_LO, IR_HI = -95.0, 45.0
IR_LUT = _lut([(-95, "#FFFFFF"), (-85, "#FF4FD8"), (-78, "#E0303D"), (-70, "#FF8A3D"), (-62, "#E8E05A"),
               (-55, "#2FB37A"), (-45, "#2B6FE0"), (-31, "#7FD4FF"), (-30, "#D8D8D8"), (45, "#141414")],
              IR_LO, IR_HI)


def colorize(kind, values):
    """values: float array (reflectance 0..1 for vis, Kelvin for IR), NaN = no data."""
    out = np.empty(values.shape + (3,), np.uint8)
    out[:] = NODATA
    ok = np.isfinite(values)
    if kind == "vis":
        g = (np.clip(values[ok], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        out[ok] = np.stack([g, g, g], axis=1)
    else:
        tc = values[ok] - 273.15
        idx = np.clip(np.rint((tc - IR_LO) / (IR_HI - IR_LO) * 255), 0, 255).astype(np.int64)
        out[ok] = IR_LUT[idx]
    return out


# ---------------------------------------------------------------- one file -> per-domain images

def render(blob, kind, domains):
    """Return {domain_id: PIL.Image} from one CMIPC file."""
    import netCDF4
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(blob)
        path = tmp.name
    try:
        ds = netCDF4.Dataset(path)
        p = ds.variables["goes_imager_projection"]
        proj = (float(p.semi_major_axis), float(p.semi_minor_axis),
                float(p.perspective_point_height) + float(p.semi_major_axis),
                math.radians(float(p.longitude_of_projection_origin)))
        xs = np.asarray(ds.variables["x"][:], dtype="float64")
        ys = np.asarray(ds.variables["y"][:], dtype="float64")
        var = ds.variables["CMI"]
        var.set_auto_maskandscale(False)
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        fill = getattr(var, "_FillValue", None)
        unsigned = str(getattr(var, "_Unsigned", "false")).lower() == "true"
        out = {}
        for dom in domains:
            row, col, ok = _index_map(dom, xs, ys, proj)
            if not ok.any():
                continue
            r0, r1 = int(row[ok].min()), int(row[ok].max()) + 1
            c0, c1 = int(col[ok].min()), int(col[ok].max()) + 1
            raw = np.asarray(var[r0:r1, c0:c1])
            if unsigned and raw.dtype.kind == "i":
                raw = raw.view(raw.dtype.str.replace("i", "u"))
            vals = np.full(row.shape, np.nan, "float32")
            sample = raw[row[ok] - r0, col[ok] - c0]
            good = np.ones(sample.shape, bool) if fill is None else sample != np.asarray(fill).astype(raw.dtype)
            v = sample.astype("float32") * scale + offset
            v[~good] = np.nan
            vals[ok] = v
            out[dom["id"]] = Image.fromarray(colorize(kind, vals), "RGB")
        ds.close()
        return out
    finally:
        os.unlink(path)


def save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.jpg"
    img.save(tmp, "JPEG", quality=84, optimize=True, progressive=True)
    os.replace(tmp, path)


def path_for(site, dom_id, kind, key):
    return os.path.join(site, "data", "goes", dom_id, kind, f"{key}.jpg")


def process_hour(session, site, t, kinds, keys=None):
    """Render the requested kinds for frame time t. Returns the kinds written."""
    keys = keys if keys is not None else list_hour(session, t)
    done = []
    for kind in kinds:
        k = pick(keys, C.ABI_BANDS[kind])
        if not k:
            continue
        r = session.get(f"{C.GLM_BUCKET}/{k}", timeout=180)
        r.raise_for_status()
        imgs = render(r.content, kind, C.DOMAINS)
        for dom_id, img in imgs.items():
            save(img, path_for(site, dom_id, kind, C.key(t)))
        if imgs:
            done.append(kind)
    return done
