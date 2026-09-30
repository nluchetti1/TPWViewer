"""Transparent PNG layers, all pixel-aligned to the same equirectangular frame.

Every image spans exactly [W, E] x [S, N] at IMG_W x IMG_H, so the browser can stack them
with plain absolute positioning.
"""
import io
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.patheffects as pe                   # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from PIL import Image                                 # noqa: E402
from scipy.ndimage import gaussian_filter             # noqa: E402

from . import config as C                             # noqa: E402
from . import rap                                     # noqa: E402

log = logging.getLogger("render")

ICE = "#8FEFFF"
AMBER = "#FFB547"
HALO = [pe.withStroke(linewidth=2.6, foreground=(0.02, 0.06, 0.12, 0.75))]
FLOW = LinearSegmentedColormap.from_list("flow", ["#9FE9FF", "#E6FBFF", "#FFE08A", "#FF8A5B", "#FF4F8B"])
THETA = LinearSegmentedColormap.from_list("theta", ["#7FB2FF", "#8FEFFF", "#E8FFF6", "#FFD36B", "#FF7A45", "#FF3D7F"])

# Overlay catalog: written to the manifest and used to build the dropdown.
OVERLAYS = [
    {"id": "mflux925", "group": "Moisture transport", "name": "925 hPa moisture transport",
     "legend": "Vectors of q·V at 925 hPa (RAP analysis).", "opacity": 1.0,
     "ref": {"value": 150, "deg": 1.0, "units": "g kg⁻¹ m s⁻¹"}},
    {"id": "mflux850", "group": "Moisture transport", "name": "850 hPa moisture transport",
     "legend": "Vectors of q·V at 850 hPa (RAP analysis).", "opacity": 1.0,
     "ref": {"value": 150, "deg": 1.0, "units": "g kg⁻¹ m s⁻¹"}},
    {"id": "mfc", "group": "Moisture transport", "name": "925–700 hPa moisture flux convergence",
     "legend": "Layer-integrated MFC in mm/day. Blue-white = convergence, amber = divergence.",
     "opacity": 0.85},
    {"id": "stream925", "group": "Flow", "name": "925 hPa streamlines",
     "legend": "Streamlines colored by speed: pale blue calm, pink 40 kt+.", "opacity": 0.95},
    {"id": "stream850", "group": "Flow", "name": "850 hPa streamlines",
     "legend": "Streamlines colored by speed: pale blue calm, pink 40 kt+.", "opacity": 0.95},
    {"id": "h500", "group": "Flow", "name": "500 hPa heights and wind",
     "legend": "Heights every 3 dam, barbs in knots.", "opacity": 0.95},
    {"id": "thetae850", "group": "Thermodynamics", "name": "850 hPa theta-e",
     "legend": "Contours every 4 K, blue low to pink high; 340 K bold.", "opacity": 0.95},
    {"id": "mlcape", "group": "Thermodynamics", "name": "Mixed-layer CAPE (90–0 hPa)",
     "legend": "Contours at 250, 500, 1000, 2000, 3000, 4000 J/kg.", "opacity": 0.95},
    {"id": "rappw", "group": "Model check", "name": "RAP precipitable water contours",
     "legend": "RAP PW every 5 mm; 50 mm bold. Compare against the MIMIC shading.", "opacity": 0.95},
]


# ---------------------------------------------------------------- helpers

def _grid():
    lons = C.target_lons()
    lats = C.target_lats()[::-1]              # ascending for matplotlib
    return lons, lats


def _up(a):
    """Flip a north-first target-grid array to south-first for plotting."""
    return None if a is None else a[::-1, :]


def _fig():
    fig = plt.figure(figsize=(C.IMG_W / C.DPI, C.IMG_H / C.DPI), dpi=C.DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(C.W, C.E)
    ax.set_ylim(C.S, C.N)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return fig, ax


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=C.DPI, transparent=True, format="png")
    plt.close(fig)
    # 8-bit palette PNG with alpha: ~4-8x smaller than RGBA, keeps the loop light.
    img = Image.open(buf).convert("RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    tmp = path + ".tmp.png"
    img.save(tmp, optimize=True)
    os.replace(tmp, path)


def _smooth(a, sigma=1.5):
    if a is None:
        return None
    m = ~np.isfinite(a)
    out = gaussian_filter(np.where(m, np.nanmean(a), a), sigma)
    out[m] = np.nan
    return out


def _clabel(ax, cs, fmt, color="white"):
    labels = ax.clabel(cs, fmt=fmt, fontsize=10, inline=True, inline_spacing=3, colors=color)
    for lb in labels:
        lb.set_path_effects(HALO)


# ---------------------------------------------------------------- layers

def _vectors(ax, fx, fy, ref):
    lons, lats = _grid()
    step = 10
    X, Y = np.meshgrid(lons[::step], lats[::step])
    U = _up(fx)[::step, ::step]
    V = _up(fy)[::step, ::step]
    ok = np.isfinite(U) & np.isfinite(V)
    q = ax.quiver(X[ok], Y[ok], U[ok], V[ok], angles="xy", scale_units="xy",
                  scale=ref["value"] / ref["deg"], width=0.0024, headwidth=3.6, headlength=4, headaxislength=3.6, minlength=0.3,
                  color="white", edgecolor=(0.02, 0.06, 0.12, 0.8), linewidth=0.5)
    return q


def draw_mflux(f, p, path, ref):
    fx, fy = rap.moisture_flux(f, p)
    fig, ax = _fig()
    _vectors(ax, fx, fy, ref)
    _save(fig, path)


def draw_stream(f, p, path):
    lons, lats = _grid()
    u, v = _up(f["u"][p]), _up(f["v"][p])
    spd = np.hypot(u, v) * 1.943844
    fig, ax = _fig()
    sp = ax.streamplot(lons, lats, np.nan_to_num(u), np.nan_to_num(v), density=1.4,
                       color=np.nan_to_num(spd), cmap=FLOW, norm=Normalize(0, 40),
                       linewidth=1.5, arrowsize=1.0, arrowstyle="-|>")
    sp.lines.set_path_effects(HALO)
    _save(fig, path)


def draw_mfc(f, path):
    mfc = rap.mfc_layer(f)
    fig, ax = _fig()
    if mfc is not None:
        lons, lats = _grid()
        m = _up(mfc)
        ax.contourf(lons, lats, m, levels=[5, 10, 20, 40, 80, 1e6],
                    colors=["#3D7BD9", "#4FA6F0", "#7FD4FF", "#BFEFFF", "#FFFFFF"], alpha=0.7)
        ax.contourf(lons, lats, m, levels=[-1e6, -40, -20, -10, -5],
                    colors=["#B8611A", "#D9822B", "#F0A447", "#FFC978"], alpha=0.55)
    _save(fig, path)


def draw_h500(f, path):
    lons, lats = _grid()
    gh = _smooth(_up(f["gh"].get(500)), 2.0)
    fig, ax = _fig()
    if gh is not None:
        dam = gh / 10.0
        lo = np.floor(np.nanmin(dam) / 3) * 3
        cs = ax.contour(lons, lats, dam, levels=np.arange(lo, 610, 3), colors="white",
                        linewidths=1.5)
        for c in [cs]:
            c.set_path_effects(HALO)
        _clabel(ax, cs, "%d")
        step = 15
        u = _up(f["u"][500])[::step, ::step] * 1.943844
        v = _up(f["v"][500])[::step, ::step] * 1.943844
        X, Y = np.meshgrid(lons[::step], lats[::step])
        ok = np.isfinite(u)
        ax.barbs(X[ok], Y[ok], u[ok], v[ok], length=5.2, linewidth=0.9, color=ICE,
                 path_effects=HALO)
    _save(fig, path)


def draw_thetae(f, p, path):
    lons, lats = _grid()
    te = _smooth(_up(rap.theta_e(f["t"][p], f["rh"][p], float(p))), 2.0)
    fig, ax = _fig()
    levels = np.arange(300, 372, 4)
    cs = ax.contour(lons, lats, te, levels=levels, cmap=THETA, norm=Normalize(316, 356),
                    linewidths=[2.6 if lv == 340 else 1.3 for lv in levels])
    cs.set_path_effects(HALO)
    _clabel(ax, cs, "%d", color=None)
    _save(fig, path)


def draw_mlcape(f, path):
    fig, ax = _fig()
    cape = _smooth(_up(f.get("cape")), 1.5)
    if cape is not None:
        lons, lats = _grid()
        levels = [250, 500, 1000, 2000, 3000, 4000]
        colors = ["#9FE9FF", "#E6FBFF", "#FFE08A", "#FFB547", "#FF7A45", "#FF3D7F"]
        cs = ax.contour(lons, lats, cape, levels=levels, colors=colors, linewidths=1.5)
        cs.set_path_effects(HALO)
        _clabel(ax, cs, "%d", color=None)
    _save(fig, path)


def draw_rappw(f, path):
    fig, ax = _fig()
    pw = _smooth(_up(f.get("pw")), 1.2)
    if pw is not None:
        lons, lats = _grid()
        levels = np.arange(10, 80, 5)
        cs = ax.contour(lons, lats, pw, levels=levels, colors="white",
                        linewidths=[2.4 if lv == 50 else 1.0 for lv in levels])
        cs.set_path_effects(HALO)
        _clabel(ax, cs, "%d")
    _save(fig, path)


def render_all(f, key, site):
    """Render every overlay for one valid time. Returns the list of overlay ids written."""
    done = []
    refs = {o["id"]: o.get("ref") for o in OVERLAYS}
    jobs = {
        "mflux925": lambda p: draw_mflux(f, 925, p, refs["mflux925"]),
        "mflux850": lambda p: draw_mflux(f, 850, p, refs["mflux850"]),
        "mfc": lambda p: draw_mfc(f, p),
        "stream925": lambda p: draw_stream(f, 925, p),
        "stream850": lambda p: draw_stream(f, 850, p),
        "h500": lambda p: draw_h500(f, p),
        "thetae850": lambda p: draw_thetae(f, 850, p),
        "mlcape": lambda p: draw_mlcape(f, p),
        "rappw": lambda p: draw_rappw(f, p),
    }
    for oid, fn in jobs.items():
        path = os.path.join(site, "data", "ovl", oid, f"{key}.png")
        try:
            fn(path)
            done.append(oid)
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            log.warning("overlay %s %s failed: %s", oid, key, exc)
    return done


def draw_coast(path):
    """Coastlines, borders, states and lakes as one transparent layer."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(C.IMG_W / C.DPI, C.IMG_H / C.DPI), dpi=C.DPI)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([C.W, C.E, C.S, C.N], crs=ccrs.PlateCarree())
    ax.set_xlim(C.W, C.E)
    ax.set_ylim(C.S, C.N)
    ax.set_aspect("auto")
    ax.spines["geo"].set_visible(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    line = (0.93, 0.98, 1.0, 0.9)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor=(0.93, 0.98, 1.0, 0.45),
                   facecolor="none", linewidth=0.6, path_effects=HALO)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor=line, linewidth=0.8, path_effects=HALO)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=line, linewidth=0.9, path_effects=HALO)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), edgecolor=line, facecolor="none", linewidth=0.6)
    _save(fig, path)
