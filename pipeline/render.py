"""Transparent PNG layers, pixel-aligned per display domain.

Every image for a domain spans exactly that domain's extent at IMG_W x IMG_H, so the browser
can stack layers with plain absolute positioning. Fields arrive on the Southeast 0.1° target
grid; each domain plots a sliced window of it with its own vector/barb spacing.
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
HALO = [pe.withStroke(linewidth=2.6, foreground=(0.02, 0.06, 0.12, 0.75))]
FLOW = LinearSegmentedColormap.from_list("flow", ["#9FE9FF", "#E6FBFF", "#FFE08A", "#FF8A5B", "#FF4F8B"])
THETA = LinearSegmentedColormap.from_list("theta", ["#7FB2FF", "#8FEFFF", "#E8FFF6", "#FFD36B", "#FF7A45", "#FF3D7F"])

# Overlay catalog: written to the manifest and used to build the dropdown.
# Vector reference: an arrow of `value` spans the domain's `ref_deg` degrees of longitude.
OVERLAYS = [
    {"id": "mflux925", "group": "Moisture transport", "name": "925 hPa moisture transport",
     "legend": "Vectors of q·V at 925 hPa (RAP analysis).", "opacity": 1.0,
     "ref": {"value": 150, "units": "g kg⁻¹ m s⁻¹"}},
    {"id": "mflux850", "group": "Moisture transport", "name": "850 hPa moisture transport",
     "legend": "Vectors of q·V at 850 hPa (RAP analysis).", "opacity": 1.0,
     "ref": {"value": 150, "units": "g kg⁻¹ m s⁻¹"}},
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


def overlay_path(site, dom, oid, key):
    return os.path.join(site, "data", dom["dir"], oid, f"{key}.png")


# ---------------------------------------------------------------- domain windowing

class Win:
    """Ascending lon/lat vectors and a slicer for one domain (with a 1° margin)."""

    def __init__(self, dom):
        w, e, s, n = dom["extent"]
        lons = C.target_lons()
        lats = C.target_lats()[::-1]                 # ascending for matplotlib
        self.ix = np.flatnonzero((lons >= w - 1) & (lons <= e + 1))
        self.iy = np.flatnonzero((lats >= s - 1) & (lats <= n + 1))
        self.lons = lons[self.ix]
        self.lats = lats[self.iy]
        self.dom = dom

    def __call__(self, a):
        """North-first target-grid array -> south-first window."""
        if a is None:
            return None
        up = a[::-1, :]
        return up[self.iy[0]:self.iy[-1] + 1, self.ix[0]:self.ix[-1] + 1]


def _fig(dom):
    w, e, s, n = dom["extent"]
    fig = plt.figure(figsize=(C.IMG_W / C.DPI, C.IMG_H / C.DPI), dpi=C.DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
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
    if m.all():
        return a
    out = gaussian_filter(np.where(m, np.nanmean(a), a), sigma)
    out[m] = np.nan
    return out


def _clabel(ax, cs, fmt, color="white"):
    labels = ax.clabel(cs, fmt=fmt, fontsize=10, inline=True, inline_spacing=3, colors=color)
    for lb in labels:
        lb.set_path_effects(HALO)


# ---------------------------------------------------------------- layers

def draw_mflux(f, p, dom, path):
    win = Win(dom)
    fx, fy = rap.moisture_flux(f, p)
    fig, ax = _fig(dom)
    st = dom["vec_step"]
    X, Y = np.meshgrid(win.lons[::st], win.lats[::st])
    U = win(fx)[::st, ::st]
    V = win(fy)[::st, ::st]
    ok = np.isfinite(U) & np.isfinite(V)
    ref = next(o["ref"] for o in OVERLAYS if o["id"] == f"mflux{p}")
    ax.quiver(X[ok], Y[ok], U[ok], V[ok], angles="xy", scale_units="xy",
              scale=ref["value"] / dom["ref_deg"], width=0.0024, headwidth=3.6, headlength=4,
              headaxislength=3.6, minlength=0.3, color="white",
              edgecolor=(0.02, 0.06, 0.12, 0.8), linewidth=0.5)
    _save(fig, path)


def draw_stream(f, p, dom, path):
    win = Win(dom)
    u, v = win(f["u"][p]), win(f["v"][p])
    spd = np.hypot(u, v) * 1.943844
    fig, ax = _fig(dom)
    sp = ax.streamplot(win.lons, win.lats, np.nan_to_num(u), np.nan_to_num(v),
                       density=dom["stream_density"], color=np.nan_to_num(spd), cmap=FLOW,
                       norm=Normalize(0, 40), linewidth=1.5, arrowsize=1.0, arrowstyle="-|>")
    sp.lines.set_path_effects(HALO)
    _save(fig, path)


def draw_mfc(f, dom, path, cache):
    if "mfc" not in cache:
        cache["mfc"] = rap.mfc_layer(f)
    mfc = cache["mfc"]
    fig, ax = _fig(dom)
    if mfc is not None:
        win = Win(dom)
        m = win(mfc)
        ax.contourf(win.lons, win.lats, m, levels=[5, 10, 20, 40, 80, 1e6],
                    colors=["#3D7BD9", "#4FA6F0", "#7FD4FF", "#BFEFFF", "#FFFFFF"], alpha=0.7)
        ax.contourf(win.lons, win.lats, m, levels=[-1e6, -40, -20, -10, -5],
                    colors=["#B8611A", "#D9822B", "#F0A447", "#FFC978"], alpha=0.55)
    _save(fig, path)


def draw_h500(f, dom, path):
    win = Win(dom)
    gh = _smooth(win(f["gh"].get(500)), 2.0)
    fig, ax = _fig(dom)
    if gh is not None and np.isfinite(gh).any():
        dam = gh / 10.0
        lo = np.floor(np.nanmin(dam) / 3) * 3
        cs = ax.contour(win.lons, win.lats, dam, levels=np.arange(lo, 610, 3), colors="white",
                        linewidths=1.5)
        cs.set_path_effects(HALO)
        _clabel(ax, cs, "%d")
        st = dom["barb_step"]
        u = win(f["u"][500])[::st, ::st] * 1.943844
        v = win(f["v"][500])[::st, ::st] * 1.943844
        X, Y = np.meshgrid(win.lons[::st], win.lats[::st])
        ok = np.isfinite(u) & np.isfinite(v)
        ax.barbs(X[ok], Y[ok], u[ok], v[ok], length=5.2, linewidth=0.9, color=ICE,
                 path_effects=HALO)
    _save(fig, path)


def draw_thetae(f, p, dom, path):
    win = Win(dom)
    te = _smooth(win(rap.theta_e(f["t"][p], f["rh"][p], float(p))), 2.0)
    fig, ax = _fig(dom)
    levels = np.arange(300, 372, 4)
    cs = ax.contour(win.lons, win.lats, te, levels=levels, cmap=THETA, norm=Normalize(316, 356),
                    linewidths=[2.6 if lv == 340 else 1.3 for lv in levels])
    cs.set_path_effects(HALO)
    _clabel(ax, cs, "%d", color=None)
    _save(fig, path)


def draw_mlcape(f, dom, path):
    fig, ax = _fig(dom)
    win = Win(dom)
    cape = _smooth(win(f.get("cape")), 1.5)
    if cape is not None:
        levels = [250, 500, 1000, 2000, 3000, 4000]
        colors = ["#9FE9FF", "#E6FBFF", "#FFE08A", "#FFB547", "#FF7A45", "#FF3D7F"]
        cs = ax.contour(win.lons, win.lats, cape, levels=levels, colors=colors, linewidths=1.5)
        cs.set_path_effects(HALO)
        _clabel(ax, cs, "%d", color=None)
    _save(fig, path)


def draw_rappw(f, dom, path):
    fig, ax = _fig(dom)
    win = Win(dom)
    pw = _smooth(win(f.get("pw")), 1.2)
    if pw is not None:
        levels = np.arange(10, 80, 5)
        cs = ax.contour(win.lons, win.lats, pw, levels=levels, colors="white",
                        linewidths=[2.4 if lv == 50 else 1.0 for lv in levels])
        cs.set_path_effects(HALO)
        _clabel(ax, cs, "%d")
    _save(fig, path)


def render_all(f, key, site, todo):
    """Render the (domain, overlay_id) pairs in `todo` for one valid time.

    Returns the list of (domain_id, overlay_id) pairs written.
    """
    cache = {}
    jobs = {
        "mflux925": lambda d, p: draw_mflux(f, 925, d, p),
        "mflux850": lambda d, p: draw_mflux(f, 850, d, p),
        "mfc": lambda d, p: draw_mfc(f, d, p, cache),
        "stream925": lambda d, p: draw_stream(f, 925, d, p),
        "stream850": lambda d, p: draw_stream(f, 850, d, p),
        "h500": lambda d, p: draw_h500(f, d, p),
        "thetae850": lambda d, p: draw_thetae(f, 850, d, p),
        "mlcape": lambda d, p: draw_mlcape(f, d, p),
        "rappw": lambda d, p: draw_rappw(f, d, p),
    }
    done = []
    for dom, oid in todo:
        try:
            jobs[oid](dom, overlay_path(site, dom, oid, key))
            done.append((dom["id"], oid))
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            log.warning("overlay %s/%s %s failed: %s", dom["id"], oid, key, exc)
    return done


def draw_coast(dom, path):
    """Coastlines, borders, states, lakes (and counties when zoomed) as one transparent layer."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    w, e, s, n = dom["extent"]
    fig = plt.figure(figsize=(C.IMG_W / C.DPI, C.IMG_H / C.DPI), dpi=C.DPI)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([w, e, s, n], crs=ccrs.PlateCarree())
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect("auto")
    ax.spines["geo"].set_visible(False)
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    line = (0.93, 0.98, 1.0, 0.9)

    if dom.get("counties"):
        # Optional: US counties. Fetch geometries up front so a download failure only drops
        # this layer instead of breaking the whole coastline image.
        try:
            counties = cfeature.NaturalEarthFeature("cultural", "admin_2_counties", "10m")
            geoms = list(counties.geometries())
            ax.add_geometries(geoms, ccrs.PlateCarree(), edgecolor=(0.93, 0.98, 1.0, 0.18),
                              facecolor="none", linewidth=0.35)
        except Exception as exc:  # noqa: BLE001
            log.warning("county layer skipped: %s", exc)

    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor=(0.93, 0.98, 1.0, 0.45),
                   facecolor="none", linewidth=0.6, path_effects=HALO)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor=line, linewidth=0.8, path_effects=HALO)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor=line,
                   linewidth=1.1 if dom.get("counties") else 0.9, path_effects=HALO)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), edgecolor=line, facecolor="none", linewidth=0.6)
    _save(fig, path)
