"""Rolling MIMIC PW mean for the anomaly view.

Cold start: average 00Z and 12Z composites over the last CLIM_DAYS days. After that, each new
MIMIC hour updates an exponential moving average with an e-folding of CLIM_EMA_HOURS.
Stored as raw float32 (north-first rows) so the browser can read it directly.
"""
import datetime as dt
import json
import logging
import os

import numpy as np

from . import config as C
from . import mimic

log = logging.getLogger("clim")


def paths(site):
    d = os.path.join(site, "data", "clim")
    return os.path.join(d, "mean.f32"), os.path.join(d, "meta.json")


def load(site):
    fp, mp = paths(site)
    if not (os.path.exists(fp) and os.path.exists(mp)):
        return None, None
    mean = np.fromfile(fp, dtype="<f4")
    if mean.size != C.NX * C.NY:
        return None, None
    with open(mp) as fh:
        return mean.reshape(C.NY, C.NX), json.load(fh)


def save(site, mean, meta):
    fp, mp = paths(site)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    mean.astype("<f4").tofile(fp + ".tmp")
    os.replace(fp + ".tmp", fp)
    with open(mp, "w") as fh:
        json.dump(meta, fh)


def cold_start(session, site):
    now = C.utc_now_hour()
    start = now - dt.timedelta(days=C.CLIM_DAYS)
    files = {t: u for t, u in mimic.list_range(session, start, now).items() if t.hour in (0, 12)}
    log.info("climatology cold start from %d composites", len(files))
    total = np.zeros((C.NY, C.NX), "float64")
    count = np.zeros((C.NY, C.NX), "float64")
    for t, url in sorted(files.items()):
        try:
            tpw, _ = mimic.fetch_subset(session, url)
        except Exception as exc:  # noqa: BLE001
            log.warning("clim %s failed: %s", C.key(t), exc)
            continue
        ok = np.isfinite(tpw)
        total[ok] += tpw[ok]
        count[ok] += 1
    if not count.any():
        return None, None
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype("float32")
    meta = {"last": C.key(max(files)), "since": C.key(min(files)), "samples": int(count.max()),
            "method": f"{C.CLIM_DAYS}-day 00/12Z mean, then {C.CLIM_EMA_HOURS} h EMA"}
    save(site, mean, meta)
    return mean, meta


def update(site, mean, meta, new_fields):
    """Fold hourly MIMIC fields {valid_time: tpw} newer than meta['last'] into the EMA."""
    alpha = 1.0 / C.CLIM_EMA_HOURS
    last = meta["last"]
    n = 0
    for t in sorted(new_fields):
        if C.key(t) <= last:
            continue
        x = new_fields[t]
        ok = np.isfinite(x) & np.isfinite(mean)
        mean[ok] += alpha * (x[ok] - mean[ok])
        fill = np.isfinite(x) & ~np.isfinite(mean)
        mean[fill] = x[fill]
        last = C.key(t)
        n += 1
    if n:
        meta["last"] = last
        save(site, mean, meta)
    return n
