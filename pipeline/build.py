"""Build or update the static site in --site.

Incremental: existing frames in the site directory are kept; only missing hours (plus the
newest MIMIC_REFRESH_HOURS of MIMIC, which SSEC re-issues) are fetched. Anything older than
HOURS_BACK is pruned.

    python -m pipeline.build --site site
"""
import argparse
import datetime as dt
import json
import logging
import os
import shutil
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config as C
from . import mimic, rap, render

log = logging.getLogger("build")


def session():
    s = requests.Session()
    s.headers["User-Agent"] = C.USER_AGENT
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def encode(arr):
    code = np.where(np.isfinite(arr), np.clip(np.round(arr * C.SCALE), 0, 254), C.MISSING)
    return code.astype(np.uint8).tobytes()


def write_bin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(encode(arr))
    os.replace(tmp, path)


def parse_key(k):
    return dt.datetime.strptime(k, "%Y%m%d%H").replace(tzinfo=dt.timezone.utc)


def prune(site, keep_keys):
    for sub in ("tpw", "rappw"):
        d = os.path.join(site, "data", sub)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.split(".")[0] not in keep_keys:
                    os.remove(os.path.join(d, fn))
    ovl = os.path.join(site, "data", "ovl")
    if os.path.isdir(ovl):
        for oid in os.listdir(ovl):
            d = os.path.join(ovl, oid)
            if oid not in {o["id"] for o in render.OVERLAYS}:
                shutil.rmtree(d)
                continue
            for fn in os.listdir(d):
                if fn.split(".")[0] not in keep_keys:
                    os.remove(os.path.join(d, fn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--hours", type=int, default=C.HOURS_BACK)
    ap.add_argument("--max-rap", type=int, default=60, help="cap on RAP hours processed per run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    site = args.site
    data = os.path.join(site, "data")
    os.makedirs(data, exist_ok=True)
    s = session()
    now = C.utc_now_hour()
    times = [now - dt.timedelta(hours=h) for h in range(args.hours)]
    keys = {C.key(t) for t in times}
    t0 = time.time()

    # ---- MIMIC
    try:
        available = mimic.list_recent(s, args.hours)
    except Exception as exc:  # noqa: BLE001
        log.error("MIMIC listing failed: %s", exc)
        available = {}
    refresh_after = now - dt.timedelta(hours=C.MIMIC_REFRESH_HOURS)
    for t, url in sorted(available.items()):
        path = os.path.join(data, "tpw", f"{C.key(t)}.bin")
        if os.path.exists(path) and t < refresh_after:
            continue
        try:
            write_bin(path, mimic.fetch_subset(s, url))
            log.info("MIMIC %s ok", C.key(t))
        except Exception as exc:  # noqa: BLE001
            log.warning("MIMIC %s failed: %s", C.key(t), exc)

    # ---- RAP analyses + overlays
    n_rap = 0
    for t in sorted(times, reverse=True):
        k = C.key(t)
        pw_path = os.path.join(data, "rappw", f"{k}.bin")
        have = all(os.path.exists(os.path.join(data, "ovl", o["id"], f"{k}.png")) for o in render.OVERLAYS)
        if have and os.path.exists(pw_path):
            continue
        if n_rap >= args.max_rap:
            break
        grib = rap.fetch(s, t)
        if grib is None:
            log.info("RAP %s not available yet", k)
            continue
        n_rap += 1
        try:
            f = rap.decode(grib)
        except Exception as exc:  # noqa: BLE001
            log.warning("RAP %s decode failed: %s", k, exc)
            continue
        if f.get("pw") is not None:
            write_bin(pw_path, f["pw"])
        done = render.render_all(f, k, site)
        log.info("RAP %s: %d overlays", k, len(done))

    # ---- static layers
    coast = os.path.join(data, "coast.png")
    if not os.path.exists(coast):
        try:
            render.draw_coast(coast)
        except Exception as exc:  # noqa: BLE001
            log.error("coastline layer failed: %s", exc)

    # ---- prune + manifest
    prune(site, keys)
    frames = []
    for t in sorted(times):
        k = C.key(t)
        tpw = os.path.exists(os.path.join(data, "tpw", f"{k}.bin"))
        rpw = os.path.exists(os.path.join(data, "rappw", f"{k}.bin"))
        ovl = [o["id"] for o in render.OVERLAYS
               if os.path.exists(os.path.join(data, "ovl", o["id"], f"{k}.png"))]
        if tpw or rpw or ovl:
            frames.append({"t": t.strftime("%Y-%m-%dT%H:%MZ"), "key": k,
                           "tpw": tpw, "rappw": rpw, "ovl": ovl})

    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extent": [C.W, C.E, C.S, C.N],
        "nx": C.NX, "ny": C.NY, "dx": C.DX,
        "scale": C.SCALE, "missing": C.MISSING,
        "img": [C.IMG_W, C.IMG_H],
        "markers": C.MARKERS,
        "overlays": render.OVERLAYS,
        "frames": frames,
    }
    with open(os.path.join(data, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, separators=(",", ":"), ensure_ascii=False)

    # ---- web page
    web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    for fn in os.listdir(web):
        shutil.copy2(os.path.join(web, fn), os.path.join(site, fn))

    log.info("done: %d frames (%d with MIMIC) in %.0f s",
             len(frames), sum(fr["tpw"] for fr in frames), time.time() - t0)


if __name__ == "__main__":
    main()
