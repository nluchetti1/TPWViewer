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
import sys
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
    ids = {o["id"] for o in render.OVERLAYS}
    for dom in C.DOMAINS:
        root = os.path.join(site, "data", dom["dir"])
        if not os.path.isdir(root):
            continue
        for oid in os.listdir(root):
            d = os.path.join(root, oid)
            if oid not in ids or not os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
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

    # ---- RAP analyses + overlays (only the domain/overlay pairs still missing)
    n_rap = 0
    for t in sorted(times, reverse=True):
        k = C.key(t)
        pw_path = os.path.join(data, "rappw", f"{k}.bin")
        todo = [(d, o["id"]) for d in C.DOMAINS for o in render.OVERLAYS
                if not os.path.exists(render.overlay_path(site, d, o["id"], k))]
        if not todo and os.path.exists(pw_path):
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
        done = render.render_all(f, k, site, todo)
        log.info("RAP %s: %d/%d overlays", k, len(done), len(todo))

    # ---- static layers
    for dom in C.DOMAINS:
        coast = os.path.join(data, dom["coast"])
        if not os.path.exists(coast):
            try:
                render.draw_coast(dom, coast)
            except Exception as exc:  # noqa: BLE001
                log.error("coastline layer %s failed: %s", dom["id"], exc)

    # ---- prune + manifest
    prune(site, keys)
    frames = []
    for t in sorted(times):
        k = C.key(t)
        tpw = os.path.exists(os.path.join(data, "tpw", f"{k}.bin"))
        rpw = os.path.exists(os.path.join(data, "rappw", f"{k}.bin"))
        ovl = {d["id"]: [o["id"] for o in render.OVERLAYS
                         if os.path.exists(render.overlay_path(site, d, o["id"], k))]
               for d in C.DOMAINS}
        if tpw or rpw or any(ovl.values()):
            frames.append({"t": t.strftime("%Y-%m-%dT%H:%MZ"), "key": k,
                           "tpw": tpw, "rappw": rpw, "ovl": ovl})

    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extent": [C.W, C.E, C.S, C.N],
        "nx": C.NX, "ny": C.NY, "dx": C.DX,
        "scale": C.SCALE, "missing": C.MISSING,
        "img": [C.IMG_W, C.IMG_H],
        "markers": C.MARKERS,
        "domains": [{k2: d[k2] for k2 in ("id", "name", "extent", "dir", "coast", "grid", "ref_deg")}
                    for d in C.DOMAINS],
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
    # Everything is written by now. Skip interpreter teardown: a native library (eccodes or
    # GEOS/cartopy) segfaults during shutdown on the Actions runner, which turned a good build
    # into exit 139 and skipped the publish step. Exceptions in main() still exit non-zero.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
