"""Build or update the static site in --site.

Incremental: existing frames in the site directory are kept; only missing hours (plus the
newest MIMIC_REFRESH_HOURS of MIMIC, which SSEC re-issues) are fetched. Anything older than
HOURS_BACK is pruned. Each run also refreshes the HRRR forecast tail, the observed-sounding
PW list, GLM flash density and the PW climatology.

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

from . import clim, glm, mimic, rap, render, soundings
from . import config as C

log = logging.getLogger("build")


def session():
    s = requests.Session()
    s.headers["User-Agent"] = C.USER_AGENT
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("https://", adapter)
    return s


def encode(arr, scale=C.SCALE):
    code = np.where(np.isfinite(arr), np.clip(np.round(arr * scale), 0, 254), C.MISSING)
    return code.astype(np.uint8).tobytes()


def write_bin(path, arr, scale=C.SCALE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(encode(arr, scale))
    os.replace(tmp, path)


PER_HOUR_DIRS = ("tpw", "age", "rappw", "glm")


def prune(site, keep_keys):
    for sub in PER_HOUR_DIRS:
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


# ---------------------------------------------------------------- HRRR forecast tail

def build_forecast(s, site, last_obs):
    data = os.path.join(site, "data")
    meta_path = os.path.join(data, "fc", "meta.json")
    old = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            old = json.load(fh)

    valid = [last_obs + dt.timedelta(hours=h) for h in range(1, C.FC_HOURS + 1)]
    cycle, fhs = None, None
    for back in range(1, 6):
        c = C.utc_now_hour() - dt.timedelta(hours=back)
        f = [int((v - c).total_seconds() // 3600) for v in valid]
        limit = 48 if c.hour % 6 == 0 else 18
        if max(f) > limit:
            continue
        if rap.hrrr_available(s, c, max(f)):
            cycle, fhs = c, f
            break
    if cycle is None:
        log.warning("HRRR: no cycle covers %s..%s yet; keeping previous tail", C.key(valid[0]), C.key(valid[-1]))
        return old

    keys = [C.key(v) for v in valid]
    if old and old.get("cycle") == C.key(cycle) and old.get("keys") == keys:
        log.info("HRRR tail unchanged (%sZ cycle)", C.key(cycle))
        return old

    shutil.rmtree(os.path.join(data, "fcpw"), ignore_errors=True)
    for dom in C.DOMAINS:
        shutil.rmtree(os.path.join(data, dom["dir"] + "_fc"), ignore_errors=True)

    todo = [(d, o["id"]) for d in C.DOMAINS for o in render.OVERLAYS]
    frames = []
    for v, fh in zip(valid, fhs):
        k = C.key(v)
        grib = rap.fetch_hrrr(s, cycle, fh)
        if grib is None:
            continue
        try:
            fld = rap.decode(grib, stride=C.HRRR_STRIDE)
        except Exception as exc:  # noqa: BLE001
            log.warning("HRRR f%02d decode failed: %s", fh, exc)
            continue
        pw = fld.get("pw") is not None
        if pw:
            write_bin(os.path.join(data, "fcpw", f"{k}.bin"), fld["pw"])
        done = render.render_all(fld, k, site, todo, fc=True)
        ovl = {d["id"]: [o for (di, o) in done if di == d["id"]] for d in C.DOMAINS}
        frames.append({"t": v.strftime("%Y-%m-%dT%H:%MZ"), "key": k, "fh": fh, "pw": pw, "ovl": ovl})
        log.info("HRRR %sZ f%02d -> %s: %d overlays", C.key(cycle), fh, k, len(done))

    meta = {"cycle": C.key(cycle), "keys": keys, "frames": frames}
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return meta


# ---------------------------------------------------------------- main

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

    def step(name, fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", name, exc)
            return None

    # ---- MIMIC (TPW + time-from-nearest-pass)
    available = step("MIMIC listing", mimic.list_recent, s, args.hours) or {}
    refresh_after = now - dt.timedelta(hours=C.MIMIC_REFRESH_HOURS)
    fetched = {}
    for t, url in sorted(available.items()):
        k = C.key(t)
        tpw_path = os.path.join(data, "tpw", f"{k}.bin")
        age_path = os.path.join(data, "age", f"{k}.bin")
        if os.path.exists(tpw_path) and os.path.exists(age_path) and t < refresh_after:
            continue
        try:
            tpw, age = mimic.fetch_subset(s, url)
            write_bin(tpw_path, tpw)
            if age is not None:
                write_bin(age_path, age, C.AGE_SCALE)
            fetched[t] = tpw
            log.info("MIMIC %s ok", k)
        except Exception as exc:  # noqa: BLE001
            log.warning("MIMIC %s failed: %s", k, exc)

    # ---- climatology for the anomaly view
    def do_clim():
        mean, meta = clim.load(site)
        if mean is None:
            mean, meta = clim.cold_start(s, site)
        if mean is not None:
            n = clim.update(site, mean, meta, fetched)
            log.info("climatology: %d new hours folded in (last %s)", n, meta["last"])
        return meta
    clim_meta = step("climatology", do_clim)

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

    # ---- GLM flash density (hour ending at each frame time)
    real_now = dt.datetime.now(dt.timezone.utc)
    n_glm = 0
    for t in sorted(times, reverse=True):
        path = os.path.join(data, "glm", f"{C.key(t)}.bin")
        if os.path.exists(path) or real_now < t + dt.timedelta(minutes=3):
            continue
        if n_glm >= C.GLM_MAX_HOURS_PER_RUN:
            break
        n_glm += 1
        counts, nfiles = step("GLM", glm.density_for, s, t) or (None, 0)
        if counts is None:
            log.info("GLM %s: no files yet", C.key(t))
            continue
        if nfiles >= 150 or real_now - t > dt.timedelta(hours=2):
            write_bin(path, counts, 1.0)
            log.info("GLM %s: %d flashes from %d files", C.key(t), int(counts.sum()), nfiles)
        else:
            log.info("GLM %s: only %d files so far, retrying next run", C.key(t), nfiles)

    # ---- HRRR forecast tail after the newest MIMIC frame
    obs_times = [t for t in times if os.path.exists(os.path.join(data, "tpw", f"{C.key(t)}.bin"))]
    last_obs = max(obs_times) if obs_times else now - dt.timedelta(hours=1)
    fc_meta = step("HRRR tail", build_forecast, s, site, last_obs)

    # ---- observed sounding PW
    raob = step("soundings", soundings.fetch, s, times[-1] - dt.timedelta(hours=12),
                now + dt.timedelta(hours=1)) or [{**st, "obs": []} for st in C.SOUNDINGS]

    # ---- static layers
    for dom in C.DOMAINS:
        coast = os.path.join(data, dom["coast"])
        if not os.path.exists(coast):
            step(f"coastline {dom['id']}", render.draw_coast, dom, coast)

    # ---- prune + manifest
    prune(site, keys)
    frames = []
    for t in sorted(times):
        k = C.key(t)
        has = {sub: os.path.exists(os.path.join(data, sub, f"{k}.bin")) for sub in PER_HOUR_DIRS}
        ovl = {d["id"]: [o["id"] for o in render.OVERLAYS
                         if os.path.exists(render.overlay_path(site, d, o["id"], k))]
               for d in C.DOMAINS}
        if has["tpw"] or has["rappw"] or any(ovl.values()):
            frames.append({"t": t.strftime("%Y-%m-%dT%H:%MZ"), "key": k, "tpw": has["tpw"],
                           "age": has["age"], "rappw": has["rappw"], "glm": has["glm"], "ovl": ovl})

    manifest = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extent": [C.W, C.E, C.S, C.N],
        "nx": C.NX, "ny": C.NY, "dx": C.DX,
        "scale": C.SCALE, "missing": C.MISSING, "age_scale": C.AGE_SCALE,
        "img": [C.IMG_W, C.IMG_H],
        "markers": C.MARKERS,
        "domains": [{k2: d[k2] for k2 in ("id", "name", "extent", "dir", "coast", "grid", "ref_deg")}
                    for d in C.DOMAINS],
        "overlays": render.OVERLAYS,
        "frames": frames,
        "fc": fc_meta,
        "clim": clim_meta,
        "soundings": raob,
    }
    with open(os.path.join(data, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, separators=(",", ":"), ensure_ascii=False)

    # ---- web page
    web = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
    for fn in os.listdir(web):
        shutil.copy2(os.path.join(web, fn), os.path.join(site, fn))

    log.info("done: %d frames (%d with MIMIC), %d forecast frames, in %.0f s",
             len(frames), sum(fr["tpw"] for fr in frames),
             len(fc_meta["frames"]) if fc_meta else 0, time.time() - t0)


if __name__ == "__main__":
    main()
    # Everything is written by now. Skip interpreter teardown: a native library (eccodes or
    # GEOS/cartopy) segfaults during shutdown on the Actions runner, which turned a good build
    # into exit 139 and skipped the publish step. Exceptions in main() still exit non-zero.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
