"""Observed precipitable water from SPC's observed-sounding archive.

SPC keeps ~7 days of soundings at  {SPC}/YYMMDDHH_OBS/STATION.txt  in SHARPpy text format:
a %RAW% block of "pres, hght, temp, dwpt, wdir, wspd" rows with -9999 for missing. Every
launch hour gets its own _OBS folder, so off-hour launches (XMR's 10Z/15Z, specials) are
included. Folders are discovered from SPC's index page, with the usual launch hours tried
as a fallback.
"""
import datetime as dt
import logging
import re

import numpy as np

from . import config as C

log = logging.getLogger("raob")
G = 9.80665
SPC = getattr(C, "SPC_SOUNDINGS", "https://www.spc.noaa.gov/exper/soundings")
FALLBACK_HOURS = (0, 10, 12, 15)
_DIR = re.compile(r"(\d{8})_OBS")


def parse_spc(text):
    """Return (pressure_hPa, dewpoint_C) arrays from an SPC SHARPpy text sounding."""
    if "%RAW%" not in text:
        return None
    body = text.split("%RAW%", 1)[1].split("%END%", 1)[0]
    p, td = [], []
    for line in body.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            pres, dwpt = float(parts[0]), float(parts[3])
        except ValueError:
            continue
        if pres > 0 and dwpt > -9000:
            p.append(pres)
            td.append(dwpt)
    return np.array(p), np.array(td)


def pw_mm(p, td, top=300.0):
    """Surface-to-300 hPa precipitable water (mm). Matches SPC's PW to ~0.01 in."""
    m = p >= top
    p, td = p[m], td[m]
    order = np.argsort(-p)
    p, td = p[order], td[order]
    if p.size < 5 or p[0] < 850 or p[-1] > 500:
        return None
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    q = 0.622 * e / (p - 0.378 * e)
    integ = getattr(np, "trapezoid", None) or np.trapz
    return float(integ(q[::-1], p[::-1] * 100.0) / G)


def _candidates(session, start, end):
    dirs = set()
    try:
        r = session.get(SPC + "/", timeout=60)
        if r.ok:
            dirs |= set(_DIR.findall(r.text))
    except Exception as exc:  # noqa: BLE001
        log.warning("SPC index unavailable: %s", exc)
    found_index = len(dirs)
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        if t.hour in FALLBACK_HOURS:
            dirs.add(t.strftime("%y%m%d%H"))
        t += dt.timedelta(hours=1)
    keep = []
    for d in sorted(dirs):
        try:
            ts = dt.datetime.strptime(d, "%y%m%d%H").replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if start <= ts <= end:
            keep.append((ts, d))
    return keep, found_index


def fetch(session, start, end):
    """Return manifest-ready station dicts with a list of {t, pw} observations."""
    cands, n_index = _candidates(session, start, end)
    out = []
    for st in C.SOUNDINGS:
        obs, missing = [], 0
        for ts, d in cands:
            try:
                r = session.get(f"{SPC}/{d}_OBS/{st['id']}.txt", timeout=30)
                if r.status_code != 200:
                    missing += 1
                    continue
                parsed = parse_spc(r.text)
                pw = pw_mm(*parsed) if parsed else None
                if pw is not None:
                    obs.append({"t": ts.strftime("%Y-%m-%dT%H:%MZ"), "pw": round(pw, 1)})
            except Exception as exc:  # noqa: BLE001
                log.debug("SPC %s %s failed: %s", d, st["id"], exc)
        log.info("SPC soundings %s: %d with PW (%d hours checked, %d from index, %d without a launch)",
                 st["id"], len(obs), len(cands), n_index, missing)
        out.append({**st, "obs": obs})
    return out
