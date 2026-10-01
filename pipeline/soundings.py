"""Observed precipitable water from rawinsondes via the IEM RAOB archive (CSV)."""
import csv
import datetime as dt
import io
import logging

import numpy as np

from . import config as C

log = logging.getLogger("raob")
G = 9.80665
# IEM identifiers vary by site; try these in order until one returns rows.
ALT_IDS = {"KXMR": ["KXMR", "XMR", "74794"]}


def _pw_mm(levels):
    """Integrate specific humidity over pressure from the lowest level up to 300 hPa.

    Needs at least 5 levels with dewpoints spanning 850 to 500 hPa (the moist layer that
    dominates PW); levels above the highest dewpoint contribute almost nothing.
    """
    lv = sorted(((p, td) for p, td in levels if p is not None and td is not None and p >= 300),
                key=lambda x: -x[0])
    if len(lv) < 5 or lv[0][0] < 850 or lv[-1][0] > 500:
        return None
    p = np.array([x[0] for x in lv])
    td = np.array([x[1] for x in lv])
    e = 6.112 * np.exp(17.67 * td / (td + 243.5))
    q = 0.622 * e / (p - 0.378 * e)
    integ = getattr(np, "trapezoid", None) or np.trapz
    return float(integ(q[::-1], p[::-1] * 100.0) / G)


def _num(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def _query(session, station, start, end):
    r = session.get(C.RAOB_URL, params={
        "station": station,
        "sts": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ets": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, timeout=60)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def fetch(session, start, end):
    """Return manifest-ready station dicts with a list of {t, pw} observations."""
    out = []
    for st in C.SOUNDINGS:
        obs = []
        try:
            rows, used = [], st["id"]
            for sid in ALT_IDS.get(st["id"], [st["id"]]):
                rows = _query(session, sid, start, end)
                log.info("RAOB %s: %d rows from IEM", sid, len(rows))
                if rows:
                    used = sid
                    break
            by_time = {}
            for row in rows:
                t = row.get("validUTC") or row.get("valid")
                if not t:
                    continue
                by_time.setdefault(t, []).append((_num(row.get("pressure_mb")), _num(row.get("dwpc"))))
            rejected = 0
            for t, levels in sorted(by_time.items()):
                pw = _pw_mm(levels)
                if pw is None:
                    rejected += 1
                    continue
                ts = dt.datetime.fromisoformat(t.replace("Z", "").replace(" ", "T")).replace(tzinfo=dt.timezone.utc)
                obs.append({"t": ts.strftime("%Y-%m-%dT%H:%MZ"), "pw": round(pw, 1)})
            log.info("RAOB %s (as %s): %d soundings, %d with PW, %d rejected by QC",
                     st["id"], used, len(by_time), len(obs), rejected)
        except Exception as exc:  # noqa: BLE001
            log.warning("RAOB %s failed: %s", st["id"], exc)
        out.append({**st, "obs": obs})
    return out
