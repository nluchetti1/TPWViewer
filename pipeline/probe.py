"""Live probe of every data source the pipeline touches. Run before the first build:

    python -m pipeline.probe
"""
import datetime as dt

import numpy as np

from . import config as C
from .build import session


def ok(msg):
    print("  OK   " + msg)


def bad(msg):
    print("  FAIL " + msg)


def probe_mimic(s):
    print("== MIMIC-TPW2 ==")
    from . import mimic
    files = mimic.list_recent(s, 12)
    if not files:
        return bad("no composites in the last 12 h (check MIMIC_BASE)")
    newest = max(files)
    lag = (dt.datetime.now(dt.timezone.utc) - newest).total_seconds() / 3600
    ok(f"{len(files)} composites; newest {newest:%Y-%m-%d %HZ} ({lag:.1f} h old)")
    tpw, age = mimic.fetch_subset(s, files[newest])
    ok(f"TPW {np.nanmin(tpw):.0f}-{np.nanmax(tpw):.0f} mm over the domain")
    if age is None:
        bad("no timeAwayGrid fields; the confidence hatch will be empty")
    else:
        ok(f"pass age {np.nanmin(age):.1f}-{np.nanmax(age):.1f} h, median {np.nanmedian(age):.1f} h")


def probe_rap(s):
    print("== RAP (observed overlays) ==")
    from . import rap
    t = C.utc_now_hour() - dt.timedelta(hours=2)
    for src in ("aws", "nomads"):
        C.RAP_SOURCE = src
        b = rap.fetch(s, t)
        if b is None:
            bad(f"{src}: nothing for {C.key(t)}")
            continue
        f = rap.decode(b)
        ok(f"{src}: {len(b)/1e6:.1f} MB, levels {sorted(f['u'])}, PW {'ok' if f['pw'] is not None else 'MISSING'}, "
           f"CAPE {'ok' if f['cape'] is not None else 'MISSING'}")
        iy, ix = int((C.N - 28.49) / C.DX), int((-80.58 - C.W) / C.DX)
        u, v = f["u"][850][iy, ix], f["v"][850][iy, ix]
        print(f"       850 hPa wind near the Cape: {np.degrees(np.arctan2(-u, -v)) % 360:.0f}° "
              f"{np.hypot(u, v) * 1.944:.0f} kt (compare with the XMR sounding)")
    C.RAP_SOURCE = "aws"


def probe_hrrr(s):
    print("== HRRR (forecast tail) ==")
    from . import rap
    for back in range(1, 4):
        c = C.utc_now_hour() - dt.timedelta(hours=back)
        if rap.hrrr_available(s, c, 12):
            b = rap.fetch_hrrr(s, c, 3)
            if b is None:
                return bad(f"{C.key(c)}Z f03 idx listed but records missing")
            f = rap.decode(b, stride=C.HRRR_STRIDE)
            cov = np.isfinite(f["pw"]).mean() * 100 if f["pw"] is not None else 0
            return ok(f"{C.key(c)}Z f03: {len(b)/1e6:.1f} MB, levels {sorted(f['u'])}, PW covers {cov:.0f}% "
                      f"of the domain (the far-south strip is outside HRRR), CAPE "
                      f"{'ok' if f['cape'] is not None else 'MISSING'}")
    bad("no HRRR cycle with f12 in the last 3 h")


def probe_glm(s):
    print("== GLM (GOES-East lightning) ==")
    from . import glm
    t = C.utc_now_hour()
    keys = glm._list_hour(s, t - dt.timedelta(hours=1))
    if not keys:
        return bad(f"no {C.GLM_SAT} files for the last hour (check GLM_BUCKET / GLM_SAT)")
    ok(f"{len(keys)} files in the hour ending {t:%HZ}")
    blob = glm._download(s, keys[len(keys) // 2])
    lat, lon = glm._flashes(blob)
    ok(f"one file parsed: {lat.size} flashes")


def probe_raob(s):
    print("== Soundings (SPC) ==")
    from . import soundings
    now = C.utc_now_hour()
    for st in soundings.fetch(s, now - dt.timedelta(days=3), now + dt.timedelta(hours=1)):
        if st["obs"]:
            last = st["obs"][-1]
            ok(f"{st['id']}: {len(st['obs'])} soundings in 3 days; latest {last['t']} PW {last['pw']} mm")
        else:
            bad(f"{st['id']}: no soundings with PW in 3 days (check the station id on SPC)")


def probe_long(s):
    print("== GFS / ECMWF long-range PW ==")
    from . import longrange
    iy, ix = int((C.N - 28.49) / C.DX), int((-80.58 - C.W) / C.DX)
    for src in longrange.SOURCES:
        steps = C.GFS_STEPS if src == "gfs" else C.ECMWF_STEPS
        for c in longrange._cycles(src, C.utc_now_hour()):
            if longrange.available(s, src, c, steps[-1]):
                b = longrange.fetch_bytes(s, src, c, 24, full=True)
                f = longrange.decode(b)
                pw = f["pw"]
                u, v = f["u"].get(850), f["v"].get(850)
                wind = (f"850 hPa wind at the Cape {np.degrees(np.arctan2(-u[iy, ix], -v[iy, ix])) % 360:.0f}° "
                        f"{np.hypot(u[iy, ix], v[iy, ix]) * 1.944:.0f} kt" if u is not None else "850 hPa wind MISSING")
                ok(f"{src.upper()} {C.key(c)}Z complete to {steps[-1]} h; f024 {len(b)/1e6:.1f} MB, levels "
                   f"{sorted(f['u'])}, CAPE {'ok' if f['cape'] is not None else 'MISSING'}; PW at the Cape "
                   f"{pw[iy, ix]:.1f} mm; {wind}")
                break
        else:
            bad(f"{src.upper()}: no complete cycle in the last 36 h")


def probe_goes(s):
    print("== GOES-19 ABI imagery ==")
    from . import goes
    t = C.utc_now_hour() - dt.timedelta(hours=1)
    keys = goes.list_hour(s, t)
    if not keys:
        return bad(f"no {C.ABI_PRODUCT} files for {t:%HZ} (check GLM_BUCKET / ABI_PRODUCT)")
    k = goes.pick(keys, C.ABI_BANDS["ir"])
    ok(f"{len(keys)} files in the {t:%HZ} hour; IR file {k.rsplit('/', 1)[-1][:60]}")
    blob = s.get(f"{C.GLM_BUCKET}/{k}", timeout=180).content
    imgs = goes.render(blob, "ir", C.DOMAINS)
    ok(f"IR rendered for {', '.join(imgs)} ({len(blob)/1e6:.1f} MB file)")


if __name__ == "__main__":
    s = session()
    for fn in (probe_mimic, probe_rap, probe_hrrr, probe_glm, probe_raob, probe_long, probe_goes):
        try:
            fn(s)
        except Exception as exc:  # noqa: BLE001
            bad(f"{fn.__name__}: {exc}")
