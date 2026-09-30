"""Live probe of every data source the pipeline touches. Run before the first build:

    python -m pipeline.probe
"""
import datetime as dt
import gzip
import os
import tempfile

import numpy as np
import xarray as xr

from . import config as C
from .build import session


def probe_mimic(s):
    print("== MIMIC-TPW2 ==")
    files = __import__("pipeline.mimic", fromlist=["x"]).list_recent(s, 12)
    if not files:
        print("  no composites found in the last 12 h  <-- check MIMIC_BASE / directory layout")
        return
    newest = max(files)
    lag = (dt.datetime.now(dt.timezone.utc) - newest).total_seconds() / 3600
    print(f"  {len(files)} hourly composites; newest {newest:%Y-%m-%d %HZ} ({lag:.1f} h old)")
    url = files[newest]
    raw = s.get(url, timeout=180).content
    raw = gzip.decompress(raw) if url.endswith(".gz") else raw
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as t:
        t.write(raw)
    with xr.open_dataset(t.name, decode_times=False, decode_timedelta=False) as ds:
        print("  variables:", list(ds.data_vars))
        lat, lon = ds["latArr"].values, ds["lonArr"].values
        print(f"  grid {ds['tpwGrid'].shape}, dlat {lat[1]-lat[0]:.4f}, dlon {lon[1]-lon[0]:.4f}, "
              f"lon range {lon.min():.1f}..{lon.max():.1f}, units {ds['tpwGrid'].attrs.get('units')}")
        print(f"  file size {len(raw)/1e6:.1f} MB")
    os.unlink(t.name)


def probe_rap(s):
    print("== RAP ==")
    from . import rap
    t = C.utc_now_hour() - dt.timedelta(hours=2)
    for src in ("aws", "nomads"):
        C.RAP_SOURCE = src
        b = rap.fetch(s, t)
        if b is None:
            print(f"  {src}: nothing for {C.key(t)}")
            continue
        print(f"  {src}: {len(b)/1e6:.1f} MB for {C.key(t)}")
        f = rap.decode(b)
        print(f"    levels {sorted(f['u'])}; PW {'ok' if f['pw'] is not None else 'MISSING'}; "
              f"CAPE {'ok' if f['cape'] is not None else 'MISSING'}")
        ksc = (int((C.N - 28.49) / C.DX), int((-80.58 - C.W) / C.DX))
        u, v = f["u"][850][ksc], f["v"][850][ksc]
        print(f"    850 hPa wind near the Cape: {np.degrees(np.arctan2(-u, -v)) % 360:.0f}° "
              f"{np.hypot(u, v) * 1.944:.0f} kt (sanity-check vs. the XMR sounding)")


if __name__ == "__main__":
    s = session()
    probe_mimic(s)
    probe_rap(s)
