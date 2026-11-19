"""Write the site's weekly station list: web/data/midas_us.<FRAME>.json.

Run by .github/workflows/pages.yml at every deploy and on a weekly schedule
(NGL updates MIDAS weekly). Usage: python scripts/build_web_data.py OUTDIR [FRAME ...]
"""
import datetime
import json
import sys
from pathlib import Path

from gps_strain import data, web


def main(outdir, frames=("NA",)):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.date.today().isoformat()
    for frame in frames:
        d = web.midas_us_json(data.load_midas(frame, max_age_days=0), frame=frame, generated=stamp)
        path = out / f"midas_us.{frame}.json"
        path.write_text(json.dumps(d, separators=(",", ":")))
        print(f"{path}: {d['count']} US stations, {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ("NA",))
