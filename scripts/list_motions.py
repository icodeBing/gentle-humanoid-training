"""List prepared dataset motion IDs, durations and available source metadata."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="interx_small")
    parser.add_argument("--step-dt", type=float, default=.02)
    parser.add_argument("--min-seconds", type=float, default=0.)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    root = Path(os.environ.get("MEMPATH", Path(__file__).resolve().parents[1] / "dataset"))
    path = root / args.dataset / "meta_motion.json"
    meta = json.loads(path.read_text())
    info = meta.get("info", {})
    shown = 0
    for mid, (start, end) in enumerate(zip(meta["starts"], meta["ends"])):
        duration = (end-start-1)*args.step_dt
        if duration < args.min_seconds:
            continue
        details = {key: value[mid] for key, value in info.items()} if isinstance(info, dict) else info[mid]
        print(f"motion-id={mid:5d}  duration={duration:7.2f}s  metadata={json.dumps(details, ensure_ascii=False)}")
        shown += 1
        if shown >= args.limit:
            break
    print("IDs identify dataset segments. Metadata may not contain action labels; preview candidates before choosing an upper-body motion.")


if __name__ == "__main__":
    main()
