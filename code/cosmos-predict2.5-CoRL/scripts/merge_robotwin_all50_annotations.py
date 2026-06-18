"""
Merge all 50 RobotWin clean50 task annotations into a single flat directory.

Output:
  datasets/robotwin_clean50_all50/
    annotation/train/<global_id>.json
    annotation/val/<global_id>.json

Each JSON's video_path is prefixed with the task name so it resolves correctly
when Dataset_3D uses video_path=datasets/robotwin_clean50/ as base.

Usage:
    python scripts/merge_robotwin_all50_annotations.py
"""

import json
import os
from pathlib import Path

BASE = Path("datasets/robotwin_clean50")
OUT  = Path("datasets/robotwin_clean50_all50")

tasks = sorted([d.name for d in BASE.iterdir() if d.is_dir()])
print(f"Found {len(tasks)} tasks: {tasks[:3]}...")

for split in ("train", "val"):
    out_dir = OUT / "annotation" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    global_id = 0
    for task in tasks:
        ann_dir = BASE / task / "annotation" / split
        if not ann_dir.exists():
            print(f"  [WARN] missing {ann_dir}, skipping")
            continue

        json_files = sorted(ann_dir.glob("*.json"), key=lambda p: int(p.stem))
        for jf in json_files:
            with open(jf) as f:
                ann = json.load(f)

            # rewrite video_path to include task prefix
            ann["videos"] = [
                {"video_path": f"{task}/{v['video_path']}"}
                for v in ann["videos"]
            ]
            ann["episode_id"] = global_id
            ann["task_name"] = task

            with open(out_dir / f"{global_id}.json", "w") as f:
                json.dump(ann, f)

            global_id += 1

    print(f"[{split}] wrote {global_id} annotations to {out_dir}")

print("Done.")
