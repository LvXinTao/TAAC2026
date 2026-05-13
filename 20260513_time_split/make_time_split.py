#!/usr/bin/env python3
"""Generate a time-ordered train/val split map from parquet data.

Usage:
    python make_time_split.py --data_dir /path/to/parquet --output time_split_map.json [--valid_ratio 0.1]

Each parquet file is read once to extract the timestamp column (~2M rows, completes in ~10-30s).
The output JSON maps each file to row ranges for train and val splits.
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate time-ordered train/val split map")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing *.parquet files')
    parser.add_argument('--output', type=str, default='time_split_map.json',
                        help='Output JSON path')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of rows for validation')
    args = parser.parse_args()

    import glob
    pq_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {args.data_dir}")

    # Step 1: Collect all (timestamp, file_path, row_idx) tuples.
    print("Scanning timestamps...")
    all_rows = []  # list of (timestamp, file_path, row_idx_within_file)
    for fpath in pq_files:
        table = pq.read_table(fpath, columns=['timestamp'])
        ts = table.column('timestamp').to_numpy(zero_copy_only=False).astype(np.int64)
        for idx, t in enumerate(ts):
            all_rows.append((int(t), fpath, idx))
        print(f"  {os.path.basename(fpath)}: {len(ts)} rows, "
              f"ts range [{datetime.fromtimestamp(ts.min())}, "
              f"{datetime.fromtimestamp(ts.max())}]")

    # Step 2: Sort by timestamp (stable sort preserves file order for ties).
    print(f"Sorting {len(all_rows):,} rows...")
    all_rows.sort(key=lambda x: x[0])

    # Step 3: Compute split point.
    total = len(all_rows)
    split_idx = int(total * (1 - args.valid_ratio))
    train_cutoff_ts = all_rows[split_idx - 1][0]
    print(f"Split point: row {split_idx:,} (ts={train_cutoff_ts}, "
          f"{datetime.fromtimestamp(train_cutoff_ts)})")

    # Step 4: Group by file into contiguous ranges.
    def group_by_file(rows_in_split):
        """Group sorted (ts, file, row_idx) into contiguous (file, start, end) ranges."""
        ranges = []
        if not rows_in_split:
            return ranges
        cur_file = rows_in_split[0][1]
        cur_start = rows_in_split[0][2]
        cur_end = cur_start
        for ts, fpath, row_idx in rows_in_split[1:]:
            if fpath == cur_file and row_idx == cur_end:
                cur_end = row_idx + 1
            else:
                ranges.append((cur_file, cur_start, cur_end))
                cur_file, cur_start, cur_end = fpath, row_idx, row_idx + 1
        ranges.append((cur_file, cur_start, cur_end))
        return ranges

    train_ranges = group_by_file(all_rows[:split_idx])
    valid_ranges = group_by_file(all_rows[split_idx:])

    # Step 5: Build per-file JSON structure.
    split = {}
    for fpath, start, end in train_ranges:
        if fpath not in split:
            split[fpath] = {"train_rows": [], "valid_rows": []}
        split[fpath]["train_rows"].append([start, end])

    for fpath, start, end in valid_ranges:
        if fpath not in split:
            split[fpath] = {"train_rows": [], "valid_rows": []}
        split[fpath]["valid_rows"].append([start, end])

    # Step 6: Write output.
    train_total = split_idx
    valid_total = total - split_idx
    output = {
        "version": 1,
        "split_method": "time_ordered",
        "total_rows": total,
        "train_rows": train_total,
        "valid_rows": valid_total,
        "valid_ratio": args.valid_ratio,
        "timestamp_cutoff": int(train_cutoff_ts),
        "files": split,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWritten to {args.output}")
    print(f"  Train: {train_total:,} rows in {len(train_ranges)} ranges "
          f"across {len(set(r[0] for r in train_ranges))} files")
    print(f"  Valid: {valid_total:,} rows in {len(valid_ranges)} ranges "
          f"across {len(set(r[0] for r in valid_ranges))} files")


if __name__ == "__main__":
    main()
