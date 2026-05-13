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

    # Step 4: Find timestamp cutoff, then split each file at that cutoff.
    # After global sort, rows from the same file are scattered, so we cannot
    # merge ranges in sorted order. Instead, find the cutoff timestamp, then
    # for each file determine which rows go to train vs valid based on their
    # individual timestamp — this yields at most 2 ranges per file.

    # Collect timestamps per file for cutoff-based splitting.
    print("Building per-file split ranges...")
    file_data = {}  # fpath -> (np.array timestamps, total_rows)
    for fpath in pq_files:
        table = pq.read_table(fpath, columns=['timestamp'])
        ts = table.column('timestamp').to_numpy(zero_copy_only=False).astype(np.int64)
        file_data[fpath] = ts

    # Use the exact cutoff timestamp. Rows with ts < cutoff → train.
    # Rows with ts == cutoff are split to achieve exact 90/10.
    cutoff_ts = train_cutoff_ts
    train_total = 0
    valid_total = 0
    split = {}

    # Count how many rows are exactly at the cutoff — we need to allocate
    # some to train and the rest to valid to hit the exact split_idx.
    rows_before = 0  # rows with ts < cutoff
    rows_at = 0      # rows with ts == cutoff
    for fpath, ts in file_data.items():
        below = int((ts < cutoff_ts).sum())
        at = int((ts == cutoff_ts).sum())
        rows_before += below
        rows_at += at

    # We need exactly split_idx rows in train. We already have rows_before
    # (ts < cutoff), so we need split_idx - rows_before more from the
    # rows_at group.
    from_at_to_train = split_idx - rows_before
    assert 0 <= from_at_to_train <= rows_at, (
        f"Invalid split: need {from_at_to_train} from {rows_at} rows at cutoff")

    for fpath, ts in file_data.items():
        below_mask = ts < cutoff_ts
        at_mask = ts == cutoff_ts

        # Train rows: all below cutoff + a fraction of those at cutoff.
        # Merge into contiguous ranges.
        train_mask = below_mask.copy()
        if from_at_to_train > 0:
            at_indices = np.where(at_mask)[0]
            # Take the first N indices at cutoff for train.
            need = min(from_at_to_train, len(at_indices))
            for idx in at_indices[:need]:
                train_mask[idx] = True
                from_at_to_train -= 1

        valid_mask = ~train_mask

        def to_ranges(mask):
            """Convert boolean mask to list of contiguous (start, end) ranges."""
            ranges = []
            start = None
            for i in range(len(mask)):
                if mask[i]:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        ranges.append([start, i])
                        start = None
            if start is not None:
                ranges.append([start, len(mask)])
            return ranges

        tr = to_ranges(train_mask)
        vr = to_ranges(valid_mask)
        if tr or vr:
            split[fpath] = {"train_rows": tr, "valid_rows": vr}

    # Recompute totals from masks (cleaner than accumulating above).
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
    print(f"  Train: {train_total:,} rows")
    print(f"  Valid: {valid_total:,} rows")
    print(f"  Files with split ranges: {len(split)}")


if __name__ == "__main__":
    main()
