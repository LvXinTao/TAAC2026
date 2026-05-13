#!/usr/bin/env python3
"""Server-side EDA for PCVR parquet data.

Reads raw multi-column Parquet files and outputs structured text analysis
to the log. Designed to replace train.py in run.sh for data exploration.

Usage:
    python eda.py --data_dir /path/to/parquet --log_dir /path/to/output

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_LOG_PATH   Log/output directory
"""

import os
import sys
import json
import argparse
import glob
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [EDA] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

MAX_SEEN_VALUES = 50000  # Cap for cardinality tracking to avoid OOM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVR Dataset EDA")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Parquet data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Output directory for summary files (env: TRAIN_LOG_PATH)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of Row Groups used for validation')

    args = parser.parse_args()
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)

    if not args.data_dir:
        parser.error("--data_dir or TRAIN_DATA_PATH is required")
    if not os.path.isdir(args.data_dir):
        parser.error(f"data_dir does not exist: {args.data_dir}")

    if args.schema_path is None:
        args.schema_path = os.path.join(args.data_dir, 'schema.json')
    if not os.path.exists(args.schema_path):
        parser.error(f"schema.json not found: {args.schema_path}")

    return args


def section_header(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}"


def main() -> None:
    args = parse_args()
    log.info(f"EDA starting: data_dir={args.data_dir}")
    log.info(f"  schema_path={args.schema_path}")
    log.info(f"  log_dir={args.log_dir}")
    log.info(f"  valid_ratio={args.valid_ratio}")

    # ── Discover files ──────────────────────────────────────────────────
    pq_files = sorted(glob.glob(os.path.join(args.data_dir, '*.parquet')))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {args.data_dir}")
    log.info(f"Found {len(pq_files)} parquet file(s)")

    rg_list = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

    total_rgs = len(rg_list)
    total_rows = sum(r[2] for r in rg_list)
    log.info(f"Total: {total_rows:,} rows in {total_rgs} row groups")

    # Split (same logic as dataset.py:get_pcvr_data)
    n_valid_rgs = max(1, int(total_rgs * args.valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs
    train_rows = sum(r[2] for r in rg_list[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_list[n_train_rgs:])

    # Load schema
    with open(args.schema_path, 'r') as f:
        schema = json.load(f)

    # Build column sets from schema
    seq_cfg = schema.get('seq', {})
    seq_col_set = set()
    for domain, cfg in seq_cfg.items():
        prefix = cfg['prefix']
        for fid, _ in cfg['features']:
            seq_col_set.add(f'{prefix}_{fid}')

    # Read first row group schema
    first_pf = pq.ParquetFile(rg_list[0][0])
    first_rg_table = first_pf.read_row_group(rg_list[0][1])
    schema_names = set(first_rg_table.schema.names)

    # ── Section 1: Data Scale ───────────────────────────────────────────
    log.info(section_header("Section 1: Data Scale"))
    log.info(f"  Parquet files:  {len(pq_files)}")
    log.info(f"  Row groups:     {total_rgs}")
    log.info(f"  Total rows:     {total_rows:,}")

    per_row_bytes = first_rg_table.nbytes / len(first_rg_table)
    est_total_mb = (per_row_bytes * total_rows) / (1024 * 1024)
    log.info(f"  Est. memory:    ~{est_total_mb:.0f} MB ({per_row_bytes:.0f} bytes/row x {total_rows:,} rows)")

    log.info(f"\n  File breakdown:")
    file_stats = {}
    for f, rg_idx, n_rows in rg_list:
        fname = os.path.basename(f)
        if fname not in file_stats:
            file_stats[fname] = {'rgs': 0, 'rows': 0}
        file_stats[fname]['rgs'] += 1
        file_stats[fname]['rows'] += n_rows
    for fname, st in sorted(file_stats.items()):
        log.info(f"    {fname:40s}: {st['rgs']:4d} RGs, {st['rows']:>10,} rows")

    # ── Section 2: Time Range ───────────────────────────────────────────
    log.info(section_header("Section 2: Time Range"))

    g_min = g_max = tr_min = tr_max = va_min = va_max = None

    for idx, (f, rg_idx, n_rows) in enumerate(rg_list):
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(rg_idx, columns=['timestamp'])
        ts_col = table.column('timestamp').to_numpy()
        rg_min = int(ts_col.min())
        rg_max = int(ts_col.max())

        if g_min is None or rg_min < g_min:
            g_min = rg_min
        if g_max is None or rg_max > g_max:
            g_max = rg_max

        if idx < n_train_rgs:
            if tr_min is None or rg_min < tr_min:
                tr_min = rg_min
            if tr_max is None or rg_max > tr_max:
                tr_max = rg_max
        else:
            if va_min is None or rg_min < va_min:
                va_min = rg_min
            if va_max is None or rg_max > va_max:
                va_max = rg_max

    duration = g_max - g_min
    tr_dur = tr_max - tr_min if tr_min else 0
    va_dur = va_max - va_min if va_min else 0

    log.info(f"  Global time range:")
    log.info(f"    {datetime.fromtimestamp(g_min)}  ->  {datetime.fromtimestamp(g_max)}")
    log.info(f"    Duration: {duration}s = {duration/60:.1f}min = {duration/3600:.2f}h")
    log.info(f"  Train split ({n_train_rgs} RGs, {train_rows:,} rows):")
    log.info(f"    {datetime.fromtimestamp(tr_min)}  ->  {datetime.fromtimestamp(tr_max)}")
    log.info(f"    Duration: {tr_dur}s = {tr_dur/60:.1f}min")
    log.info(f"  Valid split ({n_valid_rgs} RGs, {valid_rows:,} rows):")
    log.info(f"    {datetime.fromtimestamp(va_min)}  ->  {datetime.fromtimestamp(va_max)}")
    log.info(f"    Duration: {va_dur}s = {va_dur/60:.1f}min")

    overlap_start = max(tr_min, va_min)
    overlap_end = min(tr_max, va_max)
    if overlap_start <= overlap_end:
        log.info(f"  WARNING: Time overlap detected — train/valid overlap by {overlap_end - overlap_start}s")
    else:
        log.info(f"  OK: No time overlap between train and valid splits")

    # ── Section 3: Dataset Counts ───────────────────────────────────────
    log.info(section_header("Section 3: Dataset Counts (user_id / item_id)"))

    user_ids = set()
    item_ids = set()
    user_exposure = Counter()
    item_exposure = Counter()

    for f, rg_idx, n_rows in rg_list:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(rg_idx, columns=['user_id', 'item_id'])
        uids = table.column('user_id').to_pylist()
        iids = table.column('item_id').to_pylist()
        user_ids.update(uids)
        item_ids.update(iids)
        user_exposure.update(uids)
        item_exposure.update(iids)

    unique_user_count = len(user_ids)
    unique_item_count = len(item_ids)

    log.info(f"  Unique user_id:  {unique_user_count:,}")
    log.info(f"  Unique item_id:  {unique_item_count:,}")
    log.info(f"  Total rows:      {total_rows:,}")
    log.info(f"  Rows per user:   mean={total_rows/unique_user_count:.2f}, "
             f"median={np.median(list(user_exposure.values())):.0f}, "
             f"max={user_exposure.most_common(1)[0][1]}")
    log.info(f"  Rows per item:   mean={total_rows/unique_item_count:.2f}, "
             f"median={np.median(list(item_exposure.values())):.0f}, "
             f"max={item_exposure.most_common(1)[0][1]}")

    repeat_users = sum(1 for c in user_exposure.values() if c > 1)
    repeat_items = sum(1 for c in item_exposure.values() if c > 1)
    log.info(f"  Users with >1 exposure: {repeat_users:,} ({repeat_users/unique_user_count*100:.1f}%)")
    log.info(f"  Items with >1 exposure: {repeat_items:,} ({repeat_items/unique_item_count*100:.1f}%)")

    log.info(f"\n  Top-10 most exposed items:")
    for iid, count in item_exposure.most_common(10):
        log.info(f"    item_id={iid}: {count} exposures")

    del user_ids, item_ids, user_exposure, item_exposure

    # ── Section 4: Label Statistics ─────────────────────────────────────
    log.info(section_header("Section 4: Label Statistics"))

    click_count = 0
    conversion_count = 0
    delay_stats = {'sum': 0.0, 'sum2': 0.0, 'count': 0, 'min': None, 'max': None}

    for f, rg_idx, n_rows in rg_list:
        pf = pq.ParquetFile(f)
        table = pf.read_row_group(rg_idx, columns=['label_type', 'label_time', 'timestamp'])
        lt = table.column('label_type').to_numpy()
        click_count += int((lt == 1).sum())
        conversion_count += int((lt == 2).sum())

        lt_col = table.column('label_time').to_numpy().astype(np.int64)
        ts_col = table.column('timestamp').to_numpy().astype(np.int64)
        delay = lt_col - ts_col

        delay_stats['sum'] += float(delay.sum())
        delay_stats['sum2'] += float((delay ** 2).sum())
        delay_stats['count'] += len(delay)
        if delay_stats['min'] is None or delay.min() < delay_stats['min']:
            delay_stats['min'] = int(delay.min())
        if delay_stats['max'] is None or delay.max() > delay_stats['max']:
            delay_stats['max'] = int(delay.max())

    total_labels = click_count + conversion_count
    pos_rate = conversion_count / total_labels if total_labels > 0 else 0
    imbalance = (1 - pos_rate) / pos_rate if pos_rate > 0 else float('inf')

    mean_delay = delay_stats['sum'] / delay_stats['count'] if delay_stats['count'] > 0 else 0
    var_delay = (delay_stats['sum2'] / delay_stats['count']) - mean_delay ** 2
    std_delay = float(np.sqrt(max(var_delay, 0)))

    log.info(f"  label_type=1 (click):       {click_count:,} ({click_count/total_labels*100:.1f}%)")
    log.info(f"  label_type=2 (conversion):  {conversion_count:,} ({conversion_count/total_labels*100:.1f}%)")
    log.info(f"  Positive rate:              {pos_rate:.4f} ({pos_rate*100:.2f}%)")
    log.info(f"  Class imbalance ratio:      ~1:{imbalance:.1f}")
    log.info(f"\n  Conversion delay (label_time - timestamp):")
    log.info(f"    mean={mean_delay:.1f}s, std={std_delay:.1f}s")
    log.info(f"    min={delay_stats['min']}s, max={delay_stats['max']}s")

    # ── Section 5: Feature Quality & Missingness ────────────────────────
    log.info(section_header("Section 5: Feature Quality & Missingness"))

    # Build feature group column lists
    user_int_cols = [f'user_int_feats_{fid}' for fid, _, _ in schema.get('user_int', [])]
    item_int_cols = [f'item_int_feats_{fid}' for fid, _, _ in schema.get('item_int', [])]
    user_dense_cols = [f'user_dense_feats_{fid}' for fid, _ in schema.get('user_dense', [])]

    # Initialize per-column stats
    col_stats = {}

    def init_col_stats(col_name: str, col_type: str):
        col_stats[col_name] = {
            'type': col_type,
            'null_count': 0, 'total_count': 0,
        }
        if col_type == 'scalar':
            col_stats[col_name]['seen_values'] = set()
            col_stats[col_name]['value_count'] = 0
            col_stats[col_name]['min'] = None
            col_stats[col_name]['max'] = None
        elif col_type == 'list':
            col_stats[col_name]['lengths'] = []

    # Detect column types from first row group
    for col_name in schema_names:
        if col_name in ('user_id', 'item_id', 'label_type', 'label_time', 'timestamp'):
            continue
        if col_name in seq_col_set:
            continue  # handled in Section 6

        col = first_rg_table.column(col_name)
        # Check if it's a list type
        is_list = pa.types.is_list(col.type) or pa.types.is_large_list(col.type)

        if col_name in user_int_cols or col_name in item_int_cols:
            col_type = 'list' if is_list else 'scalar'
        elif col_name in user_dense_cols:
            col_type = 'list'
        else:
            col_type = 'list' if is_list else 'scalar'

        init_col_stats(col_name, col_type)

    # Streaming accumulation
    for f, rg_idx, n_rows in rg_list:
        pf = pq.ParquetFile(f)
        cols_to_read = [c for c in col_stats.keys() if c in schema_names]
        if not cols_to_read:
            continue
        table = pf.read_row_group(rg_idx, columns=cols_to_read)

        for col_name, st in col_stats.items():
            if col_name not in table.column_names:
                continue
            col = table.column(col_name)
            st['null_count'] += col.null_count
            st['total_count'] += len(col)

            if st['type'] == 'scalar':
                arr = col.to_numpy(zero_copy_only=False)
                # Handle nullable columns
                valid_mask = col.is_valid().to_numpy()
                non_null = arr[valid_mask]

                if len(non_null) > 0:
                    st['value_count'] += len(non_null)
                    if len(st['seen_values']) < MAX_SEEN_VALUES:
                        vals = non_null.tolist()
                        st['seen_values'].update(vals)

                    v_min = float(non_null.min())
                    v_max = float(non_null.max())
                    if st['min'] is None or v_min < st['min']:
                        st['min'] = v_min
                    if st['max'] is None or v_max > st['max']:
                        st['max'] = v_max

            elif st['type'] == 'list':
                arr = col.combine_chunks()
                valid_mask = arr.is_valid()
                if valid_mask.to_numpy().any():
                    offsets = arr.offsets.to_numpy()
                    vm = valid_mask.to_numpy()
                    starts = offsets[:-1][vm]
                    ends = offsets[1:][vm]
                    lens = (ends - starts).tolist()
                    st['lengths'].extend(lens)

    # Output by feature group
    groups = {
        'User Int': user_int_cols,
        'Item Int': item_int_cols,
        'User Dense': user_dense_cols,
    }

    for group_name, cols in groups.items():
        existing = [c for c in cols if c in col_stats]
        if not existing:
            continue

        scalar_cols = [c for c in existing if col_stats[c]['type'] == 'scalar']
        list_cols = [c for c in existing if col_stats[c]['type'] == 'list']

        log.info(f"\n  {group_name} ({len(existing)} cols: {len(scalar_cols)} scalar, {len(list_cols)} list):")

        if scalar_cols:
            log.info(f"    {'Feature':<30s} {'Null%':>6s} {'N-unique':>9s} {'Min':>12s} {'Max':>12s}")
            for c in sorted(scalar_cols):
                s = col_stats[c]
                null_pct = s['null_count'] / s['total_count'] * 100 if s['total_count'] > 0 else 0
                if s['value_count'] <= MAX_SEEN_VALUES:
                    nunique = f"{len(s['seen_values']):,}"
                else:
                    nunique = f"~{s['value_count']:,}"
                min_v = f"{s['min']:.0f}" if s['min'] is not None else '-'
                max_v = f"{s['max']:.0f}" if s['max'] is not None else '-'
                log.info(f"    {c:<30s} {null_pct:>5.1f}% {nunique:>9s} {min_v:>12s} {max_v:>12s}")

        if list_cols:
            log.info(f"    {'Feature':<30s} {'Null%':>6s} {'Len mean':>9s} {'Len max':>9s}")
            for c in sorted(list_cols):
                s = col_stats[c]
                null_pct = s['null_count'] / s['total_count'] * 100 if s['total_count'] > 0 else 0
                lens = s['lengths']
                len_mean = np.mean(lens) if lens else 0
                len_max = max(lens) if lens else 0
                log.info(f"    {c:<30s} {null_pct:>5.1f}% {len_mean:>9.1f} {len_max:>9,}")

    # High-missingness alert
    high_null = []
    for c, s in col_stats.items():
        if s['total_count'] > 0 and s['null_count'] / s['total_count'] > 0.5:
            high_null.append((c, s['null_count'] / s['total_count'] * 100))
    if high_null:
        log.info(f"\n  WARNING: High-missingness features (>50%):")
        for c, pct in sorted(high_null, key=lambda x: -x[1]):
            log.info(f"    {c:30s}: {pct:.1f}%")

    # Cleanup large accumulators
    for c in col_stats:
        col_stats[c].pop('seen_values', None)
        col_stats[c].pop('lengths', None)

    # ── Section 6: Sequence Feature Statistics ──────────────────────────
    log.info(section_header("Section 6: Sequence Feature Statistics"))

    seq_stats = {}
    for domain, cfg in sorted(seq_cfg.items()):
        prefix = cfg['prefix']
        seq_stats[domain] = {}
        for fid, _ in cfg['features']:
            col_name = f'{prefix}_{fid}'
            if col_name not in schema_names:
                continue
            seq_stats[domain][col_name] = {
                'null_count': 0, 'total_count': 0,
                'lengths': [], 'value_counter': Counter(),
                'val_min': None, 'val_max': None,
            }

    seq_cols_in_schema = [c for c in schema_names if c in seq_col_set]

    for f, rg_idx, n_rows in rg_list:
        pf = pq.ParquetFile(f)
        available = [c for c in seq_cols_in_schema if c in pf.schema_arrow.names]
        if not available:
            continue
        table = pf.read_row_group(rg_idx, columns=available)

        for domain, cols in seq_stats.items():
            for col_name, s in cols.items():
                if col_name not in table.column_names:
                    continue
                col = table.column(col_name)
                s['null_count'] += col.null_count
                s['total_count'] += len(col)

                arr = col.combine_chunks()
                valid_mask = arr.is_valid()
                if valid_mask.to_numpy().any():
                    offsets = arr.offsets.to_numpy()
                    values = arr.values.to_numpy()
                    vm = valid_mask.to_numpy()
                    starts = offsets[:-1][vm]
                    ends = offsets[1:][vm]
                    lens = (ends - starts).tolist()
                    s['lengths'].extend(lens)

                    # Cap value counting to avoid OOM
                    if len(s['value_counter']) < 100000:
                        all_vals = values.tolist()
                        s['value_counter'].update(all_vals[:200000])

                    v_min = int(values.min())
                    v_max = int(values.max())
                    if s['val_min'] is None or v_min < s['val_min']:
                        s['val_min'] = v_min
                    if s['val_max'] is None or v_max > s['val_max']:
                        s['val_max'] = v_max

    # Output per domain
    for domain in sorted(seq_stats.keys()):
        cols = seq_stats[domain]
        log.info(f"\n  Domain {domain} ({len(cols)} features):")

        for col_name in sorted(cols.keys()):
            s = cols[col_name]
            null_pct = s['null_count'] / s['total_count'] * 100 if s['total_count'] > 0 else 0
            fid = col_name.rsplit('_', 1)[-1]

            lens = s['lengths']
            len_info = ""
            if lens:
                len_info = (f"  len: mean={np.mean(lens):.0f}, "
                           f"median={np.median(lens):.0f}, "
                           f"max={max(lens):,}")

            val_info = ""
            if s['val_min'] is not None:
                val_info = f"  vals: [{s['val_min']}, {s['val_max']}], "
                val_info += f"unique~{len(s['value_counter']):,}"
                top3 = s['value_counter'].most_common(3)
                val_info += f", top3={top3}"

            log.info(f"    fid={fid:>4s}: null={null_pct:.1f}%{len_info}  {val_info}")

    # Cleanup
    for domain in seq_stats:
        for col_name in seq_stats[domain]:
            seq_stats[domain][col_name].pop('value_counter', None)
            seq_stats[domain][col_name].pop('lengths', None)

    # ── Summary JSON ────────────────────────────────────────────────────
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        summary_path = os.path.join(args.log_dir, 'eda_summary.json')
        summary = {
            'total_rows': total_rows,
            'total_rgs': total_rgs,
            'n_train_rgs': n_train_rgs,
            'n_valid_rgs': n_valid_rgs,
            'train_rows': train_rows,
            'valid_rows': valid_rows,
            'click_count': click_count,
            'conversion_count': conversion_count,
            'positive_rate': round(pos_rate, 6),
            'unique_users': unique_user_count,
            'unique_items': unique_item_count,
            'mean_delay': round(mean_delay, 1),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log.info(f"\n  Summary written to {summary_path}")

    log.info("\nEDA complete!")


if __name__ == "__main__":
    main()
