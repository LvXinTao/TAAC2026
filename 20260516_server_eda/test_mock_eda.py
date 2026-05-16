#!/usr/bin/env python3
"""Generate mock parquet data matching the real schema, then run eda.py against it."""
import os
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'mock_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ROWS = 500  # small enough for quick run
SEED = 42
rng = np.random.RandomState(SEED)

# ── Build schema.json ──────────────────────────────────────────────
schema = {
    "user_int": [[1, 100, 1], [3, 10, 1], [48, 50, 1]],
    "item_int": [[5, 200, 1], [11, 80, 1]],
    "user_dense": [[61, 10], [62, 5]],
    "seq": {
        "a": {
            "prefix": "seq_a",
            "features": [[100, 1], [101, 1]],
            "max_len": 32,
        },
        "b": {
            "prefix": "seq_b",
            "features": [[200, 1], [201, 1]],
            "max_len": 16,
        },
    },
}

with open(os.path.join(OUTPUT_DIR, 'schema.json'), 'w') as f:
    json.dump(schema, f, indent=2)

# ── Generate parquet columns ──────────────────────────────────────
base_ts = 1740000000  # ~2025-02

data = {
    'user_id': rng.randint(1, 200, size=N_ROWS).astype(np.int64),
    'item_id': rng.randint(1, 50, size=N_ROWS).astype(np.int64),
    'timestamp': (base_ts + rng.randint(0, 100000, size=N_ROWS)).astype(np.int64),
    'label_type': rng.choice([1, 2], size=N_ROWS, p=[0.9, 0.1]).astype(np.int64),
    'label_time': (base_ts + 100000 + rng.randint(0, 50000, size=N_ROWS)).astype(np.int64),

    # scalar int features (some with nulls)
    'user_int_feats_1': [int(x) if rng.random() > 0.05 else None for x in rng.randint(1, 100, size=N_ROWS)],
    'user_int_feats_3': rng.randint(1, 10, size=N_ROWS).astype(np.int64),
    'user_int_feats_48': [int(x) if rng.random() > 0.10 else None for x in rng.randint(1, 50, size=N_ROWS)],

    'item_int_feats_5': rng.randint(1, 200, size=N_ROWS).astype(np.int64),
    'item_int_feats_11': [int(x) if rng.random() > 0.03 else None for x in rng.randint(1, 80, size=N_ROWS)],
}

# Dense features — variable-length float lists
for fid in [61, 62]:
    col_name = f'user_dense_feats_{fid}'
    lists = []
    for _ in range(N_ROWS):
        if rng.random() < 0.08:  # 8% null
            lists.append(None)
        else:
            length = int(rng.randint(1, 8))
            lists.append(rng.uniform(-1.0, 1.0, size=length).tolist())
    data[col_name] = lists

# Sequence features — variable-length int lists
for prefix, fids, max_l in [
    ('seq_a', [100, 101], 32),
    ('seq_b', [200, 201], 16),
]:
    for fid in fids:
        col_name = f'{prefix}_{fid}'
        lists = []
        for _ in range(N_ROWS):
            if rng.random() < 0.12:  # 12% null
                lists.append(None)
            else:
                length = int(rng.randint(0, max_l))
                if length == 0:
                    lists.append([])
                else:
                    vocab = 5000
                    lists.append(rng.randint(1, vocab, size=length).tolist())
        data[col_name] = lists

# ── Build pyarrow table with proper types ─────────────────────────
fields = []
arrays = []
for col_name, values in data.items():
    if col_name == 'user_id':
        fields.append(pa.field(col_name, pa.int64()))
        arrays.append(pa.array(values, type=pa.int64()))
    elif col_name == 'item_id':
        fields.append(pa.field(col_name, pa.int64()))
        arrays.append(pa.array(values, type=pa.int64()))
    elif col_name == 'timestamp':
        fields.append(pa.field(col_name, pa.int64()))
        arrays.append(pa.array(values, type=pa.int64()))
    elif col_name == 'label_type':
        fields.append(pa.field(col_name, pa.int64()))
        arrays.append(pa.array(values, type=pa.int64()))
    elif col_name == 'label_time':
        fields.append(pa.field(col_name, pa.int64()))
        arrays.append(pa.array(values, type=pa.int64()))
    elif 'dense' in col_name:
        # variable-length float list
        fields.append(pa.field(col_name, pa.list_(pa.float32())))
        arrays.append(pa.array(
            [v if v is not None else None for v in values],
            type=pa.list_(pa.float32()),
        ))
    elif any(col_name.startswith(p) for p in ['seq_a', 'seq_b']):
        # variable-length int list
        fields.append(pa.field(col_name, pa.list_(pa.int64())))
        arrays.append(pa.array(
            [v if v is not None else None for v in values],
            type=pa.list_(pa.int64()),
        ))
    else:
        # scalar int (nullable)
        fields.append(pa.field(col_name, pa.int64()))
        arrays.append(pa.array(values, type=pa.int64()))

table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

# Write as 2 parquet files to test multi-file logic
n1 = N_ROWS // 2
pq.write_table(table.slice(0, n1), os.path.join(OUTPUT_DIR, 'part-00000.parquet'))
pq.write_table(table.slice(n1), os.path.join(OUTPUT_DIR, 'part-00001.parquet'))

print(f"Mock data written: {OUTPUT_DIR}")
print(f"  {N_ROWS} rows, {len(data)} columns")
print(f"  Columns: {list(data.keys())}")

# ── Run eda.py ─────────────────────────────────────────────────────
print("\n--- Running eda.py ---\n")
import subprocess
# Find the same python that's running this script (should be conda agent env)
python = os.path.realpath('/proc/self/exe') if os.path.exists('/proc/self/exe') else '/Users/lvxintao/miniconda3/envs/agent/bin/python3'
result = subprocess.run(
    [python, '-u', os.path.join(os.path.dirname(__file__), 'eda.py'),
     '--data_dir', OUTPUT_DIR,
     '--schema_path', os.path.join(OUTPUT_DIR, 'schema.json'),
     '--log_dir', OUTPUT_DIR],
    capture_output=True, text=True,
    env={**os.environ, 'TRAIN_DATA_PATH': OUTPUT_DIR, 'TRAIN_LOG_PATH': OUTPUT_DIR},
)
print(result.stdout)
if result.returncode != 0:
    print(f"\nSTDERR:\n{result.stderr}")
    print(f"\nExit code: {result.returncode}")
else:
    print(f"\nSUCCESS: eda.py exited with code 0")
