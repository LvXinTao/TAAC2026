#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     --weight_decay 0.01 \
#     --warmup_steps 100 \
#     "$@"

# ---- Active config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# num_ns=12 (7 user_int + 1 user_dense + 4 item_int), num_sequences=4.
# T = num_queries * num_sequences + num_ns = 2*4 + 12 = 20.
# d_model=80 satisfies 80 % 20 == 0.
#
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type group \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --num_queries 2 \
    --d_model 80 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    "$@"
