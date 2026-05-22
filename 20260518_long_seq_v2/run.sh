#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer (no ns_groups.json required) ----
USER_DENSE_UE_FIDS="${USER_DENSE_UE_FIDS:-61,87,89,90,91}"
USER_INT_DENSE_PAIR_FIDS="${USER_INT_DENSE_PAIR_FIDS:-62,63,64,65,66}"
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --user_dense_ue_fids "${USER_DENSE_UE_FIDS}" \
    --user_int_dense_pair_fids "${USER_INT_DENSE_PAIR_FIDS}" \
    --num_workers 8 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --d_model 128 \
    --seq_max_lens "seq_a:256,seq_b:1024,seq_c:1024,seq_d:2048" \
    --seq_domain_encoder_types "transformer,longer,longer,longer" \
    --seq_top_k 512 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
