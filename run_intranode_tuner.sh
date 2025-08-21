#!/bin/bash
set -euo pipefail

HIDDEN_SIZE=2048
NUM_TOKENS=4096
NUM_EXPERTS=128
TOPK=8
OUTPUT="tuned_results"

MIN_SMS=40 # default DeepEP setting
MAX_SMS=148 # Max number of SMS on B200
MIN_DISPATCH_NVL=10
MAX_DISPATCH_NVL=64
MIN_COMBINE_NVL=10
MAX_COMBINE_NVL=64

CMD="python -m tests.test_intranode \
--hidden ${HIDDEN_SIZE} \
--num-tokens ${NUM_TOKENS} \
--num-experts ${NUM_EXPERTS} \
--num-topk ${TOPK} \
--num-sms-range ${MIN_SMS} ${MAX_SMS} \
--dispatch-nvl-send-size-range ${MIN_DISPATCH_NVL} ${MAX_DISPATCH_NVL} \
--combine-nvl-send-size-range ${MIN_COMBINE_NVL} ${MAX_COMBINE_NVL} \
--output-path ${OUTPUT}
"

echo "<< ${CMD}"
mkdir -p logs

eval ${CMD} 2>&1 | tee logs/tuning_hs${HIDDEN_SIZE}_tok${NUM_TOKENS}_exp${NUM_EXPERTS}_topk${TOPK}_SMS${MIN_SMS}-${MAX_SMS}_DISPATCHNVL${MIN_DISPATCH_NVL}-${MAX_DISPATCH_NVL}_COMBINENVL${MIN_COMBINE_NVL}-${MAX_COMBINE_NVL}.log