#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

lang=en
root_path=/path/to/fairseq/examples/tshubert
config_name=infer_viterbi_no_enroll
config_dir=${root_path}/config/decode/
expdir=${root_path}/exp/finetune_${lang}/ll_1h/tshubert_base_iter1_ll_1h_8x1gpu_finetune_ll_1h_from_checkpoint_best
checkpoint=checkpoint_best.pt
test_sets="ls_test_clean ls_test_other"
data=${root_path}/../wavlm/data
device=0

. utils/parse_options.sh

checkpoint="${expdir}/checkpoints/${checkpoint}"

for test_set in $test_sets; do
    CUDA_VISIBLE_DEVICES=$device python ../speech_recognition/new/infer.py \
      --config-dir "${config_dir}" \
      --config-name "${config_name}" \
      task.data="${data}/${test_set}" \
      task.normalize=false \
      task.load_enrollment_and_emb=false \
      common_eval.results_path="${expdir}/decode/${test_set}" \
      common_eval.path="${checkpoint}" \
      dataset.gen_subset=test
done
