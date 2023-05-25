#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

lang=en
root_path=/path/to/fairseq/examples/tshubert
config_name=infer_viterbi
config_dir=${root_path}/config/decode/
expdir=${root_path}/exp/finetune_${lang}/train_2mix/tshubert_base_iter1_libri2mix_train100_8x1gpu_finetune_train_2mix_from_checkpoint_best
checkpoint=checkpoint_best.pt
test_sets="test_2mix"
data=${root_path}/data
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
