#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# before srun this file, run the following command:
# salloc -J tshubert_iter1 -p 2080ti -x gqxx-01-125,gqxx-01-111 --gres=gpu:4 --nodes 2 --ntasks-per-node 4 --mem=28G --cpus-per-task 5

root_path=/path/to/fairseq/examples/tshubert
# This config file is intended for using 2 nodes with 4 RTX 2080 Ti GPUs each.
# It takes around 3 weeks to finish 400k steps.
config_name=tshubert_base_iter1_librispeech_2x4gpu
config_dir=${root_path}/config/pretrain/
expdir=${root_path}/exp
data=${root_path}/data/tsv_files
label=${root_path}/kmeans_model/km_train960_HuBERT9/kmeans_labels
exp_tag=

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1


. ./path.sh
. utils/parse_options.sh

data="$(realpath $data)"
label="$(realpath $label)"

if [ -z "${exp_tag}" ]; then
    exp_tag=${config_name}_train_$(basename "${data}")_label_$(basename "${label}")
fi
expname=${expdir}/${exp_tag}

# CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train \
fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${label} task.labels='["km"]' model.label_rate=50 hydra.run.dir=${expname}
