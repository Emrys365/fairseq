#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=en
root_path=/path/to/fairseq/examples/wavlm
config_name=wavlm_base_iter1_libri2mix_train100_8x1gpu
config_dir=${root_path}/config/finetune/
exp_dir=
exp_tag=
data=${root_path}/../tshubert/data/train_wsj0_2mix
init=${root_path}/exp/wavlm_base_iter1_librispeech_2x4gpu_train_tsv_files_label_kmeans_labels/checkpoints/checkpoint_best.pt

export NCCL_DEBUG=INFO
export NCCP_P2P_DISABLE=1

. utils/parse_options.sh

data=$(realpath $data)
init=$(realpath $init)

if [ -z "${exp_dir}" ]; then
    exp_dir=${root_path}/exp/finetune_${lang}/$(basename "${data}")
fi

if [ -z "${exp_tag}" ]; then
    exp_tag=${config_name}_finetune_$(basename "${data}")_from_$(basename ${init} .pt)
fi

exp=${exp_dir}/${exp_tag}

# CUDA_VISIBLE_DEVICES=1 fairseq-hydra-train \
fairseq-hydra-train \
  --config-dir ${config_dir} \
  --config-name ${config_name} \
  task.data=${data} task.label_dir=${data} \
  dataset.max_tokens=500000 optimization.update_freq=[4] \
  model.w2v_path=${init} hydra.run.dir=${exp}
