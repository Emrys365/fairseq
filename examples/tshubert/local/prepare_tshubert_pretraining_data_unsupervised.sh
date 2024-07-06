#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_pretraining_data_upsupervised.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --datadir <datadir> --feature_type <feature_type> --nj <nj> --layer_index <layer_index> --max_chunk <max_chunk> --sample_rate <sample_rate> --n_clusters <n_clusters> --kmrootdir <kmrootdir> --percent <percent> --km_batch_size <km_batch_size> --tsv_dir <tsv_dir> --train_set <train_set> --dev_set <dev_set> --spk_model_ckpt <spk_model_ckpt>
  optional argument:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to python binary
    [--datadir]: path the data root directory

    [--feature_type]: feature type, either "mfcc" or "WavLM*"
    [--nj]: number of parallel processes
    [--layer_index]: Index of the network layer for outputting features
    [--max_chunk]: maximum chunk size
    [--sample_rate]: sample rate of the data (in Hz)

    [--n_clusters]: number of clusters to form with K-Means
    [--kmrootdir]: output directory for storing the K-Means model
    [--percent]: percent of the data to train a K-Means model
      if it is -1, all data will be used.
    [--km_batch_size]: batch size when training a K-Means model

    [--tsv_dir]: directory for storing the tsv files

    [--train_set]: name of the training set
    [--dev_set]: name of the development set

    [--spk_model_ckpt]: path to the pretrained speaker model checkpoint
EOF
)


stage=0
stop_stage=100

python=python3
datadir=data
# Path to the Librispeech directory
librispeech=
# Path to the pretrained HuBERT model
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models"
# Feature type: "mfcc" or "WavLM" or "HuBERT"
feature_type=HuBERT
nj=4
layer_index=9   # the 9-th TransformerEncoder layer, index starts from 1
max_chunk=1600000
sample_rate=16000

n_clusters=500
kmrootdir=kmeans_model
percent=0.1    # 960 * 0.1 = 96 hr
km_batch_size=10000

tsv_dir=${datadir}/tsv_files

train_set=train960
dev_set=dev

# spk_model_ckpt=local/spk_embs/model.th
spk_model_ckpt=/path/to/voxceleb_resnet293_LM.onnx

. utils/parse_options.sh


log "$0 $*"
if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "$librispeech" ]; then
    echo "Please fill the value of librispeech= manually"
    exit 1;
fi

if [ "$sample_rate" = "16k" ]; then
    sample_rate="16000"
elif [ "$sample_rate" = "8k" ]; then
    sample_rate="8000"
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Download HuBERT pretraining model"

    mkdir -p "${hubert_dir_path}"
    if [ ! -e "${hubert_dir_path}/hubert_base_ls960.pt" ]; then
        wget "$hubert_url" -O "${hubert_dir_path}/hubert_base_ls960.pt"
    else
        log "Skipping as the file already exists: ${hubert_dir_path}/hubert_base_ls960.pt"
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Preparing Librispeech data"

    local/data.sh \
        --python ${python} \
        --stage 1 \
        --stop-stage 4 \
        --datadir "${datadir}" \
        --librispeech "${librispeech}" \
        --ckpt-path "${hubert_dir_path}/hubert_base_ls960.pt" \
        --feature-type "${feature_type}" \
        --layer-index ${layer_index} \
        --nj ${nj} \
        --max-chunk ${max_chunk} \
        --sample-rate ${sample_rate} \
        --n-clusters ${n_clusters} \
        --kmrootdir "${kmrootdir}" \
        --percent ${percent} \
        --km_batch_size ${km_batch_size} \
        --tsv_dir "${tsv_dir}" \
        --train_set ${train_set} \
        --dev_set ${dev_set}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Preparing unsupervised speaker labels"

    for x in ${train_set} ${dev_set}; do
        # ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        #     "${tsv_dir}/${x}.tsv" \
        #     --model_init "${spk_model_ckpt}" \
        #     --outdir spk_embs/out_ll/${x} \
        #     --cuda False
        ${python} local/spk_embs/prepare_spk_embs_scp.py \
            "${tsv_dir}/${x}.tsv" \
            --onnx_path "${spk_model_ckpt}" \
            --outdir spk_embs/out_ll/${x} \
            --max_workers 20

        ${python} local/unsupervised_speaker_clustering.py \
            spk_embs/out_ll/${x}/spk2emb.scp \
            --outfile "${tsv_dir}/${x}.sid"
    done
fi
