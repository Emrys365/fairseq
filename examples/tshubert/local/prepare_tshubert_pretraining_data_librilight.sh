#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_pretraining_data.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --seed <seed> --datadir <datadir> --feature_type <feature_type> --nj <nj> --layer_index <layer_index> --max_chunk <max_chunk> --sample_rate <sample_rate> --n_clusters <n_clusters> --kmrootdir <kmrootdir> --percent <percent> --km_batch_size <km_batch_size> --tsv_dir <tsv_dir> --train_set <train_set>
  optional argument:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to python binary
    [--seed]: random seed
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
EOF
)


stage=0
stop_stage=100

python=python3
seed=0
datadir=data
# Path to the Librilight directory containing tar files
librilight=
# Path to the Librispeech directory containing audio files
librispeech=
# Path to the pretrained HuBERT model
hubert_url="https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
hubert_dir_path="./downloads/hubert_pretrained_models"
# Feature type: "mfcc" or "WavLM" or "HuBERT"
feature_type=HuBERT
nj=32
layer_index=9   # the 9-th TransformerEncoder layer, index starts from 1
max_chunk=1600000
sample_rate=16000

n_clusters=500
kmrootdir=kmeans_model
percent=0.1    # 960 * 0.1 = 96 hr
km_batch_size=10000

tsv_dir=${datadir}/librilight_60k

train_set=train
dev_set=valid

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh

feature_type=${feature_type}${layer_index}


log "$0 $*"
if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "$librilight" ]; then
    echo "Please fill the value of librilight= manually"
    echo "The data can be downloaded from https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md"
    exit 1;
fi

if [ -z "$librispeech" ]; then
    echo "Please fill the value of librispeech= manually"
    echo "The data can be downloaded from https://www.openslr.org/12"
    exit 1;
fi

if [ "$sample_rate" = "16k" ]; then
    sample_rate="16000"
elif [ "$sample_rate" = "8k" ]; then
    sample_rate="8000"
fi

km_path="${kmrootdir}/km_${train_set}_${feature_type}/km_${n_clusters}clusters.mdl"
mkdir -p "$(dirname ${km_path})"
feat_dir="${kmrootdir}/feat_${feature_type}"
mkdir -p "${feat_dir}"
lab_dir="${kmrootdir}/km_${train_set}_${feature_type}/kmeans_labels"
mkdir -p "${lab_dir}"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Download HuBERT pretraining model"

    mkdir -p "${hubert_dir_path}"
    if [ ! -e "${hubert_dir_path}/hubert_base_ls960.pt" ]; then
        wget "$hubert_url" -O "${hubert_dir_path}/hubert_base_ls960.pt"
    else
        log "Skipping as the file already exists: ${hubert_dir_path}/hubert_base_ls960.pt"
    fi
fi
# Path to the pretrained SSL model
ckpt_path=${hubert_dir_path}/hubert_base_ls960.pt


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Prepare tsv files"
    mkdir -p "${tsv_dir}"

    #  This takes ~8 hours to finish
    ${python} local/librilight/list_tar_files.py \
        "${librilight}"/large.tar \
        "${librilight}"/medium.tar \
        "${librilight}"/small.tar \
        --outdir "${librilight}"

    # This takes 30 minutes ~ 2 hours to finish with nj=32
    ${python} local/librilight/cut_by_vad_from_tar.py \
        --input_tars "${librilight}"/{large,medium,small}.tar \
        --output_dir "${tsv_dir}" \
        --write_audio False \
        --target_len_sec 15 \
        --n_workers ${nj} \
        --chunk_size 10 \
        --out_name train

    # We just use dev-clean + dev-other from Librispeech for validation
    ${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/dev-clean" "${librispeech}/dev-other" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 200 > "${tsv_dir}/${dev_set}.tsv"

    for dset in ${dev_set}; do
        tail -n +2 "${tsv_dir}/${dset}.tsv" | awk 'function basename(file) {
            sub(".*/", "", file)
            return file
        }
        {print basename($1)}' | sed -e 's/.flac$//g' > "${tsv_dir}/${dset}.uid"

        awk -F "-" '{print $1}' "${tsv_dir}/${dset}.uid" > "${tsv_dir}/${dset}.sid"
    done

    # "${tsv_dir}/
	#  ├── train.tsv
	#  ├── train.uid
	#  ├── train.sid
	#  ├── valid.tsv
	#  ├── valid.uid
	#  └── valid.sid
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Dump ${feature_type} features"

    # Prepare a subset with ${percent} percent of data for K-Means
    ${python} - << EOF
import random

random.seed(${seed})

total = 0.0
fs = ${sample_rate}
lines = []
with open("${tsv_dir}/train.tsv", "r") as tsv:
    root = next(tsv)
    for line in tsv:
        path, nsamples = line.strip().split("\t")
        total += int(nsamples) / fs
        lines.append(line)

desired = total * ${percent}
count = 0.0
random.shuffle(lines)
with open("${tsv_dir}/train_kmeans.tsv", "w") as f:
    f.write(root)
    for line in lines:
        path, nsamples = line.strip().split("\t")
        dur = int(nsamples) / fs
        f.write(line)
        if count + dur >= desired:
            break
EOF

    mkdir -p "${feat_dir}/log"
    feat_type=$(echo "${feature_type}" | sed -e 's/^wavlm.*/wavlm/i' | sed -e 's/^hubert.*/hubert/i')
    for tsv_file_name in train_kmeans; do
		if [ "${feat_type}" = "hubert" ]; then
        # Features will be saved as ${feat_dir}/${tsv_file_name}_${rank}_${nshard}.{npy,len}.
        # For tsv_file_name=train_kmeans:
        #   ~495 GB data will be generated.
        ${cuda_cmd} --gpu 1 JOB=1:${nj} "${feat_dir}"/log/dump_hubert_feature.JOB.log \
            ${python} simple_kmeans/dump_wavlm_feature.py \
                "${tsv_dir}" \
                "${tsv_file_name}" \
                --ckpt_path "${ckpt_path}" \
                --layer ${layer_index} \
                --nshard ${nj} \
                --rank JOB \
                --feat_dir "${feat_dir}" \
                --max_chunk ${max_chunk} \
                --device "gpu"

        elif [ "${feat_type}" = "wavlm" ]; then
        # Features will be saved as ${feat_dir}/${tsv_file_name}_${rank}_${nshard}.{npy,len}.
        ${cuda_cmd} --gpu 1 JOB=1:${nj} "${feat_dir}"/log/dump_wavlm_feature.JOB.log \
            ${python} simple_kmeans/dump_wavlm_feature.py \
                "${tsv_dir}" \
                "${tsv_file_name}" \
                --ckpt_path "${ckpt_path}" \
                --layer ${layer_index} \
                --nshard ${nj} \
                --rank JOB \
                --feat_dir "${feat_dir}" \
                --max_chunk ${max_chunk} \
                --device "gpu"

        elif [ "${feat_type}" = "mfcc" ]; then
            # Features will be saved as ${feat_dir}/${tsv_file_name}_${rank}_${nshard}.{npy,len}.
            ${train_cmd} JOB=1:${nj} "${feat_dir}"/log/dump_mfcc_feature.JOB.log \
                ${python} simple_kmeans/dump_mfcc_feature.py \
                    "${tsv_dir}" \
                    "${tsv_file_name}" \
                    --nshard ${nj} \
                    --rank JOB \
                    --feat_dir "${feat_dir}" \
                    --sample_rate ${sample_rate}

        else
            log "Error: Invalid feature_type: ${feature_type}"
            exit 2
        fi
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: K-Means clustering on extracted features"

    logdir="$(dirname ${km_path})"/log
    mkdir -p "${logdir}"
    # Fit a K-Means model with ${n_clusters} clusters on ${percent}% of the ${split} data
    # It took ~1 hour.
    ${train_cmd} --mem 150G "${logdir}"/learn_kmeans.log \
        ${python} simple_kmeans/learn_kmeans.py \
            "${feat_dir}" \
            train_kmeans \
            ${n_clusters} \
            --km_path "${km_path}" \
            --nshard ${nj} \
            --seed 0 \
            --percent 1.0 \
            --batch_size ${km_batch_size}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Dump K-Means clustering labels on data"

    mkdir -p "${lab_dir}/log"
    # This will extract labels for the ${rank}-th shard out of ${nshard} shards
    # and dump them to ${lab_dir}/${tsv_file_name}_${rank}_${shard}.km
    # It took ~12 hours with nj=32.
    for tsv_file_name in ${train_set} ${dev_set}; do
        ${decode_cmd} --gpu 1 JOB=1:${nj} "${lab_dir}"/log/dump_km_label.JOB.log \
            ${python} simple_kmeans/dump_km_label_from_audio.py \
                "${tsv_dir}" \
                ${tsv_file_name} \
                "${km_path}" \
                --ckpt_path "${ckpt_path}" \
                --layer ${layer_index} \
                --nshard ${nj} \
                --rank JOB \
                --lab_dir "${lab_dir}" \
                --device gpu
    done

    # Merge shards for ${tsv_file_name}
    for tsv_file_name in ${train_set} ${dev_set}; do
        for rank in $(seq 0 $((nj - 1))); do
            cat ${lab_dir}/${tsv_file_name}_${rank}_${nj}.km
        done > "${lab_dir}/${tsv_file_name}.km"

        rm ${lab_dir}/${tsv_file_name}_${rank}_${nj}.km
    done

    # generate the dict file (https://github.com/facebookresearch/fairseq/issues/2514)
    fairseq-preprocess --dataset-impl mmap --trainpref "${lab_dir}/${train_set}.km" --only-source --thresholdsrc 0
    # only keep the dict file
    mv data-bin/dict.txt "${lab_dir}/dict.km.txt"
    rm -r data-bin/
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
