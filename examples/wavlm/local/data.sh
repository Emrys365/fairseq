#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/data.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --datadir <datadir> --ckpt_path <ckpt_path> --feature_type <feature_type> --nj <nj> --layer_index <layer_index> --max_chunk <max_chunk> --sample_rate <sample_rate> --n_clusters <n_clusters> --kmrootdir <kmrootdir> --percent <percent> --km_batch_size <km_batch_size> --lab_dir <lab_dir> --tsv_dir <tsv_dir> --train_set <train_set> --dev_set <dev_set>
  optional argument:
    [--stage]: start stage, default is 1
    [--stop_stage]: stop stage, default is 100
    [--python]: path to pthon binary
    [--datadir]: path the data root directory

    [--ckpt_path]: directory of the pretrained WavLM model
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

    [--lab_dir]: directory for storing the label files
    [--tsv_dir]: directory for storing the tsv files

    [--train_set]: name of the training set
    [--dev_set]: name of the development set
EOF
)


stage=1
stop_stage=100

python=python3
datadir=data
# Path to the Librispeech directory
librispeech=
# Path to the pretrained WavLM model
ckpt_path=/path/to/hubert_pretrained_models/hubert_base_ls960.pt
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

. utils/parse_options.sh

feature_type=${feature_type}${layer_index}

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


km_path="${kmrootdir}/km_${train_set}_${feature_type}/km_${n_clusters}clusters.mdl"
mkdir -p "$(dirname ${km_path})"
feat_dir="${kmrootdir}/feat_${feature_type}"
mkdir -p "${feat_dir}"
lab_dir="${kmrootdir}/km_${train_set}_${feature_type}/kmeans_labels"
mkdir -p "${lab_dir}"



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Prepare tsv files"
    mkdir -p "${tsv_dir}"

    ${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/train-clean-100" "${librispeech}/train-clean-360" "${librispeech}/train-other-500" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 500 > "${tsv_dir}/${train_set}.tsv"

    ${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/dev-clean" "${librispeech}/dev-other" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 200 > "${tsv_dir}/${dev_set}.tsv"

    for dset in ${train_set} ${dev_set}; do
        tail -n +2 "${tsv_dir}/${dset}.tsv" | awk 'function basename(file) {
            sub(".*/", "", file)
            return file
        }
        {print basename($1)}' | sed -e 's/.flac$//g' > "${tsv_dir}/${dset}.uid"

        awk -F "-" '{print $1}' "${tsv_dir}/${dset}.uid" > "${tsv_dir}/${dset}.sid"
    done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Dump ${feature_type} features"

    mkdir -p "${feat_dir}/log"
    feat_type=$(echo "${feature_type}" | sed -e 's/^wavlm.*/wavlm/i' | sed -e 's/^hubert.*/hubert/i')
    for tsv_file_name in ${train_set} ${dev_set}; do
		if [ "${feat_type}" = "hubert" ]; then
        # Features will be saved as ${feat_dir}/${tsv_file_name}_${rank}_${nshard}.{npy,len}.
        # For tsv_file_name=train960:
        #   ~495 GB data will be generated.
        # For tsv_file_name=dev:
        #   ~5.5 GB data will be generated.
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
        # For tsv_file_name=train960:
        #   ~495 GB data will be generated.
        # For tsv_file_name=dev:
        #   ~5.5 GB data will be generated.
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
            # For tsv_file_name=train960:
            #   This takes ~45 minutes with nj=8, and xx GB data will be generated.
            # For tsv_file_name=dev:
            #   This takes ~45 minutes with nj=8, and xx GB data will be generated.
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
    tsv_file_name=train960
    # Fit a K-Means model with ${n_clusters} clusters on ${percent}% of the ${split} data
    # It took ~3 hours.
    ${train_cmd} --mem 100G "${logdir}"/learn_kmeans.log \
        ${python} simple_kmeans/learn_kmeans.py \
            "${feat_dir}" \
            ${tsv_file_name} \
            ${n_clusters} \
            --km_path "${km_path}" \
            --nshard ${nj} \
            --seed 0 \
            --percent ${percent} \
            --batch_size ${km_batch_size}
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Dump K-Means clustering labels on data"

    mkdir -p "${lab_dir}/log"
    # This will extract labels for the ${rank}-th shard out of ${nshard} shards
    # and dump them to ${lab_dir}/${tsv_file_name}_${rank}_${shard}.km
    # It took ~12 minutes with nj=8.
    for tsv_file_name in ${train_set} ${dev_set}; do
        ${decode_cmd} JOB=1:${nj} "${lab_dir}"/log/dump_km_label.JOB.log \
            ${python} simple_kmeans/dump_km_label.py \
                "${feat_dir}" \
                ${tsv_file_name} \
                "${km_path}" \
                --nshard ${nj} \
                --rank JOB \
                --lab_dir "${lab_dir}"
    done

    # Merge shards for ${tsv_file_name}
    for tsv_file_name in ${train_set} ${dev_set}; do
        for rank in $(seq 0 $((nj - 1))); do
            cat ${lab_dir}/${tsv_file_name}_${rank}_${nj}.km
        done > "${lab_dir}/${tsv_file_name}.km"
    done

    # generate the dict file (https://github.com/facebookresearch/fairseq/issues/2514)
    fairseq-preprocess --dataset-impl mmap --trainpref "${lab_dir}/${train_set}.km" --only-source --thresholdsrc 0
    # only keep the dict file
    mv data-bin/dict.txt "${lab_dir}/dict.km.txt"
    rm -r data-bin/

    # Convert km labels into a text scp file
    for tsv_file_name in ${train_set} ${dev_set}; do
        # move ${datadir}/${tsv_file_name}/ to new folders and rename ptext
        plabel_dir="${datadir}/${tsv_file_name}_${feature_type}_km${n_clusters}"
        if [[ -d "${plabel_dir}" ]]; then
            echo "${plabel_dir} already exists, will remove it"
            rm -r ${plabel_dir}
        fi
        mkdir -p ${plabel_dir}
        cp -r ${datadir}/${tsv_file_name}/* ${plabel_dir}

        paste -d " " <(tail -n +2 ${tsv_dir}/${tsv_file_name}.tsv | awk 'function basename(file) {sub(".*/", "", file); return file} {print basename($1)}' | sed -e 's/\(.flac\|.wav\)$//g') ${lab_dir}/${tsv_file_name}.km > ${plabel_dir}/text

        utils/fix_data_dir.sh "${plabel_dir}"
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Evaluate label purity"

    alignment_dir=LibriSpeech_phone_alignment
    # Download Librispeech alignment
    wget -c "https://zenodo.org/record/2619474/files/librispeech_alignments.zip?download=1" -O "librispeech_alignments.zip"
    unzip librispeech_alignments.zip -d "${alignment_dir}"

    ${python} local/prepare_librispeech_alignment.py \
        "${alignment_dir}/train-clean-100" "${alignment_dir}/train-clean-360" \
        --align_format ".TextGrid" \
        --unit_type "phn" \
        --with_timestamp False \
        --max_workers ${nj} \
        --max_chunksize 500 \
        --outfile "${alignment_dir}/train.tsv"

    for dset in dev test; do
        ${python} local/prepare_librispeech_alignment.py \
            "${alignment_dir}/${dset}-clean" \
            --align_format ".TextGrid" \
            --unit_type "phn" \
            --with_timestamp False \
            --max_workers ${nj} \
            --max_chunksize 500 \
            --outfile "${alignment_dir}/${dset}.tsv"
    done

    lab_name="km"
    ${python} measure_teacher_quality.py \
        "${tsv_dir}" \
        "${lab_dir}" \
        "${lab_name}" \
        --lab_sets ${dev_set} \
        --phn_dir "${alignment_dir}" \
        --phn_sets dev_clean dev_other \
        --pad_len 0 \
        --upsample 2 \
        --verbose
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
