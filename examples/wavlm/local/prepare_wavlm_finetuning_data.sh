#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_wavlm_finetuning_data.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --datadir <datadir> --librispeech <librispeech> --librispeech_ft <librispeech_ft> --nj <nj> --tsv_dir <tsv_dir>
  optional argument:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to pthon binary
    [--datadir]: path the data root directory
    [--librispeech]: path to the root directory of Librispeech data
    [--librispeech_ft]: path to the output directory for storing Librispeech finetuning data
    [--nj]: number of parallel workers
    [--tsv_dir]: path to the output directory for storing tsv data files (default is "\${datadir}/tsv_files")
EOF
)


stage=0
stop_stage=100

python=python3
datadir=data
# Path to the Librispeech directory
librispeech=
# Path to the output directory
librispeech_ft=downloads
nj=4

tsv_dir=${datadir}/tsv_files


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


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Download Librispeech finetuning data (10h)"

    mkdir -p "${librispeech_ft}"
    if [ ! -e "${librispeech_ft}/librispeech_finetuning.tgz" ]; then
        wget https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz -O "${librispeech_ft}"
    else
        log "Skipping as the file already exists: ${librispeech_ft}/librispeech_finetuning.tgz"
    fi

	tar xfg "${librispeech_ft}/librispeech_finetuning.tgz" -C "${librispeech_ft}"
	# librispeech_finetuning (600 MB)
	# ├── 1h
	# │   ├── 0
	# │   │   ├── clean
	# │   │   └── other
	# │   ├── 1
	# │   ├── 2
	# │   ├── 3
	# │   ├── 4
	# │   └── 5
	# ├── 9h
	# │   ├── clean
	# │   └── other
	# └── phones
	# 	  ├── 10h_phones.txt
	# 	  ├── 10min_phones.txt
	# 	  ├── 1h_phones.txt
	# 	  ├── dev-clean.txt
	# 	  ├── dev-other.txt
	# 	  ├── phones_mapping.json
	# 	  ├── test-clean.txt
	# 	  └── test-other.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Preparing data files"

	mkdir -p "${datadir}/ls_clean_100"
	${python} local/prepare_tsv.py \
        "${librispeech_ft}/librispeech_finetuning" \
        --audio_paths "${librispeech}/train-clean-100" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 500 > "${datadir}/ls_clean_100/train.tsv"

	mkdir -p "${datadir}/ll_10h"
	${python} local/prepare_tsv.py \
        "${librispeech_ft}/librispeech_finetuning" \
        --audio_paths "${librispeech_ft}/librispeech_finetuning" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 500 > "${datadir}/ll_10h/train.tsv"

	mkdir -p "${datadir}/ll_1h"
	${python} local/prepare_tsv.py \
        "${librispeech_ft}/librispeech_finetuning" \
        --audio_paths "${librispeech_ft}/librispeech_finetuning/1h" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 500 > "${datadir}/ll_1h/train.tsv"

	mkdir -p "${datadir}/ll_10min"
	${python} local/prepare_tsv.py \
        "${librispeech_ft}/librispeech_finetuning" \
        --audio_paths "${librispeech_ft}/librispeech_finetuning/1h/0" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 500 > "${datadir}/ll_10min/train.tsv"

	mkdir -p "${tsv_dir}"
	${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/dev-clean" "${librispeech}/dev-other" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 500 > "${tsv_dir}/valid.tsv"

	for subset in ll_10h ll_1h ll_10min ls_clean_100; do
		${python} libri_labels.py \
			"${datadir}/${subset}/train.tsv" \
			--output-dir "${datadir}/${subset}" --output-name "train"

		cp "${tsv_dir}/valid.tsv" "${datadir}/${subset}"
		${python} libri_labels.py \
			"${tsv_dir}/valid.tsv" \
			--output-dir "${datadir}/${subset}" --output-name "valid"
	done

	for subset in ll_10h ll_1h ll_10min ls_clean_100; do
		# generate the dict file (https://github.com/facebookresearch/fairseq/issues/2514)
		fairseq-preprocess --dataset-impl mmap --trainpref "${datadir}/${subset}/train.ltr" --only-source --thresholdsrc 0
		# only keep the dict file
		mv data-bin/dict.txt "${datadir}/${subset}/dict.ltr.txt"
		rm -r data-bin/
	done

	# "${datadir}/
	# ├── ll_10h
	# │   ├── dict.ltr.txt
	# │   ├── train.ltr
	# │   ├── train.tsv
	# │   ├── train.wrd
	# │   ├── valid.ltr
	# │   ├── valid.tsv
	# │   └── valid.wrd
	# ├── ll_1h
	# ├── ll_10min
	# └── ls_clean_100
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
