#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_wavlm_decoding_data.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --datadir <datadir> --librispeech <librispeech> --nj <nj>
  optional argument:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to pthon binary
    [--datadir]: path the data root directory
    [--librispeech]: path to the root directory of Librispeech data
    [--nj]: number of parallel workers
EOF
)


stage=1
stop_stage=100

python=python3
datadir=data
# Path to the Librispeech directory
librispeech=
nj=4


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


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Preparing data files"

	mkdir -p "${datadir}/ls_dev_clean"
	${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/dev-clean" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 200 > "${datadir}/ls_dev_clean/test.tsv"

	mkdir -p "${datadir}/ls_dev_other"
	${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/dev-other" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 200 > "${datadir}/ls_dev_other/test.tsv"

	mkdir -p "${datadir}/ls_test_clean"
	${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/test-clean" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 200 > "${datadir}/ls_test_clean/test.tsv"

	mkdir -p "${datadir}/ls_test_other"
	${python} local/prepare_tsv.py \
        "${librispeech}" \
        --audio_paths "${librispeech}/test-other" \
        --audio_format ".flac" \
        --max_workers ${nj} \
        --max_chunksize 200 > "${datadir}/ls_test_other/test.tsv"

	for subset in ls_test_clean ls_test_other; do
		${python} libri_labels.py \
			"${datadir}/${subset}/test.tsv" \
			--output-dir "${datadir}/${subset}" --output-name "test"
	done

	# "${datadir}/
	# ├── ls_test_clean
	# │   ├── test.ltr
	# │   ├── test.tsv
	# │   └── test.wrd
	# ├── ls_test_other
	# ├── ls_dev_clean
	# └── ls_dev_other
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
