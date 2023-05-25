#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_finetuning_data.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --datadir <datadir> --librispeech <librispeech> --librimix <librimix> --sample_rate <sample_rate> --nj <nj>
  optional argument:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to pthon binary
    [--datadir]: path the data root directory
    [--librispeech]: path to the root directory of Librispeech data
    [--librimix]: path to the root directory of Librimix data
    [--sample_rate]: sample rate of the LibriMix data (in Hz)
    [--nj]: number of parallel workers
EOF
)


stage=0
stop_stage=100

python=python3
datadir=data
# Path to the Librispeech directory
librispeech=
# Path to the LibriMix directory
librimix=
sample_rate=16k
nj=4


. utils/parse_options.sh


log "$0 $*"
if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "$librimix" ]; then
    echo "Please fill the value of librimix= manually"
    exit 1;
fi

if [ -z "$librispeech" ]; then
    echo "Please fill the value of librispeech= manually"
    exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Preparing data files"

	mkdir -p "${datadir}/train_2mix"
	${python} local/prepare_tsv_mix.py \
        "${librimix}/wav${sample_rate}/max/train-100/mix_both" \
        --audio_paths "${librimix}/wav${sample_rate}/max/train-100/mix_both" \
        --audio_format ".wav" \
        --max_workers ${nj} \
        --max_chunksize 500 | sort > "${datadir}/train_2mix/train.tsv"

	${python} local/prepare_tsv_mix.py \
        "${librimix}/wav${sample_rate}/max/dev/mix_both" \
        --audio_paths "${librimix}/wav${sample_rate}/max/dev/mix_both" \
        --audio_format ".wav" \
        --max_workers ${nj} \
        --max_chunksize 500 | sort > "${datadir}/train_2mix/valid.tsv"


	awk '{if (NR != 1) {print "*"}}' "${datadir}/train_2mix/train.tsv" > "${datadir}/train_2mix/train.enroll"
    cp "${datadir}/train_2mix/train.enroll" "${datadir}/train_2mix/train.emb"
    paste "${datadir}/train_2mix/train.enroll" "${datadir}/train_2mix/train.emb" > "${datadir}/train_2mix/train.enroll_emb"
	${python} local/prepare_spk2enroll_librispeech.py \
        "${librimix}/wav${sample_rate}/max/train-100" \
        --is_librimix True \
        --outfile ${datadir}/train_2mix/train.utt2enroll.json \
        --audio_format wav

	awk 'BEGIN {count=1; prev="";} {
		if (NR != 1) {split($0, a, "."); split(a[1], parts, "_"); if (prev == $0) {count=count+1} else {count=1} print(parts[count]); prev=$0;}
	}' "${datadir}/train_2mix/train.tsv" > "${datadir}/train_2mix/train.uid"
	awk -F "-" '{print $1}' "${datadir}/train_2mix/train.uid" > "${datadir}/train_2mix/train.sid"

	wget -O "${datadir}/train_2mix/valid_mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment
	${python} local/prepare_librimix_enroll.py \
		"${datadir}/train_2mix/valid.tsv" \
		--librimix_dir "${librimix}/wav${sample_rate}/max" \
		--mix2enroll "${datadir}/train_2mix/valid_mixture2enrollment" \
		--output_dir ${datadir}/train_2mix

	awk '{print $2}' "${datadir}/train_2mix/uid.scp" > "${datadir}/train_2mix/valid.uid"
	awk '{print $2}' "${datadir}/train_2mix/sid.scp" > "${datadir}/train_2mix/valid.sid"
	awk '{print $2}' "${datadir}/train_2mix/enroll.scp" > "${datadir}/train_2mix/valid.enroll"
	rm "${datadir}/train_2mix/uid.scp" "${datadir}/train_2mix/sid.scp" "${datadir}/train_2mix/enroll.scp"

    ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        ${datadir}/train_2mix/train.utt2enroll.json \
        --model_init spk_embs/model.th \
        --outdir spk_embs/out/train \
        --cuda False
    cp spk_embs/out/train/spk2emb.json ${datadir}/train_2mix/train.utt2emb.json

    # create a hybrid json file containing both enrollment audio and embedding paths
    ${python} local/merge_spk2enroll_jsons.py \
        ${datadir}/train_2mix/train.utt2enroll.json \
        ${datadir}/train_2mix/train.utt2emb.json \
        --outfile ${datadir}/train_2mix/train.utt2enroll_emb.json

    ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        ${datadir}/train_2mix/valid.enroll \
        --model_init spk_embs/model.th \
        --outdir spk_embs/out/dev \
        --cuda False
    cp spk_embs/out/dev/spk2emb.scp ${datadir}/train_2mix/valid.emb
    paste ${datadir}/train_2mix/valid.enroll ${datadir}/train_2mix/valid.emb > ${datadir}/train_2mix/valid.enroll_emb

	${python} local/librimix_labels.py \
		"${datadir}/train_2mix/train.tsv" \
		--librispeech-root "${librispeech}/train-clean-100" \
		--output-dir "${datadir}/train_2mix" --output-name "train"

	${python} local/librimix_labels.py \
		"${datadir}/train_2mix/valid.tsv" \
		--librispeech-root "${librispeech}/dev-clean" \
		--output-dir "${datadir}/train_2mix" --output-name "valid"

	# generate the dict file (https://github.com/facebookresearch/fairseq/issues/2514)
	fairseq-preprocess --dataset-impl mmap --trainpref "${datadir}/train_2mix/train.ltr" --only-source --thresholdsrc 0
	# only keep the dict file
	mv data-bin/dict.txt "${datadir}/train_2mix/dict.ltr.txt"
	rm -r data-bin/

	# "${datadir}/
	# └── train_2mix
	#     ├── dict.ltr.txt
	#     ├── train.utt2enroll.json / train.utt2emb.json
	#     ├── train.enroll / train.emb
	#     ├── train.ltr
	#     ├── train.tsv
	#     ├── train.uid
	#     ├── train.sid
	#     ├── train.wrd
	#     ├── valid.enroll / valid.emb
	#     ├── valid.ltr
	#     ├── valid.tsv
	#     ├── valid.uid
	#     ├── valid.sid
	#     └── valid.wrd
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
