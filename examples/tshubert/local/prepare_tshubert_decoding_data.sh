#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_decoding_data.sh
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

	mkdir -p "${datadir}/test_2mix"

	${python} local/prepare_tsv_mix.py \
        "${librimix}/wav${sample_rate}/max/test/mix_both" \
        --audio_paths "${librimix}/wav${sample_rate}/max/test/mix_both" \
        --audio_format ".wav" \
        --max_workers ${nj} \
        --max_chunksize 500 | sort > "${datadir}/test_2mix/test.tsv"

	wget -O "${datadir}/test_2mix/test_mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/test/map_mixture2enrollment
	${python} local/prepare_librimix_enroll.py \
		"${datadir}/test_2mix/test.tsv" \
		--librimix_dir "${librimix}/wav${sample_rate}/max" \
		--mix2enroll "${datadir}/test_2mix/test_mixture2enrollment" \
		--output_dir ${datadir}/test_2mix

	awk '{print $2}' "${datadir}/test_2mix/uid.scp" > "${datadir}/test_2mix/test.uid"
	awk '{print $2}' "${datadir}/test_2mix/sid.scp" > "${datadir}/test_2mix/test.sid"
	awk '{print $2}' "${datadir}/test_2mix/enroll.scp" > "${datadir}/test_2mix/test.enroll"
	rm "${datadir}/test_2mix/uid.scp" "${datadir}/test_2mix/sid.scp" "${datadir}/test_2mix/enroll.scp"

    ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        ${datadir}/test_2mix/test.enroll \
        --model_init spk_embs/model.th \
        --outdir spk_embs/out/test \
        --cuda False
    cp spk_embs/out/test/spk2emb.scp ${datadir}/test_2mix/test.emb
    paste ${datadir}/test_2mix/test.enroll ${datadir}/test_2mix/test.emb > ${datadir}/test_2mix/test.enroll_emb

	${python} local/librimix_labels.py \
		"${datadir}/test_2mix/test.tsv" \
		--librispeech-root "${librispeech}/test-clean" \
		--output-dir "${datadir}/test_2mix" --output-name "test"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Preparing n-gram language model for decoding"

    wget --continue -O "${datadir}/test_2mix/4-gram.arpa.gz" https://www.openslr.org/resources/11/4-gram.arpa.gz
    gunzip -c "${datadir}/test_2mix/4-gram.arpa.gz" > "${datadir}/test_2mix/4-gram.arpa"

    # wget --continue -O "${datadir}/test_2mix/librispeech-vocab.txt" https://www.openslr.org/resources/11/librispeech-vocab.txt
    # ${python} local/prepare_lexicon.py \
    #     "${datadir}/test_2mix/librispeech-vocab.txt" \
    #     --delim "|" \
    #     --outfile "${datadir}/test_2mix/librispeech_lexicon.lst"
    wget --continue -O "${datadir}/test_2mix/librispeech_lexicon.lst" https://dl.fbaipublicfiles.com/fairseq/wav2vec/librispeech_lexicon.lst
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Installing kenlm & flashlight-text"

    module load kenlm/kenlm

    git clone https://github.com/kpu/kenlm.git
    cd kenlm
    git checkout 716251e7cac9feebba1138639420089a73d008a5
    ${python} -m pip install .
    cd ..

    # Better to use a new GCC version such as 9.3.0
    # so that at least C++17 is used (std::optional has been added since then)
    git clone https://github.com/flashlight/text.git flashlight_text
    cd flashlight_text
    git checkout 8282dc71bb2da531b876557169121ddfaa52f35c
    ${python} -m pip install .
    cd ..
fi

# "${datadir}/
	# └── test_2mix
    #     ├── 4-gram.arpa
    #     ├── librispeech_lexicon.lst
	#     ├── test.enroll / test.emb
	#     ├── test.ltr
	#     ├── test.tsv
	#     ├── test.uid
	#     ├── test.sid
	#     └── test.wrd

log "Successfully finished. [elapsed=${SECONDS}s]"
