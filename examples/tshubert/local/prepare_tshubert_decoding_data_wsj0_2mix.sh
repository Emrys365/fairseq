#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_decoding_data_wsj0_2mix.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --stage <stage> --stop_stage <stop_stage> --python <python> --datadir <datadir> --librispeech <librispeech> --librimix <librimix> --sample_rate <sample_rate> --nj <nj> --wsj_data_dir <wsj_data_dir>
  optional argument:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to python binary
    [--datadir]: path the data root directory
    [--wsj0_full_wav]: path to the root directory of WSJ0 data
    [--wsj0_2mix]: path to the root directory of WSJ0-2mix data
    [--sample_rate]: sample rate of the WSJ0-2mix data (in Hz)
    [--nj]: number of parallel workers
    [--wsj_data_dir]: path to the data directory of the Kaldi-style WSJ recipe
EOF
)


stage=0
stop_stage=100

python=python3
datadir=data
# Path to the root directory containing WSJ0 audios (in wav format)
wsj0_full_wav=
# Path to the root directory containing original WSJ0 data
wsj0_root=
# Path to the generated WSJ0-2mix directory
wsj0_2mix=
sample_rate=16k
nj=4

# Path to the data directory of the Kaldi-style WSJ recipe
wsj_data_dir=


. utils/parse_options.sh


log "$0 $*"
if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "$wsj0_2mix" ]; then
    echo "Please fill the value of wsj0_2mix= manually"
    exit 1;
fi

if [ -z "$wsj0_full_wav" ]; then
    echo "Please fill the value of wsj0_full_wav= manually"
    exit 1;
fi

if [ -z "$wsj0_root" ]; then
    echo "Use wsj0_root=${wsj0_full_wav} since wsj0_root is not specified"
    wsj0_root="${wsj0_full_wav}"
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Preparing data files"

    mkdir -p "${datadir}/test_wsj0_2mix"
    wget --continue -O ${datadir}/test_wsj0_2mix/tt_mix.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/tt/mix.scp
    wget --continue -O ${datadir}/test_wsj0_2mix/tt_aux.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/tt/aux.scp

    echo "${wsj0_2mix}/wav${sample_rate}/max/tt/mix" > ${datadir}/test_wsj0_2mix/test.tsv
    awk '{print $2}' ${datadir}/test_wsj0_2mix/tt_mix.scp | \
        sed -e "s#/export/home/clx214/data/wsj0_2mix/min/tt/mix/##g" >> ${datadir}/test_wsj0_2mix/test.tsv
    ${python} local/add_nsamples_to_tsv.py \
        ${datadir}/test_wsj0_2mix/test.tsv \
        --max_workers ${nj} \
        --max_chunksize 500

    awk 'BEGIN {count=1; prev="";} {
        if (NR != 1) {split($0, a, ".wav"); split(a[1], parts, "_"); if (prev == $0) {count=3} else {count=1} print(parts[count]); prev=$0;}
    }' "${datadir}/test_wsj0_2mix/test.tsv" > "${datadir}/test_wsj0_2mix/test.uid"
    awk '{print substr($1, 1, 3)}' "${datadir}/test_wsj0_2mix/test.uid" > "${datadir}/test_wsj0_2mix/test.sid"

    awk '{print $2}' ${datadir}/test_wsj0_2mix/tt_aux.scp | \
        sed -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_tr_s_8k_all/#${wsj0_full_wav}/wsj0/si_tr_s/#g" \
            -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_dt_05_8k/#${wsj0_full_wav}/wsj0/si_dt_05/#g" \
            -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_et_05_8k/#${wsj0_full_wav}/wsj0/si_et_05/#g" > "${datadir}/test_wsj0_2mix/test.enroll"

    ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        ${datadir}/test_wsj0_2mix/test.enroll \
        --model_init local/spk_embs/model.th \
        --outdir spk_embs/out_wsj/tt \
        --cuda False
    cp spk_embs/out_wsj/tt/spk2emb.scp ${datadir}/test_wsj0_2mix/test.emb
    paste ${datadir}/test_wsj0_2mix/test.enroll ${datadir}/test_wsj0_2mix/test.emb > ${datadir}/test_wsj0_2mix/test.enroll_emb

    ${python} local/wsj0_2mix_labels.py \
        "${datadir}/test_wsj0_2mix/test.tsv" \
        --wsj0_root "$wsj0_root" \
        --output_dir "${datadir}/test_wsj0_2mix" \
        --output_name "test"

    rm ${datadir}/test_wsj0_2mix/tt_{mix,aux}.scp
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Installing kenlm & flashlight-text"

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


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Preparing n-gram language model for decoding"

    for lm_txt in "${wsj_data_dir}"/{train_si284/text,local/other_text/text}; do
        suffix=$(basename "${lm_txt}" | sed 's/text//')
        <${lm_txt} awk -v suffix=${suffix} ' { if( NF != 1 ) {$1=$1 suffix; print $0; }} '
    done | cut -f 2- -d " " > "${datadir}/test_wsj0_2mix/wsj_lm_train.txt"

    ngram_num=4
    <"${datadir}/test_wsj0_2mix/wsj_lm_train.txt" lmplz -S "20%" --discount_fallback -o ${ngram_num} - > "${datadir}/test_wsj0_2mix/${ngram_num}-gram.arpa"

    ${python} local/prepare_lexicon.py \
        "${datadir}/test_wsj0_2mix/wsj_lm_train.txt" \
        --delim "|" \
        --outfile "${datadir}/test_wsj0_2mix/wsj_lexicon.lst"
fi

# "${datadir}/
# └── test_wsj0_2mix
#     ├── 4-gram.arpa
#     ├── wsj_lexicon.lst
#     ├── test.enroll / test.emb
#     ├── test.ltr
#     ├── test.tsv
#     ├── test.uid
#     ├── test.sid
#     └── test.wrd

log "Successfully finished. [elapsed=${SECONDS}s]"
