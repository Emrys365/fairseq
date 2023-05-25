#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_finetuning_data_wsj0_2mix.sh
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
    [--wsj0_full_wav]: path to the root directory of WSJ0 data
    [--wsj0_2mix]: path to the root directory of WSJ0-2mix data
    [--sample_rate]: sample rate of the WSJ0-2mix data (in Hz)
    [--nj]: number of parallel workers
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

    mkdir -p "${datadir}/train_wsj0_2mix"
    for x in tr cv; do
        wget --continue -O ${datadir}/train_wsj0_2mix/${x}_mix.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/${x}/mix.scp
        wget --continue -O ${datadir}/train_wsj0_2mix/${x}_aux.scp https://raw.githubusercontent.com/gemengtju/SpEx_Plus/master/data/wsj0_2mix/${x}/aux.scp

        if [ "$x" = "tr" ]; then
            dset=train
        else
            dset=valid
        fi

        echo "${wsj0_2mix}/wav${sample_rate}/max/${x}/mix" > ${datadir}/train_wsj0_2mix/${dset}.tsv
        awk '{print $2}' ${datadir}/train_wsj0_2mix/${x}_mix.scp | \
            sed -e "s#/export/home/clx214/data/wsj0_2mix/min/${x}/mix/##g" >> ${datadir}/train_wsj0_2mix/${dset}.tsv
        ${python} local/add_nsamples_to_tsv.py \
            ${datadir}/train_wsj0_2mix/${dset}.tsv \
            --max_workers ${nj} \
            --max_chunksize 500

        awk 'BEGIN {count=1; prev="";} {
            if (NR != 1) {split($0, a, ".wav"); split(a[1], parts, "_"); if (prev == $0) {count=3} else {count=1} print(parts[count]); prev=$0;}
        }' "${datadir}/train_wsj0_2mix/${dset}.tsv" > "${datadir}/train_wsj0_2mix/${dset}.uid"
        awk '{print substr($1, 1, 3)}' "${datadir}/train_wsj0_2mix/${dset}.uid" > "${datadir}/train_wsj0_2mix/${dset}.sid"
    done

    awk '{if (NR != 1) {print "*"}}' "${datadir}/train_wsj0_2mix/train.tsv" > "${datadir}/train_wsj0_2mix/train.enroll"
    cp "${datadir}/train_wsj0_2mix/train.enroll" "${datadir}/train_wsj0_2mix/train.emb"
    paste "${datadir}/train_wsj0_2mix/train.enroll" "${datadir}/train_wsj0_2mix/train.emb" > "${datadir}/train_wsj0_2mix/train.enroll_emb"
    ${python} local/prepare_spk2enroll_wsj.py \
        "${wsj0_2mix}/wav${sample_rate}/max/tr" \
        --is_wsj0_2mix True \
        --outfile "${datadir}/train_wsj0_2mix/train.utt2enroll.json" \
        --audio_format wav

    awk '{print $2}' ${datadir}/train_wsj0_2mix/cv_aux.scp | \
        sed -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_tr_s_8k_all/#${wsj0_full_wav}/wsj0/si_tr_s/#g" \
            -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_dt_05_8k/#${wsj0_full_wav}/wsj0/si_dt_05/#g" \
            -e "s#/home/clx214/works/tfext_wsj/data/wsj/wsj0/si_et_05_8k/#${wsj0_full_wav}/wsj0/si_et_05/#g" > "${datadir}/train_wsj0_2mix/valid.enroll"

    ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        ${datadir}/train_wsj0_2mix/train.utt2enroll.json \
        --model_init local/spk_embs/model.th \
        --outdir spk_embs/out_wsj/tr \
        --cuda False
    cp spk_embs/out_wsj/tr/spk2emb.json ${datadir}/train_wsj0_2mix/train.utt2emb.json

    # create a hybrid json file containing both enrollment audio and embedding paths
    ${python} local/merge_spk2enroll_jsons.py \
        ${datadir}/train_wsj0_2mix/train.utt2enroll.json \
        ${datadir}/train_wsj0_2mix/train.utt2emb.json \
        --outfile ${datadir}/train_wsj0_2mix/train.utt2enroll_emb.json

    ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
        ${datadir}/train_wsj0_2mix/valid.enroll \
        --model_init local/spk_embs/model.th \
        --outdir spk_embs/out_wsj/cv \
        --cuda False
    cp spk_embs/out_wsj/cv/spk2emb.scp ${datadir}/train_wsj0_2mix/valid.emb
    paste ${datadir}/train_wsj0_2mix/valid.enroll ${datadir}/train_wsj0_2mix/valid.emb > ${datadir}/train_wsj0_2mix/valid.enroll_emb

    ${python} local/wsj0_2mix_labels.py \
        "${datadir}/train_wsj0_2mix/train.tsv" \
        --wsj0_root "$wsj0_root" \
        --output_dir "${datadir}/train_wsj0_2mix" \
        --output_name "train"

    ${python} local/wsj0_2mix_labels.py \
        "${datadir}/train_wsj0_2mix/valid.tsv" \
        --wsj0_root "$wsj0_root" \
        --output_dir "${datadir}/train_wsj0_2mix" \
        --output_name "valid"

    # generate the dict file (https://github.com/facebookresearch/fairseq/issues/2514)
    fairseq-preprocess --dataset-impl mmap --trainpref "${datadir}/train_wsj0_2mix/train.ltr" --only-source --thresholdsrc 0
    # only keep the dict file
    mv data-bin/dict.txt "${datadir}/train_wsj0_2mix/dict.ltr.txt"
    rm -r data-bin/
    rm ${datadir}/train_wsj0_2mix/{tr,cv}_{mix,aux}.scp

    # "${datadir}/
    # └── train_wsj0_2mix
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
