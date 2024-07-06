#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=local/prepare_tshubert_data_for_kmeans.sh
  #local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 --ckpt_path <ckpt_path> --librispeech <librispeech> --librimix <librimix>

  required arguments:
    [--ckpt_path]: path to the pretrained TS-HuBERT model
    [--librispeech]: path to the root directory of Librispeech data
    [--librimix]: path to the root directory of Librimix data

  optional arguments:
    [--stage]: start stage, default is 0
    [--stop_stage]: stop stage, default is 100
    [--python]: path to python binary
    [--datadir]: path the data root directory
    [--espnet_datadir]: path to the data directory for writing ESPnet-style data files

    [--nj]: number of parallel processes
    [--layer_index]: Index of the network layer for outputting features
    [--max_chunk]: maximum chunk size
    [--sample_rate]: sample rate of the data (in Hz)

    [--n_clusters]: number of clusters to form with K-Means
    [--kmrootdir]: output directory for storing the K-Means model
    [--percent]: percent of the data to train a K-Means model
      if it is -1, all data will be used.
    [--km_batch_size]: batch size when training a K-Means model

    [--src_case]: determine whether or not to deduplicate the discrete token sequence
        if src_case is "rm", do deduplicate;
        if src_case is "ts", do not deduplicate.
    [--token_type]: bpe or char
    [--nbpe]: number of BPE tokens
    [--bpemode]: unigram or bpe
    [--bpe_input_sentence_size]: Size of input sentence for BPE
    [--bpe_char_cover]: character coverage when modeling BPE
    [--nlsyms]: non-linguistic symbols list, separated by a comma

    [--oov]: Out of vocabulary symbol
    [--blank]: CTC blank symbol
    [--sos_eos]: sos and eos symbole
EOF
)


# This script is expected to be run after `local/prepare_tshubert_finetuning_data.sh`
# and `local/prepare_tshubert_decoding_data.sh`.

stage=0
stop_stage=100
python=python3
datadir=data
espnet_datadir=data

ckpt_path=
nj=8
layer_index=12   # the 12-th TransformerEncoder layer, index starts from 1
max_chunk=1600000
sample_rate=16k

n_clusters=4000
kmrootdir=kmeans_model
percent=0.2    # train-100 + train-360 subset: (56.4 * 2 + 205.2 * 2) * 0.2 = 104.6 hr
km_batch_size=10000

librimix=
librispeech=

src_case="rm"
token_type=bpe
nbpe=400
bpemode=unigram
bpe_input_sentence_size=100000000
bpe_char_cover=1.0
nlsyms=

oov="<unk>"
blank="<blank>"
sos_eos="<sos/eos>"

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


feat_dir="${kmrootdir}/feat_TSHuBERT_L${layer_index}"
km_path="${kmrootdir}/TSHuBERT_L${layer_index}_km_${n_clusters}clusters.mdl"


log "$0 $*"
if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1
fi
if [ -z "${ckpt_path}" ]; then
    log "Fill set --ckpt_path"
    exit 1
fi
if [ "$sample_rate" != "16k" ] && [ "$sample_rate" != "8k" ]; then
    log "Invalid sample rate"
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Preparing LibriMix dataset"

    ${python} local/prepare_tsv_mix.py \
        "${librimix}/wav${sample_rate}/max" \
        --audio_paths "${librimix}/wav${sample_rate}/max/train-100/mix_both" "${librimix}/wav${sample_rate}/max/train-360/mix_both" \
        --audio_format ".wav" \
        --max_workers ${nj} \
        --max_chunksize 500 | sort > "${datadir}/train_460_2mix/train.tsv"

    awk 'BEGIN {prev="";} {
        if (NR != 1) {tgt = (prev == $0) ? "/s2/" : "/s1/"; prev=$0; gsub(/\/mix_both\//, tgt, $0);}
        print($0);
    }' "${datadir}/train_460_2mix/train.tsv" > "${datadir}/train_460_2mix/train_ref.tsv"

    ${python} local/prepare_tsv_mix.py \
        "${librimix}/wav${sample_rate}/max/dev/mix_both" \
        --audio_paths "${librimix}/wav${sample_rate}/max/dev/mix_both" \
        --audio_format ".wav" \
        --max_workers ${nj} \
        --max_chunksize 500 | sort > "${datadir}/train_460_2mix/valid.tsv"

    awk '{if (NR != 1) {print "*"}}' "${datadir}/train_460_2mix/train.tsv" > "${datadir}/train_460_2mix/train.enroll"
    ${python} local/prepare_spk2enroll_librispeech.py \
        "${librimix}/wav${sample_rate}/max/train-100" "${librimix}/wav${sample_rate}/max/train-360" \
        --is_librimix True \
        --outfile ${datadir}/train_460_2mix/train.utt2enroll.json \
        --audio_format wav

    awk 'BEGIN {count=1; prev="";} {
        if (NR != 1) {split($0, a, "."); n=split(a[1], paths, "/"); split(paths[n], parts, "_"); if (prev == $0) {count=count+1} else {count=1} print(parts[count]); prev=$0;}
    }' "${datadir}/train_460_2mix/train.tsv" > "${datadir}/train_460_2mix/train.uid"
    awk -F "-" '{print $1}' "${datadir}/train_460_2mix/train.uid" > "${datadir}/train_460_2mix/train.sid"

    wget -O "${datadir}/train_460_2mix/valid_mixture2enrollment" https://raw.githubusercontent.com/BUTSpeechFIT/speakerbeam/main/egs/libri2mix/data/wav8k/min/dev/map_mixture2enrollment
	${python} local/prepare_librimix_enroll.py \
		"${datadir}/train_460_2mix/valid.tsv" \
		--librimix_dir "${librimix}/wav${sample_rate}/max" \
		--mix2enroll "${datadir}/train_460_2mix/valid_mixture2enrollment" \
		--output_dir ${datadir}/train_460_2mix

	awk '{print $2}' "${datadir}/train_460_2mix/uid.scp" > "${datadir}/train_460_2mix/valid.uid"
	awk '{print $2}' "${datadir}/train_460_2mix/sid.scp" > "${datadir}/train_460_2mix/valid.sid"
	awk '{print $2}' "${datadir}/train_460_2mix/enroll.scp" > "${datadir}/train_460_2mix/valid.enroll"
	rm "${datadir}/train_460_2mix/uid.scp" "${datadir}/train_460_2mix/sid.scp" "${datadir}/train_460_2mix/enroll.scp"

    # cp "${datadir}/train_460_2mix/train.enroll" "${datadir}/train_460_2mix/train.emb"
    # paste "${datadir}/train_460_2mix/train.enroll" "${datadir}/train_460_2mix/train.emb" > "${datadir}/train_460_2mix/train.enroll_emb"
    # ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
    #     ${datadir}/train_460_2mix/train.utt2enroll.json \
    #     --model_init spk_embs/model.th \
    #     --outdir spk_embs/out/train \
    #     --cuda False
    # cp spk_embs/out/train/spk2emb.json ${datadir}/train_460_2mix/train.utt2emb.json

    # # create a hybrid json file containing both enrollment audio and embedding paths
    # ${python} local/merge_spk2enroll_jsons.py \
    #     ${datadir}/train_460_2mix/train.utt2enroll.json \
    #     ${datadir}/train_460_2mix/train.utt2emb.json \
    #     --outfile ${datadir}/train_460_2mix/train.utt2enroll_emb.json

    # ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
    #     ${datadir}/train_460_2mix/valid.enroll \
    #     --model_init spk_embs/model.th \
    #     --outdir spk_embs/out/dev \
    #     --cuda False
    # cp spk_embs/out/dev/spk2emb.scp ${datadir}/train_460_2mix/valid.emb
    # paste ${datadir}/train_460_2mix/valid.enroll ${datadir}/train_460_2mix/valid.emb > ${datadir}/train_460_2mix/valid.enroll_emb

    # ${python} local/librimix_labels.py \
	# 	"${datadir}/train_460_2mix/valid.tsv" \
	# 	--librispeech-root "${librispeech}" \
	# 	--output-dir "${datadir}/train_460_2mix" --output-name "valid"

    ${python} local/librimix_labels.py \
        "${datadir}/train_460_2mix/train.tsv" \
        --librispeech-root "${librispeech}" \
        --output-dir "${datadir}/train_460_2mix" --output-name "train"

    fairseq-preprocess --dataset-impl mmap --trainpref "${datadir}/train_460_2mix/train.ltr" --only-source --thresholdsrc 0
    # only keep the dict file
    mv data-bin/dict.txt "${datadir}/train_460_2mix/dict.ltr.txt"
    rm -r data-bin/

	# "${datadir}/
	# └── train_460_2mix
	#     ├── dict.ltr.txt
	#     ├── train.utt2enroll.json / train.utt2emb.json
	#     ├── train.enroll / train.emb
	#     ├── train.ltr
	#     ├── train.tsv
	#     ├── train_ref.tsv
	#     ├── train.uid
	#     ├── train.sid
	#     ├── train.wrd
	#     ├── valid.enroll / valid.emb
	#     ├── valid.ltr
	#     ├── valid.tsv
	#     ├── valid.uid
	#     ├── valid.sid
	#     └── valid.wrd

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

    # ${python} local/spk_embs/prepare_spk2emb_librispeech.py \
    #     ${datadir}/test_2mix/test.enroll \
    #     --model_init spk_embs/model.th \
    #     --outdir spk_embs/out/test \
    #     --cuda False
    # cp spk_embs/out/test/spk2emb.scp ${datadir}/test_2mix/test.emb
    # paste ${datadir}/test_2mix/test.enroll ${datadir}/test_2mix/test.emb > ${datadir}/test_2mix/test.enroll_emb

	${python} local/librimix_labels.py \
		"${datadir}/test_2mix/test.tsv" \
		--librispeech-root "${librispeech}" \
		--output-dir "${datadir}/test_2mix" --output-name "test"

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

    subset_dir="${datadir}/train_460_2mix"
    ${python} - << EOF
import json
import random
from pathlib import Path


random.seed(0)
# with open("${subset_dir}/train.utt2enroll.json", "r") as f:
#     spk2enrolls = json.load(f)

with open("${subset_dir}/train_ref.tsv", "r") as tsv, open("${subset_dir}/train.uid", "r") as uid, open("${subset_dir}/train.sid", "r") as sid:
    root = next(tsv)
    lines = list(zip(enumerate(tsv), uid, sid))

random.shuffle(lines)
length = int(len(lines) * ${percent})
lines = sorted(lines[:length], key=lambda tup: tup[0][0])
with open("${subset_dir}/train_kmeans.tsv", "w") as tsv, open("${subset_dir}/train_kmeans.uid", "w") as uid, open("${subset_dir}/train_kmeans.sid", "w") as sid, open("${subset_dir}/train_kmeans.enroll", "w") as en:
    tsv.write(root)
    for (i, tsv_line), uid_line, sid_line in lines:
        tsv.write(tsv_line)
        uid.write(uid_line)
        sid.write(sid_line)
        utt, spk = uid_line.strip(), sid_line.strip()
        # enrollments = spk2enrolls[spk]
        # assert len(enrollments) > 1, (spk, enrollments)
        # enroll_uid, enroll = random.choice(enrollments)
        # while enroll_uid == utt:
        #     enroll_uid, enroll = random.choice(enrollments)
        enroll = Path(root.strip()) / tsv_line.strip().split("\t")[0]
        en.write(str(enroll) + "\n")
EOF
    rm "${datadir}/train_460_2mix/train_kmeans.uid" "${datadir}/train_460_2mix/train_kmeans.sid"
fi

if [ "$sample_rate" = "16k" ]; then
    sample_rate="16000"
elif [ "$sample_rate" = "8k" ]; then
    sample_rate="8000"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Dumping TS-HuBERT features for K-Means clustering"

    mkdir -p "${feat_dir}/log"
    # Features will be saved as ${feat_dir}/${train}_${rank}_${nshard}.{npy,len}.
    #   This took ~15 minutes, and ~54 GB data will be generated.
    ${cuda_cmd} --gpu 1 JOB=1:${nj} "${feat_dir}"/log/dump_tshubert_feature.JOB.log \
        ${python} simple_kmeans/tshubert_extract_kmeans_features.py \
            "${datadir}/train_460_2mix" \
            "train_kmeans" \
            --ckpt_path "${ckpt_path}" \
            --layer ${layer_index} \
            --nshard ${nj} \
            --rank JOB \
            --feat_dir "${feat_dir}" \
            --max_chunk ${max_chunk} \
            --device "gpu" \
            --seed 0 \
            --enroll_len 64000

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: K-Means clustering on extracted features"

    logdir="${kmrootdir}"/log
    mkdir -p "${logdir}"
    # Fit a K-Means model with ${n_clusters} clusters on ${percent}% of the ${split} data
    # It took ~1.5 hours.
    ${train_cmd} --mem 150G "${logdir}"/learn_kmeans.log \
        ${python} simple_kmeans/learn_kmeans.py \
            "${feat_dir}" \
            "train_kmeans" \
            ${n_clusters} \
            --km_path "${km_path}" \
            --nshard ${nj} \
            --seed 0 \
            --percent 1.0 \
            --batch_size ${km_batch_size}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Extracting K-Means labels"

    # Prepare train_ref data that enumerates all clean signals with themselves as enrollment
    awk 'BEGIN {prev=""; root="";} {
        if (NR == 1) {gsub(/\n/, "", $0); root=$0;} else {print(root "/" $1);}
    }' "${datadir}/train_460_2mix/train_ref.tsv" | sed -e 's#//#/#g' > "${datadir}/train_460_2mix/train_ref.enroll"
    # Prepare train_all data that enumerates all combinations of enrollments and mixtures
    python - << EOF
import json
from pathlib import Path

with open("${datadir}/train_460_2mix/train.utt2enroll.json", "r") as f:
    spk2enrolls = json.load(f)

prev = ""
with open("${datadir}/train_460_2mix/train.tsv", "r") as f, open("${datadir}/train_460_2mix/train_all.tsv", "w") as tsv, open("${datadir}/train_460_2mix/train_all.enroll", "w") as en:
    root = next(f)
    tsv.write(root)
    for line in f:
        path, nsamples = line.strip().split("\t")
        if prev != path:
            uid = Path(path).stem.split("_")[0]
        else:
            uid = Path(path).stem.split("_")[1]
        prev = path
        sid = uid.split("-")[0]
        for utt, wav in spk2enrolls[sid]:
            if utt == uid:
                continue
            tsv.write(line)
            en.write(wav + "\n")
EOF

    for dset in valid test train_ref train_all; do
        lab_dir="${kmrootdir}/${dset}_TSHuBERT_L${layer_index}_km_${n_clusters}clusters"
        if [ "${dset}" = "test" ]; then
            tsv_dir="${datadir}/test_2mix"
        else
            tsv_dir="${datadir}/train_460_2mix"
        fi

        # With nj=16,
        #  it took ~47 minutes to finish for dset=train_ref and ~419 MB data was generated
        # With nj=24,
        #  it took ~48 hours to finish for dset=train_all and ~47 GB data was generated
        ${cuda_cmd} --gpu 1 JOB=1:${nj} "${kmrootdir}"/log/extract_kmeans_label.JOB.log \
            ${python} simple_kmeans/tshubert_dump_km_label_from_audio.py \
                "${tsv_dir}" \
                ${dset} \
                "${km_path}" \
                --ckpt_path "${ckpt_path}" \
                --layer ${layer_index} \
                --nshard ${nj} \
                --rank JOB \
                --lab_dir "${lab_dir}" \
                --max_chunk ${max_chunk} \
                --device "gpu" \
                --seed 0
    done

    # Merge shards for ${tsv_file_name}
    for dset in valid test train_ref train_all; do
        lab_dir="${kmrootdir}/${dset}_TSHuBERT_L${layer_index}_km_${n_clusters}clusters"
        for rank in $(seq 0 $((nj - 1))); do
            cat "${lab_dir}/${dset}_${rank}_${nj}.km"
        done > "${lab_dir}/${dset}.km"
        rm "${lab_dir}"/${dset}_*_${nj}.km
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Preparing token_list and convert number indices to CJK tokens"

    ref_lab_dir="${kmrootdir}"/train_ref_TSHuBERT_L${layer_index}_km_${n_clusters}clusters
    # Get uniq chars
    if [ ! -f "${ref_lab_dir}"/distinct_cjk_token_lists ]; then
        if [ ${n_clusters} -ge 20900 ]; then
            log "Warning: too many clusters, be careful with the distinct token list."
        fi
        ${python} -c "for i in range(${n_clusters}): print(i, chr(int('4e00', 16) + i))" \
            > "${ref_lab_dir}"/distinct_cjk_token_lists
    fi

    if [ "${src_case}" = "ts" ]; then
        log "keep the original discrete token sequence"
        for dset in valid test train_ref train_all; do
            lab_dir="${kmrootdir}/${dset}_TSHuBERT_L${layer_index}_km_${n_clusters}clusters"
            if [ "$dset" = "test" ]; then
                out_dir="${datadir}/test_2mix"
            else
                out_dir="${datadir}/train_460_2mix"
            fi
            awk '
                (FILENAME==ARGV[1]) {a[$1]=$2}
                (FILENAME==ARGV[2]) {
                    out="";
                    for (i=1; i<=NF; i++) {
                        out=out""a[$i];
                    }
                    print(out);
                }' "${ref_lab_dir}"/distinct_cjk_token_lists \
                ${lab_dir}/${dset}.km \
                > "${out_dir}"/${dset}.${src_case}.text
        done
    elif [ "${src_case}" = "rm" ]; then
        log "remove repetitions in the discrete token sequence"
        for dset in valid test train_ref train_all; do
            lab_dir="${kmrootdir}/${dset}_TSHuBERT_L${layer_index}_km_${n_clusters}clusters"
            if [ "$dset" = "test" ]; then
                out_dir="${datadir}/test_2mix"
            else
                out_dir="${datadir}/train_460_2mix"
            fi
            awk '
                (FILENAME==ARGV[1]) {a[$1]=$2}
                (FILENAME==ARGV[2]) {
                    out="";
                    for (i=1; i<=NF; i++) {
                        if ($i != $(i-1)) {out=out""a[$i]}
                    }
                    print(out);
                }' "${ref_lab_dir}"/distinct_cjk_token_lists \
                ${lab_dir}/${dset}.km \
                > "${out_dir}"/${dset}.${src_case}.text
        done
    else
        log "Unrecognized src_case ${src_case}" && exit 1;
    fi
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Prepare tokenizer for discrete labels"

    if [ "$token_type" = "bpe" ]; then
        if ! ${python} -c 'import sentencepiece' &> /dev/null; then
            log "Please install sentencepiece first: pip install sentencepiece"
            exit 1
        fi

        tokenlist_dir="${datadir}/train_460_2mix/token_list/src_${token_type}_${bpemode}${nbpe}_${src_case}_tshubert_${layer_index}_km${n_clusters}"
        mkdir -p "${tokenlist_dir}"

        _opts_spm=${nlsyms:+ "--user_defined_symbols=$nlsyms"}
        ${python} - << EOF
import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    "--input=${datadir}/train_460_2mix/train_ref.${src_case}.text "
    "--vocab_size=${nbpe} "
    "--model_type=${bpemode} "
    "--model_prefix=${tokenlist_dir}/bpe "
    "--character_coverage=${bpe_char_cover} "
    "--input_sentence_size=${bpe_input_sentence_size} "
    "${_opts_spm}"
)
EOF

        {
        echo "${blank}"
        echo "${oov}"
        # Remove <unk>, <s>, </s> from the vocabulary
        <"${tokenlist_dir}/bpe.vocab" awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
        echo "${sos_eos}"
        } > "${tokenlist_dir}/tokens.txt"

    elif [ "$token_type" = "char" ]; then
        if ! ${python} -c 'import espnet' &> /dev/null; then
            log "Please install espnet first: pip install espnet"
            exit 1
        fi

        tokenlist_dir="${datadir}/train_460_2mix/token_list/char_tshubert_${layer_index}_km${n_clusters}"
        mkdir -p "${tokenlist_dir}"
        cat "${datadir}/train_460_2mix/train_ref.${src_case}.text" | tr '\t' ' ' > "${datadir}/train_460_2mix/train_ref.train.txt"

        # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
        # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
        ${python} -m espnet2.bin.tokenize_text  \
            --token_type "${token_type}" \
            --input "${datadir}/train_460_2mix/train_ref.train.txt" \
            --output "${tokenlist_dir}/src_tokens.txt" \
            --non_linguistic_symbols ${nlsyms_txt:-"none"} \
            --field 1- \
            --cleaner "none" \
            --g2p "none" \
            --write_vocabulary true \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"

        rm "${datadir}/train_460_2mix/train_ref.train.txt"

    else
        log "Error: not supported --token_type '${token_type}'"
        exit 1
    fi
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    log "Stage 6: Prepare ESPnet style data files"

    if [ "$token_type" = "bpe" ]; then
        suffix=src_${token_type}_${bpemode}${nbpe}_${src_case}_tshubert_${layer_index}_km${n_clusters}
        tokenlist_dir="${datadir}/train_460_2mix/token_list/${suffix}"
        mkdir -p ${espnet_datadir}/token_list/${suffix}
        cp "${tokenlist_dir}/tokens.txt" "${espnet_datadir}/token_list/${suffix}/"
        cp -r "${tokenlist_dir}"/bpe.* "${espnet_datadir}/token_list/${suffix}/"
    elif [ "$token_type" = "char" ]; then
        suffix=char_tshubert_${layer_index}_km${n_clusters}
        mkdir -p ${espnet_datadir}/token_list/${suffix}
        tokenlist_dir="${datadir}/train_460_2mix/token_list/${suffix}"
        cp "${tokenlist_dir}/src_tokens.txt" "${espnet_datadir}/token_list/${suffix}/"
    else
        log "Error: not supported --token_type '${token_type}'"
        exit 1
    fi
    for dset in valid test train_all; do
        mkdir -p ${espnet_datadir}/${dset}

        if [ "$dset" = "test" ]; then
                srcdir="${datadir}/test_2mix"
            else
                srcdir="${datadir}/train_460_2mix"
            fi
        ${python} - << EOF
from pathlib import Path

utt2text = {}
fname = "train" if "train" in "${dset}" else "${dset}"
with open(f"${srcdir}/{fname}.wrd", "r") as text, open(f"${srcdir}/{fname}.tsv", "r") as tsv:
    prev = ""
    root = next(tsv)
    for line, label in zip(tsv, text):
        suffix = "_s2" if line == prev else "_s1"
        prev = line
        path, nsamples = line.strip().split("\t")
        mix_uid = Path(path).stem
        utt2text[mix_uid + suffix] = label

    with open("${srcdir}/${dset}.tsv", "r") as tsv, \
        open("${srcdir}/${dset}.enroll", "r") as en, \
        open("${srcdir}/${dset}.${src_case}.text", "r") as text_in, \
        open("${espnet_datadir}/${dset}/text.${src_case}.tshubert_${layer_index}_km${n_clusters}", "w") as src, \
        open("${espnet_datadir}/${dset}/utt2spk", "w") as utt2spk, \
        open("${espnet_datadir}/${dset}/text.ts.en", "w") as tgt:
        root = next(tsv)
        for line, enroll, txt in zip(tsv, en, text_in):
            path, nsamples = line.strip().split("\t")
            enroll = Path(enroll.strip())
            label = label.strip()
            mix_uid = Path(path).stem
            mix_sid = "_".join([uid.split("-")[0] for uid in mix_uid.split("_")])
            spk = enroll.parent.stem
            label = utt2text[mix_uid + "_" + spk]
            if spk == "s1":
                enroll_uid = enroll.stem.split("_")[0]
            elif spk == "s2":
                enroll_uid = enroll.stem.split("_")[1]
            else:
                raise ValueError(f"{spk} must be either 'spk1' or 'spk2'")

            src.write(f"{mix_uid}__{enroll_uid} {txt}")
            utt2spk.write(f"{mix_uid}__{enroll_uid} {mix_sid}\n")
            tgt.write(f"{mix_uid}__{enroll_uid} {label}")
EOF
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
