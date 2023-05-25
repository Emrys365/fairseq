#!/usr/bin/env python3

"""
Helper script to pre-compute embeddings for a flashlight (previously called wav2letter++) dataset
"""

import argparse
import re
from pathlib import Path


def find_transcripts(dot_files_list, scp):
    # ported from https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/local/find_transcripts.pl
    spk2dot = {}
    for dot in dot_files_list:
        match = re.fullmatch(r"(\w{6})00\.dot", dot.name)
        if not match:
            raise ValueError("Bad file name in dot file list: " + dot.name)
        spk = match.group(1)
        spk2dot[spk] = dot

    utt2trans_all = {}
    utt2trans = {}
    for uid in scp:
        match = re.fullmatch(r"(\w{6})\w\w", uid)
        if not match:
            raise ValueError("Bad utterance id in scp: " + uid)
        spk = match.group(1)
        dot = spk2dot[spk]
        if uid not in utt2trans_all:
            with dot.open("r") as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    match = re.fullmatch(r"(.+)\((\w{8})\)\s*", line.strip())
                    if not match:
                        raise ValueError(
                            f"Bad line in dot file {dot} (line {i}): {line}"
                        )
                    trans = match.group(1).rstrip()
                    utt = match.group(2).strip()
                    utt2trans_all[utt] = trans
        if uid not in utt2trans_all:
            raise ValueError(
                f"No transcript for utterance {uid} (current dot file is {dot})"
            )
        utt2trans[uid] = utt2trans_all[uid]
    return utt2trans


def normalize_transcript(utt2trans, noiseword="<NOISE>"):
    # ported from https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/local/normalize_transcript.pl
    for uid in utt2trans.keys():
        words = []
        for w in utt2trans[uid].split(" "):
            w = w.upper()  # Upcase everything to match the CMU dictionary.
            w = w.replace("\\", "")  # Remove backslashes.  We don't need the quoting.
            # Normalization for Nov'93 test transcripts.
            w = re.sub(r"^\%PERCENT$", "PERCENT", w)
            w = re.sub(r"^\.POINT$", "POINT", w)
            if (
                # E.g. [<door_slam], this means a door slammed in the preceding word.
                re.match(r"^\[\<\w+\]$", w)
                or
                # E.g. [door_slam>], this means a door slammed in the next word.
                re.match(r"^\[\w+\>\]$", w)
                or
                # E.g. [phone_ring/], which indicates the start of this phenomenon.
                re.match(r"\[\w+/\]$", w)
                or
                # E.g. [/phone_ring], which indicates the end of this phenomenon.
                re.match(r"\[/\w+\]$", w)
                or
                # This is used to indicate truncation of an utterance.  Not a word.
                w == "~"
                or
                # "." is used to indicate a pause.  Silence is optional anyway so not much
                # point including this in the transcript.
                w == "."
            ):
                continue  # we won't print this word.
            if re.match(r"\[\w+\]", w):
                # Other noises, e.g. [loud_breath], [lip_smack], [bad_recording], etc.
                words.append(noiseword)
            elif re.match(r"^\<([\w\']+)\>$", w):
                # e.g. replace <and> with and.  (the <> means verbal deletion of a word)
                # but it's pronounced.
                words.append(w)
            elif w == "--DASH":
                # This is a common issue; the CMU dictionary has it as -DASH.
                words.append("-DASH")
            # elif re.match(r"(.+)\-DASH", w):
            #     # E.g. INCORPORATED-DASH... seems the DASH gets combined with
            #     # previous word
            #     words.append(w)
            #     words.append("-DASH")
            else:
                words.append(w)
        utt2trans[uid] = " ".join(words)
    return utt2trans


def words2chars(word_list, nlsyms, sep="|"):
    char_list = []
    for word in word_list:
        if word in nlsyms:
            char_list.append(word)
        else:
            char_list.extend(list(word))
        char_list.append(sep)
    return char_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--wsj0_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--output_name", choices=["train", "valid", "test"], required=True
    )
    parser.add_argument("--nlsyms", nargs="+", default=["<*IN*>", "<*MR.*>", "<NOISE>"])
    args = parser.parse_args()

    wsj0_root = Path(args.wsj0_root)
    links = [f for f in wsj0_root.iterdir() if re.fullmatch("..-(.|..)\..", f.name)]
    # Do some basic checks that we have what we expected.
    if "11-13.1" not in [l.name for l in links]:
        raise FileNotFoundError("WSJ0 directory may be in a noncompatible form.")

    if args.output_name in ("train", "valid"):
        names_tup = ("si_tr_s",)
        disks_tup = ([f for f in links if f.name in ("11-1.1", "11-2.1", "11-3.1")],)
    elif args.output_name == "test":
        names_tup = ("si_dt_05", "si_et_05")
        disks_tup = (
            [f for f in links if f.name == "11-6.1"],
            [f for f in links if f.name == "11-14.1"],
        )
    else:
        raise ValueError("Invalid output name: " + args.output_name)

    # ported from https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/local/flist2scp.pl
    scp = {}
    for name, disks in zip(names_tup, disks_tup):
        for disk in disks:
            for spk in (disk / "wsj0" / name).iterdir():
                for audio in spk.glob("*.[wW][vV]1"):
                    uid = audio.stem.lower()
                    scp[uid] = audio

    dot_files_list = [dot for disk in links for dot in disk.rglob("*.dot")]
    utt2trans = find_transcripts(dot_files_list, scp)
    # Do some basic normalization steps.  At this point we don't remove OOVs--
    # that will be done inside the training scripts, as we'd like to make the
    # data-preparation stage independent of the specific lexicon used.
    utt2trans = normalize_transcript(utt2trans)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_line = ""
    count = 0
    with open(args.tsv, "r") as tsv, open(
        output_dir / (args.output_name + ".ltr"), "w"
    ) as ltr_out, open(output_dir / (args.output_name + ".wrd"), "w") as wrd_out:
        root = next(tsv).strip()
        for line in tsv:
            line = line.strip()
            if line == prev_line:
                count = 2
            else:
                count = 0
            mix_uid = Path(line).stem
            uid = mix_uid.split("_")[count]
            assert uid in utt2trans, uid

            print(utt2trans[uid], file=wrd_out)
            print(
                " ".join(
                    words2chars(utt2trans[uid].split(" "), nlsyms=args.nlsyms, sep="|")
                ),
                file=ltr_out,
            )
            prev_line = line


if __name__ == "__main__":
    main()
