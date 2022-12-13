#!/usr/bin/env python
from distutils.util import strtobool
from fractions import Fraction
from functools import partial
from itertools import chain
from multiprocessing import Manager
from pathlib import Path

import textgrid
from tqdm.contrib.concurrent import process_map


def int_or_float_or_numstr(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        assert 0 < value < 1, value
        return Fraction(value)
    elif isinstance(value, (str, Fraction)):
        num = Fraction(value)
        if num.denominator == 1:
            return num.numerator  # int
        else:
            return num
    else:
        raise TypeError("Unsupported value type: %s" % type(value))


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def prepare_alignment_tsv(
    alignment_paths,
    outfile,
    align_format=".TextGrid",
    delim="\t",
    unit_type="phn",
    with_timestamp=False,
    frame_level=False,
    frame_size=Fraction("0.032"),
    frame_shift=Fraction("0.02"),
    utt_samples_tsv="",
    fs=8000,
    max_workers=1,
    chunksize=1000,
):
    all_alignments = list(
        chain(*[list(Path(p).rglob("*" + align_format)) for p in alignment_paths])
    )
    name = "phones" if unit_type == "phn" else "words"
    if len(all_alignments) // max_workers < chunksize:
        chunksize = len(all_alignments) // max_workers

    utt2samples = {}
    if frame_level:
        assert Path(utt_samples_tsv).exists()
        with open(utt_samples_tsv, "r") as f:
            f.readline()    # ignore the first line
            for line in f:
                audio_path, nsamples = line.strip().split("\t")
                uid = Path(audio_path).stem
                utt2samples[uid] = int(nsamples)

    if frame_level:
        with_timestamp = False
    utt2samples = Manager().dict(utt2samples)
    utt2ali = dict(
        process_map(
            partial(
                worker,
                name=name,
                with_timestamp=with_timestamp,
                frame_level=frame_level,
                frame_size=frame_size,
                frame_shift=frame_shift,
                utt2samples=utt2samples,
                fs=fs,
            ),
            all_alignments,
            max_workers=max_workers,
            chunksize=chunksize,
        )
    )

    with Path(outfile).open("w") as f:
        for uid, ali in utt2ali.items():
            alignment = [",".join(seq) for seq in ali]
            if with_timestamp:
                alignment = ['"{}"'.format(s) for s in alignment]
            alignment = delim.join(alignment)
            f.write(f"{uid}{delim}{alignment}\n")


def worker(
    alignment,
    name="phn",
    with_timestamp=False,
    frame_level=False,
    frame_size=Fraction("0.032"),
    frame_shift=Fraction("0.02"),
    utt2samples=None,
    fs=8000,
):
    uid = alignment.stem
    tg = textgrid.TextGrid.fromFile(alignment)
    tier = None
    for interval_tier in tg.tiers:
        if interval_tier.name == name:
            tier = interval_tier
            break
    if tier is None:
        raise ValueError("No matched IntervalTier found for %s" % name)
    units = [intv.mark for intv in tier.intervals]
    if frame_level:
        frame_size = int(frame_size * fs)
        frame_shift = int(frame_shift * fs)
        assert frame_size >= frame_shift, (frame_size, frame_shift)
        frame_overlap = frame_size - frame_shift
        endsamples = [round(intv.maxTime * fs) for intv in tier.intervals]
        assert len(units) == len(endsamples), (len(units), len(endsamples))
        total_length = utt2samples[uid]
        total_frames = (total_length - frame_overlap) // frame_shift
        start = 0
        repeated_units = []
        for i, stop in enumerate(endsamples):
            num_frames = (stop - start - frame_overlap) // frame_shift
            start = start + num_frames * frame_shift + frame_overlap
            assert start <= stop, (start, stop)
            if stop - start > frame_size // 2:
                num_frames += 1
                start += frame_shift
            repeated_units.extend([units[i] for _ in range(num_frames)])
        units = repeated_units
        if len(units) < total_frames:
            pad_length = total_frames - len(units)
            units.extend([units[-1] for _ in range(pad_length)])
        else:
            assert len(units) == total_frames, (len(units), total_frames)
        return uid, (units,)
    elif with_timestamp:
        endtimes = [str(intv.maxTime) for intv in tier.intervals]
        return uid, (units, endtimes)
    else:
        return uid, (units,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "alignment_paths",
        type=str,
        nargs="+",
        help="Paths to the directories containing TextGrid files",
    )
    parser.add_argument(
        "--align_format",
        type=str,
        required=".TextGrid",
        help="Suffix of the alignment files",
    )
    parser.add_argument(
        "--unit_type",
        type=str,
        default="phn",
        choices=("phn", "wrd"),
        help="modeling unit (phone or word) for parsing the alignment information",
    )
    parser.add_argument(
        "--with_timestamp",
        type=str2bool,
        default=True,
        help="Output tsv file for storing alignment information",
    )

    parser.add_argument(
        "--frame_level",
        type=str2bool,
        default=False,
        help="Whether to obtain frame-level alignment; otherwise, utterance-level "
        "alignment will be stored. (This option will overwrite `with_timestamp` "
        "to False)",
    )
    parser.add_argument(
        "--frame_size",
        type=int_or_float_or_numstr,
        default=int_or_float_or_numstr("0.032"),
        help="Frame size in seconds (only used when `frame_level` is True)",
    )
    parser.add_argument(
        "--frame_shift",
        type=int_or_float_or_numstr,
        default=int_or_float_or_numstr("0.02"),
        help="Frame shift in seconds (only used when `frame_level` is True)",
    )
    parser.add_argument(
        "--utt_samples_tsv",
        type=str,
        default="",
        help="tsv file containing audio paths and the corresponding sample number "
        "(only used when `frame_level` is True)",
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=8000,
        help="Sampling rate (only used when `frame_level` is True)",
    )

    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Output tsv file for storing alignment information",
    )
    parser.add_argument(
        "--delim", type=str, default="\t", help="Delimiter for separating columns"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of workers to process audio files in parallel",
    )
    parser.add_argument(
        "--max_chunksize",
        type=int,
        default=1000,
        help="Maximum size of chunks sent to worker processes",
    )
    args = parser.parse_args()

    prepare_alignment_tsv(
        args.alignment_paths,
        args.outfile,
        unit_type=args.unit_type,
        with_timestamp=args.with_timestamp,
        frame_level=args.frame_level,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        fs=args.fs,
        utt_samples_tsv=args.utt_samples_tsv,
        delim=args.delim,
        max_workers=args.max_workers,
        chunksize=args.max_chunksize,
    )
