#!/usr/bin/env python
from itertools import chain
from pathlib import Path

import soundfile as sf
from tqdm.contrib.concurrent import process_map  # or thread_map


def prepare_tsv(
    root_path,
    audio_paths,
    audio_format=".wav",
    delim="\t",
    max_workers=8,
    chunksize=1000,
):
    root_path = Path(root_path)
    audios = list(
        chain(*[list(Path(p).rglob("*" + audio_format)) for p in audio_paths])
    )
    if len(audios) // max_workers < chunksize:
        chunksize = len(audios) // max_workers
    # List[Tuple(path, nsamples)]
    ret = process_map(worker, audios, max_workers=max_workers, chunksize=chunksize)
    # tsv file:
    # the first row containing the root directory of audios
    # and the rest rows listing each audio path
    print(str(root_path), flush=True)
    for path, nsamples in ret:
        rel_path = path.relative_to(root_path)
        print(f"{rel_path}{delim}{nsamples}", flush=True)


def worker(audio):
    with sf.SoundFile(audio) as f:
        nsamples = f.frames
    return audio, nsamples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", type=str, help="Root path of all `audio_paths`")
    parser.add_argument(
        "--audio_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing audio files",
    )
    parser.add_argument(
        "--delim", type=str, default="\t", help="Delimiter for separating columns"
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        default=".wav",
        help="Specify the audio format to search",
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

    prepare_tsv(
        args.root_path,
        args.audio_paths,
        audio_format=args.audio_format,
        delim=args.delim,
        max_workers=args.max_workers,
        chunksize=args.max_chunksize,
    )
