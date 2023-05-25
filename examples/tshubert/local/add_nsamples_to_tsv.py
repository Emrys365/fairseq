#!/usr/bin/env python
from pathlib import Path

import soundfile as sf
from tqdm.contrib.concurrent import process_map  # or thread_map


def update_tsv_inplace(tsv_path, delim="\t", max_workers=8, chunksize=1000):
    tsv_path = Path(tsv_path)
    with open(tsv_path, "r") as f:
        root_path = Path(f.readline().strip())
        audios = [root_path / line.split(delim)[0].strip() for line in f]
    if len(audios) // max_workers < chunksize:
        chunksize = len(audios) // max_workers
    # List[Tuple(path, nsamples)]
    ret = process_map(worker, audios, max_workers=max_workers, chunksize=chunksize)
    # tsv file:
    # the first row containing the root directory of audios
    # and the rest rows listing each audio path
    with open(tsv_path, "w") as f:
        f.write(f"{root_path}\n")
        for path, nsamples in ret:
            rel_path = path.relative_to(root_path)
            f.write(f"{rel_path}{delim}{nsamples}\n")


def worker(audio):
    with sf.SoundFile(audio) as f:
        nsamples = f.frames
    return audio, nsamples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tsv_path", type=str, help="Path to the tsv file containing audio paths"
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

    update_tsv_inplace(
        args.tsv_path,
        delim=args.delim,
        max_workers=args.max_workers,
        chunksize=args.max_chunksize,
    )
