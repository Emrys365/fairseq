import json
from collections import defaultdict
from itertools import chain
from pathlib import Path

from espnet2.utils.types import str2bool


def get_spk2utt(paths, audio_format="flac"):
    spk2utt = defaultdict(list)
    for path in paths:
        for audio in Path(path).rglob("*.{}".format(audio_format)):
            uid = audio.stem
            sid = uid[:3]
            spk2utt[sid].append((uid, str(audio)))

    return spk2utt


def get_spk2utt_wsj0_2mix(paths, audio_format="wav"):
    spk2utt = defaultdict(list)
    for path in paths:
        for audio in chain(
            Path(path).rglob("s1/*.{}".format(audio_format)),
            Path(path).rglob("s2/*.{}".format(audio_format)),
            Path(path).rglob("s3/*.{}".format(audio_format)),
        ):
            spk_idx = int(audio.parent.stem[1:]) - 1
            mix_uid = audio.stem
            uid = mix_uid.split("_")[::2][spk_idx]
            sid = uid[:3]
            spk2utt[sid].append((uid, str(audio)))

    return spk2utt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_paths",
        type=str,
        nargs="+",
        help="Paths to WSJ subsets",
    )
    parser.add_argument(
        "--is_wsj0_2mix",
        type=str2bool,
        default=False,
        help="Whether the provided audio_paths points to WSJ0-2mix data",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="spk2utt_tse.json",
        help="Path to the output spk2utt json file",
    )
    parser.add_argument("--audio_format", type=str, default="wav")
    args = parser.parse_args()

    if args.is_wsj0_2mix:
        # use clean sources from LibriMix as enrollment
        spk2utt = get_spk2utt_wsj0_2mix(args.audio_paths, audio_format=args.audio_format)
    else:
        # use Librispeech as enrollment
        spk2utt = get_spk2utt(args.audio_paths, audio_format=args.audio_format)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", encoding="utf-8") as f:
        json.dump(spk2utt, f)
