from itertools import chain
from pathlib import Path

from espnet2.fileio.datadir_writer import DatadirWriter


def prepare_librimix_enroll(tsv, librimix_dir, map_mix2enroll, output_dir):
    # noqa E501: ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/egs/libri2mix/local/create_enrollment_csv_fixed.py
    mixtures = []
    prev_line = ""
    count = 0
    with Path(tsv).open("r", encoding="utf-8") as f:
        root = next(f).strip()
        for line in f:
            if not line.strip():
                continue
            if line == prev_line:
                count += 1
            else:
                count = 0
            mixtureID = Path(line.strip().split(maxsplit=1)[0]).stem
            mixtures.append((mixtureID, count))
            prev_line = line

    utt2path = {}
    for audio in chain(
        Path(librimix_dir).rglob("s1/*.wav"),
        Path(librimix_dir).rglob("s2/*.wav"),
        Path(librimix_dir).rglob("s3/*.wav"),
    ):
        pdir = audio.parent.stem
        utt2path[pdir + "/" + audio.stem] = str(audio.resolve())

    mix2enroll = {}
    with open(map_mix2enroll) as f:
        for line in f:
            mix_id, utt_id, enroll_id = line.strip().split()
            sid = mix_id.split("_").index(utt_id) + 1
            mix2enroll[mix_id, f"s{sid}"] = enroll_id

    with DatadirWriter(Path(output_dir)) as writer:
        for mixtureID, spk in mixtures:
            # 100-121669-0004_3180-138043-0053
            enroll_id = mix2enroll[mixtureID, f"s{spk + 1}"]
            uid = mixtureID.split("_")[spk]
            writer[f"enroll.scp"][mixtureID + f"_s{spk + 1}"] = utt2path[enroll_id]
            writer[f"uid.scp"][mixtureID + f"_s{spk + 1}"] = uid
            writer[f"sid.scp"][mixtureID + f"_s{spk + 1}"] = uid.split("-")[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wav_scp",
        type=str,
        help="Path to the wav.scp file",
    )
    parser.add_argument(
        "--librimix_dir",
        type=str,
        default=None,
        help="Path to the generated LibriMix directory. "
        "If `train` is False, this value is required.",
    )
    parser.add_argument(
        "--mix2enroll",
        type=str,
        default=None,
        help="Path to the downloaded map_mixture2enrollment file. "
        "If `train` is False, this value is required.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory for storing output files",
    )
    args = parser.parse_args()

    prepare_librimix_enroll(
        args.wav_scp, args.librimix_dir, args.mix2enroll, args.output_dir
    )
