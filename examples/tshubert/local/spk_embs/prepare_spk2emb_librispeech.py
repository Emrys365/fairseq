from collections import defaultdict
from distutils.util import strtobool
from functools import partial
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.contrib.concurrent import process_map  # or thread_map

from resnet import ResNet34


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def worker(spk2utt_kv, dirpath, model_init, cuda=False):
    model = ResNet34(80, 256)
    state_dict = torch.load(model_init, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.eval()
    if cuda:
        model.cuda()

    spk2emb = []
    spkid, utts = spk2utt_kv
    for uid, path in utts:
        name = Path(path).stem
        folder = Path(path).parent.stem
        (dirpath / folder).mkdir(parents=True, exist_ok=True)
        outfile = Path(str(dirpath / folder / name) + ".npy").resolve()
        if not outfile.exists():
            wav, sr = sf.read(path)
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
            if cuda:
                wav = wav.cuda()
            with torch.no_grad():
                emb = model(wav)

            np.save(str(dirpath / folder / name), emb[0].cpu().numpy())
        spk2emb.append((uid, str(outfile)))

    return [spkid] + spk2emb


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("spk2utt_json", type=str, help="Path to the spk2utt.json file or spk2enroll.scp file")
    parser.add_argument(
        "--model_init", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--cuda", type=str2bool, default=False, help="Whether to run on GPU"
    )
    args = parser.parse_args()

    if args.spk2utt_json.endswith(".json"):
        with open(args.spk2utt_json, "r") as f:
            spk2utt = json.load(f)
    else:
        spk2utt = {}
        has_uid = None
        with open(args.spk2utt_json, "r") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if has_uid is None:
                    has_uid = len(line.strip().split()) > 1
                if has_uid:
                    uid, path = line.strip().split(maxsplit=1)
                else:
                    uid = i
                    path = line.strip()
                spk2utt[uid] = [(uid, path)]

    p = Path(f"{args.outdir}/spk_embs")
    p.mkdir(parents=True, exist_ok=True)
 
    key_values = list(spk2utt.items())
    ret = process_map(
        partial(worker, dirpath=p, model_init=args.model_init, cuda=args.cuda),
        key_values,
        chunksize=10,
        max_workers=4,
    )
    spk2emb = defaultdict(list)
    for lst in ret:
        spkid, *lst = lst
        spk2emb[spkid].extend(lst)

    if args.spk2utt_json.endswith(".json"):
        with open(f"{args.outdir}/spk2emb.json", "w") as f:
            json.dump(spk2emb, f)
    else:
        with open(f"{args.outdir}/spk2emb.scp", "w") as f:
            for sid, utts in spk2emb.items():
                for uid, path in utts:
                    if has_uid:
                        f.write(f"{uid} {path}\n")
                    else:
                        f.write(f"{path}\n")
