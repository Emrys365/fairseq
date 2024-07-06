from functools import partial
import numpy as np
import onnxruntime as ort
from pathlib import Path
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map  # or thread_map

from espnet2.fileio.npy_scp import NpyScpWriter


def compute_fbank(
    wav_path, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
):
    """Extract fbank.

    Simlilar to the one in wespeaker.dataset.processor,
    While integrating the wave reading and CMN.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat


def worker(uid_path, session, outdir, has_uid=True):
    outdir = Path(outdir).absolute()

    uid, wav_path = uid_path
    feats = compute_fbank(wav_path)
    feats = feats.unsqueeze(0).numpy()  # add batch dimension
    embeddings = session.run(output_names=["embs"], input_feed={"feats": feats})
    key, value = Path(wav_path).stem, np.squeeze(embeddings[0])
    p = str(outdir / f"{key}.npy")
    np.save(p, value)
    return f"{uid} {p}\n" if has_uid else f"{p}\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scp", type=str, help="scp file containing paths to utterances")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the pretrained model in ONNX format\n"
        "(e.g., https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md"
        "#model-list)",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to the output directory"
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

    # so = ort.SessionOptions()
    # so.inter_op_num_threads = 1
    # so.intra_op_num_threads = 1
    # session = ort.InferenceSession(args.onnx_path, sess_options=so)

    # writer = NpyScpWriter(args.outdir, f"{args.outdir}/embs.scp")
    # with open(args.scp, "r") as f:
    #     for line in tqdm(f):
    #         if not line.strip():
    #             continue
    #         uid, path = line.strip().split(maxsplit=1)
    #         feats = compute_fbank(path)
    #         feats = feats.unsqueeze(0).numpy()  # add batch dimension
    #         embeddings = session.run(output_names=["embs"], input_feed={"feats": feats})
    #         writer[uid] = np.squeeze(embeddings[0])
    # writer.close()

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(
        args.onnx_path, sess_options=so, #providers=["CUDAExecutionProvider"]
    )

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    tup = []
    has_uid = True
    if args.scp.endswith(".scp"):
        with open(args.scp, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                uid, path = line.strip().split(maxsplit=1)
                tup.append((uid, path))
    elif args.scp.endswith(".tsv"):
        has_uid = False
        with open(args.scp, "r") as f:
            root = Path(next(f).strip())
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                path = line.strip().split(maxsplit=1)[0]
                tup.append((i, str(root / path)))
    else:
        raise ValueError(f"Unsupported file format: {args.scp}")
    # List[str]
    ret = thread_map(
        partial(worker, session=session, outdir=args.outdir, has_uid=has_uid),
        tup,
        max_workers=args.max_workers,
        chunksize=args.max_chunksize,
    )
    with open(f"{args.outdir}/embs.scp", "w") as f:
        for line in ret:
            f.write(line)
