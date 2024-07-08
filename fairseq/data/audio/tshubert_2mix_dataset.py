# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import itertools
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import kaldiio
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)


def load_audio(manifest_path, uid_path, sid_path, enroll_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    intervals = []
    to_skip = []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) in (2, 4), line
            if len(items) == 4:
                st, et = map(int, items[2:])
                sz = et - st
            else:
                sz = int(items[1])
                st, et = 0, sz
            if min_keep is not None and sz < min_keep:
                n_short += 1
                to_skip.append(ind)
            elif max_keep is not None and sz > max_keep:
                n_long += 1
                to_skip.append(ind)
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
                intervals.append((st, et))
    tot = ind + 1
    uids = []
    with open(uid_path, "r") as f:
        for i, line in enumerate(f):
            if i in to_skip:
                continue
            items = line.strip().split("\t")
            uids.append(items[0])
    assert i == ind, (i, ind, manifest_path, uid_path)
    sids = []
    with open(sid_path, "r") as f:
        for i, line in enumerate(f):
            if i in to_skip:
                continue
            items = line.strip().split("\t")
            sids.append(items[0])
    assert i == ind, (i, ind)
    enrolls = []
    with open(enroll_path, "r") as f:
        for i, line in enumerate(f):
            if i in to_skip:
                continue
            items = line.strip().split("\t")
            if len(items) == 1:
                enrolls.append(items[0])
            elif len(items) == 2:
                enrolls.append((items[0], items[1]))
            else:
                raise ValueError("Expected 1 or 2 items, but got %d" % len(items))
    assert i == ind, (i, ind)

    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, uids, sids, enrolls, inds, tot, sizes, intervals


def load_noise_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, (ind, line, root)
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, loaded {len(names)} "
            f"noise audios, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, tot, sizes


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


def read_file_from_tar(tar_path, offset, size):
    """Read a single file (specified by offset and size) from a tar file.

    Args:
        tar_path (str): Path to the tar file
        offset (int): Offset of the target file to read in the tar file
        size (int): Total size of the target file
    Returns:
        ret (BytesIO): Bytes of the target file
    """
    with open(tar_path, "rb") as tar:
        tar.seek(offset)
        buffer = tar.read(size)
    return io.BytesIO(buffer)


def read_audio_from_tar(path):
    # read audio from a tar file

    # format:
    # (1) /path/to/tar_file.tar:[offset]:[size]
    # (2) /path/to/tar_file.tar:[offset]:[size]#[start1]:[end1],[start2]:[end2],...
    assert ":" in path, path
    if "#" in path:
        # read audio with start and end indices
        tar_path_info, start_end = path.split("#")
        tar_path, offset, size = tar_path_info.split(":")
        start_end = sorted(
            [tuple(map(int, x.split(":"))) for x in start_end.split(",")],
            key=lambda x: x[0],
        )
    else:
        tar_path, offset, size = path.split(":")
        start_end = None
    offset = int(offset)
    size = int(size)
    with sf.SoundFile(read_file_from_tar(tar_path, offset, size)) as f:
        sr = f.samplerate
        if start_end is None:
            wav = f.read()
        else:
            lst = []
            for st, et in start_end:
                assert et > st, (st, et)
                f.seek(st)
                lst.append(f.read(et - st))
            wav = np.hstack(lst)
    return wav, sr


def read_audio_from_ark(path):
    # read audio from an ark file
    sr, wav = kaldiio.load_mat(path)
    return wav, sr


class TsHubert2MixDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        manifest_uid_path: str,
        manifest_sid_path: str,
        manifest_enroll_path: str,
        manifest_utt2enroll: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        # TS-HuBERT specific arguments
        enrollment_max_length: Optional[int] = None,
        autoregressive: bool = False,
    ):
        (
            audio_root,
            audio_names,
            uids,
            sids,
            enrolls,
            inds,
            tot,
            sizes,
            intervals,
        ) = load_audio(
            manifest_path,
            manifest_uid_path,
            manifest_sid_path,
            manifest_enroll_path,
            max_keep_sample_size,
            min_keep_sample_size,
        )
        self.audio_root = audio_root
        self.audio_names = audio_names
        if (
            "train" not in Path(manifest_utt2enroll).stem
            or not Path(manifest_utt2enroll).exists()
        ):
            self.audio_spk2enroll = {}
            self.has_enroll_and_emb = isinstance(enrolls[0], tuple)
        else:
            # dynamically sample the enrollment for training
            with open(manifest_utt2enroll, "r") as f:
                self.audio_spk2enroll = json.load(f)
            key0 = next(iter(self.audio_spk2enroll))
            # load enrollment audio and the corresponding embedding at the same time
            self.has_enroll_and_emb = (
                isinstance(self.audio_spk2enroll[key0][0], (tuple, list))
                and len(self.audio_spk2enroll[key0][0]) > 2
            )
        self.audio_uids = uids
        self.audio_sids = sids
        self.audio_enrolls = enrolls
        self.sizes = sizes
        self.intervals = intervals

        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"manifest_path={manifest_path}, "
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"autoregressive={autoregressive}, "
            f"has_enroll_and_emb={self.has_enroll_and_emb}"
        )

        if enrollment_max_length is None:
            self.enrollment_max_length = np.inf
        else:
            self.enrollment_max_length = enrollment_max_length

        self.autoregressive = autoregressive

    def get_audio(self, index):
        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        if wav_path.split(":")[0].endswith(".tar"): # audio in a tar file
            wav, cur_sample_rate = read_audio_from_tar(wav_path)
        elif re.match(r".*\.ark:\d+", wav_path):  # kaldi ark style audio path
            wav, cur_sample_rate = read_audio_from_ark(wav_path)
        else:
            wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_enroll(self, wav_path):
        if wav_path.endswith("npy"):
            wav = torch.from_numpy(np.load(wav_path)).float()
            norm = np.linalg.norm(wav, keepdims=True)
            if norm != 0:
                wav /= norm
        else:
            if wav_path.split(":")[0].endswith(".tar"): # audio in a tar file
                wav, cur_sample_rate = read_audio_from_tar(wav_path)
            elif re.match(r".*\.ark:\d+", wav_path):  # kaldi ark style audio path
                wav, cur_sample_rate = read_audio_from_ark(wav_path)
            else:
                wav, cur_sample_rate = sf.read(wav_path)

            wav = torch.from_numpy(wav).float()
            wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](
                label, append_eos=self.autoregressive
            )
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        if self.has_enroll_and_emb:
            enroll, emb = self.audio_enrolls[index]
        else:
            enroll = self.audio_enrolls[index]
        if enroll.startswith("*"):
            # choose enrollment randomly
            sid = self.audio_sids[index]
            idx = np.random.randint(0, len(self.audio_spk2enroll[sid]))
            if self.has_enroll_and_emb:
                uid, enroll, emb = self.audio_spk2enroll[sid][idx]
            else:
                uid, enroll = self.audio_spk2enroll[sid][idx]
            while uid == self.audio_uids[index] and len(self.audio_spk2enroll[sid]) > 1:
                idx = np.random.randint(0, len(self.audio_spk2enroll[sid]))
                if self.has_enroll_and_emb:
                    uid, enroll, emb = self.audio_spk2enroll[sid][idx]
                else:
                    uid, enroll = self.audio_spk2enroll[sid][idx]
        enroll = self.get_enroll(enroll)
        labels = self.get_labels(index)
        ret = {"id": index, "source": wav, "enrollment": enroll, "label_list": labels}
        if self.has_enroll_and_emb:
            ret["enrollment_emb"] = self.get_enroll(emb)
        return ret

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )
        enrolls = [s["enrollment"] for s in samples]
        collated_enrolls = self.collater_audio(
            enrolls, min(min([len(s) for s in enrolls]), self.enrollment_max_length)
        )[0]
        if self.has_enroll_and_emb:
            enroll_embs = [s["enrollment_emb"] for s in samples]
            collated_enroll_embs = self.collater_audio(
                enroll_embs,
                min(min([len(s) for s in enroll_embs]), self.enrollment_max_length),
            )[0]

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        if self.autoregressive:
            (
                targets_list,
                lengths_list,
                ntokens_list,
                prev_output_tokens_list,
            ) = self.collater_label_s2s(targets_by_label, audio_size, audio_starts)
        else:
            targets_list, lengths_list, ntokens_list = self.collater_label(
                targets_by_label, audio_size, audio_starts
            )

        net_input = {
            "source": collated_audios,
            "padding_mask": padding_mask,
            "enrollment": collated_enrolls,
        }
        if self.has_enroll_and_emb:
            net_input["enrollment_emb"] = collated_enroll_embs
        if self.autoregressive:
            if self.single_target:
                net_input["prev_output_tokens"] = prev_output_tokens_list[0]
            else:
                net_input["prev_output_tokens"] = prev_output_tokens_list
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad, eos=None
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, eos_idx=eos, left_pad=False
        )
        prev_output_tokens = data_utils.collate_tokens(
            targets,
            pad_idx=pad,
            eos_idx=eos,
            left_pad=False,
            move_eos_to_beginning=True,
        )
        return targets, lengths, ntokens, prev_output_tokens

    def collater_seq_label(self, targets, pad, eos=None):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, eos_idx=eos, left_pad=False
        )
        prev_output_tokens = data_utils.collate_tokens(
            targets,
            pad_idx=pad,
            eos_idx=eos,
            left_pad=False,
            move_eos_to_beginning=True,
        )
        return targets, lengths, ntokens, prev_output_tokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                targets, lengths, ntokens, _ = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens, _ = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def collater_label_s2s(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        prev_output_tokens_list = []
        itr = zip(targets_by_label, self.label_rates, self.pad_list, self.eos_list)
        for targets, label_rate, pad, eos in itr:
            if label_rate == -1:
                targets, lengths, ntokens, prev_output_tokens = self.collater_seq_label(
                    targets, pad, eos
                )
            else:
                targets, lengths, ntokens, prev_output_tokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad, eos
                )
            targets_list.append(targets.long())
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
            prev_output_tokens_list.append(prev_output_tokens.long())
        return targets_list, lengths_list, ntokens_list, prev_output_tokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
