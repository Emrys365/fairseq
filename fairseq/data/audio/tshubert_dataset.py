# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import itertools
import logging
import os
import re
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)


def load_audio(manifest_path, uid_path, sid_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    to_skip = []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
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
    tot = ind + 1
    uids = []
    with open(uid_path, "r") as f:
        for i, line in enumerate(f):
            if i in to_skip:
                continue
            items = line.strip().split("\t")
            uids.append(items[0])
    assert i == ind, (i, ind)
    sids = []
    with open(sid_path, "r") as f:
        for i, line in enumerate(f):
            if i in to_skip:
                continue
            items = line.strip().split("\t")
            sids.append(items[0])
    assert i == ind, (i, ind)
    spk2indexes = defaultdict(list)
    for i, sid in enumerate(sids):
        spk2indexes[sid].append(i)

    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, dict(spk2indexes), sids, inds, tot, sizes


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


class TsHubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        manifest_uid_path: str,
        manifest_sid_path: str,
        manifest_noise: str,
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
        noise_apply_prob: float = 0.1,
        noise_db_range: str = "-5_20",
        interf_speech_db_range: str = "-5_5",
        enrollment_max_length: Optional[int] = None,
    ):
        audio_root, audio_names, spk2indexes, sids, inds, tot, sizes = load_audio(
            manifest_path,
            manifest_uid_path,
            manifest_sid_path,
            max_keep_sample_size,
            min_keep_sample_size,
        )
        self.audio_root = audio_root
        self.audio_names = audio_names
        self.audio_spk2indexes = spk2indexes
        self.audio_sids = sids
        self.sizes = sizes

        if noise_apply_prob > 0:
            noise_root, noise_names, tot_noise, noise_sizes = load_noise_audio(
                manifest_noise, max_keep_sample_size, min_keep_sample_size
            )
            self.noise_root = noise_root
            self.noise_names = noise_names
            self.noise_sizes = noise_sizes

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
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}, "
            f"len(audio_spk2indexes)={len(self.audio_spk2indexes)}"
        )

        self.noise_apply_prob = noise_apply_prob
        sps = noise_db_range.strip().split("_")
        if len(sps) == 1:
            self.noise_db_low = self.noise_db_high = float(sps[0])
        elif len(sps) == 2:
            self.noise_db_low, self.noise_db_high = float(sps[0]), float(sps[1])
        else:
            raise ValueError("Format error: '{noise_db_range}' e.g. -3_4 -> [-3db,4db]")
        sps = interf_speech_db_range.strip().split("_")
        if len(sps) == 1:
            self.interf_speech_db_low = self.interf_speech_db_high = float(sps[0])
        elif len(sps) == 2:
            self.interf_speech_db_low = float(sps[0])
            self.interf_speech_db_high = float(sps[1])
        else:
            raise ValueError(
                "Format error: '{interf_speech_db_range}' e.g. -3_4 -> [-3db,4db]"
            )
        if enrollment_max_length is None:
            self.enrollment_max_length = np.inf
        else:
            self.enrollment_max_length = enrollment_max_length

    def get_audio(self, index):
        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        if re.match(r".*\.ark:\d+", wav_path):  # kaldi ark style audio path
            import kaldiio
            cur_sample_rate, wav = kaldiio.load_mat(wav_path)
        else:
            import soundfile as sf

            wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_noise_audio(self, index):
        wav_path = os.path.join(self.noise_root, self.noise_names[index])
        if re.match(r".*\.ark:\d+", wav_path):  # kaldi ark style audio path
            import kaldiio
            cur_sample_rate, wav = kaldiio.load_mat(wav_path)
        else:
            import soundfile as sf

            wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def adjust_noise_energy(self, wav, noise, target_snr):
        power = (wav**2).mean()
        noise_power = (noise**2).mean()
        scale = (
            10 ** (-target_snr / 20) * np.sqrt(power) / np.sqrt(max(noise_power, 1e-10))
        )
        return noise * scale

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        # choose enrollment
        idx = np.random.choice(self.audio_spk2indexes[self.audio_sids[index]])
        while idx == index and len(self.audio_spk2indexes[self.audio_sids[index]]) > 1:
            idx = np.random.choice(self.audio_spk2indexes[self.audio_sids[index]])
        enroll = self.get_audio(idx)
        # choose noise
        if self.noise_apply_prob >= np.random.random():
            noise = self.get_noise_audio(np.random.randint(0, len(self.noise_names)))
            noise_db = np.random.uniform(self.noise_db_low, self.noise_db_high)
        else:
            index_interf = np.random.randint(0, len(self.audio_names))
            while self.audio_sids[index_interf] == self.audio_sids[index]:
                index_interf = np.random.randint(0, len(self.audio_names))
            noise = self.get_audio(index_interf)
            noise_db = np.random.uniform(
                self.interf_speech_db_low, self.interf_speech_db_high
            )
        # adjust noise length and overlap position
        L = min(np.random.randint(1, len(wav)), len(noise))
        wav_start = np.random.randint(0, len(wav) - L + 1)
        noise_start = np.random.randint(0, len(noise) - L + 1)
        # adjust noise energy
        noise = self.adjust_noise_energy(wav, noise, noise_db)
        # mix with noise
        wav[wav_start : wav_start + L] += noise[noise_start : noise_start + L]

        labels = self.get_labels(index)
        return {"id": index, "source": wav, "enrollment": enroll, "label_list": labels}

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

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {
            "source": collated_audios,
            "padding_mask": padding_mask,
            "enrollment": collated_enrolls,
        }
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

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
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
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
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
