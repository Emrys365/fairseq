import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset, TsHubertDataset, TsHubert2MixDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str, append_eos: bool = False) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=append_eos,
            add_if_not_exist=False,
        )

    def decode(self, label_tokens, include_eos: bool = False):
        return self.dictionary.string(label_tokens, include_eos=include_eos)


@dataclass
class TsHubertPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning TS-HuBERT"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )

    # TS-HuBERT specific fields
    noise_apply_prob: float = field(
        default=0.1,
        metadata={
            "help": "The probability applying Noise adding instead of speech mixing."
        },
    )
    noise_db_range: str = field(
        default="-5_20",
        metadata={"help": "The range of noise decibel level."},
    )
    interf_speech_db_range: str = field(
        default="-5_5",
        metadata={"help": "The range of noise decibel level."},
    )
    enrollment_max_length: Optional[int] = field(
        default=None, metadata={"help": "Maximum length of the enrollment audio."}
    )
    load_enrollment_and_emb: bool = field(
        default=False,
        metadata={"help": "Whether to load both enrollment and embedding."},
    )
    autoregressive: bool = field(
        default=False, metadata={"help": "used for Seq2Seq finetuning"}
    )
    max_keep_sample_size: Optional[int] = field(
        default=None, metadata={"help": "Maximum length of the input audio."}
    )
    fine_tuning_no_enroll: bool = field(
        default=False,
        metadata={"help": "set to true if fine-tuning TS-HuBERT without enrollment"},
    )


@register_task("tshubert_pretraining", dataclass=TsHubertPretrainingConfig)
class TsHubertPretrainingTask(FairseqTask):
    cfg: TsHubertPretrainingConfig

    def __init__(
        self,
        cfg: TsHubertPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"TsHubertPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning
        self.fine_tuning_no_enroll = getattr(cfg, "fine_tuning_no_enroll", False)

        if self.fine_tuning or self.fine_tuning_no_enroll:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self._source_dictionary = None

        self.blank_symbol = "<s>"

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return self._source_dictionary

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: TsHubertPretrainingConfig, **kwargs
    ) -> "TsHubertPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return (
            dictionaries[0]
            if self.fine_tuning or self.fine_tuning_no_enroll
            else dictionaries
        )

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        manifest_uid = f"{self.cfg.data}/{split}.uid"
        manifest_sid = f"{self.cfg.data}/{split}.sid"
        manifest_noise = f"{self.cfg.data}/{split}.noise.tsv"
        dicts = (
            [self.target_dictionary]
            if self.fine_tuning or self.fine_tuning_no_enroll
            else self.dictionaries
        )
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

        # hubert v1: pad_audio=True, random_crop=False;
        if self.fine_tuning:
            if self.cfg.load_enrollment_and_emb:
                manifest_enroll = f"{self.cfg.data}/{split}.enroll_emb"
                manifest_utt2enroll = f"{self.cfg.data}/{split}.utt2enroll_emb.json"
            else:
                manifest_enroll = f"{self.cfg.data}/{split}.enroll"
                manifest_utt2enroll = f"{self.cfg.data}/{split}.utt2enroll.json"
            self.datasets[split] = TsHubert2MixDataset(
                manifest,
                manifest_uid,
                manifest_sid,
                manifest_enroll,
                manifest_utt2enroll,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=getattr(self.cfg, "max_keep_sample_size", None),
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
                # TS-HuBERT specific arguments
                enrollment_max_length=self.cfg.enrollment_max_length,
                autoregressive=self.cfg.autoregressive,
            )
        elif self.fine_tuning_no_enroll:
            self.datasets[split] = HubertDataset(
                manifest,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=getattr(self.cfg, "max_keep_sample_size", None),
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
            )
        else:
            self.datasets[split] = TsHubertDataset(
                manifest,
                manifest_uid,
                manifest_sid,
                manifest_noise,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=getattr(self.cfg, "max_keep_sample_size", None),
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                single_target=self.cfg.single_target,
                # TS-HuBERT specific arguments
                noise_apply_prob=self.cfg.noise_apply_prob,
                noise_db_range=self.cfg.noise_db_range,
                interf_speech_db_range=self.cfg.interf_speech_db_range,
                enrollment_max_length=self.cfg.enrollment_max_length,
            )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
