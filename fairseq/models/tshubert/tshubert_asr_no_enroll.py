# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
import torch.nn as nn
from omegaconf import II, MISSING

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.tshubert.tshubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)

ACTIVATION_CHOICES = ChoiceEnum(["relu", "gelu", "gelu_accurate", "tanh", "linear"])
ENROLL_TYPE_CHOICES = ChoiceEnum(["none", "zero", "same"])


@dataclass
class TsHubertAsrNoEnrollConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING, metadata={"help": "path to TSHuBERT model"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside tshubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside tshubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside tshubert model"
        },
    )
    encoder_embed_dim: Optional[int] = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune tshubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in tshubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in tshubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    freeze_layers: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "indexes of Transformer layers to always freeze during fine-tuning",
        },
    )

    # weighted sum of TS-HuBERT features in different Transformer layers
    ssl_layer_selections: Optional[str] = field(
        default=None,
        metadata={
            "help": "To select a subset of hidden states from the given upstream by "
            "layer ids (0-index). "
            "If None (default), than all the layer of hidden states are selected"
        },
    )
    use_weighted_sum: bool = field(
        default=False,
        metadata={"help": "The method for fusing different features"},
    )

    # no enrollment based fine-tuning
    enroll_type: ENROLL_TYPE_CHOICES = field(
        default="none", metadata={"help": "Determines how to get the fake enrollment"}
    )

    # this holds the loaded tshubert args
    w2v_args: Any = None


@dataclass
class TsHubertCtcNoEnrollConfig(TsHubertAsrNoEnrollConfig):
    pass


@register_model("tshubert_ctc_no_enroll", dataclass=TsHubertCtcNoEnrollConfig)
class TsHubertCtcNoEnroll(BaseFairseqModel):
    def __init__(self, cfg: TsHubertCtcNoEnrollConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: TsHubertCtcNoEnrollConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = TsHubertNoEnrollEncoder(cfg, task)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][..., 0] = 0
            logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


class TsHubertNoEnrollEncoder(FairseqEncoder):
    def __init__(self, cfg: TsHubertAsrNoEnrollConfig, task):
        # torch.autograd.set_detect_anomaly(True)
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "freeze_layers": getattr(cfg, "freeze_layers", None),
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)

        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()
        # force overwriting in case `freeze_layers` is not stored in the checkpoint
        freeze_layers = getattr(cfg, "freeze_layers", None)
        self.freeze_layers = freeze_layers if freeze_layers else ()

        super().__init__(pretrain_task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if task.target_dictionary is not None and not cfg.get("autoregressive", False):
            self.proj = Linear(d, len(task.target_dictionary))
            self.out_dim = len(task.target_dictionary)
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
            self.out_dim = cfg.decoder_embed_dim
        else:
            self.proj = None
            self.out_dim = d

        self.enroll_type = getattr(cfg, "enroll_type", "none")
        logging.info("Using fake enrollment type: %s", self.enroll_type)

        self.use_weighted_sum = getattr(cfg, "use_weighted_sum", False)
        if self.use_weighted_sum:
            num_layers = len(self.w2v_model.encoder.layers)
            if getattr(cfg, "ssl_layer_selections", None) is None:
                self.layer_selections = list(range(num_layers))
            else:
                self.layer_selections = sorted(
                    map(int, cfg.ssl_layer_selections.split(","))
                )
                assert num_layers > self.layer_selections[-1]
            self.frontend_weights = nn.Parameter(torch.ones(len(self.layer_selections)))
        else:
            self.frontend_weights = None

    def init_linear_layer(self, dim_in, dim_out, weight_std=1e-03, bias_value=0):
        layer = nn.Linear(dim_in, dim_out)
        bias = torch.full_like(layer.bias, bias_value)
        weight = torch.normal(
            torch.zeros_like(layer.weight), torch.full_like(layer.weight, weight_std)
        )
        layer.bias = nn.Parameter(bias)
        layer.weight = nn.Parameter(weight)
        return layer

    def init_cat_linear_layer(self, dim_in, dim_out, weight_std=1e-03, bias_value=0):
        # identity mapping for the first sub-layer
        layer1 = nn.Linear(dim_out, dim_out, bias=False)
        layer1.weight = nn.Parameter(torch.eye(dim_out))
        # near-`bias_value` mapping for the second sub-layer
        layer2 = nn.Linear(dim_in, dim_out)
        bias2 = torch.full_like(layer2.bias, bias_value)
        weight2 = torch.normal(
            torch.zeros_like(layer2.weight), torch.full_like(layer2.weight, weight_std)
        )
        layer2.bias = nn.Parameter(bias2)
        layer2.weight = nn.Parameter(weight2)
        return layer1, layer2

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def freeze_w2v_model(self):
        for param in self.w2v_model.parameters():
            param.requires_grad = False

    def unfreeze_w2v_model(self):
        for param in self.w2v_model.parameters():
            param.requires_grad = True
        for i in self.freeze_layers:
            for param in self.w2v_model.encoder.layers[i].parameters():
                param.requires_grad = False

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        if self.enroll_type == "none":
            # discard the enrollment completely
            enroll = None
        elif self.enroll_type == "zero":
            # use a chunk of zeros as enrollment (max length is 48000)
            enroll = torch.zeros_like(source)[:, :48000]
        elif self.enroll_type == "same":
            # partially reuse the same input speech as enrollment (max length is 48000)
            if self.training:
                ilens = (~padding_mask).sum(dim=-1)
                l = min(ilens.min(), 48000)
                enroll = []
                for b, ilen in enumerate(ilens):
                    st = torch.randint(0, ilen - l + 1, (1,)).item()
                    enroll.append(source[b, st:st+l])
                enroll = torch.stack(enroll)
            else:
                enroll = source
        else:
            raise ValueError(f"Unknown enrollment type: {self.enroll_type}")

        if self.use_weighted_sum:
            w2v_args = {
                "source": source,
                "enrollment": enroll,
                "padding_mask": padding_mask,
                "mask": self.apply_mask and self.training,
                "output_layer": -1,
                "ret_layer_results": True,
            }
        else:
            w2v_args = {
                "source": source,
                "enrollment": enroll,
                "padding_mask": padding_mask,
                "mask": self.apply_mask and self.training,
            }

        ft = self.freeze_finetune_updates <= self.num_updates
        if ft:
            self.unfreeze_w2v_model()
        else:
            self.freeze_w2v_model()

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

        if self.use_weighted_sum:
            x = x[1][1:]  # drop the Transformer input data
            all_hs = [h for idx, h in enumerate(x) if idx in self.layer_selections]
            stacked_hs = torch.stack(all_hs, dim=0)
            _, *origin_shape = stacked_hs.shape
            stacked_hs = stacked_hs.view(len(self.layer_selections), -1)
            norm_weights = nn.functional.softmax(self.frontend_weights, dim=-1)
            weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
            x = weighted_hs.view(*origin_shape)

        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out["padding_mask"].index_select(
                0, new_order
            )
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
