# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import II, MISSING, open_dict

from fairseq import checkpoint_utils, tasks, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
)
from fairseq.models.wavlm.wavlm import MASKING_DISTRIBUTION_CHOICES
from fairseq.models.wavlm.wavlm_asr import Embedding, Linear, TransformerDecoder
from fairseq.modules import (
    FairseqDropout,
    GradMultiply,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.tasks import FairseqTask

logger = logging.getLogger(__name__)

ACTIVATION_CHOICES = ChoiceEnum(["relu", "gelu", "gelu_accurate", "tanh", "linear"])
TSE_MODE_CHOICES = ChoiceEnum(["add", "cat", "film", "cln"])


@dataclass
class WavLMAsrMixConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING, metadata={"help": "path to wavlm model"})
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
        metadata={"help": "dropout probability inside wavlm model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wavlm model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wavlm model"
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
        metadata={"help": "dont finetune wavlm for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in wavlm to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in wavlm"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # speaker embedding related
    spk_emb_dim: int = field(
        default=256,
        metadata={"help": "Dimension of speaker embeddings"},
    )
    spk_adapter_type: TSE_MODE_CHOICES = field(
        default="add",
        metadata={"help": "Determine how the speaker embedding is fused"},
    )
    adapter_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply adapted feature var grads by this"},
    )
    cln_layers: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "indexes of Transformer layers to apply conditional layer "
            "normalization to (only used when `spk_adapter_type` is `cln`)"
        },
    )
    freeze_layers: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "indexes of Transformer layers to always freeze during fine-tuning",
        },
    )

    # weighted sum of wavLM features in different Transformer layers
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

    # this holds the loaded wavlm args
    w2v_args: Any = None


@dataclass
class WavLMCtcMixConfig(WavLMAsrMixConfig):
    pass


@register_model("wavlm_ctc_mix", dataclass=WavLMCtcMixConfig)
class WavLMCtcMix(BaseFairseqModel):
    def __init__(self, cfg: WavLMCtcMixConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: WavLMCtcMixConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = WavLMEncoder(cfg, task)
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


@dataclass
class WavLMSeq2SeqMixConfig(WavLMAsrMixConfig):
    # Encoder
    encoder_conv_kernel_sizes: str = field(
        default="5,5", metadata={"help": "kernel sizes of Conv1d subsampling layers"}
    )
    encoder_conv_channels: int = field(
        default=1024, metadata={"help": "# of channels in Conv1d subsampling layers"}
    )
    encoder_activation_fn: ACTIVATION_CHOICES = field(
        default="relu", metadata={"help": "activation function to use inside encoder"}
    )
    encoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside encoder"}
    )
    encoder_attention_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for attention weights inside encoder"},
    )
    encoder_activation_dropout: str = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN inside encoder"},
    )
    encoder_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=True, metadata={"help": "apply layernorm before each encoder block"}
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    encoder_freezing_updates: int = field(
        default=0, metadata={"help": "freeze encoder for first N updates"}
    )
    max_source_positions: int = field(
        default=6000, metadata={"help": "max source positions"}
    )

    # Decoder
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    autoregressive: bool = II("task.autoregressive")
    seq2seq_path: str = field(
        default="",
        metadata={"help": "reset_dict"},
    )
    reset_dict: bool = field(
        default=False,
        metadata={"help": "reset_dict"},
    )


@register_model("wavlm_seq2seq_mix", dataclass=WavLMSeq2SeqMixConfig)
class WavLMSeq2SeqMixModel(FairseqEncoderDecoderModel):
    def __init__(self, ssl_encoder, encoder, decoder):
        super().__init__(encoder, decoder)
        self.ssl_encoder = ssl_encoder

    @classmethod
    def build_model(cls, cfg: WavLMSeq2SeqMixConfig, task: FairseqTask):
        """Build a new model instance."""

        assert (
            cfg.autoregressive
        ), "Please set task.autoregressive=true for seq2seq asr models"

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)

        ssl_encoder = cls.build_ssl_encoder(cfg, task)
        encoder = cls.build_encoder(cfg, ssl_encoder.out_dim)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)

        model = WavLMSeq2SeqMixModel(ssl_encoder, encoder, decoder)

        if cfg["seq2seq_path"]:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.seq2seq_path)
            state = state["model"]
            if cfg["reset_dict"]:
                del state["decoder.embed_out"]
                del state["decoder.embed_tokens.weight"]
            model.load_state_dict(state, strict=False)
        return model

    @classmethod
    def build_ssl_encoder(cls, cfg: WavLMAsrMixConfig, task):
        return WavLMEncoder(cfg, task)

    @classmethod
    def build_encoder(cls, cfg: WavLMAsrMixConfig, input_dim: int):
        return MixTransformerEncoder(cfg, input_dim)

    @classmethod
    def build_decoder(cls, cfg: WavLMSeq2SeqMixConfig, tgt_dict, embed_tokens):
        return TransformerDecoder(cfg, tgt_dict, embed_tokens)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward_frontend_torchscript(self, kwargs):
        return self.ssl_encoder(**kwargs, tbc=False)

    def forward(self, **kwargs):
        ssl_encoder_out = self.forward_frontend_torchscript(kwargs)
        encoder_out = self.encoder(**ssl_encoder_out)
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args: Optional[Namespace] = None,
    ):
        if model_cfg.reset_dict:
            logger.warn("Overriding loading strict state dict!")
            del state_dict["decoder.embed_out"]
            del state_dict["decoder.embed_tokens.weight"]
            return super().load_state_dict(state_dict, False, model_cfg, args)
        return super().load_state_dict(state_dict, strict, model_cfg, args)


class WavLMEncoder(FairseqEncoder):
    def __init__(self, cfg: WavLMAsrMixConfig, task):
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
            "cln_layers": getattr(cfg, "cln_layers", None),
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
        # force overwriting in case `cln_layers` is not stored in the checkpoint
        cln_layers = getattr(cfg, "cln_layers", None)
        model.encoder.cln_layers = cln_layers if cln_layers else ()
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

        self.spk_adapter_type = cfg.spk_adapter_type
        self.spk_emb_dim = cfg.spk_emb_dim
        self.adapter_grad_mult = cfg.adapter_grad_mult
        if self.spk_adapter_type == "add":
            self.spk_adapter_proj = self.init_linear_layer(self.spk_emb_dim, d)
        elif self.spk_adapter_type == "cat":
            # implement CAT as summation of two linear sub-layer outputs
            self.spk_adapter_projs = self.init_cat_linear_layer(self.spk_emb_dim, d)
        elif self.spk_adapter_type == "film":
            self.spk_adapter_proj_scale = self.init_linear_layer(
                self.spk_emb_dim, d, bias_value=1
            )
            self.spk_adapter_proj_bias = self.init_linear_layer(
                self.spk_emb_dim, d, bias_value=0
            )
        elif self.spk_adapter_type == "cln":
            assert cfg.cln_layers is not None
            cln_layers = tuple(set(cfg.cln_layers))
            self.spk_adapter_projs_scale = nn.ModuleList()
            self.spk_adapter_projs_bias = nn.ModuleList()
            for _ in cln_layers:
                self.spk_adapter_projs_scale.append(
                    nn.ModuleList(
                        [
                            self.init_linear_layer(self.spk_emb_dim, d, bias_value=1),
                            self.init_linear_layer(self.spk_emb_dim, d, bias_value=1),
                        ]
                    )
                )
                self.spk_adapter_projs_bias.append(
                    nn.ModuleList(
                        [
                            self.init_linear_layer(self.spk_emb_dim, d, bias_value=0),
                            self.init_linear_layer(self.spk_emb_dim, d, bias_value=0),
                        ]
                    )
                )
        else:
            raise NotImplementedError(self.spk_adapter_type)

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

    def forward(self, source, enrollment, padding_mask, tbc=True, **kwargs):
        spk_emb = enrollment  # B x Dim
        ft = self.freeze_finetune_updates <= self.num_updates
        if ft:
            self.unfreeze_w2v_model()
        else:
            self.freeze_w2v_model()

        # with torch.no_grad() if not ft else contextlib.ExitStack():
        feat, feat_pen, _, padding_mask = self.w2v_model.forward_conv(
            source, padding_mask=padding_mask
        )
        if self.spk_adapter_type == "add":
            feat = feat + self.spk_adapter_proj(spk_emb).unsqueeze(1)
            if self.adapter_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.adapter_grad_mult).clone()
        elif self.spk_adapter_type == "cat":
            spk_emb = (
                self.spk_adapter_projs[1](spk_emb)
                .unsqueeze(1)
                .expand(-1, feat.size(1), -1)
            )
            feat = self.spk_adapter_projs[0](feat) + spk_emb
            if self.adapter_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.adapter_grad_mult).clone()
        elif self.spk_adapter_type == "film":
            gamma = self.spk_adapter_proj_scale(spk_emb).unsqueeze(1)
            beta = self.spk_adapter_proj_bias(spk_emb).unsqueeze(1)
            feat = gamma * feat + beta
            if self.adapter_grad_mult != 1.0:
                feat = GradMultiply.apply(feat, self.adapter_grad_mult).clone()
        elif self.spk_adapter_type == "cln":
            spk_cln_weights = [
                tuple(proj(spk_emb) for proj in projs)
                for projs in self.spk_adapter_projs_scale
            ]
            spk_cln_biases = [
                tuple(proj(spk_emb) for proj in projs)
                for projs in self.spk_adapter_projs_bias
            ]
            if self.adapter_grad_mult != 1.0:
                spk_cln_weights = [
                    tuple(GradMultiply.apply(w, self.adapter_grad_mult) for w in ws)
                    for ws in spk_cln_weights
                ]
                spk_cln_biases = [
                    tuple(GradMultiply.apply(b, self.adapter_grad_mult) for b in bs)
                    for bs in spk_cln_biases
                ]
        else:
            raise NotImplementedError(self.spk_adapter_type)

        if self.use_weighted_sum:
            w2v_args = {
                "features": feat,
                "features_pen": feat_pen,
                "padding_mask": padding_mask,
                "mask": self.apply_mask and self.training,
                "output_layer": -1,
                "ret_layer_results": True,
            }
        else:
            w2v_args = {
                "features": feat,
                "features_pen": feat_pen,
                "padding_mask": padding_mask,
                "mask": self.apply_mask and self.training,
            }
        if self.spk_adapter_type == "cln":
            w2v_args["cln_weights"] = spk_cln_weights
            w2v_args["cln_biases"] = spk_cln_biases
        x, padding_mask = self.w2v_model.extract_features_spk(**w2v_args)
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


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


class MixTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, cfg: WavLMAsrMixConfig, input_dim: int):
        super().__init__(None)

        self.encoder_freezing_updates = cfg.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=cfg.encoder_dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(cfg.encoder_embed_dim)
        if cfg.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            input_dim,
            cfg.encoder_conv_channels,
            cfg.encoder_embed_dim,
            [int(k) for k in cfg.encoder_conv_kernel_sizes.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            cfg.max_source_positions, cfg.encoder_embed_dim, self.padding_idx
        )

        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.encoder_dropout
            transformer_cfg.activation_fn = transformer_cfg.encoder_activation_fn
            transformer_cfg.attention_dropout = (
                transformer_cfg.encoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.encoder_activation_dropout
            )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg) for _ in range(cfg.encoder_layers)]
        )
        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(cfg.encoder_embed_dim)
        else:
            self.layer_norm = None

        if cfg.decoder_embed_dim != cfg.encoder_embed_dim:
            self.proj = Linear(cfg.encoder_embed_dim, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def _forward(self, encoder_out, encoder_padding_mask, padding_mask):
        encoder_out_lengths = (~encoder_padding_mask).sum(dim=-1)
        x, input_lengths = self.subsample(encoder_out, encoder_out_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
            "padding_mask": encoder_padding_mask,
        }

    def forward(self, encoder_out, encoder_padding_mask, padding_mask, **kwargs):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(encoder_out, encoder_padding_mask, padding_mask)
        else:
            x = self._forward(encoder_out, encoder_padding_mask, padding_mask)
        return x

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

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
