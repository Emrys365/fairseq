# --------------------------------------------------------
# TS-HuBERT
# --------------------------------------------------------
# Note: this implementation is modified based on
# https://github.com/microsoft/unilm/tree/master/wavlm
# and adpated to fit the fairseq module definition.
# --------------------------------------------------------

import logging
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass, field
import numpy as np
from omegaconf import II
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import GradMultiply
from fairseq.models.wavlm.encoder import (
    ConvFeatureExtractionModel,
    rel_positional_encoding,
    TransformerEncoder,
)
from fairseq.tasks.tshubert_pretraining import (
    TsHubertPretrainingConfig,
    TsHubertPretrainingTask,
)

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


@dataclass
class TsHubertConfig(FairseqDataclass):
    label_rate: int = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
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
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    #######################################################
    # new parameters (in addition to HuBERT) for TsHuBERT #
    #######################################################
    relative_position_embedding: bool = field(
        default=False, metadata={"help": "apply relative position embedding"}
    )
    num_buckets: int = field(
        default=320,
        metadata={"help": "number of buckets for relative position embedding"},
    )
    max_distance: int = field(
        default=1280,
        metadata={"help": "maximum distance for relative position embedding"},
    )
    gru_rel_pos: bool = field(
        default=False, metadata={"help": "apply gated relative position embedding"}
    )
    expand_attention_head_size: int = field(default=-1, metadata={"help": "not used"})

    # only used for cLN-based speaker adapter
    cln_layers: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "indexes of Transformer layers to apply conditional layer "
            "normalization to (only used when `spk_adapter_type` is `cln`)"
        },
    )


@register_model("tshubert", dataclass=TsHubertConfig)
class TsHubert(BaseFairseqModel):
    def __init__(
        self,
        cfg: TsHubertConfig,
        task_cfg: Optional[TsHubertPretrainingConfig] = None,
        dictionaries: List[Dictionary] = [None],
    ) -> None:
        super().__init__()
        logger.info(f"TS-HuBERT Config: {cfg}")
        is_finetuning_or_inference = any([d is None for d in dictionaries])

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        if not is_finetuning_or_inference:
            feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
            self.feat2tar_ratio = (
                cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
            )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        # modality embedding for distinguishing enrollment and input
        if self.post_extract_proj is not None:
            self.aux_post_extract_proj = torch.nn.Linear(
                self.embed, cfg.encoder_embed_dim
            )
        else:
            self.aux_post_extract_proj = None
        dim = cfg.encoder_embed_dim if self.post_extract_proj else self.embed
        self.aux_modality_emb = torch.nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.org_modality_emb = torch.nn.Parameter(torch.FloatTensor(dim).uniform_())

        # modality-specific positional encodings
        self.aux_pos = rel_positional_encoding(dim, cfg.conv_pos, cfg.conv_pos_groups)
        self.org_pos = rel_positional_encoding(dim, cfg.conv_pos, cfg.conv_pos_groups)

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        if not is_finetuning_or_inference:
            self.logit_temp = cfg.logit_temp
            self.skip_masked = cfg.skip_masked
            self.skip_nomask = cfg.skip_nomask

            final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        if not is_finetuning_or_inference:
            self.target_glu = None
            if cfg.target_glu:
                self.target_glu = nn.Sequential(
                    nn.Linear(final_dim, final_dim * 2), nn.GLU()
                )

            self.untie_final_proj = cfg.untie_final_proj
            if self.untie_final_proj:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, final_dim * len(dictionaries)
                )
            else:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if is_finetuning_or_inference:
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: TsHubertConfig, task: TsHubertPretrainingTask):
        """Build a new model instance."""

        if task is not None:
            model = TsHubert(cfg, task.cfg, task.dictionaries)
        else:
            model = TsHubert(cfg)
        return model

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        enrollment: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        cln_weights: Optional[List[Tuple[torch.Tensor]]] = None,
        cln_biases: Optional[List[Tuple[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        feats, feats_aux, feats_pen, target_list, padding_mask = self.forward_conv(
            source,
            enrollment=enrollment,
            target_list=target_list,
            padding_mask=padding_mask,
        )
        return self.forward_transformer(
            feats,
            feats_aux,
            feats_pen,
            target_list=target_list,
            padding_mask=padding_mask,
            mask=mask,
            features_only=features_only,
            output_layer=output_layer,
            cln_weights=cln_weights,
            cln_biases=cln_biases,
        )

    def forward_conv(
        self,
        source: torch.Tensor,
        enrollment: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # (B, T, D), float
        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)

        if enrollment is None:
            # only for fine-tuning without enrollment
            return features, None, features_pen, target_list, padding_mask

        features_aux = self.forward_features(enrollment).transpose(1, 2)
        features_aux = self.layer_norm(features_aux)
        if self.aux_post_extract_proj is not None:
            features_aux = self.aux_post_extract_proj(features_aux)
        # (B, T', D), float
        features_aux = self.dropout_input(features_aux)
        return features, features_aux, features_pen, target_list, padding_mask

    def forward_transformer(
        self,
        features: torch.Tensor,
        features_aux: torch.Tensor,
        features_pen: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        cln_weights: Optional[List[Tuple[torch.Tensor]]] = None,
        cln_biases: Optional[List[Tuple[torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None

        # (B, T', D), float
        x = self.org_pos(x.transpose(1, 2)).transpose(1, 2)
        if features_aux is None:
            # only for fine-tuning without enrollment
            features_concat = x + self.org_modality_emb
            padding_mask2 = padding_mask
        else:
            x_aux = self.aux_pos(features_aux.transpose(1, 2)).transpose(1, 2)
            # (B, T' + T, D), float
            features_concat = torch.cat(
                [x_aux + self.aux_modality_emb, x + self.org_modality_emb], dim=1
            )
            # (B, T' + T), bool
            if padding_mask is not None:
                padding_mask2 = nn.functional.pad(padding_mask, (x_aux.size(1), 0))
            else:
                padding_mask2 = None

        # target: (B, T), long
        # features_concat: (B, T' + T, D), float
        # padding_mask2: (B, T' + T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            features_concat,
            padding_mask=padding_mask2,
            layer=None if output_layer is None else output_layer - 1,
            cln_weights=cln_weights,
            cln_biases=cln_biases,
        )
        x = x[:, -features.size(1) :]

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": features,
                "features_aux": features_aux,
                "layer_results": layer_results,
            }

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        enrollment: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):
        res = self.forward(
            source,
            enrollment,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )

        feature = res["features"] if ret_conv else res["x"]
        if ret_layer_results:
            layer_results = [
                ll[0][-res["features"].size(1) :].transpose(0, 1)
                for ll in res["layer_results"]
            ]
            feature = (feature, layer_results)
        return feature, res["padding_mask"]

    def extract_features_spk(
        self,
        features: torch.Tensor,
        features_aux: torch.Tensor,
        features_pen: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
        cln_weights: Optional[List[Tuple[torch.Tensor]]] = None,
        cln_biases: Optional[List[Tuple[torch.Tensor]]] = None,
        **kwargs,
    ):
        res = self.forward_transformer(
            features,
            features_aux,
            features_pen,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
            cln_weights=cln_weights,
            cln_biases=cln_biases,
        )

        feature = res["features"] if ret_conv else res["x"]
        if ret_layer_results:
            layer_results = [
                ll[0][-res["features"].size(1) :].transpose(0, 1)
                for ll in res["layer_results"]
            ]
            feature = (feature, layer_results)
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
