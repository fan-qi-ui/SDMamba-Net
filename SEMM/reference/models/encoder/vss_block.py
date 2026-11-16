# -*- coding: utf-8 -*-
"""
Final Grad-CAM ablation-aware replacement for:
    MixerCSeg-main/models/encoder/vss_block.py

Confirmed from the uploaded current code:
- MixerCSeg build uses 4 VSS stages with depths=[1,1,1,1].
- Default CSA candidate_ratio remains 0.125.
- For this visualization experiment only, runtime overrides are supported via:
      MIXER_CSA_RATIO
      MIXER_CSA_MAX_TOKENS

The original four-step experiment remains available through:
    MIXER_VSS_VARIANT=baseline | fhe | fhe_csa

The paper ablation table can be selected with the composable settings:
    MIXER_SEMM_VARIANT=none | fhe | fhe_mamba | fhe_mamba_gl
    MIXER_DASR_VARIANT=none | ccsa | ccsa_deg

``fhe_mamba_gl`` is the complete SEMM, while ``ccsa_deg`` is the
complete DASR.  These variables are read when the model is constructed.
"""

import os
from timm.models.vision_transformer import trunc_normal_
import torch.nn as nn
import torch
from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from models.utils import LayerNorm2D, Stem, PatchMerging
from models.layers import (
    HoGEdgeGateConv,
    FrequencyHighEnhancement,
    CrackCandidateSparseAttention,
)
from VMamba.models.vmamba import TransMixer


def _get_csa_ratio(default=0.125):
    value = float(
        os.environ.get(
            "MIXER_CSA_RATIO",
            str(default),
        )
    )

    if not (0.0 < value <= 1.0):
        raise ValueError(
            f"MIXER_CSA_RATIO must be in (0, 1], got {value}"
        )

    return value


def _get_csa_max_tokens(default=1024):
    value = int(
        os.environ.get(
            "MIXER_CSA_MAX_TOKENS",
            str(default),
        )
    )

    if value < 1:
        raise ValueError(
            f"MIXER_CSA_MAX_TOKENS must be >= 1, got {value}"
        )

    return value


def _get_variant():
    value = os.environ.get("MIXER_VSS_VARIANT", "fhe_csa").strip().lower()
    aliases = {
        "baseline": "baseline",
        "base": "baseline",
        "original": "baseline",
        "fhe": "fhe",
        "fhe_csa": "fhe_csa",
        "fhe+csa": "fhe_csa",
        "full": "fhe_csa",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown MIXER_VSS_VARIANT={value!r}. "
            "Use baseline / fhe / fhe_csa."
        )
    return aliases[value]


def _get_named_variant(name, default, aliases):
    value = os.environ.get(name, default).strip().lower()
    if value not in aliases:
        allowed = " / ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"Unknown {name}={value!r}. Use {allowed}."
        )
    return aliases[value]


def _get_ablation_variants():
    """Resolve the composable SEMM/DASR ablation configuration.

    If either new variable is present, the paper-style composable mode is
    used.  Otherwise the historical MIXER_VSS_VARIANT behavior is preserved.
    """
    composable = (
        "MIXER_SEMM_VARIANT" in os.environ
        or "MIXER_DASR_VARIANT" in os.environ
    )
    if not composable:
        return None, None

    semm = _get_named_variant(
        "MIXER_SEMM_VARIANT",
        "fhe_mamba_gl",
        {
            "none": "none",
            "off": "none",
            "fhe": "fhe",
            "fhe_mamba": "fhe_mamba",
            "fhe+mamba": "fhe_mamba",
            "fhe_mamba_gl": "fhe_mamba_gl",
            "fhe+mamba+global_local": "fhe_mamba_gl",
            "full": "fhe_mamba_gl",
        },
    )
    dasr = _get_named_variant(
        "MIXER_DASR_VARIANT",
        "ccsa_deg",
        {
            "none": "none",
            "off": "none",
            "ccsa": "ccsa",
            "ccsa_deg": "ccsa_deg",
            "ccsa+deg": "ccsa_deg",
            "full": "ccsa_deg",
        },
    )
    return semm, dasr


class VSS(nn.Module):
    def __init__(
        self,
        in_dim,
        depth,
        mlp_ratio=4.0,
        state_dim=64,
        nbins=180,
        candidate_ratio=0.125,
        max_tokens=1024,
    ):
        super().__init__()
        self.depth = depth
        self.variant = _get_variant()
        self.semm_variant, self.dasr_variant = _get_ablation_variants()

        # Read at model-construction time so different ablation variants can
        # use different inference-time CSA sparsity in the same Python process.
        self.csa_ratio = _get_csa_ratio(
            candidate_ratio
        )
        self.csa_max_tokens = _get_csa_max_tokens(
            max_tokens
        )

        self.blocks = nn.ModuleList()

        for _ in range(depth):
            if self.semm_variant is not None:
                modules = []

                if self.semm_variant in {
                    "fhe",
                    "fhe_mamba",
                    "fhe_mamba_gl",
                }:
                    modules.append(
                        FrequencyHighEnhancement(
                            dim=in_dim,
                            cutoff=0.18,
                            temperature=18.0,
                        )
                    )

                if self.semm_variant in {
                    "fhe_mamba",
                    "fhe_mamba_gl",
                }:
                    modules.append(
                        TransMixer(
                            hidden_dim=in_dim,
                            ssm_d_state=state_dim,
                            mlp_ratio=mlp_ratio,
                            channel_first=True,
                            use_global_local=(
                                self.semm_variant == "fhe_mamba_gl"
                            ),
                        )
                    )

                if self.dasr_variant in {"ccsa", "ccsa_deg"}:
                    modules.append(
                        CrackCandidateSparseAttention(
                            dim=in_dim,
                            candidate_ratio=self.csa_ratio,
                            min_tokens=64,
                            max_tokens=self.csa_max_tokens,
                            max_heads=4,
                        )
                    )

                if self.dasr_variant == "ccsa_deg":
                    modules.append(
                        HoGEdgeGateConv(
                            in_dim=in_dim,
                            nbins=nbins,
                        )
                    )

                if not modules:
                    raise ValueError(
                        "The composable ablation must keep at least one "
                        "SEMM or DASR component."
                    )

            elif self.variant == "baseline":
                modules = [
                    TransMixer(
                        hidden_dim=in_dim,
                        ssm_d_state=state_dim,
                        mlp_ratio=mlp_ratio,
                        channel_first=True,
                    ),
                    HoGEdgeGateConv(
                        in_dim=in_dim,
                        nbins=nbins,
                    ),
                ]

            elif self.variant == "fhe":
                modules = [
                    FrequencyHighEnhancement(
                        dim=in_dim,
                        cutoff=0.18,
                        temperature=18.0,
                    ),
                    TransMixer(
                        hidden_dim=in_dim,
                        ssm_d_state=state_dim,
                        mlp_ratio=mlp_ratio,
                        channel_first=True,
                    ),
                    HoGEdgeGateConv(
                        in_dim=in_dim,
                        nbins=nbins,
                    ),
                ]

            elif self.variant == "fhe_csa":
                modules = [
                    FrequencyHighEnhancement(
                        dim=in_dim,
                        cutoff=0.18,
                        temperature=18.0,
                    ),
                    TransMixer(
                        hidden_dim=in_dim,
                        ssm_d_state=state_dim,
                        mlp_ratio=mlp_ratio,
                        channel_first=True,
                    ),
                    CrackCandidateSparseAttention(
                        dim=in_dim,
                        candidate_ratio=self.csa_ratio,
                        min_tokens=64,
                        max_tokens=self.csa_max_tokens,
                        max_heads=4,
                    ),
                    HoGEdgeGateConv(
                        in_dim=in_dim,
                        nbins=nbins,
                    ),
                ]

            else:
                raise AssertionError(self.variant)

            self.blocks.append(
                nn.Sequential(*modules)
            )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class VSSEncoder(nn.Module):
    def __init__(
        self,
        in_dim=3,
        embed_dim=[128, 256, 512],
        depths=[2, 2, 2],
        mlp_ratio=4.0,
        state_dim=[49, 25, 9],
        distillation=False,
        is_patch_embed=True,
        nbins=180,
        candidate_ratio=0.125,
        max_tokens=1024,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.distillation = distillation

        if is_patch_embed:
            self.patch_embed = Stem(
                in_dim=in_dim,
                dim=embed_dim[0],
            )
        self.is_patch_embed = is_patch_embed

        self.vss_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_layer in range(self.num_layers):
            vss = VSS(
                in_dim=int(embed_dim[i_layer]),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                state_dim=state_dim[i_layer],
                nbins=nbins,
                candidate_ratio=candidate_ratio,
                max_tokens=max_tokens,
            )
            self.vss_layers.append(vss)

            if i_layer < self.num_layers - 1:
                self.downsamples.append(
                    PatchMerging(
                        in_dim=int(embed_dim[i_layer]),
                        out_dim=int(embed_dim[i_layer + 1]),
                    )
                )

        self.apply(self._init_weights)
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        logger = get_root_logger()

        if self.init_cfg is None and pretrained is None:
            logger.warn(
                f"No pre-trained weights for {self.__class__.__name__}, "
                "training start from scratch"
            )
        else:
            assert "checkpoint" in self.init_cfg, (
                f"Only support specify `Pretrained` in init_cfg "
                f"in {self.__class__.__name__}"
            )

            if self.init_cfg is not None:
                ckpt_path = self.init_cfg["checkpoint"]
            else:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path,
                logger=logger,
                map_location="cpu",
            )

            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt

            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict,
                False,
            )
            logger.info(f"Miss {missing_keys}")
            logger.info(f"Unexpected {unexpected_keys}")

    def train(self, mode=True):
        super(VSSEncoder, self).train(mode)

        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        if self.is_patch_embed:
            x = self.patch_embed(x)

        outs = []

        for i in range(self.num_layers):
            x = self.vss_layers[i](x)
            outs.append(x)

            if i < self.num_layers - 1:
                x = self.downsamples[i](x)

        return outs
