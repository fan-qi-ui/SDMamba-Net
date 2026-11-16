# -*- coding: utf-8 -*-
"""
Final Grad-CAM ablation-aware replacement for:
    MixerCSeg-main/models/decoder/SRF.py

Important:
- MSDWConv is ALWAYS instantiated so historical checkpoints that contain
  ms_conv parameters can still load with strict=True.
- Whether MSDWConv participates in forward() is controlled by:
      MIXER_USE_MSCONV_FORWARD=0/1

- The paper ablation decoder is selected through:
      MIXER_DECODER_VARIANT=shallow|full

  ``shallow`` keeps projected shallow cross-scale fusion and the prediction
  head. ``full`` adds MSDWConv, channel fusion, and BoundaryRefine.
"""

import os
import torch.nn as nn
import torch


def _flag(name, default):
    value = os.environ.get(name)

    if value is None:
        return bool(default)

    value = value.strip().lower()

    if value in ("1", "true", "yes", "on"):
        return True

    if value in ("0", "false", "no", "off"):
        return False

    raise ValueError(
        f"{name} must be 0/1 or true/false, got {value!r}"
    )


def _get_decoder_variant():
    value = os.environ.get("MIXER_DECODER_VARIANT")
    if value is None:
        return None

    value = value.strip().lower()
    aliases = {
        "shallow": "shallow",
        "cross_scale": "shallow",
        "cross-scale": "shallow",
        "full": "full",
        "mcfbr": "full",
    }
    if value not in aliases:
        raise ValueError(
            "Unknown MIXER_DECODER_VARIANT={!r}. Use shallow / full.".format(
                value
            )
        )
    return aliases[value]


class Conv2d_GN(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                inc,
                outc,
                ks,
                stride,
                pad,
                dilation,
                groups,
                bias,
            ),
            nn.GroupNorm(
                outc // 8,
                outc,
            ),
        )

    def forward(self, x):
        return self.block(x)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        embed_dim=768,
        identity=False,
    ):
        super().__init__()

        self.proj = nn.Linear(
            input_dim,
            embed_dim,
        )

        if identity:
            self.proj = nn.Identity()

    def forward(self, x):
        n, _, h, w = x.shape

        x = x.flatten(
            2
        ).transpose(
            1,
            2,
        )

        x = self.proj(x)

        x = x.permute(
            0,
            2,
            1,
        ).reshape(
            n,
            -1,
            h,
            w,
        )

        return x


class SRModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.att = nn.Conv2d(
            in_channels=dim,
            out_channels=1,
            kernel_size=1,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x1,
        x2,
        x3,
        x4,
    ):
        _, _, h, w = x1.shape

        att_map = self.sigmoid(
            self.att(x1)
        )

        x2 = nn.Upsample(
            size=(h, w),
            mode="bilinear",
        )(x2)

        x3 = nn.Upsample(
            size=(h, w),
            mode="bilinear",
        )(x3)

        x4 = nn.Upsample(
            size=(h, w),
            mode="bilinear",
        )(x4)

        x2 = x2 * att_map
        x3 = x3 * att_map
        x4 = x4 * att_map

        return x2, x3, x4


class MSDWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.branch1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=dim,
            bias=False,
        )

        self.branch2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=dim,
            bias=False,
        )

        self.branch3 = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=dim,
            bias=False,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                dim * 3,
                dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                max(1, dim // 8),
                dim,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat(
            [x1, x2, x3],
            dim=1,
        )

        return self.fuse(x)


class BoundaryRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.GroupNorm(
                max(1, dim // 8),
                dim,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self.refine(x)


class SRFModule(nn.Module):
    def __init__(
        self,
        embed_dims,
        mid_dim=8,
        size=(512, 512),
    ):
        super(SRFModule, self).__init__()

        self.size = size
        self.decoder_variant = _get_decoder_variant()

        if self.decoder_variant is None:
            # Historical behavior for the original command files.
            self.use_msconv_forward = _flag(
                "MIXER_USE_MSCONV_FORWARD",
                True,
            )
        else:
            self.use_msconv_forward = self.decoder_variant == "full"

        self.mlp1 = MLP(
            input_dim=embed_dims[0],
            embed_dim=mid_dim,
        )
        self.mlp2 = MLP(
            input_dim=embed_dims[1],
            embed_dim=mid_dim,
        )
        self.mlp3 = MLP(
            input_dim=embed_dims[2],
            embed_dim=mid_dim,
        )
        self.mlp4 = MLP(
            input_dim=embed_dims[3],
            embed_dim=mid_dim,
        )

        self.conv1 = Conv2d_GN(
            inc=mid_dim,
            outc=mid_dim,
        )
        self.conv2 = Conv2d_GN(
            inc=mid_dim,
            outc=mid_dim,
        )
        self.conv3 = Conv2d_GN(
            inc=mid_dim,
            outc=mid_dim,
        )
        self.conv4 = Conv2d_GN(
            inc=mid_dim,
            outc=mid_dim,
        )

        self.sr_module = SRModule(
            dim=mid_dim,
        )

        if self.decoder_variant == "shallow":
            # The projection is required to feed the common prediction head;
            # optional multi-scale/context/boundary refinement is disabled.
            self.shallow_fuse = Conv2d_GN(
                inc=mid_dim * 4,
                outc=mid_dim,
            )
            self.ms_conv = None
            self.block = None
            self.boundary_refine = None
        else:
            # In legacy mode these historical module names are retained for
            # checkpoint compatibility.
            self.ms_conv = MSDWConv(
                mid_dim * 4
            )

            self.block = nn.Sequential(
                Conv2d_GN(
                    inc=mid_dim * 4,
                    outc=mid_dim * 4,
                ),
                Conv2d_GN(
                    inc=mid_dim * 4,
                    outc=mid_dim,
                ),
                nn.ReLU(),
            )

            self.boundary_refine = BoundaryRefine(
                mid_dim
            )
            self.shallow_fuse = None

        self.linear_pred = nn.Sequential(
            nn.Conv2d(
                mid_dim,
                mid_dim // 4,
                1,
            ),
            nn.Conv2d(
                mid_dim // 4,
                mid_dim // 4,
                1,
                groups=mid_dim // 4,
            ),
            nn.Conv2d(
                mid_dim // 4,
                1,
                kernel_size=1,
            ),
            nn.Conv2d(
                1,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs

        x1 = self.conv1(
            self.mlp1(x1)
        )
        x2 = self.conv2(
            self.mlp2(x2)
        )
        x3 = self.conv3(
            self.mlp3(x3)
        )
        x4 = self.conv4(
            self.mlp4(x4)
        )

        x2, x3, x4 = self.sr_module(
            x1,
            x2,
            x3,
            x4,
        )

        # Keep original feature-visualization outputs.
        self.outs = [
            x1,
            x2,
            x3,
            x4,
        ]

        x1 = nn.Upsample(
            size=self.size,
            mode="bilinear",
        )(x1)

        x2 = nn.Upsample(
            size=self.size,
            mode="bilinear",
        )(x2)

        x3 = nn.Upsample(
            size=self.size,
            mode="bilinear",
        )(x3)

        x4 = nn.Upsample(
            size=self.size,
            mode="bilinear",
        )(x4)

        x = torch.cat(
            [x1, x2, x3, x4],
            dim=1,
        )

        if self.decoder_variant == "shallow":
            x = self.shallow_fuse(x)
        else:
            if self.use_msconv_forward:
                x = self.ms_conv(x)

            x = self.block(x)
            x = self.boundary_refine(x)

        x = self.linear_pred(x)

        return x
