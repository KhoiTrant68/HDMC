import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import update_registered_buffers

# Uses the ConvNeXt Blocks defined in your updated modules
from modules.conv_module import (
    ConvBottleneckBlockWithStride,
    ConvBottleneckBlockWithUpsample,
)
from modules.swin_module import (
    ResScaleConvGateBlock,
    SpectralMoEDictionaryCrossAttention,
    SwinBlockWithConvMulti,
)


def ste_round(x):
    return torch.round(x) - x.detach() + x


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


# =========================================================================
#  HELPER MODULES (Spatial Checkerboard & SimpleGate)
# =========================================================================


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them. Needed for NAFBlock."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CheckerboardSplitter(nn.Module):
    """
    Splits a (B, C, H, W) tensor into:
    1. Anchor: Top-Left pixel of 2x2 block -> (B, C, H/2, W/2)
    2. Non-Anchor: The other 3 pixels stacked -> (B, 3*C, H/2, W/2)
    """

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to isolate 2x2 blocks: (B, C, H/2, 2, W/2, 2)
        # Permute to (B, C, H/2, W/2, 2, 2) to group spatial dims
        x_reshaped = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5)

        # Anchor is at local index (0, 0)
        anchor = x_reshaped[..., 0, 0]  # (B, C, H/2, W/2)

        # Non-Anchors are (0,1), (1,0), (1,1)
        na1 = x_reshaped[..., 0, 1]
        na2 = x_reshaped[..., 1, 0]
        na3 = x_reshaped[..., 1, 1]

        # Concatenate neighbors into channels
        non_anchor = torch.cat([na1, na2, na3], dim=1)  # (B, 3C, H/2, W/2)
        return anchor, non_anchor


class CheckerboardMerger(nn.Module):
    """Reverses the split to reconstruction (B, C, H, W)"""

    def forward(self, anchor, non_anchor):
        B, C, H_half, W_half = anchor.shape

        # Split non_anchor back to 3 parts
        na1, na2, na3 = torch.split(non_anchor, C, dim=1)

        # Stack into 2x2 grid: [[Anchor, na1], [na2, na3]]
        row0 = torch.stack([anchor, na1], dim=-1)  # (..., 2)
        row1 = torch.stack([na2, na3], dim=-1)  # (..., 2)
        grid = torch.stack([row0, row1], dim=-2)  # (..., 2, 2)

        # Permute back: (B, C, H/2, 2, W/2, 2)
        x = grid.permute(0, 1, 2, 4, 3, 5)

        # Merge dims: (B, C, H, W)
        x = x.reshape(B, C, H_half * 2, W_half * 2)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NAFBlock(nn.Module):
    def __init__(self, dim, inter_dim=None):
        super().__init__()
        self.dim = inter_dim if inter_dim is not None else dim
        dw_channel = self.dim * 2
        ffn_channel = self.dim * 2

        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dim, dw_channel, 1),
            nn.Conv2d(dw_channel, dw_channel, 3, 1, padding=1, groups=dw_channel),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.FFN = nn.Sequential(
            nn.Conv2d(self.dim, ffn_channel, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channel // 2, self.dim, 1),
        )
        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel // 2, self.dim, 1)

        self.beta = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)

        self.in_conv = (
            nn.Conv2d(dim, inter_dim, 1) if inter_dim is not None else nn.Identity()
        )
        self.out_conv = (
            nn.Conv2d(inter_dim, dim, 1) if inter_dim is not None else nn.Identity()
        )

    def forward(self, x):
        x_in = self.in_conv(x)
        identity = x_in
        x = self.norm1(x_in)

        x_dw = self.dwconv(x)
        x1, x2 = x_dw.chunk(2, dim=1)
        x = x1 * x2

        x = x * self.sca(x)
        x = self.conv1(x)
        out = identity + x * self.beta

        identity = out
        out = self.norm2(out)
        out = self.FFN(out)
        out = identity + out * self.gamma

        out = self.out_conv(out)
        return out


class HDMC(CompressionModel):
    def __init__(
        self,
        head_dim=None,
        N=192,
        M=320,
    ):
        super().__init__()

        if head_dim is None:
            self.head_dim = [8, 16, 32, 32, 16, 8]
        else:
            self.head_dim = head_dim

        self.N = N
        self.M = M

        # Uneven Groups: [0, 16, 16, 32, 64, 192]
        # Slices 0-3 (Scale 1/2) | Slice 4 (Scale 3 HPCM)
        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_standard_slices = len(self.groups) - 2
        self.last_slice_dim = self.groups[-1]

        # ==================================================
        # PART 1: BACKBONE (Updated Window Size=16)
        # ==================================================
        self.window_size = 16
        feature_dim = [96, 144, 256]
        basic_block = ResScaleConvGateBlock
        swin_block = SwinBlockWithConvMulti
        block_counts = [1, 2, 12]

        # Encoder
        self.m_down1 = nn.Sequential(
            swin_block(
                feature_dim[0],
                feature_dim[0],
                self.head_dim[0],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[0],
            ),
            ConvBottleneckBlockWithStride(feature_dim[0], feature_dim[1]),
        )
        self.m_down2 = nn.Sequential(
            swin_block(
                feature_dim[1],
                feature_dim[1],
                self.head_dim[1],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[1],
            ),
            ConvBottleneckBlockWithStride(feature_dim[1], feature_dim[2]),
        )
        self.m_down3 = nn.Sequential(
            swin_block(
                feature_dim[2],
                feature_dim[2],
                self.head_dim[2],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[2],
            ),
            nn.Conv2d(feature_dim[2], M, kernel_size=5, stride=2, padding=2),
        )
        self.g_a = nn.Sequential(
            ConvBottleneckBlockWithStride(3, feature_dim[0]),
            self.m_down1,
            self.m_down2,
            self.m_down3,
        )

        # Decoder
        self.m_up1 = nn.Sequential(
            swin_block(
                feature_dim[2],
                feature_dim[2],
                self.head_dim[3],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[2],
            ),
            ConvBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1]),
        )
        self.m_up2 = nn.Sequential(
            swin_block(
                feature_dim[1],
                feature_dim[1],
                self.head_dim[4],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[1],
            ),
            ConvBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0]),
        )
        self.m_up3 = nn.Sequential(
            swin_block(
                feature_dim[0],
                feature_dim[0],
                self.head_dim[5],
                self.window_size,
                0,
                basic_block,
                block_num=block_counts[0],
            ),
            ConvBottleneckBlockWithUpsample(feature_dim[0], 3),
        )
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                M, feature_dim[2], kernel_size=5, stride=2, output_padding=1, padding=2
            ),
            self.m_up1,
            self.m_up2,
            self.m_up3,
        )

        # Hyper-Prior
        self.h_a = nn.Sequential(
            ConvBottleneckBlockWithStride(M, N),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            nn.Conv2d(N, 192, kernel_size=3, stride=2, padding=1),
        )
        self.h_z_s1 = nn.Sequential(
            nn.ConvTranspose2d(
                192, N, kernel_size=3, stride=2, output_padding=1, padding=1
            ),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            ConvBottleneckBlockWithUpsample(N, M),
        )
        self.h_z_s2 = nn.Sequential(
            nn.ConvTranspose2d(
                192, N, kernel_size=3, stride=2, output_padding=1, padding=1
            ),
            SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvGateBlock, block_num=1),
            ConvBottleneckBlockWithUpsample(N, M),
        )

        # ==================================================
        # PART 2: ENTROPY MODULES
        # ==================================================
        self.dt_cross_attention = nn.ModuleList()
        self.context_transforms = nn.ModuleList()
        self.mean_transforms = nn.ModuleList()
        self.scale_transforms = nn.ModuleList()
        self.lrp_transforms = nn.ModuleList()

        cum_channels = 0

        # --- A. Standard Slices (Scale 1 & 2) ---
        for i in range(self.num_standard_slices):
            current_dim = self.groups[i + 1]
            moe_input_dim = (M * 2) + cum_channels

            self.dt_cross_attention.append(
                SpectralMoEDictionaryCrossAttention(
                    input_dim=moe_input_dim,
                    output_dim=M,
                    head_num=8,
                    mlp_rate=4,
                    num_experts=4,
                )
            )
            support_dim = M + (M * 2) + cum_channels
            self.context_transforms.append(NAFBlock(support_dim, inter_dim=128))

            self.mean_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )
            self.scale_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )
            self.lrp_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim + current_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )
            cum_channels += current_dim

        # --- B. Checkerboard Split (Scale 3) ---
        self.checkerboard_split = CheckerboardSplitter()
        self.checkerboard_merge = CheckerboardMerger()

        # 1. Anchor Modules
        self.moe_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=(M * 2) + cum_channels,
            output_dim=M,
            head_num=8,
            mlp_rate=4,
            num_experts=4,
        )
        support_dim_anc = M + (M * 2) + cum_channels
        self.naf_anchor = NAFBlock(support_dim_anc, inter_dim=128)

        self.mean_anchor = nn.Sequential(
            nn.Conv2d(support_dim_anc, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim, 3, 1, 1),
        )
        self.scale_anchor = nn.Sequential(
            nn.Conv2d(support_dim_anc, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim, 3, 1, 1),
        )
        self.lrp_anchor = nn.Sequential(
            nn.Conv2d(support_dim_anc + self.last_slice_dim, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim, 3, 1, 1),
        )

        # 2. Non-Anchor Modules (FUSION)
        fusion_input_dim = (M * 2) + cum_channels + self.last_slice_dim

        self.moe_non_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=fusion_input_dim,
            output_dim=M,
            head_num=8,
            mlp_rate=4,
            num_experts=4,
        )
        support_dim_na = M + fusion_input_dim
        self.naf_non_anchor = NAFBlock(support_dim_na, inter_dim=128)

        out_na_dim = self.last_slice_dim * 3

        self.mean_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, out_na_dim, 3, 1, 1),
        )
        self.scale_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, out_na_dim, 3, 1, 1),
        )
        self.lrp_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na + out_na_dim, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, out_na_dim, 3, 1, 1),
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, training_mode="noise"):
        # 1. Transform
        y = self.g_a(x)
        y_shape = y.shape[2:]

        # 2. Hyper-Prior
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        hyper_info = torch.cat([latent_means, latent_scales], dim=1)

        # 3. Entropy Modeling
        y_slices = y.split(self.groups[1:], 1)
        y_hat_slices = []
        y_likelihood = []
        mu_list = []
        scale_list = []
        all_logits = []

        # --- A. Standard Slices (0 to 3) ---
        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            if i == 0:
                query = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([hyper_info, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            if hasattr(self.dt_cross_attention[i], "last_routing_logits"):
                all_logits.append(
                    (
                        self.dt_cross_attention[i].last_routing_logits,
                        self.dt_cross_attention[i].last_routing_indices,
                    )
                )

            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)

            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            if self.training and training_mode == "noise":
                noise = torch.empty_like(y_slice).uniform_(-0.5, 0.5)
                y_hat_slice = y_slice + noise
            else:
                y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))

            y_hat_slices.append(y_hat_slice)
            mu_list.append(mu)
            scale_list.append(scale)

        # --- B. Checkerboard (Slice 4) ---
        last_slice = y_slices[-1]
        y_anchor, y_non_anchor = self.checkerboard_split(last_slice)

        # Prepare Context (Downsampled for Anchor)
        prev_slices_full = torch.cat(y_hat_slices, dim=1)
        prev_slices_down = F.avg_pool2d(prev_slices_full, 2)
        hyper_down = F.avg_pool2d(hyper_info, 2)

        # --- Anchor Pass ---
        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_info_anc = self.moe_anchor(query_anc)

        if hasattr(self.moe_anchor, "last_routing_logits"):
            all_logits.append(
                (
                    self.moe_anchor.last_routing_logits,
                    self.moe_anchor.last_routing_indices,
                )
            )

        support_anc = torch.cat([dict_info_anc, query_anc], dim=1)
        feat_anc = self.naf_anchor(support_anc)
        mu_anc = self.mean_anchor(feat_anc)
        scale_anc = self.scale_anchor(feat_anc)

        _, y_lik_anc = self.gaussian_conditional(y_anchor, scale_anc, mu_anc)

        if self.training and training_mode == "noise":
            y_hat_anc = y_anchor + torch.empty_like(y_anchor).uniform_(-0.5, 0.5)
        else:
            y_hat_anc = ste_round(y_anchor - mu_anc) + mu_anc

        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        # --- Non-Anchor Pass (FUSION of Global + Local Anchor) ---
        query_na = torch.cat([query_anc, y_hat_anc], dim=1)  # Cross-Scale Fusion
        dict_info_na = self.moe_non_anchor(query_na)

        if hasattr(self.moe_non_anchor, "last_routing_logits"):
            all_logits.append(
                (
                    self.moe_non_anchor.last_routing_logits,
                    self.moe_non_anchor.last_routing_indices,
                )
            )

        support_na = torch.cat([dict_info_na, query_na], dim=1)
        feat_na = self.naf_non_anchor(support_na)
        mu_na = self.mean_non_anchor(feat_na)
        scale_na = self.scale_non_anchor(feat_na)

        _, y_lik_na = self.gaussian_conditional(y_non_anchor, scale_na, mu_na)

        if self.training and training_mode == "noise":
            y_hat_na = y_non_anchor + torch.empty_like(y_non_anchor).uniform_(-0.5, 0.5)
        else:
            y_hat_na = ste_round(y_non_anchor - mu_na) + mu_na

        lrp_na = self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1))
        y_hat_na = y_hat_na + (0.5 * torch.tanh(lrp_na))

        # --- Merge & Restore ---
        y_hat_last = self.checkerboard_merge(y_hat_anc, y_hat_na)
        mu_last = self.checkerboard_merge(mu_anc, mu_na)
        scale_last = self.checkerboard_merge(scale_anc, scale_na)
        y_lik_last = self.checkerboard_merge(y_lik_anc, y_lik_na)

        y_hat_slices.append(y_hat_last)
        y_likelihood.append(y_lik_last)
        mu_list.append(mu_last)
        scale_list.append(scale_last)

        # 4. Reconstruction
        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para": {"means": means, "scales": scales, "y": y},
            "dict_info": dict_info,
            "router_logits": tuple(all_logits) if all_logits else None,
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        hyper_info = torch.cat([latent_means, latent_scales], dim=1)

        y_slices = y.split(self.groups[1:], 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        all_symbols = []
        all_indexes = []

        # A. Standard Slices
        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            if i == 0:
                query = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([hyper_info, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)

            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]
            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            all_symbols.append(y_q_slice.reshape(-1))
            all_indexes.append(index.reshape(-1))

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        # B. Checkerboard Slice
        last_slice = y_slices[-1]
        y_anc, y_na = self.checkerboard_split(last_slice)
        prev_slices_down = F.avg_pool2d(torch.cat(y_hat_slices, dim=1), 2)
        hyper_down = F.avg_pool2d(hyper_info, 2)

        # Anchor
        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_anc = self.moe_anchor(query_anc)
        feat_anc = self.naf_anchor(torch.cat([dict_anc, query_anc], dim=1))
        mu_anc = self.mean_anchor(feat_anc)
        scale_anc = self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        y_q_anc = self.gaussian_conditional.quantize(y_anc, "symbols", mu_anc)
        y_hat_anc = y_q_anc + mu_anc
        all_symbols.append(y_q_anc.reshape(-1))
        all_indexes.append(index_anc.reshape(-1))

        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        # Non-Anchor
        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        dict_na = self.moe_non_anchor(query_na)
        feat_na = self.naf_non_anchor(torch.cat([dict_na, query_na], dim=1))
        mu_na = self.mean_non_anchor(feat_na)
        scale_na = self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        y_q_na = self.gaussian_conditional.quantize(y_na, "symbols", mu_na)
        all_symbols.append(y_q_na.reshape(-1))
        all_indexes.append(index_na.reshape(-1))

        encoder.encode_with_indexes(
            torch.cat(all_symbols).tolist(),
            torch.cat(all_indexes).tolist(),
            cdf,
            cdf_lengths,
            offsets,
        )
        y_string = encoder.flush()
        return {"strings": [[y_string], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        hyper_info = torch.cat([latent_means, latent_scales], dim=1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        y_hat_slices = []

        # A. Standard Slices
        for i in range(self.num_standard_slices):
            if i == 0:
                query = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([hyper_info, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]
            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1]).to(mu.device)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        # B. Checkerboard Slice
        prev_slices_down = F.avg_pool2d(torch.cat(y_hat_slices, dim=1), 2)
        hyper_down = F.avg_pool2d(hyper_info, 2)

        # Anchor
        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_anc = self.moe_anchor(query_anc)
        feat_anc = self.naf_anchor(torch.cat([dict_anc, query_anc], dim=1))
        mu_anc = self.mean_anchor(feat_anc)
        scale_anc = self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        rv_anc = decoder.decode_stream(
            index_anc.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv_anc = (
            torch.Tensor(rv_anc)
            .reshape(1, self.last_slice_dim, y_shape[0] // 2, y_shape[1] // 2)
            .to(mu_anc.device)
        )
        y_hat_anc = self.gaussian_conditional.dequantize(rv_anc, mu_anc)
        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        # Non-Anchor
        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        dict_na = self.moe_non_anchor(query_na)
        feat_na = self.naf_non_anchor(torch.cat([dict_na, query_na], dim=1))
        mu_na = self.mean_non_anchor(feat_na)
        scale_na = self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        rv_na = decoder.decode_stream(
            index_na.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv_na = (
            torch.Tensor(rv_na)
            .reshape(1, self.last_slice_dim * 3, y_shape[0] // 2, y_shape[1] // 2)
            .to(mu_na.device)
        )
        y_hat_na = self.gaussian_conditional.dequantize(rv_na, mu_na)
        lrp_na = self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1))
        y_hat_na = y_hat_na + (0.5 * torch.tanh(lrp_na))

        # Merge
        y_hat_last = self.checkerboard_merge(y_hat_anc, y_hat_na)
        y_hat_slices.append(y_hat_last)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp(0, 1)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        try:
            N = state_dict["g_a.0.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
        except KeyError:
            N = 192
            M = 320
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net
