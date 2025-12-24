import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

# ==========================================
# PART 1: OPTIMIZED BLOCKS (Flash Attention)
# ==========================================


class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type

        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)

        # Relative position bias
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        trunc_normal_(self.relative_position_params, std=0.02)

        # Pre-calculate relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self):
        relative_position_bias = self.relative_position_params[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        return relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # [nH, Wh*Ww, Wh*Ww]

    def forward(self, x):
        B, H, W, C = x.shape
        # Cyclic Shift
        if self.type == "W":
            shifted_x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        # Partition Windows
        x_windows = rearrange(
            shifted_x,
            "b (h p1) (w p2) c -> b h w p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        # Flatten to [B*num_windows, window_size*window_size, C]
        x_windows = rearrange(x_windows, "b h w p1 p2 c -> (b h w) (p1 p2) c")

        # QKV
        qkv = self.embedding_layer(x_windows)
        q, k, v = rearrange(
            qkv, "b n (qkv h c) -> qkv b h n c", h=self.n_heads, qkv=3
        ).unbind(0)

        # Relative Bias
        rel_bias = self.get_relative_position_bias().unsqueeze(0)  # [1, nH, N, N]

        # Masking for SW-MSA
        attn_mask = None
        if self.type != "W":
            # Lazy mask generation
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.window_size // 2),
                slice(-self.window_size // 2, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.window_size // 2),
                slice(-self.window_size // 2, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Rearrange b=1 -> nW
            mask_windows = rearrange(
                img_mask,
                "b (h p1) (w p2) c -> (b h w) (p1 p2) c",
                p1=self.window_size,
                p2=self.window_size,
            )
            mask_windows = mask_windows.squeeze(-1)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask.unsqueeze(1)  # [nW, 1, N, N]

        # Scaled Dot Product Attention
        scale = self.head_dim**-0.5
        q = q * scale
        attn = (q @ k.transpose(-2, -1)) + rel_bias

        if attn_mask is not None:
            # FIX: Repeat mask to match Batch dimension [nW -> B*nW]
            # attn shape: [B*nW, nH, N, N]
            # attn_mask shape: [nW, 1, N, N]
            attn_mask = attn_mask.repeat(B, 1, 1, 1)
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        x_windows = attn @ v

        # Merge Heads
        x_windows = rearrange(x_windows, "b h n c -> b n (h c)")
        x_windows = self.linear(x_windows)

        # Merge Windows
        x_windows = rearrange(
            x_windows,
            "(b h w) (p1 p2) c -> b (h p1) (w p2) c",
            h=H // self.window_size,
            w=W // self.window_size,
            p1=self.window_size,
            p2=self.window_size,
        )

        # Reverse Cyclic Shift
        if self.type == "W":
            x = torch.roll(
                x_windows,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        else:
            x = x_windows

        return x


class ResScaleConvGateBlock(nn.Module):
    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, type="W"
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvGELU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, x):
        x = self.res_scale_1(x) + self.drop_path(self.msa(self.ln1(x)))
        x = self.res_scale_2(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinBlockWithConvMulti(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        head_dim,
        window_size,
        drop_path,
        block=ResScaleConvGateBlock,
        block_num=2,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.window_size = window_size
        for i in range(block_num):
            ty = "W" if i % 2 == 0 else "SW"
            self.layers.append(
                block(input_dim, input_dim, head_dim, window_size, drop_path, type=ty)
            )
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x is NCHW
        H, W = x.size(2), x.size(3)
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # NCHW -> NHWC for Transformer blocks
        trans_x = x.permute(0, 2, 3, 1).contiguous()
        for layer in self.layers:
            trans_x = layer(trans_x)

        # NHWC -> NCHW for Conv
        trans_x = trans_x.permute(0, 3, 1, 2).contiguous()
        trans_x = self.conv(trans_x)

        if pad_b > 0 or pad_r > 0:
            trans_x = trans_x[:, :, :H, :W]
        return trans_x + x[:, :, :H, :W]


class DWConv(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        # NHWC -> NCHW -> Conv -> NHWC
        # Optimization: use memory_format=torch.channels_last if possible externally
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class ConvGELU(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features // 2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.dwconv(x)
        x = self.act(x) * v
        x = self.fc2(x)
        return x


class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


# ==========================================
# PART 2: OPTIMIZED WAVELET & MOE
# ==========================================


class LiftingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# JIT Compile the element-wise lifting math for fusion
@torch.jit.script
def lifting_step_script(
    even: torch.Tensor, odd: torch.Tensor, P_out: torch.Tensor, U_out: torch.Tensor
):
    # Standard: High = Odd - P(Even); Low = Even + U(High)
    high = odd - P_out
    low = even + U_out
    return low, high


class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.P_horz = LiftingBlock(in_channels)
        self.U_horz = LiftingBlock(in_channels)
        self.P_vert = LiftingBlock(in_channels)
        self.U_vert = LiftingBlock(in_channels)

    def forward(self, x):
        # 1. Horizontal
        even_h, odd_h = x[:, :, :, 0::2], x[:, :, :, 1::2]
        # h_horz = odd_h - P(even_h)
        h_horz = odd_h - self.P_horz(even_h)
        # l_horz = even_h + U(h_horz)
        l_horz = even_h + self.U_horz(h_horz)

        # 2. Vertical
        even_ll, odd_ll = l_horz[:, :, 0::2, :], l_horz[:, :, 1::2, :]
        h_ll = odd_ll - self.P_vert(even_ll)
        ll = even_ll + self.U_vert(h_ll)

        # Apply vertical to h_horz
        even_hh, odd_hh = h_horz[:, :, 0::2, :], h_horz[:, :, 1::2, :]
        h_hh = odd_hh - self.P_vert(even_hh)
        lh = even_hh + self.U_vert(h_hh)

        # Output Order: LL, [LH, HL, HH] (Note: Variable naming mixed but structure consistent)
        hf = torch.cat([h_ll, lh, h_hh], dim=1)
        return ll, hf


class InverseLearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.P_horz = LiftingBlock(in_channels)
        self.U_horz = LiftingBlock(in_channels)
        self.P_vert = LiftingBlock(in_channels)
        self.U_vert = LiftingBlock(in_channels)

    def _inverse_lifting(self, low, high, P, U, dim):
        # Inverse: Even = Low - U(High); Odd = High + P(Even)
        even = low - U(high)
        odd = high + P(even)

        if dim == 2:
            B, C, H, W = even.shape
            out = torch.empty(B, C, H * 2, W, device=even.device, dtype=even.dtype)
            out[:, :, 0::2, :] = even
            out[:, :, 1::2, :] = odd
        else:
            B, C, H, W = even.shape
            out = torch.empty(B, C, H, W * 2, device=even.device, dtype=even.dtype)
            out[:, :, :, 0::2] = even
            out[:, :, :, 1::2] = odd
        return out

    def forward(self, ll, hf):
        C = ll.shape[1]
        lh, hl, hh = torch.split(hf, C, dim=1)

        l_horz = self._inverse_lifting(ll, lh, self.P_vert, self.U_vert, dim=2)
        h_horz = self._inverse_lifting(hl, hh, self.P_vert, self.U_vert, dim=2)
        x = self._inverse_lifting(l_horz, h_horz, self.P_horz, self.U_horz, dim=3)
        return x


class SpectralMoEDictionaryCrossAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_rate=4,
        head_num=4,
        qkv_bias=True,
        num_experts=4,
        expert_entries=64,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_num = head_num
        self.num_experts = num_experts

        self.dim_low = 32 * head_num
        self.dim_high = 32 * head_num * 3

        self.dwt = LearnableWaveletTransform(32 * head_num)
        self.idwt = InverseLearnableWaveletTransform(32 * head_num)

        self.x_trans = nn.Linear(input_dim, 32 * head_num, bias=qkv_bias)
        self.output_trans = nn.Linear(32 * head_num, output_dim, bias=qkv_bias)

        # Low Freq
        c_block = 32 * head_num
        self.ln_low = nn.LayerNorm(c_block)
        self.q_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.k_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.ln_dict_low = nn.LayerNorm(c_block)
        self.scale = c_block**-0.5
        self.dict_low = nn.Parameter(torch.randn(64, c_block))

        # High Freq
        self.router = nn.Sequential(
            nn.Linear(self.dim_high + c_block, self.dim_high),
            nn.GELU(),
            nn.Linear(self.dim_high, self.dim_high // 4),
            nn.GELU(),
            nn.Linear(self.dim_high // 4, num_experts),
        )
        self.experts_high = nn.Parameter(
            torch.randn(num_experts * expert_entries, self.dim_high)
        )
        self.ln_dict_high = nn.LayerNorm(self.dim_high)
        self.ln_high = nn.LayerNorm(self.dim_high)
        self.q_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.k_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.v_all = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)

        # Utils
        self.msa = MultiScaleAggregation(c_block)
        self.ln_scale = nn.LayerNorm(c_block)
        self.res_scale_1 = Scale(c_block, init_value=1.0)
        self.ln_mlp = nn.LayerNorm(c_block)
        self.mlp = ConvGELU(c_block, c_block * mlp_rate)
        self.res_scale_2 = Scale(c_block, init_value=1.0)

        self.register_buffer("expert_biases", torch.zeros(num_experts))

        # Cache for loss
        self.last_routing_logits = None
        self.last_routing_indices = None

        trunc_normal_(self.dict_low, std=0.02)
        trunc_normal_(self.experts_high, std=0.02)

    def process_low_freq(self, x):
        # x: NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        x_norm = self.ln_low(x)
        q = self.q_low(x_norm)
        k = self.k_low(self.ln_dict_low(self.dict_low))  # [Entries, C]

        # Attention: [B, H, W, C] @ [C, Entries] -> [B, H, W, Entries]
        attn = torch.matmul(q, k.transpose(0, 1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, self.dict_low)
        return out.permute(0, 3, 1, 2)

    def process_high_freq_guided(self, hf, lf):
        # hf, lf: NHWC
        B, H, W, C_high = hf.shape

        # 1. Calculate RAW (Unbiased) Routing Logits
        routing_logits = self.router(torch.cat([hf, lf], dim=-1))
        self.last_routing_logits = routing_logits

        # 2. Add Bias ONLY for Selection (Top-K)
        biased_logits = routing_logits + self.expert_biases.view(1, 1, 1, -1)
        _, topk_indices = torch.topk(biased_logits, k=2, dim=-1)

        # Save indices for the balancer to use
        self.last_routing_indices = topk_indices

        # 3. Calculate UNBIASED Weights for the final sum
        # We calculate softmax on the raw logits
        unbiased_probs = F.softmax(routing_logits, dim=-1)  # [B, H, W, Experts]

        # 4. Create a mask to only keep the Top-K selected experts
        # This ensures we only use the experts chosen by the biased selection
        mask = torch.zeros_like(unbiased_probs).scatter_(-1, topk_indices, 1.0)

        # Apply mask and RE-NORMALIZE so the selected k-experts sum to 1.0
        # This matches the reference code: top_k_probs = top_k_probs / top_k_probs.sum()
        masked_probs = unbiased_probs * mask
        routing_weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 5. Expert Attention (Computational part remains the same)
        q = self.q_high(self.ln_high(hf)).reshape(-1, C_high)
        all_keys = self.k_high(self.ln_dict_high(self.experts_high))

        sim = torch.matmul(q, all_keys.transpose(0, 1)) * (C_high**-0.5)
        sim = sim.view(B, H * W, self.num_experts, -1)
        attn = F.softmax(sim, dim=-1)

        v_experts = self.v_all(self.experts_high).view(self.num_experts, -1, C_high)
        expert_outputs = torch.einsum("bhke,kec->bhkc", attn, v_experts)

        # 6. Final Weighted Sum using the Unbiased but Masked weights
        router_weights_flat = routing_weights.view(B, H * W, self.num_experts, 1)
        final_out = (expert_outputs * router_weights_flat).sum(dim=2)

        return final_out.view(B, H, W, C_high)

    def forward(self, x):
        shortcut = x

        # Embed (NHWC) -> NCHW
        x_emb = self.x_trans(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        ll, hf = self.dwt(x_emb)  # NCHW
        ll_processed = self.process_low_freq(ll)  # NCHW

        # Process High Freq (NHWC math)
        hf_processed = self.process_high_freq_guided(
            hf.permute(0, 2, 3, 1), ll_processed.permute(0, 2, 3, 1)
        )
        hf_processed = hf_processed.permute(0, 3, 1, 2)

        recon = self.idwt(ll_processed, hf_processed)  # NCHW

        # Post-process (NHWC)
        recon = recon.permute(0, 2, 3, 1)
        recon = recon + self.res_scale_1(self.msa(self.ln_scale(recon)))
        recon = recon + self.res_scale_2(self.mlp(self.ln_mlp(recon)))
        out = self.output_trans(recon)

        out = out.permute(0, 3, 1, 2)
        if self.input_dim == self.output_dim:
            out = out + shortcut

        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Conv2d(dim, dim, 1)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = nn.Sequential(
            nn.Sequential(
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.Conv2d(dim, dim, 1),
            ),
            nn.Sequential(
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.Conv2d(dim, dim, 1),
            ),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        x_nchw = x.permute(0, 3, 1, 2)
        s = self.s(x_nchw)
        s_out = self.dense(s)
        out = s_out * self.spatial_atte(s_out)
        return out.permute(0, 2, 3, 1)
