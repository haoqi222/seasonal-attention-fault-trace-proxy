#!/usr/bin/env python
"""阶段6：时序跨模态 Transformer（v3）——把"多时相"建模为季节序列。

v1/v2 把 4 季当作 36 个独立通道堆叠，未建模季节交互；模态分离（热/结构分 encoder）
对高度相关场景是坏偏置。v3 改动：
  - 共享 Encoder(9ch=每季 ST+结构8通道)，4 季权重共享
  - 每尺度对 4 个季节 token 做时间注意力（TemporalAttention）：每像素位置 4 季互查
  - 时间融合后 concat 回 decoder（U-Net 式 4 尺度 skip）
  - temporal="concat" 消融：TA 换恒等 → 隔离注意力增益
  - season_sel=[1] 消融：单季，S=1 → TA 退化为恒等

通道排列（band 外层 / season 内层，X 为 (B,36,H,W)）：
  season s 的 9 通道 = [4*b + s for b in range(9)]
"""
import torch
import torch.nn as nn

from cross_modal import conv_block


class Encoder(nn.Module):
    """4 级共享编码器（9ch 输入）。"""

    def __init__(self, in_channels=8, base=16, dropout=0.0):
        super().__init__()
        self.skips = nn.ModuleList()
        cin = in_channels
        for i in range(4):
            c = base * (2 ** i)
            self.skips.append(conv_block(cin, c, dropout))
            cin = c

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.skips):
            x = blk(x)
            feats.append(x)
            if i < 3:
                x = nn.functional.max_pool2d(x, 2)
        return feats


class TemporalAttention(nn.Module):
    """季节轴注意力：每空间位置 S 个季节 token 互为 Q/K/V，残差相加。"""

    def __init__(self, d, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)

    def forward(self, F):
        """F: list[S] of (B,C,H,W)。返回同形状 list[S]。"""
        S = len(F)
        B, C, H, W = F[0].shape
        N = H * W
        x = torch.stack(F, dim=2)                     # (B,C,S,H,W)
        x = x.permute(0, 3, 4, 2, 1).reshape(B * N, S, C)
        qkv = self.qkv(x).reshape(B * N, S, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B*N, H, S, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) * ((C // self.heads) ** -0.5)
        a = a.softmax(dim=-1)
        out = (a @ v).reshape(B * N, S, C)            # (B*N, S, C)
        out = self.norm(x + out)
        out = out.view(B, H, W, S, C).permute(0, 4, 3, 1, 2)   # (B,C,S,H,W)
        return [out[:, :, s] for s in range(S)]


class Decoder(nn.Module):
    def __init__(self, base=16, out_ch=1, dropout=0.0):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(8 * base, 4 * base, 2, 2)
        self.conv3 = conv_block(8 * base, 4 * base, dropout)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2)
        self.conv2 = conv_block(4 * base, 2 * base, dropout)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, 2)
        self.conv1 = conv_block(2 * base, base, dropout)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, fused3, fused2, fused1, fused0):
        x = self.up3(fused3); x = torch.cat([x, fused2], 1); x = self.conv3(x)
        x = self.up2(x); x = torch.cat([x, fused1], 1); x = self.conv2(x)
        x = self.up1(x); x = torch.cat([x, fused0], 1); x = self.conv1(x)
        return self.head(x)


class TemporalCMT(nn.Module):
    def __init__(self, season_sel=None, temporal="attn", base=16, heads=4,
                 dropout=0.0, band_sel=None, n_bands=8):
        super().__init__()
        season_sel = season_sel or [0, 1, 2, 3]
        self.season_sel = season_sel
        self.temporal = temporal
        self.n_bands = n_bands
        self.band_sel = list(range(n_bands)) if band_sel is None else list(band_sel)
        self.enc = Encoder(len(self.band_sel), base, dropout)
        self.tas = nn.ModuleList([TemporalAttention(base * (2 ** i), heads)
                                  if temporal == "attn" else nn.Identity()
                                  for i in range(4)])
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(base * (2 ** i) * len(season_sel), base * (2 ** i), 1),
                          nn.BatchNorm2d(base * (2 ** i)), nn.ReLU(inplace=True))
            for i in range(4)])
        self.dec = Decoder(base, 1, dropout)

    def _season_channels(self, s):
        return [4 * b + s for b in self.band_sel]

    def forward(self, x):
        frames = [x[:, self._season_channels(s)] for s in self.season_sel]
        # 每季节过共享 encoder，得各尺度特征
        feats_s = [self.enc(f) for f in frames]      # S × 4 个 (B,C,H,W)
        fused = []
        for i in range(4):
            outs = self.tas[i]([fs[i] for fs in feats_s])   # 均为 list[S]
            fused.append(self.projs[i](torch.cat(outs, dim=1)))
        return self.dec(*fused[::-1])


if __name__ == "__main__":
    x = torch.randn(2, 32, 128, 128)
    for kw in [dict(), dict(temporal="concat"), dict(season_sel=[1]),
               dict(base=8)]:
        m = TemporalCMT(**kw)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{kw or '默认'} → {m(x).shape}  {n:.2f}M")
