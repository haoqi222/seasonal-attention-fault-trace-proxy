#!/usr/bin/env python
"""阶段6：跨模态 Transformer 主模型（v2：多尺度 cross-attention 融合）。

v1（单点瓶颈注意力）在 TEST 上不稳定且低于基线——热模态(4ch)容量不足、全局注意力
稀释薄线结构。v2 改动：
  - base=32（默认），热/结构分支容量对等
  - cross-attention 在 32×32 与 16×16 两处尺度融合（fused2 / fused3）
  - 解码器以 fused3 为底、fused2 代替 32×32 层的 skip
  - concat 消融仍可用：两处都改为 cat + 1×1 conv

模态划分（band 外层 / season 内层）：
  热模态   : ST（band 0）                           → 每季 1 通道
  结构模态 : red/nir08/swir16/ndvi/ndbi/ndwi/bsi/count（band 1-8）→ 每季 8 通道
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, dropout=0.0):
    layers = [
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    layers += [nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """4 级编码器：每级 conv_block + maxpool，返回 4 个尺度特征（128/64/32/16）。"""

    def __init__(self, in_ch, base=32, dropout=0.0):
        super().__init__()
        self.skips = nn.ModuleList()
        cin = in_ch
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
                x = F.max_pool2d(x, 2)
        return feats


class CrossAttention(nn.Module):
    """双向 cross-attention：结构↔热 互为 Q/K/V，残差相加。

    window>0 时改为非重叠窗口内局部注意力（窗口 w×w tokens 拆入 batch），
    限制感受野，避免全局注意力稀释薄线断层信号。window=0 为全局注意力。
    """

    def __init__(self, d, heads=4, window=0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.heads = heads
        self.window = window
        self.q1 = nn.Linear(d, d); self.k1 = nn.Linear(d, d); self.v1 = nn.Linear(d, d)
        self.q2 = nn.Linear(d, d); self.k2 = nn.Linear(d, d); self.v2 = nn.Linear(d, d)

    def _attn(self, q, k, v, N):
        B = q.shape[0]; h = self.heads; dh = q.shape[-1] // h
        q = q.view(B, N, h, dh).transpose(1, 2)
        k = k.view(B, N, h, dh).transpose(1, 2)
        v = v.view(B, N, h, dh).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) * (dh ** -0.5)
        a = a.softmax(dim=-1)
        return (a @ v).transpose(1, 2).reshape(B, N, -1)

    @staticmethod
    def _to_windows(t, w):
        B, C, H, W = t.shape
        return (t.permute(0, 2, 3, 1)                          # (B,H,W,C)
                 .view(B, H // w, w, W // w, w, C)
                 .permute(0, 1, 3, 2, 4, 5)
                 .reshape(B * (H // w) * (W // w), w * w, C))

    @staticmethod
    def _from_windows(out, B, H, W, w, d):
        out = (out.view(B, H // w, W // w, w, w, d)
               .permute(0, 1, 3, 2, 4, 5)
               .reshape(B, H, W, d)
               .permute(0, 3, 1, 2))
        return out

    def forward(self, ft, fo):
        B, C, H, W = ft.shape
        w = self.window
        if w > 0 and w < H:
            N = w * w
            ft_f = self._to_windows(ft, w)
            fo_f = self._to_windows(fo, w)
            at = self._attn(self.q1(fo_f), self.k1(ft_f), self.v1(ft_f), N)
            ao = self._attn(self.q2(ft_f), self.k2(fo_f), self.v2(fo_f), N)
            ft_out = self.norm(ft_f + at)
            fo_out = self.norm(fo_f + ao)
            return self._from_windows(torch.cat([ft_out, fo_out], dim=-1),
                                      B, H, W, w, 2 * C)
        N = H * W
        ft_f = ft.flatten(2).transpose(1, 2)
        fo_f = fo.flatten(2).transpose(1, 2)
        at = self._attn(self.q1(fo_f), self.k1(ft_f), self.v1(ft_f), N)
        ao = self._attn(self.q2(ft_f), self.k2(fo_f), self.v2(fo_f), N)
        ft_out = self.norm(ft_f + at)
        fo_out = self.norm(fo_f + ao)
        return torch.cat([ft_out, fo_out], dim=-1).transpose(1, 2).view(B, 2 * C, H, W)


class Decoder(nn.Module):
    """解码器：fused3 为底，fused2 替代 32×32 skip，其余沿用双分支 skip。"""

    def __init__(self, base=32, out_ch=1, dropout=0.0):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(8 * base, 4 * base, 2, 2)     # 16→32
        self.conv3 = conv_block(8 * base, 4 * base, dropout)         # fused3(up)+fused2
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2)      # 32→64
        self.conv2 = conv_block(6 * base, 2 * base, dropout)         # +skip_t1+skip_o1
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, 2)          # 64→128
        self.conv1 = conv_block(3 * base, base, dropout)             # +skip_t0+skip_o0
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, fused3, fused2, st, so):
        x = self.up3(fused3); x = torch.cat([x, fused2], 1); x = self.conv3(x)
        x = self.up2(x); x = torch.cat([x, st[1], so[1]], 1); x = self.conv2(x)
        x = self.up1(x); x = torch.cat([x, st[0], so[0]], 1); x = self.conv1(x)
        return self.head(x)


class CrossModalTransformer(nn.Module):
    def __init__(self, season_sel=None, fusion="cross_attn", base=32, heads=4, dropout=0.0,
                 attn_window=0):
        super().__init__()
        season_sel = season_sel or [0, 1, 2, 3]
        self.t_ch = [0 * 4 + s for s in season_sel]
        self.o_ch = [b * 4 + s for b in range(1, 9) for s in season_sel]
        self.fusion = fusion
        self.enc_t = Encoder(len(self.t_ch), base, dropout)
        self.enc_o = Encoder(len(self.o_ch), base, dropout)
        self.cross3 = CrossAttention(8 * base, heads, attn_window) if fusion == "cross_attn" else None
        self.cross2 = CrossAttention(4 * base, heads, attn_window) if fusion == "cross_attn" else None
        self.fuse3 = nn.Sequential(
            nn.Conv2d(16 * base, 8 * base, 1), nn.BatchNorm2d(8 * base), nn.ReLU(inplace=True))
        self.fuse2 = nn.Sequential(
            nn.Conv2d(8 * base, 4 * base, 1), nn.BatchNorm2d(4 * base), nn.ReLU(inplace=True))
        self.dec = Decoder(base, 1, dropout)

    def _fuse(self, ft, fo, cross, fuse):
        if cross is not None:
            return fuse(cross(ft, fo))
        return fuse(torch.cat([ft, fo], 1))

    def forward(self, x):
        ft = self.enc_t(x[:, self.t_ch])
        fo = self.enc_o(x[:, self.o_ch])
        fused3 = self._fuse(ft[3], fo[3], self.cross3, self.fuse3)   # 16×16
        fused2 = self._fuse(ft[2], fo[2], self.cross2, self.fuse2)   # 32×32
        return self.dec(fused3, fused2, ft, fo)


if __name__ == "__main__":
    x = torch.randn(2, 36, 128, 128)
    for kw in [dict(), dict(fusion="concat"), dict(season_sel=[1]),
               dict(attn_window=4), dict(attn_window=8)]:
        m = CrossModalTransformer(**kw)
        n = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{kw or '默认'} → {m(x).shape}  {n:.2f}M")
