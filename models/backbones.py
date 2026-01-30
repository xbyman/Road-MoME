import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """适配官方命名的 MLP 模块"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """适配官方命名的 Attention 模块"""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PointNetEmbedding(nn.Module):
    """对应 Point_MAE.py 中的 class Encoder"""

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Block(nn.Module):
    """适配官方权重的 Transformer Block"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RoadPointMAEEncoder(nn.Module):
    def __init__(self, trans_dim=384, depth=12, num_heads=6, encoder_dims=384):
        super().__init__()
        self.trans_dim = trans_dim
        self.encoder = PointNetEmbedding(encoder_channel=encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        # 修复点：将 qkv_bias 改为 False，以匹配官方预训练权重
        self.blocks = nn.ModuleList(
            [
                Block(dim=trans_dim, num_heads=num_heads, qkv_bias=False)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, neighborhood, center):
        group_input_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)
        x = group_input_tokens + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.max(x, dim=1)[0]


def load_official_pretrain(model, ckpt_path):
    """
    更加鲁棒的权重加载函数，自动识别并剥离 module. 或 MAE_encoder. 等前缀
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("base_model", checkpoint.get("model", checkpoint))

    new_state_dict = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key.replace("module.", "")
        if key.startswith("MAE_encoder."):
            key = key.replace("MAE_encoder.", "")

        # 针对 TransformerEncoder 内部的嵌套逻辑进行微调
        if key.startswith("blocks.blocks."):
            key = key.replace("blocks.blocks.", "blocks.")

        if key in model.state_dict():
            new_state_dict[key] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    matched_count = len(new_state_dict)
    print(f"✅ 权重加载结果 | 匹配项: {matched_count} | 状态: {msg}")

    if matched_count < 100:
        print("❌ 警告：匹配项过少！请检查模型定义是否与权重完全一致。")
    return model
