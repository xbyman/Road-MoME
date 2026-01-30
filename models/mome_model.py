import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertBranch(nn.Module):
    """
    通用专家分支：将不同模态的特征映射到 128 维统一隐空间
    """

    def __init__(self, input_dim, hidden_dim=128, use_bn=True):
        super(ExpertBranch, self).__init__()
        # 针对不同维度的特征，设计自适应的 MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        # 独立的专家分类头 (Logit 输出)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [B * N, input_dim]
        feat = self.net(x)
        logit = self.classifier(feat)
        return feat, logit


class RoadMoMENet(nn.Module):
    """
    完善后的 MoME 主架构
    1. 物理分支量级对齐 (BatchNorm)
    2. 全帧视觉背景广播逻辑
    3. 协同门控路由器
    """

    def __init__(self, config):
        super(RoadMoMENet, self).__init__()

        # 1. 初始化三大专家
        # 物理专家必须开启 BatchNorm 以对齐不同量级的统计量
        self.phys_expert = ExpertBranch(
            config["features"]["phys"]["input_dim"], use_bn=True
        )
        self.geom_expert = ExpertBranch(
            config["features"]["3d"]["input_dim"], use_bn=True
        )
        self.tex_expert = ExpertBranch(
            config["features"]["2d"]["input_dim"], use_bn=True
        )

        # 2. 门控网络 (Gating Network)
        # 输入：三专家特征拼接 (128 * 3 = 384)
        self.gating_net = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, phys_feat, geom_feat, tex_feat):
        """
        phys_feat: [B, N, 8]
        geom_feat: [B, N, 384]
        tex_feat:  [B, 1, 768] (全帧特征)
        """
        B, N, _ = phys_feat.shape

        # 展平处理 Batch 内部所有的 Patch
        phys_x = phys_feat.view(B * N, -1)
        geom_x = geom_feat.view(B * N, -1)

        # 纹理特征广播：[B, 1, 768] -> [B, N, 768]
        tex_x = tex_feat.expand(B, N, -1).contiguous().view(B * N, -1)

        # --- 1. 专家独立推理 ---
        f_p, l_p = self.phys_expert(phys_x)
        f_g, l_g = self.geom_expert(geom_x)
        f_t, l_t = self.tex_expert(tex_x)

        # --- 2. 门控权重计算 ---
        gate_input = torch.cat([f_p, f_g, f_t], dim=-1)  # [B*N, 384]
        weights = self.gating_net(gate_input)  # [B*N, 3]

        # --- 3. 结果加权融合 ---
        # final_logit = w_phys * l_phys + w_geom * l_geom + w_tex * l_tex
        final_logit = (
            weights[:, 0:1] * l_p + weights[:, 1:2] * l_g + weights[:, 2:3] * l_t
        )

        return final_logit.view(B, N), weights.view(B, N, 3)


def build_mome_model(config):
    return RoadMoMENet(config)
