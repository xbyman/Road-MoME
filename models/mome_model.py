"""
Road-MoME v2.0 - 特征级协同融合架构
核心改进：
1. Synergy Block: 显式建模模态间的一致性 (Hadamard Product) 与 冲突 (Absolute Residual)。
2. Feature-level Fusion: 门控网络不再仅看原始特征，而是基于交互后的协同表征进行路由决策。
3. 维度解耦: 支持 3D (384d) 与 2D (768d) 的非对称输入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SynergyModule(nn.Module):
    """
    [核心创新点] 跨模态协同模块
    功能：通过投影对齐空间，并计算一致性特征与残差特征。
    """

    def __init__(self, dim_3d, dim_2d, shared_dim=128):
        super(SynergyModule, self).__init__()

        # 1. 投影层：将 3D (384) 和 2D (768) 映射到统一维度 (128)
        self.proj_3d = nn.Sequential(
            nn.Linear(dim_3d, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )
        self.proj_2d = nn.Sequential(
            nn.Linear(dim_2d, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )

        # 2. 协同特征提取层 (MLP)
        # 输入：[f3, f2, f3 * f2, |f3 - f2|] 共 4 个 shared_dim
        self.synergy_mlp = nn.Sequential(
            nn.Linear(shared_dim * 4, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, f3, f2):
        # f3: [B*N, 384], f2: [B*N, 768]
        f3_hat = self.proj_3d(f3)
        f2_hat = self.proj_2d(f2)

        # 计算一致性特征 (Element-wise multiplication)
        # 强调两个模态共同激活的部分
        consistency = f3_hat * f2_hat

        # 计算冲突特征 (Absolute difference)
        # 捕捉模态间的矛盾信号
        conflict = torch.abs(f3_hat - f2_hat)

        # 产生协同特征向量 f_syn
        concat_feat = torch.cat([f3_hat, f2_hat, consistency, conflict], dim=-1)
        f_syn = self.synergy_mlp(concat_feat)

        return f_syn, f3_hat, f2_hat


class ExpertHead(nn.Module):
    """单专家分类头"""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1)  # 输出 Logit
        )

    def forward(self, x):
        return self.net(x)


class RoadMoMENet(nn.Module):
    """
    升级后的 MoME 主模型
    实现了从“决策融合”到“特征协同”的跨越
    """

    def __init__(self, config):
        super(RoadMoMENet, self).__init__()

        d3 = config["features"]["3d"]["input_dim"]  # 384
        d2 = config["features"]["2d"]["input_dim"]  # 768
        dp = config["features"]["phys"]["input_dim"]  # 8
        shared = 128

        # 1. 协同特征引擎
        self.synergy_engine = SynergyModule(d3, d2, shared_dim=shared)

        # 2. 三大专家分支 (物理专家独立处理)
        self.phys_proj = nn.Sequential(
            nn.Linear(dp, shared), nn.BatchNorm1d(shared), nn.ReLU()
        )

        self.expert_phys = ExpertHead(shared)  # 物理专家
        self.expert_geom = ExpertHead(shared)  # 3D 专家 (看投影后的几何)
        self.expert_tex = ExpertHead(shared)  # 2D 专家 (看投影后的纹理)
        self.expert_syn = ExpertHead(shared)  # 协同专家 (专门看 f_syn)

        # 3. 门控网络 (Gating Network)
        # 输入：[f_phys, f_geom, f_tex, f_syn]
        self.gating_net = nn.Sequential(
            nn.Linear(shared * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 对应 4 个专家的权重
            nn.Softmax(dim=-1),
        )

    def forward(self, phys_feat, geom_feat, tex_feat):
        """
        phys_feat: [B, N, 8]
        geom_feat: [B, N, 384]
        tex_feat:  [B, 1, 768]
        """
        B, N, _ = phys_feat.shape

        # 展平处理
        p_x = phys_feat.view(B * N, -1)
        g_x = geom_feat.view(B * N, -1)
        t_x = tex_feat.expand(B, N, -1).reshape(B * N, -1)

        # 1. 特征交互产生新特征
        f_syn, f_g, f_t = self.synergy_engine(g_x, t_x)
        f_p = self.phys_proj(p_x)

        # 2. 专家独立给出结论
        l_p = self.expert_phys(f_p)
        l_g = self.expert_geom(f_g)
        l_t = self.expert_tex(f_t)
        l_s = self.expert_syn(f_syn)

        # 3. 门控权重决策 (基于协同特征流)
        gate_input = torch.cat([f_p, f_g, f_t, f_syn], dim=-1)
        weights = self.gating_net(gate_input)  # [B*N, 4]

        # 4. 加权融合
        final_logit = (
            weights[:, 0:1] * l_p
            + weights[:, 1:2] * l_g
            + weights[:, 2:3] * l_t
            + weights[:, 3:4] * l_s
        )

        return final_logit.view(B, N), weights.view(B, N, 4)


def build_mome_model(config):
    return RoadMoMENet(config)
