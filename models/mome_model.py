"""
Road-MoME v2.6 - 模态感知唤醒版 (完整脚本)
核心升级：
1. Expert 2D Calibration: 2D 专家不再观察原始图像特征，而是观察经 FiLM 校准后的几何关联特征。
2. Multi-Path Dropout: 引入协同路径(30%)与几何路径(20%)的随机封禁，强制视觉专家（2D）在孤立状态下学习。
3. Gate Incentive Bias: 为 2D 分支提供正向路由激励，平衡伪标签的几何偏见。
4. Feature Synergy: 维持 v2.4 的去冗余设计，协同专家专注于跨模态一致性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class FiLMCalibrator(nn.Module):
    """
    [阶段一] 特征校准模块
    """

    def __init__(self, cond_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, target_dim // 2),
            nn.GELU(),
            nn.Linear(target_dim // 2, target_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f3, f2):
        params = self.net(f3)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        gamma = torch.tanh(gamma) * 0.5
        return (1.0 + gamma) * f2 + beta


class SynergyReasoningBlock(nn.Module):
    """
    [阶段二] 协同推理模块 (v2.4+ 极致去冗余)
    """

    def __init__(self, dim, low_dim=64):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Linear(dim * 2, low_dim),
            nn.LayerNorm(low_dim),
            nn.GELU(),
            nn.Linear(low_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, f3, f2_c):
        consistency = f3 * f2_c
        conflict = torch.abs(f3 - f2_c)
        combined = torch.cat([consistency, conflict], dim=-1)
        return self.reduction(combined)


class RoadMoMENetV23(nn.Module):
    def __init__(self, config):
        super().__init__()
        d3 = config["features"]["3d"]["input_dim"]
        d2 = config["features"]["2d"]["input_dim"]
        shared = config.get("features", {}).get("shared_dim", 128)

        self.T = 2.0
        self.penalty_scale = 1.0
        self.syn_drop_prob = 0.3  # 协同专家丢弃概率
        self.geom_drop_prob = 0.2  # 几何专家丢弃概率

        self.proj_3d = nn.Sequential(
            nn.Linear(d3, shared), nn.LayerNorm(shared), nn.GELU()
        )
        self.proj_2d = nn.Sequential(
            nn.Linear(d2, shared), nn.LayerNorm(shared), nn.GELU()
        )

        self.film = FiLMCalibrator(d3, shared)
        self.synergy = SynergyReasoningBlock(shared)

        self.expert_phys = nn.Linear(8, 1)
        self.expert_geom = nn.Linear(shared, 1)
        self.expert_tex = nn.Linear(shared, 1)
        self.expert_syn = nn.Linear(shared, 1)

        self.gating_net = nn.Sequential(
            nn.Linear(shared * 2 + 2, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    def forward(self, phys, geom, tex, quality_vec):
        B, N, _ = phys.shape
        f3_raw = geom.view(B * N, -1)
        f2_raw = tex.expand(B, N, -1).reshape(B * N, -1)
        q_in = quality_vec.view(B * N, -1)

        # --- (A) 特征加工 ---
        f3_h = self.proj_3d(f3_raw)
        f2_h = self.proj_2d(f2_raw)
        f2_calib = self.film(f3_raw, f2_h)  # 关键：3D引导校准后的视觉特征
        f_syn = self.synergy(f3_h, f2_calib)

        # --- (B) 计算专家结论 ---
        l_p = self.expert_phys(phys.view(B * N, -1))
        l_g = self.expert_geom(f3_h)
        # [v2.6 改进] 2D专家现在基于“已校准特征”预测，增强其对病害几何分布的敏感度
        l_t = self.expert_tex(f2_calib)
        l_s = self.expert_syn(f_syn)

        # --- (C) 路由决策 ---
        gate_input = torch.cat([f3_h, f2_h, q_in], dim=-1)
        gate_logits = self.gating_net(gate_input)

        # 1. 物理抑制
        gate_logits[:, 0:1] += torch.log(torch.tensor([0.2], device=phys.device))

        # [v2.6 改进] 路由激励：给 2D 专家 2.0 的初始 Logit 奖励，抵消 3D 标签偏见
        # 索引 2 为 2D-Tex
        gate_logits[:, 2:3] += 2.0

        # 2. 质量硬约束
        q_geo, q_img = q_in[:, 0:1], q_in[:, 1:2]
        gate_logits[:, 1:2] += self.penalty_scale * torch.log(q_geo + 1e-6)
        gate_logits[:, 2:3] += self.penalty_scale * torch.log(q_img + 1e-6)
        gate_logits[:, 3:4] += self.penalty_scale * torch.log(q_geo * q_img + 1e-6)

        # [v2.6 改进] 复合路径丢弃策略 (仅在训练模式下)
        if self.training:
            rand_val = random.random()
            if rand_val < self.syn_drop_prob:
                gate_logits[:, 3:4] -= 1e9  # 封禁协同
            elif rand_val < self.syn_drop_prob + self.geom_drop_prob:
                gate_logits[:, 1:2] -= 1e9  # 封禁 3D 几何，逼迫模型学习 2D 纹理

        weights = F.softmax(gate_logits / self.T, dim=-1)

        # --- (D) 结果合成 ---
        final_logit = (
            weights[:, 0:1] * l_p
            + weights[:, 1:2] * l_g
            + weights[:, 2:3] * l_t
            + weights[:, 3:4] * l_s
        )

        expert_logits = torch.cat([l_p, l_g, l_t, l_s], dim=-1)

        return (
            final_logit.view(B, N),
            weights.view(B, N, 4),
            expert_logits.view(B, N, 4),
        )


def build_mome_model(config):
    return RoadMoMENetV23(config)
