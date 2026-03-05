"""
Road-MoME v2.8 - 不确定性感知与防作弊路由架构 (完整版)
1. FiLM & Synergy: 维持级联交互，产生跨模态协同特征。
2. Uncertainty Heads: 专家级异方差不确定性预测 [log(sigma^2)]。
3. Anti-Cheating Gating: 不确定性预测作为“输入特征”提供给门控，而非直接乘以 Logit。
4. Stability: 强制 clamp(-5, 5) 防止异方差计算爆炸。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMCalibrator(nn.Module):
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


class RoadMoMENetV28(nn.Module):
    def __init__(self, config):
        super().__init__()
        d3, d2 = (
            config["features"]["3d"]["input_dim"],
            config["features"]["2d"]["input_dim"],
        )
        shared = config["features"]["shared_dim"]
        self.T = 1.8

        # 投影层
        self.proj_3d = nn.Sequential(
            nn.Linear(d3, shared), nn.LayerNorm(shared), nn.GELU()
        )
        self.proj_2d = nn.Sequential(
            nn.Linear(d2, shared), nn.LayerNorm(shared), nn.GELU()
        )

        # 协同组件
        self.film = FiLMCalibrator(d3, shared)
        self.synergy = SynergyReasoningBlock(shared)

        # 四大分类专家头
        self.expert_phys = nn.Linear(8, 1)
        self.expert_geom = nn.Linear(shared, 1)
        self.expert_tex = nn.Linear(shared, 1)
        self.expert_syn = nn.Linear(shared, 1)

        # [v2.8 新增] 不确定性头：映射特征到 log(sigma^2)
        self.uncert_geom = nn.Sequential(
            nn.Linear(shared, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.uncert_tex = nn.Sequential(
            nn.Linear(shared, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.uncert_syn = nn.Sequential(
            nn.Linear(shared, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # [v2.8 门控升级] 输入增加 3 维不确定性信号
        self.gating_net = nn.Sequential(
            nn.Linear(shared * 2 + 2 + 3, 128), nn.ReLU(), nn.Linear(128, 4)
        )

    def forward(self, phys, geom, tex, quality_vec):
        B, N, _ = phys.shape
        f3_raw = geom.view(B * N, -1)
        f2_raw = tex.expand(B, N, -1).reshape(B * N, -1)
        q_in = quality_vec.view(B * N, -1)

        # 1. 特征提取与协同
        f3_h = self.proj_3d(f3_raw)
        f2_h = self.proj_2d(f2_raw)
        f2_calib = self.film(f3_raw, f2_h)
        f_syn = self.synergy(f3_h, f2_calib)

        # 2. 结论与不确定性计算
        l_p = self.expert_phys(phys.view(B * N, -1))
        l_g = self.expert_geom(f3_h)
        l_t = self.expert_tex(f2_calib)
        l_s = self.expert_syn(f_syn)

        # 数值稳定性保护
        s_g = torch.clamp(self.uncert_geom(f3_h), -5, 5)
        s_t = torch.clamp(self.uncert_tex(f2_calib), -5, 5)
        s_s = torch.clamp(self.uncert_syn(f_syn), -5, 5)
        uncertainties = torch.cat([s_g, s_t, s_s], dim=-1)

        # 3. 防作弊门控路由
        # 不确定性作为环境输入的一部分，由门控决定是否降权
        gate_input = torch.cat([f3_h, f2_h, q_in, uncertainties], dim=-1)
        gate_logits = self.gating_net(gate_input)

        # 物理质量惩罚
        penalty_scale = 0.7
        gate_logits[:, 0:1] += torch.log(torch.tensor([0.2], device=phys.device))
        gate_logits[:, 1:2] += penalty_scale * torch.log(q_in[:, 0:1] + 1e-6)
        gate_logits[:, 2:3] += penalty_scale * torch.log(q_in[:, 1:2] + 1e-6)
        gate_logits[:, 3:4] += penalty_scale * torch.log(
            q_in[:, 0:1] * q_in[:, 1:2] + 1e-6
        )

        weights = F.softmax(gate_logits / self.T, dim=-1)

        # 4. 融合
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
            uncertainties.view(B, N, 3),
        )


def build_mome_model(config):
    return RoadMoMENetV28(config)
