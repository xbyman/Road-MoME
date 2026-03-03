"""
Road-MoME v2.2 - 级联调制协同架构 (FiLM + Synergy Hybrid)
核心逻辑：
1. FiLM Calibration: 利用 3D 几何特征生成 (1+gamma) 缩放因子，校准 2D 纹理分布。
2. Synergy Interaction: 在校准后的空间内，计算 [f3, f2_calib, dot, diff] 产生协同特征 f_syn。
3. Residual Prediction: 最终预测由 f_syn 主导，叠加 f3 物理残差，确保鲁棒性。
4. Quality-aware Gating: 路由决策基于原始特征与外部物理质量因子 [q_geo, q_img]。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMCalibrator(nn.Module):
    """
    [阶段一] 特征校准模块
    功能：利用 3D 先验对 2D 特征进行线性调制 (去噪/对齐)
    """

    def __init__(self, cond_dim, target_dim):
        super().__init__()
        # 极简两层 MLP 产生 gamma 和 beta
        self.net = nn.Sequential(
            nn.Linear(cond_dim, target_dim // 2),
            nn.GELU(),
            nn.Linear(target_dim // 2, target_dim * 2),
        )
        # 初始化为 0，确保训练初期是恒等映射 (Identity Map)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f3, f2):
        params = self.net(f3)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        gamma = torch.tanh(gamma) * 0.5  # 限制调制幅度
        # 校准运算
        f2_calibrated = (1.0 + gamma) * f2 + beta
        return f2_calibrated


class SynergyReasoningBlock(nn.Module):
    """
    [阶段二] 协同推理模块
    功能：在校准后的特征上提取高阶交互特征
    """

    def __init__(self, dim, low_dim=64):
        super().__init__()
        # 采用降维设计 (BottleNeck)，防止参数量爆炸
        self.reduction = nn.Sequential(
            nn.Linear(dim * 4, low_dim),
            nn.LayerNorm(low_dim),
            nn.GELU(),
            nn.Linear(low_dim, dim),  # 还原回 shared_dim
        )

    def forward(self, f3, f2_c):
        # 显式建模一致性与冲突
        dot_product = f3 * f2_c
        diff = torch.abs(f3 - f2_c)
        # 拼接产生 f_syn
        combined = torch.cat([f3, f2_c, dot_product, diff], dim=-1)
        f_syn = self.reduction(combined)
        return f_syn


class RoadMoMENetV22(nn.Module):
    def __init__(self, config):
        super().__init__()
        d3 = config["features"]["3d"]["input_dim"]  # 384
        d2 = config["features"]["2d"]["input_dim"]  # 768
        shared = 128

        # 1. 投影层
        self.proj_3d = nn.Sequential(
            nn.Linear(d3, shared), nn.LayerNorm(shared), nn.GELU()
        )
        self.proj_2d = nn.Sequential(
            nn.Linear(d2, shared), nn.LayerNorm(shared), nn.GELU()
        )

        # 2. 级联引擎
        self.film = FiLMCalibrator(d3, shared)  # 3D 调 2D
        self.synergy = SynergyReasoningBlock(shared, low_dim=64)  # 产生协同特征

        # 3. 四大专家头 (注意逻辑变化)
        self.expert_phys = nn.Linear(8, 1)  # 原始物理专家
        self.expert_geom = nn.Linear(shared, 1)  # 纯几何语义专家
        self.expert_tex = nn.Linear(shared, 1)  # 纯纹理语义专家 (调制前)
        self.expert_syn = nn.Linear(shared, 1)  # [核心] 协同专家

        # 4. 增强型门控 (接收特征 + 物理质量分 q_geo, q_img)
        self.gating_net = nn.Sequential(
            nn.Linear(shared * 2 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1),
        )

    def forward(self, phys, geom, tex, quality_vec):
        B, N, _ = phys.shape
        f3_raw = geom.view(B * N, -1)
        f2_raw = tex.expand(B, N, -1).reshape(B * N, -1)

        # --- 级联流水线 ---
        # (1) 基础特征提取
        f3_h = self.proj_3d(f3_raw)
        f2_h = self.proj_2d(f2_raw)

        # (2) FiLM 校准: 用原始 3D 引导投影后的 2D
        f2_calib = self.film(f3_raw, f2_h)

        # (3) Synergy 交互: 产生新特征 f_syn
        f_syn = self.synergy(f3_h, f2_calib)

        # --- 路由决策 ---
        q_in = quality_vec.view(B * N, -1)  # [q_geo, q_img]
        gate_input = torch.cat([f3_h, f2_h, q_in], dim=-1)
        weights = self.gating_net(gate_input)

        # --- 专家结论 ---
        l_p = self.expert_phys(phys.view(B * N, -1))
        l_g = self.expert_geom(f3_h)
        l_t = self.expert_tex(f2_h)

        # [学术改进] 协同结论使用残差连接: f_syn_logit + f3_logit
        # 即使校准完全失效，协同专家也能保留 3D 的基本判定
        l_s = self.expert_syn(f_syn) + 0.5 * l_g

        # --- 加权融合 ---
        final_logit = (
            weights[:, 0:1] * l_p
            + weights[:, 1:2] * l_g
            + weights[:, 2:3] * l_t
            + weights[:, 3:4] * l_s
        )

        return final_logit.view(B, N), weights.view(B, N, 4)


def build_mome_model(config):
    return RoadMoMENetV22(config)
