from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class TrustRegionMappingClipConfig:
    use_tokenwise_trust_region_mapping_clip: bool = True
    band_radius_delta: float = 0.1
    delta_mapping_ratio_method: str = "sub" # "sub" 或 "div"
    use_one_minus_p_for_mapping_in_div: bool = True # 仅 method="div" 时有效
    mapping_eps: float = 1e-12 # 保护 p->0 的下限
    mapping_ratio_min_cap: float = 1e-6 # 防止极端情况下 ratio 下界过小
    mapping_ratio_max_cap: float = 1e6 # 防止极端情况下 ratio 上界过大

@torch.no_grad()
def _scalar_D_H_from_delta(delta: float):
    """
    用高精度稳定公式计算:
      D = 1 - exp(-delta) = -expm1(-delta)
      H = 0.5 * sqrt(1 - exp(-2*delta)) = 0.5 * sqrt(-expm1(-2*delta))
    返回 Python float（随后会转成与old_log_prob同device/dtype的tensor）
    """
    if delta < 0:
        raise ValueError(f"delta 必须 >= 0，当前={delta}")
    # 经验警告阈值（与题述一致）
    if delta > 0.5:
        print(f"[warn] delta={delta} 偏大，已超出常用TR范围")
    elif delta > 0.1:
        print(f"[warn] delta={delta} 稍大，注意近似误差")
    D = -math.expm1(-delta)
    H = 0.5 * math.sqrt(-math.expm1(-2.0 * delta))
    # 保证 D < H <= 0.5
    if not (D < H <= 0.5 + 1e-12):
        raise ValueError(f"D/H 条件不满足: D={D}, H={H}, delta={delta}")
    return D, H

def _ensure_tensor(x: float, like: Tensor):
    return torch.tensor(x, device=like.device, dtype=like.dtype)

def _coeffs_from_D_H(D: Tensor, H: Tensor):
    """
    由 D, H 封闭解系数:
      a = -4D
      b = 8D - 4H
      c = 4H - 5D
      d = D
    """
    a = -4.0 * D
    b = 8.0 * D - 4.0 * H
    c = 4.0 * H - 5.0 * D
    d = D
    return a, b, c, d

def _stable_inv(x: Tensor, eps: float) -> Tensor:
    return 1.0 / torch.clamp_min(x, eps)

def _ratio_bounds_sub(old_log_prob: Tensor,
                      a: Tensor, b: Tensor, c: Tensor, d: Tensor,
                      eps: float,
                      ratio_min_cap: float, ratio_max_cap: float) -> tuple[Tensor, Tensor]:
    """
    'sub' 路线：
      delta_up   = f_up(p) = a p^3 + b p^2 + c p + d
      delta_down = -f_up(1-p)
      q_high = clamp(p + delta_up, max=1)
      q_low  = clamp(p + delta_down, min=0)
      bound_high = q_high / p
      bound_low  = q_low / p
    """
    # 注意：old_log_prob 来自旧策略，按 PPO 习惯应视为常量，不参与梯度
    logp = old_log_prob.detach()
    p = torch.exp(logp)
    # 检查 p 是否在 [0, 1] 范围内
    # if torch.any(logp > 0):  # log p > 0 对应 p > 1，非法
    #     raise ValueError(f"Invalid log p, log p should be <= 0, got max(log p) = {logp.max()}")
    # if not torch.all((p >= 0) & (p <= 1)):
    #     raise ValueError(f"概率 p 的值超出了有效范围。当前 p 的值: {p}")
    one_minus_p = -torch.expm1(logp)  # 1 - p，稳定算式

    # Horner 形式
    delta_up = ((a * p + b) * p + c) * p + d
    delta_down = -(((a * one_minus_p + b) * one_minus_p + c) * one_minus_p + d)

    q_high = torch.clamp(p + delta_up, max=1.0)
    q_low  = torch.clamp(p + delta_down, min=0.0)

    inv_p = _stable_inv(p, eps)
    bound_high = q_high * inv_p
    bound_low  = q_low  * inv_p

    # 合理的安全截断，避免极端数值
    bound_high = torch.clamp(bound_high, min=ratio_min_cap, max=ratio_max_cap)
    bound_low  = torch.clamp(bound_low,  min=ratio_min_cap, max=ratio_max_cap)

    # 保证下界不超过上界
    bound_low = torch.minimum(bound_low, bound_high)
    return bound_low, bound_high

def _ratio_bounds_div(old_log_prob: Tensor,
                      a: Tensor, b: Tensor, c: Tensor, d: Tensor,
                      use_one_minus_p: bool,
                      eps: float,
                      ratio_min_cap: float, ratio_max_cap: float) -> tuple[Tensor, Tensor]:
    """
    'div' 路线（直接得到 ratio= q/p 的上下界）：
      high = 1 + (a p + b) p + c + d / p
      low  = 1 + delta_down / p         （use_one_minus_p=True）
         其中 delta_down = - f_up(1-p)
      或者（use_one_minus_p=False）：
         low = 1 + ((a p + (-3a-b)) p + (3a+2b+c)) - (a+b+c+d)/p
    """
    logp = old_log_prob.detach()
    p = torch.exp(logp)
    # 检查 p 是否在 [0, 1] 范围内
    # if torch.any(logp > 0):  # log p > 0 对应 p > 1，非法
    #     raise ValueError(f"Invalid log p, log p should be <= 0, got max(log p) = {logp.max()}")
    # if not torch.all((p >= 0) & (p <= 1)):
    #     raise ValueError(f"概率 p 的值超出了有效范围。当前 p 的值: {p}")
    one_minus_p = -torch.expm1(logp)  # 1 - p
    inv_p = _stable_inv(p, eps)

    bound_high = 1.0 + (a * p + b) * p + c + d * inv_p

    if use_one_minus_p:
        delta_down = -(((a * one_minus_p + b) * one_minus_p + c) * one_minus_p + d)
        bound_low = 1.0 + delta_down * inv_p
    else:
        bound_low = (
            1.0
            + ((a * p + (-3.0 * a - b)) * p + (3.0 * a + 2.0 * b + c))
            - (a + b + c + d) * inv_p
        )

    bound_high = torch.clamp(bound_high, min=ratio_min_cap, max=ratio_max_cap)
    bound_low  = torch.clamp(bound_low,  min=ratio_min_cap, max=ratio_max_cap)
    bound_low  = torch.minimum(bound_low, bound_high)
    return bound_low, bound_high

def compute_tokenwise_ratio_bounds(
    old_log_prob: Tensor,
    *,
    cfg: TrustRegionClipConfig):
    # 标量 D,H -> 转成与 old_log_prob 一致的 dtype/device
    D_scalar, H_scalar = _scalar_D_H_from_delta(cfg.band_radius_delta)
    D = _ensure_tensor(D_scalar, like=old_log_prob)
    H = _ensure_tensor(H_scalar, like=old_log_prob)

    a, b, c, d = _coeffs_from_D_H(D, H)

    if cfg.delta_mapping_ratio_method == "sub":
        return _ratio_bounds_sub(
            old_log_prob, a, b, c, d,
            eps=cfg.mapping_eps,
            ratio_min_cap=cfg.mapping_ratio_min_cap, ratio_max_cap=cfg.mapping_ratio_max_cap
        )
    elif cfg.delta_mapping_ratio_method == "div":
        return _ratio_bounds_div(
            old_log_prob, a, b, c, d,
            use_one_minus_p=cfg.use_one_minus_p_for_mapping_in_div,
            eps=cfg.mapping_eps,
            ratio_min_cap=cfg.mapping_ratio_min_cap, ratio_max_cap=cfg.mapping_ratio_max_cap
        )
    else:
        raise ValueError(f"未知 method={cfg.delta_mapping_ratio_method}, 只能是 'sub' 或 'div'")
