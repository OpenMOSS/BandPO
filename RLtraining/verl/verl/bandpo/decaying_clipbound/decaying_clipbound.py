from __future__ import annotations
from typing import Optional, Literal, Union
import numpy as np
from numbers import Number

ArrayLike = Union[float, np.ndarray]

# -------------------------
# 数值辅助
# -------------------------
def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)

def _clip01(x: ArrayLike) -> np.ndarray:
    return np.clip(_as_array(x), 0.0, 1.0)

def _validate_start_end(start_value: float, end_value: float):
    assert np.isfinite(start_value) and np.isfinite(end_value), \
        f"start_value/end_value 必须为有限数: start={start_value}, end={end_value}"

def _progress_from_steps(step_now: int, step_total: int) -> float:
    assert isinstance(step_now, (int, np.integer)) and isinstance(step_total, (int, np.integer)), \
        f"type(step_now)={type(step_now)}, type(step_total)={type(step_total)}"
    assert step_total > 0 and 0 <= step_now <= step_total, \
        f"step_now={step_now}, step_total={step_total}（需满足 0<=step_now<=step_total 且 step_total>0）"
    s = step_now / float(step_total)
    return float(np.clip(s, 0.0, 1.0))

# 在工具函数区新增两个小工具：
def _is_scalar_like(x) -> bool:
    """判断输入是否“标量语义”：Python标量、0维ndarray、或仅含一个元素的序列/数组。"""
    if x is None:
        return False
    if isinstance(x, Number):  # int/float/bool等
        return True
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    return arr.size == 1  # 包括 0维 和 (1,) 情况

def _return_scalar_or_array(y, want_scalar: bool):
    """
    若 want_scalar=True，则断言 y 只有1个元素，并返回 float；
    否则返回 np.ndarray（float64）。
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if want_scalar:
        assert y_arr.size == 1, f"期望返回标量，但得到 shape={y_arr.shape}, size={y_arr.size}"
        return float(y_arr.reshape(-1)[0])
    else:
        return y_arr

# --------------------
# A) Linear schedules
# --------------------
def decaying_clip_linear(process_ratio: ArrayLike,
                         start_value: float,
                         end_value: float) -> np.ndarray:
    _validate_start_end(start_value, end_value)
    s = _clip01(process_ratio)
    return start_value + (end_value - start_value) * s

def decaying_clip_hold_then_linear(process_ratio: ArrayLike,
                                   start_value: float,
                                   end_value: float,
                                   hold_frac: float = 0.30) -> np.ndarray:
    _validate_start_end(start_value, end_value)
    s = _clip01(process_ratio)
    hold_frac = float(np.clip(hold_frac, 0.0, 0.999999))
    slope = (end_value - start_value) / (1.0 - hold_frac) if hold_frac < 1.0 else 0.0
    eps_linear = start_value + slope * (s - hold_frac)
    return np.where(s <= hold_frac, start_value, eps_linear)

# ----------------------
# B) Cosine schedules
# ----------------------
def decaying_clip_cosine(process_ratio: ArrayLike,
                         start_value: float,
                         end_value: float) -> np.ndarray:
    _validate_start_end(start_value, end_value)
    s = _clip01(process_ratio)
    return end_value + 0.5 * (start_value - end_value) * (1.0 + np.cos(np.pi * s))

# -------------------------
# C) Polynomial schedules
# -------------------------
def decaying_clip_poly(process_ratio: ArrayLike,
                       start_value: float,
                       end_value: float,
                       exponent: float = 2.0) -> np.ndarray:
    _validate_start_end(start_value, end_value)
    assert exponent > 0.0, f"exponent 必须 > 0，但得到 {exponent}"
    s = _clip01(process_ratio)
    return end_value + (start_value - end_value) * ((1.0 - s) ** exponent)

# ---------------------------------------------
# D) Restarts (统一版：fixed / expanding + 衰减 + 周期内函数型)
# ---------------------------------------------
def _sgdr_periods(T0: float,
                  T_mult: float,
                  s_end: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    生成覆盖 [0, s_end] 的周期序列（在 s 空间）。
    返回 (starts, lengths)，长度单调增长：T0, T0*T_mult, ...
    """
    assert T0 > 0.0 and T_mult > 0.0 and s_end > 0.0
    periods = []
    T = float(T0)
    total = 0.0
    while total < s_end - 1e-12:
        Ti = min(T, s_end - total)
        periods.append(Ti)
        total += Ti
        T *= T_mult
    starts = np.cumsum([0.0] + periods[:-1])
    return np.array(starts, dtype=np.float64), np.array(periods, dtype=np.float64)

def decaying_clip_restarts(
    process_ratio: ArrayLike,
    start_value: float,
    end_value: float,
    *,
    # 周期与扩张
    mode: Literal["fixed", "expanding"] = "expanding",
    T0: float = 0.15,
    T_mult: float = 2.0,
    # 衰减开关
    decay: bool = False,
    gamma: float = 0.8,
    # 周期内“函数型”
    shape: Literal["cosine", "constant"] = "cosine",
) -> np.ndarray:
    """
    统一的“重启”调度：
    - mode:
        "fixed"     → 固定周期长度（等于 T0）
        "expanding" → 周期长度按 T0, T0*T_mult, T0*T_mult^2, ... 扩张
    - decay:
        True/False 开关。注意：
          * shape="cosine"  时：使用几何衰减 gamma^k（与经典 SGDR 一致）
          * shape="constant"时：忽略 gamma，采用“分段阶梯式”衰减：
                第 k 段常数为 start + (k/(K-1))*(end - start)，K 为总周期数（K>=2 时保证最后段等于 end）
    """
    _validate_start_end(start_value, end_value)
    s = _clip01(process_ratio)

    # 周期生成
    T_mult_eff = 1.0 if mode == "fixed" else float(T_mult)
    starts, lengths = _sgdr_periods(T0=T0, T_mult=T_mult_eff, s_end=1.0)
    K = len(lengths)

    # 每个 s 所在的周期索引，与 s 等形
    idx = np.searchsorted(starts[1:], s, side="right")
    phase = (s - starts[idx]) / lengths[idx]  # 归一化到 [0,1]

    if shape == "cosine":
        # 余弦内核：经典 SGDR（可选几何衰减）
        amp_scale = (gamma ** idx) if decay else np.ones_like(s, dtype=np.float64)
        amp = (start_value - end_value) * amp_scale
        y = end_value + 0.5 * amp * (1.0 + np.cos(np.pi * phase))
    elif shape == "constant":
        if decay and K >= 2:
            # 分段阶梯：k=0→start, k=K-1→end，严格落到 end
            denom = (K - 1)
            step_ratio = idx / denom
            y = start_value + (end_value - start_value) * step_ratio
        elif decay and K < 2:
            # 只有一个周期无法阶梯衰减；退化为常数 start（保兼容）
            y = np.full_like(s, fill_value=start_value, dtype=np.float64)
        else:
            # 不衰减：整段常数为 start_value
            y = np.full_like(s, fill_value=start_value, dtype=np.float64)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return np.broadcast_to(np.asarray(y, dtype=np.float64), np.shape(s))

# -------------------------
# 统一分发器（工程调用）
# -------------------------
def apply_decaying_clip(
    *,
    step_now: Optional[int] = None,
    step_total: Optional[int] = None,
    process_ratio: Optional[ArrayLike] = None,  # 可直接给 s∈[0,1]；给了就忽略 step_*。
    start_value: float,
    end_value: float,
    # A/B/C 超参
    hold_frac: float = 0.25,
    exponent: float = 2.0,
    # 选择器 + D 超参
    method: Literal[
        "linear",
        "hold_linear",
        "cosine",
        "poly",
        "restarts",
    ] = "linear",
    # restarts 专属参数：
    mode: Literal["fixed", "expanding"] = "expanding",
    T0: float = 0.15,
    T_mult: float = 2.0,
    decay: bool = False,
    gamma: float = 0.8,
    shape: Literal["cosine", "constant"] = "cosine",
):
    _validate_start_end(start_value, end_value)

    # 判定“返回是否应该是标量”
    # 1) 用 step_* → 一定返回标量
    # 2) 用 process_ratio：
    #    - 标量/长度1 → 返回标量
    #    - 向量/矩阵 → 返回数组
    using_steps = (process_ratio is None)
    want_scalar = using_steps or _is_scalar_like(process_ratio)

    # 计算 s（数组语义，便于复用现有实现）
    if using_steps:
        assert step_now is not None and step_total is not None, \
            "需给出 step_now/step_total 或直接给 process_ratio"
        s = _progress_from_steps(step_now, step_total)   # 返回 float
        s = np.asarray(s, dtype=np.float64)              # 统一成数组语义（0维）
    else:
        s = _clip01(process_ratio)                       # 可能是标量或数组

    # 分发得到 y（数组语义）
    m = method.lower()
    if m == "linear":
        y = decaying_clip_linear(s, start_value, end_value)
    elif m == "hold_linear":
        y = decaying_clip_hold_then_linear(s, start_value, end_value, hold_frac=hold_frac)
    elif m == "cosine":
        y = decaying_clip_cosine(s, start_value, end_value)
    elif m == "poly":
        y = decaying_clip_poly(s, start_value, end_value, exponent=exponent)
    elif m == "restarts":
        y = decaying_clip_restarts(
            s, start_value, end_value,
            mode=mode, T0=T0, T_mult=T_mult,
            decay=decay, gamma=gamma,
            shape=shape,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # —— 返回类型按需收敛为 float 或 ndarray ——
    return _return_scalar_or_array(y, want_scalar)

