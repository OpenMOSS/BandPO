import numpy as np
import math
import warnings
from decimal import Decimal, getcontext

def compute_1_minus_exp_neg_x_by_math(x):
    """
    精确计算 1 - e^(-x)
    使用恒等式: 1 - e^(-x) = -expm1(-x)
    """
    return -math.expm1(-x)

def compute_1_minus_exp_neg_x_by_numpy(x):
    """
    精确计算 1 - e^(-x)
    使用恒等式: 1 - e^(-x) = -expm1(-x)
    """
    return -np.expm1(-x)

def get_D_H_from_delta(delta, high_precision=True, use_numpy=False):
    # 根据KL的delta计算D和H，可直接推导出公式：
    if delta < 0:
        raise ValueError(f"delta必须大于等于0，当前值为{delta}")
    elif delta > 0.5:
        warnings.warn(f"delta值过大，不符合小delta假设，当前值为{delta}", UserWarning)
    elif delta > 0.1:
        warnings.warn(f"delta值稍大，当前值为{delta}", UserWarning)

    if high_precision:
        if use_numpy:
            D = compute_1_minus_exp_neg_x_by_numpy(delta)
            H = np.sqrt(compute_1_minus_exp_neg_x_by_numpy(2*delta)) / 2
        else:
            D = compute_1_minus_exp_neg_x_by_math(delta)
            H = np.sqrt(compute_1_minus_exp_neg_x_by_math(2*delta)) / 2
    else:
        if use_numpy:
            D = 1 - np.exp(-delta)
            H = np.sqrt(1 - np.exp(-2*delta)) / 2
        else:
            D = 1 - math.exp(-delta)
            H = math.sqrt(1 - math.exp(-2*delta)) / 2
    return D, H

def cubic_polynomial_clip(D, H):
    # 验证输入参数
    if not (D < H <= 0.5):
        raise ValueError(f"参数不满足条件 D < H <= 0.5: D={D}, H={H}")
    # 变量：a, b, c, d
    # f(x) = a*x^3 + b*x^2 + c*x + d
    # f‘(x) = 3*a*x^2 + 2*b*x + c # 对称轴 x = -b/(3a)
    # 方程1: f(1) = 0 (线性)： a + b + c + d = 0      
    # 方程2：f(0.5) = H (线性)：0.125*a + 0.25*b + 0.5*c + d = H
    # 方程3: f'(0.5) = 0 (线性)：0.75*a + b + c = 0
    # 方程4:  f(0) = D (线性)： d = D
    # 其中包含了不等式约束：（x=0.5在对称轴左侧且a>0）或者（x=0.5在对称轴右侧且a<0）。但是由于需要0.5左侧速度慢，右侧速度快，所以a<0
    # 求解可得：
    a = -4*D
    b = 8*D-4*H
    c = 4*H - 5*D
    assert c>=0
    d = D
    solution = {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "polynomial": lambda x: a*x**3 + b*x**2 + c*x + d
    }
    return solution

def feasible_variation_range_up_2_clip_ration(p,a,b,c,d):
    # 计算在token概率为p时clip上界
    # 计算旧策略p向上的可移动的区间
    delta_up = a*p**3 + b*p**2 + c*p + d
    # 新策略的可移动上界
    q_high = p + delta_up
    # 计算clip_ratio上界
    epsilon_high = delta_up / p
    bound_up = 1 + epsilon_high

    # 根据镜像映射计算旧策略p向下的可移动的区间
    delta_down =  -1 * (a*(1-p)**3 + b*(1-p)**2 + c*(1-p) + d)
    # 新策略的可移动下界
    q_low = p + delta_down
    # 计算clip_ratio下界
    epsilon_low = -1 * delta_down / p
    bound_low = 1 - epsilon_low

    # 返回一个字典
    return {
        "delta_up": delta_up,
        "delta_down": delta_down,
        "q_high": q_high,
        "q_low": q_low,
        # "epsilon_high": epsilon_high,
        # "epsilon_low": epsilon_low,
        "bound_up": bound_up,
        "bound_low": bound_low
    }

def feasible_variation_range_up_2_clip_ratio_hybrid(p, a, b, c, d, high_precision=True, precision_threshold=1e-10):
    assert 0 <= p <= 1, f"p值必须在[0,1]范围内，当前p={p}"
    # 检查是否需要高精度
    need_high_precision = high_precision or abs(p) < precision_threshold or abs(1-p) < precision_threshold
    flag_too_close_to_0 = False
    if need_high_precision:
        # 使用 Decimal 高精度计算
        getcontext().prec = 50
        # 转换为 Decimal
        p_d = Decimal(str(p))
        a_d = Decimal(str(a))
        b_d = Decimal(str(b))
        c_d = Decimal(str(c))
        d_d = Decimal(str(d))
        
        # 使用 Horner 方法计算多项式
        delta_up = ((a_d * p_d + b_d) * p_d + c_d) * p_d + d_d # 计算旧策略p向上的可移动的区间
        assert delta_up >= -1e-10, f"计算得到的delta_up应为非负值，当前delta_up={delta_up}, p={p}, a={a}, b={b}, c={c}, d={d}"
        one_minus_p = Decimal('1') - p_d # 计算 1 - p
        delta_down = -((a_d * one_minus_p + b_d) * one_minus_p + c_d) * one_minus_p - d_d # 计算旧策略p向下的可移动的区间
        assert delta_down <= 1e-10, f"计算得到的delta_down应为非正值，当前delta_down={delta_down}, p={p}, a={a}, b={b}, c={c}, d={d}"
        q_high = p_d + delta_up # 新策略的可移动上界
        q_high = min(q_high, Decimal('1')) # 和1取min，避免超出概率的值域范围
        if q_high==1:
            delta_up = q_high - p_d # min改变了q_high，修正delta_up
        q_low = p_d + delta_down # 新策略的可移动下界
        q_low = max(q_low, Decimal('0')) # 和0取max，避免超出概率的值域范围
        if q_low==0:
            delta_down = q_low - p_d # max改变了q_low，修正delta_down
        # 安全除法
        if p_d != 0:
            # epsilon_high = delta_up / p_d # 计算clip_ratio上界
            # epsilon_low = -delta_down / p_d # 计算clip_ratio下界
            # bound_up = Decimal('1') + epsilon_high # 计算bound上界
            # bound_low = Decimal('1') - epsilon_low # 计算bound下界
            bound_up = q_high / p_d  # 直接 q_high / p
            bound_low = q_low / p_d  # 直接 q_low / p
        else:
            # 处理 p = 0 的特殊情况
            flag_too_close_to_0 = True
            # epsilon_high = Decimal('inf') if delta_up > 0 else Decimal('-inf')
            # epsilon_low = Decimal('inf') if delta_down < 0 else Decimal('-inf')
            # fallback：基于q_high/low的语义（>p允许inf，==p限1）
            bound_up = Decimal('inf') if q_high > p_d else Decimal('1')
            bound_low = Decimal('0') if q_low <= p_d else Decimal('1')  # q_low >=0, p=0 →1 (不变)
        # 转换回 float
        result = {
            "delta_up": float(delta_up),
            "delta_down": float(delta_down),
            "q_high": float(q_high),
            "q_low": float(q_low),
            # "epsilon_high": float(epsilon_high) if not epsilon_high.is_infinite() else float('inf'),
            # "epsilon_low": float(epsilon_low) if not epsilon_low.is_infinite() else float('inf'),
            "bound_up": float(bound_up) if not bound_up.is_infinite() else float('inf'),
            "bound_low": float(bound_low)
        }
    else:
        # 使用标准浮点计算（带稳定性优化）
        # Horner 方法
        delta_up = ((a * p + b) * p + c) * p + d # 计算旧策略p向上的可移动的区间
        assert delta_up >= 0, f"计算得到的delta_up应为非负值，当前delta_up={delta_up}, p={p}, a={a}, b={b}, c={c}, d={d}"
        one_minus_p = 1 - p # 计算 1 - p
        delta_down = -(((a * one_minus_p + b) * one_minus_p + c) * one_minus_p + d) # 计算旧策略p向下的可移动的区间
        assert delta_down <= 0, f"计算得到的delta_down应为非正值，当前delta_down={delta_down}, p={p}, a={a}, b={b}, c={c}, d={d}"
        q_high = p + delta_up # 新策略的可移动上界
        q_high = min(q_high, 1) # 和1取min，避免超出概率的值域范围
        if q_high==1:
            delta_up = q_high - p # min改变了q_high，修正delta_up
        q_low = p + delta_down # 新策略的可移动下界
        q_low = max(q_low, 0) # 和0取max，避免超出概率的值域范围
        if q_low==0:
            delta_down = q_low - p # max改变了q_low，修正delta_down
        # 安全除法        
        if abs(p) > 1e-15:
            # epsilon_high = delta_up / p # 计算clip_ratio上界
            # epsilon_low = -delta_down / p # 计算clip_ratio下界
            # bound_up = 1 + epsilon_high # 计算bound上界
            # bound_low = 1 - epsilon_low # 计算bound下界
            bound_up = q_high / p  # 直接 q_high / p
            bound_low = q_low / p  # 直接 q_low / p
        else:
            # 处理 p = 0 的特殊情况
            flag_too_close_to_0 = True
            # epsilon_high = float('inf') if delta_up > 0 else float('-inf') # 处理 p = 0 的特殊情况
            # epsilon_low = float('inf') if delta_down < 0 else float('-inf') # 处理 p = 0 的特殊情况
            # fallback：基于q_high/low的语义（>p允许inf，==p限1）
            bound_up = float('inf') if q_high > p else 1 # 处理 p = 0 的特殊情况
            bound_low = 0 if q_low <= p else 1  # q_low >=0, p=0 →1 (不变)
        # 返回结果         
        result = {
            "delta_up": delta_up,
            "delta_down": delta_down,
            "q_high": q_high,
            "q_low": q_low,
            # "epsilon_high": epsilon_high,
            # "epsilon_low": epsilon_low,
            "bound_up": bound_up,
            "bound_low": bound_low
        }
    if flag_too_close_to_0:
        # PyTorch的torch.clamp完美处理±inf边界，不会因为inf报错
        warnings.warn(f"p值过于接近0，计算可能不准确，当前p={p}", UserWarning)
    # 使用加法的变化衡量相比除法的会避免很多精度溢出问题
    return result

def get_feasible_variation_range_and_clip_ratio(logp, a, b, c, d, method="sub", use_one_minus_p=True, precision_threshold=1e-10):
    # Note: 这是一个三阶多项式的近似KL的函数：
    # 这条三次曲线可能在某些𝑝上略微“鼓出”真边界之外（这时 clip 过松，会违反信赖域）
    # 也可能偏保守（clip 过紧，安全但影响步长）
    assert logp <= 0, f"p值必须在[0,1]范围内，logp值必须在[-inf,0]范围内，当前={logp}"
    p = exp(logp) # p
    inv_p = exp(-logp) # 1/p
    one_minus_p = -expm1(logp) # 1-p
    # inv_one_minus_p = -1.0 / expm1(logp) # 1/(1-p)，不会使用

    assert method in ["sub", "div"], '衡量新旧策略概率变化只有除法（"div"）和减法（"sub"）两种方式'
    if method=="sub": # 1. 减法衡量
        # 计算加法变化边界：feasible variation range
        # 向上部分：delta_up
        # f_up(p) = a*p^3+b*p^2+c*p+d
        # horner: f_up(p) = ((a*p+b)*p+c)*p+d
        delta_up = ((a * p + b) * p + c) * p + d # Horner 方法计算旧策略p向上的可移动的区间
        # TODO: 检查delta_up非负
        # 向下部分：delta_down（根据镜像对称）
        # f_down(p) = -f_up(1-p) = a*p^3+b*p^2+c*p+d
        # horner: f_down(p) = -f_up(1-p) = -(((a*(1-p)+b)*(1-p)+c)*(1-p)+d)
        delta_down = -(((a * one_minus_p + b) * one_minus_p + c) * one_minus_p + d) # 计算旧策略p向下的可移动的区间
        # TODO: 检查delta_down非正

        # 计算加法变化终点值
        # q_high 等于 p 加上 feasible variation range的向上部分
        q_high = p + delta_up # 新策略的可移动上界
        q_high = min(q_high, 1) # 和1取min，避免超出概率的值域范围
        if q_high==1:
            delta_up = q_high - p # min改变了q_high，修正delta_up
        # q_high 等于 p 加上 feasible variation range的向下部分
        q_low = p + delta_down # 新策略的可移动下界
        q_low = max(q_low, 0) # 和0取max，避免超出概率的值域范围
        if q_low==0:
            delta_down = q_low - p # max改变了q_low，修正delta_down
        
        # 使用时，直接比较q-p的差值是否在delta中，也可以把q_high和q_low转化，用于比较q/p
        # 计算除法变化终点值
        # clip_ratio_bound_high = q_high/p
        clip_ratio_bound_high = q_high / p
        clip_ratio_bound_low = q_low / p
        result = {
            "delta_up": delta_up,
            "delta_down": delta_down,
            "q_high":q_high ,
            "q_low": q_low,
            "clip_ratio_bound_high": clip_ratio_bound_high,
            "clip_ratio_bound_low": clip_ratio_bound_low,
            "method": method,
        }
        
    elif method=="div": # 2. 除法衡量
        # 计算除法变化终点值
        # clip_ratio_bound_high = g_high(p) = (f_up(p)+p) / p = (a*p^3+b*p^2+c*p+d+p) / p
        # 拆分: clip_ratio_bound_high = g(p) = 1 + (a*p^2+b*p+c) + d/p
        # horner: clip_ratio_bound_high = g(p) = 1 + (a*p+b)*p+c + d/p
        clip_ratio_bound_high = 1 + (a * p + b) * p + c + d * inv_p
        if use_one_minus_p:
            # clip_ratio_bound_low = g_low(p) = (f_down(p)+p) / p = (-f_up(1-p)+p) / p
            delta_down = -(((a * one_minus_p + b) * one_minus_p + c) * one_minus_p + d) # 计算旧策略p向下的可移动的区间
            clip_ratio_bound_low = 1 + delta_down * inv_p
        else:
            # 因为不是在clip_ratio_bound_high中直接替换1-p，所以需要展开得到公式来计算
            # clip_ratio_bound_low = g_low(p) = (f_down(p)+p) / p = (-f_up(1-p)+p) / p
            # 代入化简：clip_ratio_bound_low = g_low(p) = 1 + a*p*p + (-3*a - b)*p + (3*a + 2*b + c) - (a + b + c + d)*inv_p
            # horner: clip_ratio_bound_low = g_low(p) = 1 + ((a*p + (-3*a - b))*p + (3*a + 2*b + c)) - (a + b + c + d)*inv_p
            # Horner 形式（只含 p 与 1/p）
            clip_ratio_bound_low = 1 + ((a*p + (-3*a - b))*p + (3*a + 2*b + c)) - (a + b + c + d)*inv_p
        result = {
            "clip_ratio_bound_high": clip_ratio_bound_high,
            "clip_ratio_bound_low": clip_ratio_bound_low,
            "method": method,
        }
        # 因为clip_ratio_up和clip_ratio_low非常不优雅，不提供计算值
        # 使用时，直接比较q/p的比值是否在clip_bound中

    # 1.sub，直接比较delta，换算log p和log q到p和q，计算差值直接比较delta
    # 2.sub，直接比较q high和q low的区间，换算log q到q，计算q是否在q high和q low的区间中
    # 3.sub，得到bound，比较ratio
    # 4.div，得到bound，比较ratio

    return result

def analyze_polynomial(a, b, c, d):
    """分析多项式的性质"""
    print("\n=== 多项式分析 ===")
    
    # 导数 f'(x) = 3ax^2 + 2bx + c
    print(f"一阶导数: f'(x) = {3*a:.4f}x² + {2*b:.4f}x + {c:.4f}")
    
    # 找临界点（f'(x) = 0）
    discriminant_derivative = 4*b**2 - 12*a*c
    if discriminant_derivative >= 0:
        x1 = (-2*b + np.sqrt(discriminant_derivative)) / (6*a)
        x2 = (-2*b - np.sqrt(discriminant_derivative)) / (6*a)
        print(f"临界点: x₁ = {x1:.4f}, x₂ = {x2:.4f}")
        
        # 计算临界点的函数值
        f_x1 = a*x1**3 + b*x1**2 + c*x1 + d
        f_x2 = a*x2**3 + b*x2**2 + c*x2 + d
        print(f"临界值: f(x₁) = {f_x1:.4f}, f(x₂) = {f_x2:.4f}")
        
        # 判断极值类型（二阶导数测试）
        f_second_x1 = 6*a*x1 + 2*b
        f_second_x2 = 6*a*x2 + 2*b
        
        if f_second_x1 > 0:
            print(f"x₁ = {x1:.4f} 是局部最小值点")
        elif f_second_x1 < 0:
            print(f"x₁ = {x1:.4f} 是局部最大值点")
        
        if f_second_x2 > 0:
            print(f"x₂ = {x2:.4f} 是局部最小值点")
        elif f_second_x2 < 0:
            print(f"x₂ = {x2:.4f} 是局部最大值点")
    else:
        print("无实数临界点（导数判别式 < 0）")

# TODO: 替换pg_losses2=-advantages*torch.clamp(ratio,1-clip_ratio_low,1+clip_ratio_high)
def clip_by_token_on_ratio_tensor(ratio, delta):
    pass
# 示例使用
if __name__ == "__main__":
    # 测试案例
    delta = 0.1  # KL散度阈值
    D, H = get_D_H_from_delta(delta, high_precision=True)
    # D = 0.0952
    # H = 0.25
    print(f"根据 delta={delta:.2f} 计算得到 D={D:.4f}, H={H:.4f}")

    try:
        solution = cubic_polynomial_clip(D, H)
        
        if solution:
            a = solution['a']
            b = solution['b']
            c = solution['c']
            d = solution['d']
            print(f"\n=== 求解结果 ===")
            print(f"a = {a:.4f}")
            print(f"b = {b:.4f}")
            print(f"c = {c:.4f}")
            print(f"d = {d:.4f}")
            
            # 分析多项式性质
            analyze_polynomial(a, b, c, d)
            # 测试在不同p值下的可行变化范围
            test_p_values = [0, 0.00001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.99999, 1]
            print("\n=== 可行变化范围测试 ===")
            for p in test_p_values:
                result = feasible_variation_range_up_2_clip_ratio_hybrid(p, a, b, c, d, high_precision=True)
                print(f"p={p:.2f} -> delta_up={result['delta_up']:.4f}, delta_down={result['delta_down']:.4f}, "
                      f"q_high={result['q_high']:.4f}, q_low={result['q_low']:.4f}, "
                      f"bound_up={result['bound_up']:.4f}, bound_low={result['bound_low']:.4f}")
            # 可视化结果
            
            # 绘制函数
            import matplotlib.pyplot as plt
            # plt.rcParams['font.sans-serif'] = ['Noto Sans CJK', 'Arial']  
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            x_down = -0.0
            x_up = 1.0
            x = np.linspace(x_down, x_up, 1000)
            y = a*x**3 + b*x**2 + c*x + d
            total = y+x
            y_mirror = -(a*(1-x)**3 + b*(1-x)**2 + c*(1-x) + d)
            bound_high = 1 + y/x
            limited_bound_high = 1/x
            bound_low = 1 + y_mirror/x
            limited_bound_low = 0/x
            
            plt.figure(figsize=(18, 6))
            
            # 子图1: 函数图像
            plt.subplot(1, 3, 1)
            plt.plot(x, y, 'b-', linewidth=2, label=f'f(x) = {a:.3f}x³ + {b:.3f}x² + {c:.3f}x + {d:.3f}')
            plt.plot(x, y_mirror, 'g-', linewidth=2, linestyle='--', label=f'-f(1-x)')
            plt.plot(x, total, 'b+', linewidth=2, linestyle='-.', label=f'f(x)+x')

            plt.plot(0, D, 'ro', markersize=8, label=f'f(0) = {D}')
            plt.plot(0.5, H, 'go', markersize=8, label=f'f(0.5) = {H}')
            plt.plot(1, 0, 'mo', markersize=8, label=f'f(1) = 0')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=0.5, color='g', linestyle='--', alpha=0.2)
            plt.axvline(x=1, color='m', linestyle='--', alpha=0.2)
            plt.grid(True, alpha=0.3)
            plt.xlabel('p')
            plt.ylabel('feasible variation range')
            plt.title(f'Cubic Polynomial Function (Delta={delta})')
            plt.legend(loc='lower left')
            plt.xlim(x_down, x_up)
            plt.ylim(-0.7, 1.2)
            
            # 子图2: 导数图像
            plt.subplot(1, 3, 2)
            y_prime = 3*a*x**2 + 2*b*x + c
            plt.plot(x, y_prime, 'r-', linewidth=2, label="f'(x)")
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            plt.grid(True, alpha=0.3)
            plt.xlabel('x')
            plt.ylabel("f'(x)")
            plt.title('First Derivative')
            plt.legend(loc='lower left')
            plt.xlim(x_down, x_up)

            # 子图3: 函数图像
            plt.subplot(1, 3, 3)
            plt.plot(x, bound_high, 'b-', linewidth=2, label=f'bound_high')
            plt.plot(x, bound_low, 'g-', linewidth=2, linestyle='--', label=f'bound_low')
            plt.plot(x, limited_bound_high, 'b-', linewidth=2, linestyle='-.', label=f'limited_bound_high')
            plt.plot(x, limited_bound_low,  'g-', linewidth=2, linestyle='-.', label=f'limited_bound_low')
                        
            plt.axhline(y=1.28, color='k', linestyle='--', alpha=0.3)  # 第一条水平线
            plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)   # 第二条水平线
            plt.axhline(y=0.8, color='k', linestyle='--', alpha=0.3)   # 第三条水平线

            plt.grid(True, alpha=0.3)
            plt.xlabel('x')
            plt.ylabel('clip ratio bound')
            plt.title('bound_high and bound_low')
            plt.legend(loc='upper right')
            plt.xlim(x_down, x_up)
            plt.ylim(0, 8)

            plt.tight_layout()
            plt.show()
            # 保存
            plt.savefig('cubic_polynomial_and_derivative.png', dpi=300)
            
        else:
            print("\n未找到满足所有约束的解")
            print("可能的原因：")
            print("1. 约束条件相互矛盾")
            print("2. 非线性约束导致无实数解")
            print("3. 数值求解器未能收敛")
            
    except ValueError as e:
        print(f"错误: {e}")
