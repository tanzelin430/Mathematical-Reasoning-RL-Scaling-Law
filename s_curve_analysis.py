#!/usr/bin/env python3
"""
S型曲线分析脚本
分析k-N关系的不同S型函数拟合效果
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# 基于你提供的数据
N_vals = np.array([5e8, 1.5e9, 3e9, 7e9, 1.4e10])
k_vals = np.array([0.0054, 0.0065, 0.0164, 0.0469, 0.0504])

print("=== S型曲线拟合分析 ===")
print(f"数据点: N = {N_vals}")
print(f"数据点: k = {k_vals}")

# 定义S型函数
def logistic_func(N, L, r, N0):
    """Logistic函数: k = L / (1 + exp(-r * (N - N0)))"""
    return L / (1 + np.exp(-r * (N - N0)))

def gompertz_func(N, a, b, c):
    """Gompertz函数: k = a * exp(-b * exp(-c * N))"""
    return a * np.exp(-b * np.exp(-c * N))

def weibull_func(N, L, lam, beta):
    """Weibull CDF: k = L * (1 - exp(-(N/lambda)^beta))"""
    return L * (1 - np.exp(-np.power(N / lam, beta)))

def tanh_func(N, a, b, c, N0):
    """双曲正切: k = a + b * tanh(c * (N - N0))"""
    return a + b * np.tanh(c * (N - N0))

def richards_func(N, L, Q, B, v):
    """Richards函数: k = L / (1 + Q * exp(-B * N))^(1/v)"""
    return L / np.power(1 + Q * np.exp(-B * N), 1/v)

# 存储拟合结果
s_curve_results = {}

print("\n=== 各种S型函数拟合结果 ===")

# 1. Logistic拟合
try:
    L_init = max(k_vals) * 1.5
    r_init = 1e-9
    N0_init = np.median(N_vals)
    
    popt_logistic, pcov_logistic = curve_fit(
        logistic_func, N_vals, k_vals, 
        p0=[L_init, r_init, N0_init],
        bounds=([0, 0, 0], [1, 1e-6, 1e12]),
        maxfev=5000
    )
    k_pred_logistic = logistic_func(N_vals, *popt_logistic)
    r2_logistic = 1 - np.sum((k_vals - k_pred_logistic)**2) / np.sum((k_vals - np.mean(k_vals))**2)
    s_curve_results['Logistic'] = (popt_logistic, r2_logistic)
    print(f"1. Logistic: L={popt_logistic[0]:.4f}, r={popt_logistic[1]:.2e}, N0={popt_logistic[2]:.2e}, R²={r2_logistic:.4f}")
except Exception as e:
    print(f"1. Logistic拟合失败: {e}")
    s_curve_results['Logistic'] = None

# 2. Gompertz拟合
try:
    a_init = max(k_vals) * 1.2
    b_init = 5.0
    c_init = 1e-9
    
    popt_gompertz, _ = curve_fit(
        gompertz_func, N_vals, k_vals,
        p0=[a_init, b_init, c_init],
        bounds=([0, 0, 0], [1, 100, 1e-6]),
        maxfev=5000
    )
    k_pred_gompertz = gompertz_func(N_vals, *popt_gompertz)
    r2_gompertz = 1 - np.sum((k_vals - k_pred_gompertz)**2) / np.sum((k_vals - np.mean(k_vals))**2)
    s_curve_results['Gompertz'] = (popt_gompertz, r2_gompertz)
    print(f"2. Gompertz: a={popt_gompertz[0]:.4f}, b={popt_gompertz[1]:.4f}, c={popt_gompertz[2]:.2e}, R²={r2_gompertz:.4f}")
except Exception as e:
    print(f"2. Gompertz拟合失败: {e}")
    s_curve_results['Gompertz'] = None

# 3. Weibull拟合
try:
    L_init = max(k_vals) * 1.2
    lam_init = max(N_vals)
    beta_init = 1.5
    
    popt_weibull, _ = curve_fit(
        weibull_func, N_vals, k_vals,
        p0=[L_init, lam_init, beta_init],
        bounds=([0, 1e6, 0.1], [1, 1e12, 10]),
        maxfev=5000
    )
    k_pred_weibull = weibull_func(N_vals, *popt_weibull)
    r2_weibull = 1 - np.sum((k_vals - k_pred_weibull)**2) / np.sum((k_vals - np.mean(k_vals))**2)
    s_curve_results['Weibull'] = (popt_weibull, r2_weibull)
    print(f"3. Weibull: L={popt_weibull[0]:.4f}, λ={popt_weibull[1]:.2e}, β={popt_weibull[2]:.4f}, R²={r2_weibull:.4f}")
except Exception as e:
    print(f"3. Weibull拟合失败: {e}")
    s_curve_results['Weibull'] = None

# 4. 双曲正切拟合
try:
    a_init = np.mean(k_vals)
    b_init = (max(k_vals) - min(k_vals)) / 2
    c_init = 1e-9
    N0_init = np.median(N_vals)
    
    popt_tanh, _ = curve_fit(
        tanh_func, N_vals, k_vals,
        p0=[a_init, b_init, c_init, N0_init],
        bounds=([-1, 0, 0, 0], [1, 1, 1e-6, 1e12]),
        maxfev=5000
    )
    k_pred_tanh = tanh_func(N_vals, *popt_tanh)
    r2_tanh = 1 - np.sum((k_vals - k_pred_tanh)**2) / np.sum((k_vals - np.mean(k_vals))**2)
    s_curve_results['Tanh'] = (popt_tanh, r2_tanh)
    print(f"4. Tanh: a={popt_tanh[0]:.4f}, b={popt_tanh[1]:.4f}, c={popt_tanh[2]:.2e}, N0={popt_tanh[3]:.2e}, R²={r2_tanh:.4f}")
except Exception as e:
    print(f"4. Tanh拟合失败: {e}")
    s_curve_results['Tanh'] = None

# 5. Richards函数拟合
try:
    L_init = max(k_vals) * 1.2
    Q_init = 10.0
    B_init = 1e-9
    v_init = 1.0
    
    popt_richards, _ = curve_fit(
        richards_func, N_vals, k_vals,
        p0=[L_init, Q_init, B_init, v_init],
        bounds=([0, 0.1, 0, 0.1], [1, 1000, 1e-6, 10]),
        maxfev=5000
    )
    k_pred_richards = richards_func(N_vals, *popt_richards)
    r2_richards = 1 - np.sum((k_vals - k_pred_richards)**2) / np.sum((k_vals - np.mean(k_vals))**2)
    s_curve_results['Richards'] = (popt_richards, r2_richards)
    print(f"5. Richards: L={popt_richards[0]:.4f}, Q={popt_richards[1]:.4f}, B={popt_richards[2]:.2e}, v={popt_richards[3]:.4f}, R²={r2_richards:.4f}")
except Exception as e:
    print(f"5. Richards拟合失败: {e}")
    s_curve_results['Richards'] = None

# 绘制比较图
plt.figure(figsize=(14, 10))

# 创建子图
plt.subplot(2, 1, 1)
plt.scatter(N_vals, k_vals, color='black', s=80, label='数据点', zorder=5, marker='o')

N_smooth = np.logspace(np.log10(N_vals.min()), np.log10(N_vals.max()), 300)
colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

plot_idx = 0
for name, result in s_curve_results.items():
    if result is not None:
        popt, r2 = result
        if name == 'Logistic':
            k_smooth = logistic_func(N_smooth, *popt)
        elif name == 'Gompertz':
            k_smooth = gompertz_func(N_smooth, *popt)
        elif name == 'Weibull':
            k_smooth = weibull_func(N_smooth, *popt)
        elif name == 'Tanh':
            k_smooth = tanh_func(N_smooth, *popt)
        elif name == 'Richards':
            k_smooth = richards_func(N_smooth, *popt)
        
        plt.plot(N_smooth, k_smooth, 
                color=colors[plot_idx], 
                linestyle=linestyles[plot_idx],
                label=f'{name} (R²={r2:.3f})', 
                linewidth=2.5)
        plot_idx += 1

plt.xscale('log')
plt.xlabel('Model Size N (parameters)')
plt.ylabel('k parameter')
plt.title('S-Curve Analysis: k = f(N) - Log Scale')
plt.legend()
plt.grid(True, alpha=0.3)

# 线性尺度子图
plt.subplot(2, 1, 2)
plt.scatter(N_vals/1e9, k_vals, color='black', s=80, label='数据点', zorder=5, marker='o')

N_smooth_linear = np.linspace(N_vals.min(), N_vals.max(), 300)
plot_idx = 0
for name, result in s_curve_results.items():
    if result is not None:
        popt, r2 = result
        if name == 'Logistic':
            k_smooth = logistic_func(N_smooth_linear, *popt)
        elif name == 'Gompertz':
            k_smooth = gompertz_func(N_smooth_linear, *popt)
        elif name == 'Weibull':
            k_smooth = weibull_func(N_smooth_linear, *popt)
        elif name == 'Tanh':
            k_smooth = tanh_func(N_smooth_linear, *popt)
        elif name == 'Richards':
            k_smooth = richards_func(N_smooth_linear, *popt)
        
        plt.plot(N_smooth_linear/1e9, k_smooth, 
                color=colors[plot_idx], 
                linestyle=linestyles[plot_idx],
                label=f'{name} (R²={r2:.3f})', 
                linewidth=2.5)
        plot_idx += 1

plt.xlabel('Model Size N (Billions of parameters)')
plt.ylabel('k parameter')
plt.title('S-Curve Analysis: k = f(N) - Linear Scale')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/s_curve_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('outputs/s_curve_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nS型曲线分析图已保存到 outputs/s_curve_analysis.pdf 和 .png")

# 找出最佳拟合
valid_s_results = {k: v[1] for k, v in s_curve_results.items() if v is not None}
if valid_s_results:
    best_s_func = max(valid_s_results, key=valid_s_results.get)
    print(f"\n=== 最佳S型拟合函数 ===")
    print(f"函数: {best_s_func}")
    print(f"R² = {valid_s_results[best_s_func]:.4f}")
    
    # 输出所有结果排序
    print(f"\n=== 所有S型函数R²排序 ===")
    sorted_results = sorted(valid_s_results.items(), key=lambda x: x[1], reverse=True)
    for i, (name, r2) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: R² = {r2:.4f}")

print(f"\n=== S型曲线函数形式总结 ===")
print("1. Logistic: k = L / (1 + exp(-r * (N - N0)))")
print("   - 经典S型，对称，有上下渐近线")
print("   - 参数: L(上限), r(增长率), N0(拐点)")

print("\n2. Gompertz: k = a * exp(-b * exp(-c * N))")
print("   - 非对称S型，增长前期较慢")
print("   - 参数: a(上限), b,c(形状参数)")

print("\n3. Weibull: k = L * (1 - exp(-(N/λ)^β))")
print("   - 从0开始的S型增长")
print("   - 参数: L(上限), λ(尺度), β(形状)")

print("\n4. Tanh: k = a + b * tanh(c * (N - N0))")
print("   - 双曲正切，光滑S型")
print("   - 参数: a(中点值), b(振幅), c(陡峭度), N0(拐点)")

print("\n5. Richards: k = L / (1 + Q * exp(-B * N))^(1/v)")
print("   - 广义Logistic，可调节非对称性")
print("   - 参数: L(上限), Q,B(形状), v(非对称)")
