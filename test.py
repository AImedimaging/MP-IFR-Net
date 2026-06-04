import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 参数设置
# ---------------------------
omega0 = 2.0                      # 频率缩放因子
z = np.linspace(-2*np.pi, 2*np.pi, 1000)
sigma = np.sin(omega0 * z)

# ---------------------------
# 绘图
# ---------------------------
plt.figure(figsize=(8, 4.5), dpi=300)
plt.plot(z, sigma, linewidth=2.2, label=rf'$\sigma(z)=\sin(\omega_0 z),\ \omega_0={omega0}$')

# 坐标轴
plt.axhline(0, linewidth=1.0)
plt.axvline(0, linewidth=1.0)

# 坐标范围
plt.xlim(z.min(), z.max())
plt.ylim(-1.2, 1.2)

# 刻度设置
xticks = [-2*np.pi, -np.pi, 0, np.pi, 2*np.pi]
xtick_labels = [r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$']
plt.xticks(xticks, xtick_labels, fontsize=11)
plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=11)

# 标签和标题
plt.xlabel(r'$z$', fontsize=13)
plt.ylabel(r'$\sigma(z)$', fontsize=13)
plt.title(r' $\sigma(z)=\sin(\omega_0 z)$', fontsize=13, pad=10)

# 网格和图例
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=10, frameon=True)

# 边距优化
plt.tight_layout()

# 保存图片
plt.savefig('sine_activation_function.png', bbox_inches='tight')
plt.show()