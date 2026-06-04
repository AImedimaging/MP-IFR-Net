import os
import cv2
import pywt
import torch
import scipy.io as spio
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
from model import MLP
from scipy.interpolate import griddata

from encoder.hashencoder.hashgrid import HashEncoder

# 确定是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------弃用-----------------------------
def inverse_log_extension(F_prime, K):
    magnitude = np.abs(F_prime)
    phase = np.angle(F_prime)
    restored_magnitude = np.exp(magnitude / K) - 1
    return restored_magnitude * np.exp(1j * phase)


# ------------------end-----------------------

# def x_func(y, mu):
#     return -mu * np.log((1 - y) / (1 + y))
def x_func(y, mu):
    """
    修改 x_func 函数，出错时，可以添加范围检查，避免 np.log 计算中的无效输入
    """
    # y = np.where(y >= 0.99, 0.99, y)
    # y = np.where(y <= -0.99, -0.99, y)
    return -mu * np.log((1 - y) / (1 + y))


# ----------弃用--------------------------
# def adaptive_center_filter(spectrum):
#     """修正后的自适应滤波器"""
#     rows, cols = spectrum.shape
#     crow, ccol = rows // 2, cols // 2
#
#     # 修正距离计算表达式
#     X, Y = np.ogrid[:rows, :cols]
#     dist = np.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
#
#     # 构建混合滤波器核
#     cos_window = np.cos(np.pi * dist / (2 * np.max(dist))) ** 0.5
#     gauss = np.exp(-(dist ** 2) / (2 * (rows / 8) ** 2))
#
#     # 动态参数计算
#     central_region = spectrum[crow - 5:crow + 5, ccol - 5:ccol + 5]
#     energy_ratio = np.mean(central_region) / np.mean(spectrum)
#
#     # 限制指数函数的输入值
#     exponent = np.clip(-10 * (energy_ratio - 0.3), -700, 700)  # 700 是一个经验值，可根据实际情况调整
#     blend_factor = 1 - 1 / (1 + np.exp(exponent))
#
#     # 检查 blend_factor 是否包含 NaN 值
#     if np.isnan(blend_factor).any():
#         print("Warning: blend_factor contains NaN values!")
#         # 可以选择对 NaN 值进行处理，例如替换为合适的值
#         blend_factor = np.nan_to_num(blend_factor, nan=0)  # 将 NaN 替换为 0
#
#     return spectrum * (blend_factor * gauss + (1 - blend_factor) * cos_window)

# --------------------------------end---------------------------


# 利用频域的对称性----------弃用-----------------
def symmetryFre(fre):
    matrix = fre
    N = matrix.shape[0]
    # print("原始矩阵:")
    # print(fre)
    # 划分四个子矩阵
    matrix_1 = matrix[:N // 2, :N // 2]  # 左上
    matrix_2 = matrix[:N // 2, N // 2 + 1:]  # 右上
    # matrix_3 = matrix[N // 2 + 1:, :N // 2]  # 左下
    # matrix_4 = matrix[N // 2 + 1:, N // 2 + 1:]  # 右下
    # 提取中心子矩阵（去掉边缘）
    matrix_5 = matrix_1[1:N // 2, 1:N // 2]  # 左上中心
    matrix_6 = matrix_2[1:N // 2, 0:N // 2 - 1]  # 右上中心
    # 矩阵5旋转180度并取共轭（实部保持，虚部取反）
    matrix_7 = np.conj(np.flipud(np.fliplr(matrix_5)))
    # 矩阵6旋转180度并取共轭（示例用简单翻转代替实际旋转）
    matrix_8 = np.conj(np.flipud(np.fliplr(matrix_6)))
    # 覆盖到原矩阵右下和左下区域
    matrix[N // 2 + 1:, N // 2 + 1:] = matrix_7  # 右下覆盖
    matrix[N // 2 + 1:, 1:N // 2] = matrix_8  # 左下覆盖
    # 处理中心行（第N//2行）
    center_row = matrix[N // 2, :].copy()
    left_part = np.conj(center_row[1:N // 2][::-1])  # 取左半逆序并共轭
    matrix[N // 2, N // 2 + 1:N // 2 * 2] = left_part
    # 处理中心列（第N//2列）
    center_col = matrix[:, N // 2].copy()
    top_part = np.conj(center_col[1:N // 2][::-1])  # 取上半逆序并共轭
    matrix[N // 2 + 1:N // 2 * 2, N // 2] = top_part
    # print("\n最终中心对称复数矩阵:")
    # print(matrix)
    return matrix


if __name__ == "__main__":
    data_kargs = {
        'ic': 2,
        'oc': 1
    }

    net_kargs = {
        'encoder_layer_num': 8,
        'decoder_layer_num': 1,
        'feature_num': 512,
        'ffm': 'linear',
        'L': 256
    }

    num_proj = 90
    num_proj_pred = 720
    num_dete = 512

    for idx in [5]:
        img = spio.loadmat('data/CT_images_preprocessed')['img_cropped'][:, :, idx]
        for num_proj in [60]:
            encoder = HashEncoder(input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19)
            net = MLP(data_kargs, net_kargs)
            # 将模型移动到指定设备（GPU 或 CPU）
            if torch.cuda.is_available():
                net.to('cuda')
            model = f'PBCT_{idx}_{num_proj}_60'
            model_path = f'proj{num_proj}/{model}/models/model_epoch_900.pth'
            net.load_state_dict(torch.load(model_path))
            for num_proj_pred in [num_proj]:
                dstx, dsty = np.meshgrid(np.arange(num_dete), np.arange(num_dete))
                dstx = dstx - 256
                dsty = dsty - 256
                dsty = - dsty
                dstx = dstx.flatten()
                dsty = dsty.flatten()
                x_pair = np.column_stack((dstx, dsty))
                x_pairs = []
                for point in x_pair:
                    x_norm = point[0] / 256
                    y_norm = point[1] / 256
                    x_pairs.append((x_norm, y_norm))
                x_pairs_tensor = torch.tensor(x_pairs, dtype=torch.float32).to(device)
                with torch.no_grad():
                    measurements = net(x_pairs_tensor)
                measurements = measurements.cpu().numpy()
                measurements = measurements.reshape((num_dete, num_dete))
                # 获取矩阵的大小
                # rows, columns = measurements.shape
                #
                # # 将四个顶点的值设置为 0+0j
                # measurements[0, 0] = 0 + 0j
                # measurements[0, columns - 1] = 0 + 0j
                # measurements[rows - 1, 0] = 0 + 0j
                # measurements[rows - 1, columns - 1] = 0 + 0j

                # measurements = symmetryFre(measurements)

                # ---------分别看实部和虚部-------------------
                # np.savetxt("/home/mengping/mlp-F/dataVisual/pred_real.txt", measurements.real)
                # np.savetxt("/home/mengping/mlp-F/dataVisual/pred_imag.txt", measurements.imag)
                # -------------------------end------------------------

                # 绘制一个幅值的热图
                mag = np.abs(measurements)
                # 创建一个图形窗口
                plt.figure(figsize=(8, 6))
                # 绘制热图
                plt.imshow(mag, cmap='hot', interpolation='nearest')
                # 添加颜色条
                plt.colorbar(label='Magnitude')
                # 设置坐标轴标签和标题
                plt.xlabel('Column Index')
                plt.ylabel('Row Index')
                plt.title('Magnitude of Measurements')
                # 显示图形
                plt.show()
                print(mag.max(), mag.min())

                # 暂时只能根据生成数据集时的超参数来写
                measurements_real = x_func(measurements.real, 165.12 / 2)
                print("np.max(f_ps.real)", np.max(measurements.real))
                print("np.min(f_ps.real)", np.min(measurements.real))
                measurements_imag = x_func(measurements.imag, 27.40 / 2)
                print("np.max(f_ps.imag)", np.max(measurements.imag))
                print("np.min(f_ps.imag)", np.min(measurements.imag))
                measurements = measurements_real + 1j * measurements_imag
                print("np.max(f_ps.real)", np.max(measurements.real))
                print("np.min(f_ps.real)", np.min(measurements.real))
                print("np.max(f_ps.imag)", np.max(measurements.imag))
                print("np.min(f_ps.imag)", np.min(measurements.imag))



                # 对频谱进行滤波，无用
                # filtered_spectrum = adaptive_center_filter(measurements)
                #
                Fourier2_shifted = np.fft.fftshift(measurements)
                # Fourier2_shifted = np.fft.fftshift(filtered_spectrum)

                result = np.fft.ifft2(Fourier2_shifted)
                inverse_result = np.fft.fftshift(result)
                # 进行对数缩放，弃用
                # inverse_result = np.log(inverse_result + 1e-10)  # 加上一个小的常数防止log(0)

                # print(np.max(inverse_result.real))
                # print(np.min(np.abs(inverse_result)))

                # 计算复数的幅值，当做最后得到的图像的像素值
                magnitude = np.abs(inverse_result)
                # print(np.max(magnitude))
                # print(np.min(magnitude))

                # ===============保存预测数据========================
                # # 归一化到 [0, 1] 范围
                # magnitude_min = np.min(magnitude)
                # magnitude_max = np.max(magnitude)
                # normalized_result = (magnitude - magnitude_min) / (magnitude_max - magnitude_min)
                # print(" max(normalized_result): ", np.max(normalized_result))
                # print(" min(normalized_result): ", np.min(normalized_result))
                # # 创建保存文件的目录（如果目录不存在）
                # truth_path = "/home/mengping/mlp-F/pred_data"
                # data_name = 'pred_{}_{}_{}_{}'.format(idx, num_proj, noiseLevel, 60)
                # # 生成保存路径
                # save_path = os.path.join(truth_path, data_name + '.txt')
                # # 如果目录不存在，创建目录
                # if not os.path.exists(truth_path):
                #     os.makedirs(truth_path)
                # # 保存归一化后的幅值结果为 txt 文件
                # np.savetxt(save_path, normalized_result, delimiter=',', fmt='%.6f')
                # print(f"Normalized magnitude saved to {save_path}")
                # ==================================end===========================



                # 可视化
                fig, ax = plt.subplots(figsize=(5.12, 5.12))
                # ax.imshow(np.abs(inverse_result), cmap='gray')
                ax.imshow(np.abs(inverse_result), cmap='gray')
                print("max(np.abs(inverse_result))", np.max(np.abs(inverse_result)))
                # 关闭坐标轴
                ax.axis('off')
                # 调整子图布局，去掉所有边距
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                # 显示图像
                plt.show()

                # 绘制相位图像
                fig_real, ax_real = plt.subplots(figsize=(5.12, 5.12))
                ax_real.imshow(np.angle(inverse_result), cmap='gray')
                print("max(np.real(inverse_result))", np.max(np.real(inverse_result)))
                ax_real.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.show()
