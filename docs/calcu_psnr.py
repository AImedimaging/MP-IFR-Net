import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import scipy.io as spio

# 定义数据文件夹路径
truth_path = "/home/mengping/mlp-F/truth_data"
pred_path = "/home/mengping/mlp-F/pred_data"
fbp_path = "/home/mengping/mlp-F/fbp_data"

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_val = np.max(img2)
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

for idx in [5]:
    for noiseLevel in [40]:
        for num_proj in [60]:
            # 生成 truth 数据的文件名
            truth_data_name = 'truth_{}_{}_{}_{}'.format(idx, num_proj, noiseLevel, 60)
            truth_save_path = os.path.join(truth_path, truth_data_name + '.txt')

            # # 生成 pred 数据的文件名
            # pred_data_name = 'pred_{}_{}_{}_{}'.format(idx, num_proj, noiseLevel, 60)
            # pred_save_path = os.path.join(pred_path, pred_data_name + '.txt')

            # 生成 pred 数据的文件名
            pred_data_name = 'fbp_{}_{}_{}_{}'.format(idx, num_proj, noiseLevel, 60)
            pred_save_path = os.path.join(fbp_path, pred_data_name + '.txt')

            # 检查文件是否存在
            if os.path.exists(truth_save_path) and os.path.exists(pred_save_path):
                # 读取 truth 数据
                truth_data = np.loadtxt(truth_save_path, delimiter=',')
                # 读取 pred 数据
                pred_data = np.loadtxt(pred_save_path, delimiter=',')

                # 计算 PSNR
                psnr_value = compute_psnr(truth_data, pred_data)
                # 计算 SSIM
                ssim_value = ssim(truth_data, pred_data, data_range=1)

                print(f"对于 idx={idx}, noiseLevel={noiseLevel}, num_proj={num_proj}:")
                print(f"PSNR: {psnr_value}")
                print(f"SSIM: {ssim_value}")
            else:
                print(f"对于 idx={idx}, noiseLevel={noiseLevel}, num_proj={num_proj}, 数据文件不存在。")
