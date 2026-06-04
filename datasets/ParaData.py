import scipy.io as spio
import numpy as np
import h5py
import odl
import os
from matplotlib import pyplot as plt, cm
from scipy.interpolate import griddata

np.random.seed(512)


# 定义函数，可逆归一化
def y_func(x, omega):
    return (1 - np.exp(-omega * x)) / (1 + np.exp(-omega * x))


def x_func(y, mu):
    return -mu * np.log((1 - y) / (1 + y))


def SanDianView(f_ps):
    # 指定要可视化的行
    row_index = 50
    # 获取指定行的数据
    row_data = f_ps[row_index, :]
    # 设置实部、虚部、幅值和相位的临界值
    real_threshold = 0.1
    imag_threshold = 0.1
    magnitude_threshold = 0.5
    phase_threshold = np.pi / 10
    # 创建实部和虚部的 2x1 子图
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))
    # 绘制实部
    real_data = row_data.real
    real_colors = np.where(real_data > real_threshold, 'red', 'blue')
    axes1[0].scatter(np.arange(len(real_data)), real_data, c=real_colors, s=3)
    axes1[0].set_title(f'Real Part (Threshold: {real_threshold})')
    axes1[0].set_xlabel('Index')
    axes1[0].set_ylabel('Value')
    # 绘制虚部
    imag_data = row_data.imag
    imag_colors = np.where(imag_data > imag_threshold, 'red', 'blue')
    axes1[1].scatter(np.arange(len(imag_data)), imag_data, c=imag_colors, s=3)
    axes1[1].set_title(f'Imaginary Part (Threshold: {imag_threshold})')
    axes1[1].set_xlabel('Index')
    axes1[1].set_ylabel('Value')
    plt.tight_layout()
    plt.show()
    # 创建幅值和相位的 2x1 子图
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
    # 绘制幅值
    magnitude_data = np.abs(row_data)
    magnitude_colors = np.where(magnitude_data > magnitude_threshold, 'red', 'blue')
    axes2[0].scatter(np.arange(len(magnitude_data)), magnitude_data, c=magnitude_colors, s=3)
    axes2[0].set_title(f'Magnitude (Threshold: {magnitude_threshold})')
    axes2[0].set_xlabel('Index')
    axes2[0].set_ylabel('Magnitude')
    # 绘制相位
    phase_data = np.angle(row_data)
    phase_colors = np.where(phase_data > phase_threshold, 'red', 'blue')
    axes2[1].scatter(np.arange(len(phase_data)), phase_data, c=phase_colors, s=3)
    axes2[1].set_title(f'Phase (Threshold: {phase_threshold})')
    axes2[1].set_xlabel('Index')
    axes2[1].set_ylabel('Phase (radians)')
    plt.tight_layout()
    plt.show()


def ZhexianView(f_ps):
    # 指定要可视化的行
    row_index = 50
    # 获取指定行的数据
    row_data = f_ps[row_index, :]
    # 设置实部、虚部、幅值和相位的临界值
    real_threshold = 0.5
    imag_threshold = 0.5
    magnitude_threshold = 1.0
    phase_threshold = np.pi / 4
    # 创建实部和虚部的 2x1 子图
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))
    # 绘制实部曲线
    real_data = row_data.real
    x = np.arange(len(real_data))
    for i in range(len(real_data) - 1):
        if real_data[i] > real_threshold and real_data[i + 1] > real_threshold:
            axes1[0].plot(x[i:i + 2], real_data[i:i + 2], color='red')
        else:
            axes1[0].plot(x[i:i + 2], real_data[i:i + 2], color='blue')
    axes1[0].set_title(f'Real Part (Threshold: {real_threshold})')
    axes1[0].set_xlabel('Index')
    axes1[0].set_ylabel('Value')
    # 绘制虚部曲线
    imag_data = row_data.imag
    for i in range(len(imag_data) - 1):
        if imag_data[i] > imag_threshold and imag_data[i + 1] > imag_threshold:
            axes1[1].plot(x[i:i + 2], imag_data[i:i + 2], color='red')
        else:
            axes1[1].plot(x[i:i + 2], imag_data[i:i + 2], color='blue')
    axes1[1].set_title(f'Imaginary Part (Threshold: {imag_threshold})')
    axes1[1].set_xlabel('Index')
    axes1[1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()
    # 创建幅值和相位的 2x1 子图
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
    # 绘制幅值曲线
    magnitude_data = np.abs(row_data)
    for i in range(len(magnitude_data) - 1):
        if magnitude_data[i] > magnitude_threshold and magnitude_data[i + 1] > magnitude_threshold:
            axes2[0].plot(x[i:i + 2], magnitude_data[i:i + 2], color='red')
        else:
            axes2[0].plot(x[i:i + 2], magnitude_data[i:i + 2], color='blue')
    axes2[0].set_title(f'Magnitude (Threshold: {magnitude_threshold})')
    axes2[0].set_xlabel('Index')
    axes2[0].set_ylabel('Magnitude')
    # 绘制相位曲线
    phase_data = np.angle(row_data)
    for i in range(len(phase_data) - 1):
        if phase_data[i] > phase_threshold and phase_data[i + 1] > phase_threshold:
            axes2[1].plot(x[i:i + 2], phase_data[i:i + 2], color='red')
        else:
            axes2[1].plot(x[i:i + 2], phase_data[i:i + 2], color='blue')
    axes2[1].set_title(f'Phase (Threshold: {phase_threshold})')
    axes2[1].set_xlabel('Index')
    axes2[1].set_ylabel('Phase (radians)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image_path = 'CT_images_preprocessed.mat'
    num_dete = 512
    num_val_proj = 60
    for num_proj in [180]:
            for img_idx in [5]:
                new_path = "/home/mengping/mlp-freq/data"
                dataset_name = 'PBCT_{}_{}_{}'.format(img_idx, num_proj, num_val_proj)
                dataset_name = os.path.join(new_path, dataset_name)
                print(f'Start Generating the Training & Validation Dataset for Image {img_idx} . . .')
                # 下载图像
                img = spio.loadmat(image_path)['img_cropped'][:, :, img_idx]


                # --------------------可视化 img（无白边设置）--------------------
                fig = plt.figure(frameon=False)
                fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)  # 去掉图像周围的边框
                plt.imshow(img, cmap='gray')  # 灰度图显示
                plt.axis('off')  # 去掉坐标轴
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉外部的白边
                #
                # 显示或保存结果
                plt.show()
                # -------------------end----------------------------



                # -----------创建保存文件的目录（如果目录不存在）-------------
                # truth_path = "/home/mengping/mlp-F/truth_data"
                # data_name = 'truth_{}_{}_{}_{}'.format(img_idx, num_proj, input_snr, num_val_proj)
                # # 生成保存路径
                # save_path = os.path.join(truth_path, data_name + '.txt')
                # # 如果目录不存在，创建目录
                # if not os.path.exists(truth_path):
                #     os.makedirs(truth_path)
                # # 保存归一化后的幅值结果为 txt 文件
                # np.savetxt(save_path, img, delimiter=',', fmt='%.4f')
                # print(f"Truth magnitude saved to {save_path}")
                # ---------------------------end---------------------------------



                # ----------------------对原图像进行的一些操作验证--------------
                # shift_img = np.fft.fft2(img)
                # shift_img = np.fft.fftshift(shift_img)
                # recon_img = np.fft.ifft2(shift_img)
                # recon = np.fft.ifftshift(recon_img)
                # # 计算幅值和相位
                # magnitude = np.abs(shift_img)
                # phase = np.angle(shift_img)
                # # 创建一个包含两个子图的画布
                # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                # # 显示幅值图像
                # axes[0].imshow(np.log(1 + magnitude), cmap='gray')
                # axes[0].set_title('Magnitude')
                # axes[0].axis('off')
                # # 显示相位图像
                # axes[1].imshow(phase, cmap='gray')
                # axes[1].set_title('Phase')
                # axes[1].axis('off')
                # # 显示图像
                # plt.show()


                # 看幅值和相位的影响------------------
                # 仅保留幅值信息，相位置为零
                # magnitude_only = magnitude * np.exp(1j * 0)
                # # 进行逆傅里叶变换
                # magnitude_only_img = np.fft.ifft2(np.fft.ifftshift(magnitude_only))
                # magnitude_only_img = np.abs(magnitude_only_img)
                # print("幅值逆变换图像原始数值范围:", np.min(magnitude_only_img), np.max(magnitude_only_img))
                # # 归一化幅值图像
                # magnitude_only_img = (magnitude_only_img - np.min(magnitude_only_img)) / (
                #             np.max(magnitude_only_img) - np.min(magnitude_only_img) + 1e-8)
                # print("幅值逆变换图像归一化后数值范围:", np.min(magnitude_only_img), np.max(magnitude_only_img))
                #
                # # 仅保留相位信息，幅值置为 1
                # phase_only = 10 * np.exp(1j * phase)
                # # 进行逆傅里叶变换
                # phase_only_img = np.fft.ifft2(np.fft.ifftshift(phase_only))
                # phase_only_img = np.abs(phase_only_img)
                # print("相位逆变换图像原始数值范围:", np.min(phase_only_img), np.max(phase_only_img))
                # # 归一化相位图像
                # phase_only_img = (phase_only_img - np.min(phase_only_img)) / (
                #             np.max(phase_only_img) - np.min(phase_only_img) + 1e-8)
                # print("相位逆变换图像归一化后数值范围:", np.min(phase_only_img), np.max(phase_only_img))
                #
                # # 使用对数变换调整数值分布
                # phase_only_img = np.log(1 + 10 * phase_only_img)  # 加上1是为了避免对0取对数
                # phase_only_img = (phase_only_img - np.min(phase_only_img)) / (
                #             np.max(phase_only_img) - np.min(phase_only_img) + 1e-8)
                #
                # print("调整后相位逆变换图像数值范围:", np.min(phase_only_img), np.max(phase_only_img))
                #
                # # 创建一个包含三个子图的画布
                # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                #
                # # 显示原始图像
                # axes[0].imshow(img, cmap='gray')
                # axes[0].set_title('Original Image')
                # axes[0].axis('off')
                # # 显示仅含幅值信息的逆变换图像
                # axes[1].imshow(magnitude_only_img, cmap='gray')
                # axes[1].set_title('Magnitude Only')
                # axes[1].axis('off')
                # # 显示仅含相位信息的逆变换图像
                # axes[2].imshow(phase_only_img, cmap='gray')
                # axes[2].set_title('Phase Only')
                # axes[2].axis('off')
                # # 显示图像
                # plt.show()
                # # 单独展示 adjusted_phase_only_img，去除白边
                # fig, ax = plt.subplots(figsize=(8, 8))
                # ax.imshow(phase_only_img, cmap='gray')
                # ax.axis('off')
                # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                # # 显示图像
                # plt.show()
                # ------------------------------------end--------------------



                # ---------------------下面这一段是在生成所谓的truth的时候写的代码，当时验证最终仍旧觉得直接对比图像来计算psnr更加合理------
                # img1 = img_as_ubyte(img)
                #
                # img_flat = img1.flatten()
                # print(min(img_flat))
                # print(max(img_flat))
                #
                # # 创建保存文件的目录（如果目录不存在）
                # truth_path = "/home/mengping/mlp-F/truth_data"
                # data_name = 'PBCT_{}_{}_{}_{}'.format(img_idx, num_proj, input_snr, num_val_proj)
                #
                # # 生成保存路径
                # save_path = os.path.join(truth_path, data_name + '.txt')
                #
                # # 如果目录不存在，创建目录
                # if not os.path.exists(truth_path):
                #     os.makedirs(truth_path)
                #
                # # 保存 img 为 txt 文件
                # np.savetxt(save_path, img1, delimiter=',', fmt='%.2f')  # 可以根据需要设置格式
                # print(f"Array saved to {save_path}")
                #
                # # 确保保存文件夹存在
                # save_folder = '/home/mengping/mlp-F/resultA'  # 保存文件夹路径
                # if not os.path.exists(save_folder):
                #     os.makedirs(save_folder)
                # # 保存图像数组
                # save_path = os.path.join(save_folder, 'truth_image.npy')
                # np.save(save_path, img)


                # f = np.fft.fft2(img)
                # # 将零频率分量移到频谱中心
                # fshift = np.fft.fftshift(f)
                # # np.savetxt("/home/mengping/mlp-F/resultA/truth_measure.txt", fshift)
                # # 计算幅度谱（取对数是为了更好的可视化）
                # magnitude_spectrum = np.log(np.abs(fshift) + 1)
                #
                # # 进行反变换
                # f_ishift = np.fft.ifftshift(fshift)
                # img_back = np.fft.ifft2(f_ishift)
                # img_back = np.real(img_back)  # 取实部，因为傅里叶反变换的结果可能有极小的虚部

                # # 可视化
                # plt.subplot(131), plt.imshow(img, cmap='gray')
                # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                # plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
                # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
                # plt.subplot(133), plt.imshow(img_back, cmap='gray')
                # plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])
                # plt.show()
                # -------------------------------end-------------------------------------------------




                # 这行代码定义了一个重建空间，使用了ODL库的uniform_discr函数，它创建了一个二维均匀离散空间
                # 范围从 (-1, -1) 到 (1, 1)，形状与图像 img 相同，数据类型为 float32个点，每一个点对应了图像的像素值
                # 如果图像的大小为512x512，那么该离散空间则可以理解为在（-1，-1）与（1,1）之间的平面中对应了512x512

                reco_space = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=img.shape, dtype='float32')
                theta = np.linspace(0.5 * np.pi, 1.5 * np.pi, num_proj, endpoint=False)
                grid = odl.RectGrid(theta)
                #　如果 theta 是一个长度为 num_proj 的一维数组，那么 RectGrid 会生成一个网格，其中每个角度都对应一个网格点。这些角度将决定射线的方向，
                # 这是个常规的图像网格，而是用于描述射线方向的几何结构。它的作用是定义在哪些方向上发射射线来获取投影数据。
                angles = odl.uniform_partition_fromgrid(grid)
                # Detector: uniformly sampled, n = 512, min = -40, max = 40
                detector_partition = odl.uniform_partition(-1, 1, num_dete)
                # Geometry with large fan angle
                geometry = odl.tomo.Parallel2dGeometry(angles, detector_partition)
                # Ray transform (= forward projection). We use the ASTRA CUDA backend.
                ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
                # projections
                projs = np.array(ray_trafo(img))



                # ------------------------------以下这段生成fbp重建的结果并将其保存-----------------
                # ray_trafo_fbp = odl.tomo.fbp_op(ray_trafo)
                # recon = np.array(ray_trafo_fbp(projs))
                # # 找到数组中的最小值和最大值
                # min_val = np.min(recon)
                # max_val = np.max(recon)
                # # 避免最大值和最小值相等时出现除零错误
                # if max_val == min_val:
                #     if min_val > 0:
                #         recon_normalized = np.ones_like(recon)
                #     else:
                #         recon_normalized = np.zeros_like(recon)
                # else:
                #     # 归一化到 0 - 1 范围
                #     recon_normalized = (recon - min_val) / (max_val - min_val)
                # # 保留小数点后 4 位
                # recon = np.round(recon_normalized, 6)
                # # 创建保存文件的目录（如果目录不存在）
                # fbp_path = "/home/mengping/mlp-F/fbp_data"
                # data_name = 'fbp_{}_{}_{}_{}'.format(img_idx, num_proj, input_snr, num_val_proj)
                # # 生成保存路径
                # save_path = os.path.join(fbp_path, data_name + '.txt')
                # # 如果目录不存在，创建目录
                # if not os.path.exists(fbp_path):
                #     os.makedirs(fbp_path)
                # # 保存归一化后的幅值结果为 txt 文件
                # np.savetxt(save_path, recon, delimiter=',', fmt='%.6f')
                # ------------------------------------end-----------------------------------------


                #  -------------可视化fbp重建的结果--------------------
                # 创建一个图形和轴对象，并设置合适的 figsize，这里设置为正方形
                # fig, ax = plt.subplots(figsize=(6, 6))
                # # 显示图像
                # im = ax.imshow(recon, cmap='gray')
                # # 隐藏轴标签和刻度
                # ax.axis('off')
                # # 调整子图参数，去除边距
                # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                # # 显示图像
                # plt.show()
                # # projs, _ = addawgn(projs, input_snr)
                #
                # plt.imshow(projs, cmap='gray', aspect='auto')  # 显示灰度图
                # plt.axis("off")  # 去掉坐标轴
                # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉白边
                # plt.show()
                # -------------------------------end-------------------------------------


                proj_sino = projs
                # print(proj_sino.min(), proj_sino.max())
                # 图像范围是0-1
                proj_tensor = np.fft.ifftshift(proj_sino)
                f_ps = np.fft.fft(proj_tensor)
                # ----------------看幅值-------------------
                # mag = np.abs(f_ps)
                # print(mag.max(), mag.min())
                # ----------------end---------------------
                f_ps = np.fft.fftshift(f_ps)
                print(np.max(f_ps.real), np.max(f_ps.imag))
                print(np.min(f_ps.real), np.min(f_ps.imag))
                # 打印实部和虚部绝对值的最小值
                print(np.min(np.abs(f_ps.real)), np.min(np.abs(f_ps.imag)))


                SanDianView(f_ps)
                ZhexianView(f_ps)



                fps_real = y_func(f_ps.real,2/np.max(f_ps.real))
                # print("np.max(f_ps.real)", np.max(f_ps.real))
                # print("np.min(f_ps.real)", np.min(f_ps.real))
                fps_imag = y_func(f_ps.imag,2/np.max(f_ps.imag))
                # print("np.max(f_ps.imag)", np.max(f_ps.imag))
                # print("np.min(f_ps.imag)", np.min(f_ps.imag))
                f_ps = fps_real + 1j * fps_imag
                # print("np.max(f_ps.real)", np.max(f_ps.real))
                # print("np.min(f_ps.real)", np.min(f_ps.real))
                # print("np.max(f_ps.imag)", np.max(f_ps.imag))
                # print("np.min(f_ps.imag)", np.min(f_ps.imag))

                SanDianView(f_ps)
                ZhexianView(f_ps)

                # ----------------看幅值-------------------
                # mag = np.abs(f_ps)
                # print(mag.max(), mag.min())
                # ----------------end---------------------

                y_values = f_ps.flatten()

                # 生成网格坐标
                pad_N = projs.shape[1]
                # 生成角度和半径数组
                a = np.linspace(0, np.pi, num_proj, endpoint=False)
                r = np.arange(pad_N) - pad_N / 2
                r, a = np.meshgrid(r, a)
                a = -a
                # 展平半径和角度数组
                r = r.flatten()
                a = a.flatten()
                # 计算源点坐标（从极坐标转换为笛卡尔坐标）
                srcx_cos = r * np.cos(a)
                srcy_sin = r * np.sin(a)

                srcx = srcx_cos
                # print(np.max(srcx))
                # print(np.min(srcx))
                srcy = srcy_sin
                # print(np.max(srcy))
                # print(np.min(srcy))
                src_point = np.column_stack((srcx, srcy))

                # # 目标点坐标生成与展平，dstx, dsty用于后面的插值检查正确性，不参与生成训练数据
                dstx, dsty = np.meshgrid(np.arange(num_dete), np.arange(num_dete))
                dstx = dstx - pad_N / 2
                dsty = dsty - pad_N / 2
                dstx = dstx.flatten()
                dsty = dsty.flatten()
                # print(np.max(dstx))
                # print(np.max(dsty))
                # print(np.min(dstx))
                # print(np.min(dsty))

                x_pairs = []
                print(x_pairs[256])
                for point in src_point:
                    # 归一化到（-1,1）
                    x_norm = point[0] / 256
                    y_norm = point[1] / 256
                    x_pairs.append((x_norm, y_norm))

                # print(np.max(x_pairs[0]))
                # print(np.min(x_pairs[0]))

                # ============================================模拟插值=======================================
                points = (src_point, (dstx, dsty))
                m = (dstx, dsty)
                f_p = f_ps.flatten()
                Fourier2_radial = griddata(points[0], f_p, points[1], method='nearest',
                                           fill_value = 0 + 1j * 0).reshape((pad_N, pad_N))


                # #-----------------------创建一个全零的结果，初始化用零来填充，是没有意义的-------------
                #
                # result = np.zeros((dstx.size), dtype=complex)
                # # 遍历目标点
                # for i, target_point in enumerate(zip(dstx, dsty)):
                #     target_point = np.array(target_point)
                #     # 查找目标点是否在源点中
                #     match_indices = np.where(np.all(src_point == target_point, axis=1))[0]
                #     if match_indices.size > 0:
                #         # 如果找到匹配的点，使用该点的值
                #         result[i] = f_p[match_indices[0]]
                #     else:
                #         # 如果未找到匹配的点，保持为零
                #         result[i] = 0 + 1j * 0
                #
                # Fourier2_radial = result.reshape((pad_N, pad_N))
                # -----------------------------end----------------------------------------


                # 频域处理和逆傅里叶变换
                Fourier2_np = Fourier2_radial
                Fourier2_shifted = np.fft.fftshift(Fourier2_np)

                phase = np.angle(Fourier2_shifted)
                # 仅保留相位信息，幅值置为 1
                phase_only = 10 * np.exp(1j * phase)
                # 进行逆傅里叶变换
                phase_only_img = np.fft.ifft2(np.fft.ifftshift(phase_only))
                phase_only_img = np.abs(phase_only_img)
                print("相位逆变换图像原始数值范围:", np.min(phase_only_img), np.max(phase_only_img))
                # 归一化相位图像
                phase_only_img = (phase_only_img - np.min(phase_only_img)) / (
                        np.max(phase_only_img) - np.min(phase_only_img) + 1e-8)
                print("相位逆变换图像归一化后数值范围:", np.min(phase_only_img), np.max(phase_only_img))

                # 使用对数变换调整数值分布
                phase_only_img = np.log(1 + 10 * phase_only_img)  # 加上1是为了避免对0取对数
                phase_only_img = (phase_only_img - np.min(phase_only_img)) / (
                        np.max(phase_only_img) - np.min(phase_only_img) + 1e-8)

                print("调整后相位逆变换图像数值范围:", np.min(phase_only_img), np.max(phase_only_img))
                # 可视化图像
                plt.imshow(phase_only_img, cmap=cm.gray)
                plt.title("Phase Only Image")
                plt.colorbar()
                plt.show()

                result = np.fft.ifft2(Fourier2_np)
                inverse_result = np.fft.fftshift(result)
                fig, ax = plt.subplots(figsize=(5.12, 5.12))
                plt.imshow(np.abs(inverse_result.real), cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去掉四周的空白边
                plt.show()

                # ============================================end=======================================

                # 训练集数据
                train_inputs = np.array(x_pairs)
                train_truths = np.array(y_values)



                # ==============================训练数据的生成=================
                valid_theta = np.linspace(0.5 * np.pi, 1.5 * np.pi, num_proj, endpoint=False)
                grid = odl.RectGrid(np.sort(valid_theta))
                valid_angles = odl.uniform_partition_fromgrid(grid)
                # Geometry with large fan angle
                geometry = odl.tomo.Parallel2dGeometry(valid_angles, detector_partition)
                # Ray transform (= forward projection). We use the ASTRA CUDA backend.
                ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')
                projs = np.array(ray_trafo(img))

                proj_sino = projs

                proj_tensor = np.fft.ifftshift(proj_sino)
                f_ps = np.fft.fft(proj_tensor)
                f_ps = np.fft.fftshift(f_ps)

                fps_real = y_func(f_ps.real, 2/np.max(f_ps.real))
                print("np.max(f_ps.real)", np.max(f_ps.real))
                fps_imag = y_func(f_ps.imag, 2/np.max(f_ps.imag))
                print("np.max(f_ps.imag)", np.max(f_ps.imag))
                f_ps = fps_real + 1j * fps_imag


                pad_N = projs.shape[1]
                # 生成角度和半径数组
                a = np.linspace(0, np.pi, num_proj, endpoint=False)
                r = np.arange(pad_N) - pad_N / 2
                r, a = np.meshgrid(r, a)
                a = -a
                # 展平半径和角度数组
                r = r.flatten()
                a = a.flatten()
                # 计算源点坐标（从极坐标转换为笛卡尔坐标）
                srcx_cos = r * np.cos(a)
                srcy_sin = r * np.sin(a)
                srcx = srcx_cos
                srcy = srcy_sin
                src_point = np.column_stack((srcx, srcy))
                x_pairs = []
                for point in src_point:
                    x_norm = point[0] / 256
                    y_norm = point[1] / 256
                    x_pairs.append((x_norm, y_norm))
                y_values = f_ps.flatten()
                valid_inputs = np.array(x_pairs)
                valid_truths = np.array(y_values)

                # 保存数据集
                # 四个数据集：tri_inputs、tri_truths、val_inputs、val_truths 分别存储训练和验证数据的坐标和幅度（或者说标签）
                print('Saving the dataset . . .')
                with h5py.File(dataset_name, 'w') as hf:
                    hf.create_dataset("tri_inputs", data=train_inputs)  # training coordinates
                    hf.create_dataset("tri_truths", data=train_truths)  # training amplitudes
                    hf.create_dataset("val_inputs", data=valid_inputs)  # testing coordinates
                    hf.create_dataset("val_truths", data=valid_truths)  # testing coordinates

                print('. . . Finished')
                print('Training Data: #{}, [1,{}] [1,{}]'.format(train_inputs.shape[0], train_inputs.shape[1],
                                                                 train_truths.shape[0]))
                print('Validation Data: #{}, [1,{}] [1,{}]'.format(valid_inputs.shape[0], valid_inputs.shape[1],
                                                                   valid_truths.shape[0]))







