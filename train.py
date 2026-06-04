# import shutil, sys
# print("python exe:", sys.executable)
# print("which ninja:", shutil.which("ninja"))
# print("which ninja-build:", shutil.which("ninja-build"))

import os
import torch
import numpy as np
import logging
from data_loader import DataLoader
import matplotlib.pyplot as plt
from model import MLP
from encoder.hashencoder.hashgrid import HashEncoder

# 配置日志格式
logging.basicConfig(level=logging.INFO, format='INFO:%(message)s')

# # 指数学习率衰减
# def exp_decay(Ns, Ne, epochs):
#     lamda = - (1 / epochs) * np.log(Ne / Ns)
#     return Ns * np.exp(-lamda * np.arange(epochs))

data_kargs = {
    'ic': 2,
    'oc': 1
}

# siren模型的隐藏层数是8 特征数是512
# 调参后的MLP隐藏层数是8，特征数是512
net_kargs = {
    'encoder_layer_num': 8,
    'decoder_layer_num': 1,
    'feature_num': 512,
    'ffm': 'linear',
    'L': 256
}


if __name__ == '__main__':
    for num_proj in [60]:
            for img_idx in [5]:
                data_root = 'data'
                ori_name = f'PBCT_{img_idx}_{num_proj}_60'
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                data_loader = DataLoader(os.path.join(data_root, ori_name))
                encoder = HashEncoder(input_dim = 2, num_levels = 16, level_dim = 2, base_resolution = 16, log2_hashmap_size = 19)
                # net = MLP(encoder, data_kargs, net_kargs).to(device)
                net = MLP(data_kargs, net_kargs).to(device)

                epochs = 4000
                # # 根据噪声级别设置学习率，这里设置的是一个可以衰减的学习率
                # if noiseLevel >= 50:
                #     start, end = 2e-4, 1e-5
                # elif noiseLevel >= 40:
                #     start, end = 1e-4, 1e-5
                # else:
                #     start, end = 1e-6, 1e-7
                # lr_schedule = exp_decay(start, end, epochs)
                # 设置固定学习率
                # optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

                optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

                # 定义输出路径
                output_path = os.path.join(f'proj{num_proj}/{ori_name}/models')
                os.makedirs(output_path, exist_ok=True)

                global_step = 0
                batch_size = 1000

                # 可视化数据
                train_losses, valid_losses = [], []
                train_snr, valid_snr = [],[]
                signal_powers, noise_powers = [],[]

                for epoch in range(epochs):
                    net.train()
                    train_loss_sum, train_snr_sum = 0, 0
                    total_batches = len(data_loader.train_data[0])//batch_size
                    # 训练循环，遍历每个batch
                    for i in range(total_batches):
                        batch_x, batch_y = data_loader.get_train_batch(batch_size)
                        batch_x = batch_x.to(device, dtype=torch.float32)
                        batch_y = batch_y.to(device, dtype=torch.complex64)
                        optimizer.zero_grad()  # 清除上一次的梯度
                        output = net(batch_x)

                        # 计算损失和SNR
                        batch_y = batch_y.unsqueeze(-1)
                        loss, snr = net.loss_train1(output, batch_y, batch_x)

                        loss.backward()
                        optimizer.step()  # 更新参数

                        train_loss_sum += loss.item()
                        train_snr_sum += snr.item()
                        logging.info(
                            f"[Global Step {global_step}] [Epoch {epoch + 1}/{epochs}: {i + 1}/{total_batches}] "
                            f"Minibatch Loss = {loss.item():.4f}, Minibatch SNR = {snr.item():.4f}")
                        global_step += 1
                    # 计算训练集平均损失和 SNR
                    avg_train_loss = train_loss_sum / total_batches
                    avg_train_snr = train_snr_sum / total_batches
                    train_losses.append(avg_train_loss)
                    train_snr.append(avg_train_snr)
                    logging.info(
                        f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train SNR: {avg_train_snr:.4f}")

                    # 验证循环
                    net.eval()  # 切换到验证模式
                    valid_x, valid_y = data_loader.get_valid_data()
                    valid_x = valid_x.to(device, dtype=torch.float32)
                    valid_y = valid_y.to(device, dtype=torch.complex64)

                    with torch.no_grad():  # 不计算梯度
                        valid_output = net(valid_x)
                        valid_y = valid_y.unsqueeze(-1)
                        valid_loss, valid_snr_tensor = net.loss_train1(valid_output, valid_y, valid_x)

                    valid_losses.append(valid_loss.item())
                    valid_snr.append(valid_snr_tensor.item())

                    logging.info(
                        f"Validation Statistics, Validation Loss= {valid_loss.item():.4f}, Validation SNR= {valid_snr_tensor.item():.4f}")

                    # 以轮次来保存模型
                    save_epoch = 100
                    if (epoch + 1) % save_epoch == 0:
                        model_save_path = os.path.join(output_path, f'model_epoch_{epoch + 1}.pth')
                        torch.save(net.state_dict(), model_save_path)
                        logging.info(f"Model saved to {model_save_path}")

                    # 在训练循环中添加
                    if epoch % 500 == 0:
                        embeddings = encoder.embeddings.detach().cpu().numpy()
                        plt.hist(embeddings.flatten(), bins=50)
                        plt.title(f"Epoch {epoch} Embedding Distribution")
                        plt.show()


                    # 可视化训练过程
                    view_epoch = 100
                    if (epoch + 1) % view_epoch == 0:
                        plt.figure(figsize=(15, 10))

                        # 绘制损失和 SNR 曲线
                        plt.subplot(2, 2, 1)
                        plt.plot(train_losses, label='Train Loss')
                        plt.plot(valid_losses, label='Validation Loss')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.legend()

                        plt.subplot(2, 2, 2)
                        plt.plot(train_snr, label='Train SNR')
                        plt.plot(valid_snr, label='Validation SNR')
                        plt.xlabel('Epoch')
                        plt.ylabel('SNR')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()