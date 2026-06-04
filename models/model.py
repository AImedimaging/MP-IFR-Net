import numpy as np
import torch
import torch.nn as nn


# siren模型，经过多方面的检验可以确定sinelayer的代码基本确定无问题
class SineLayer(nn.Module):
    # 用正弦函数做激活函数
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, is_last=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.is_last = is_last
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        # x = self.linear(input)
        # return torch.sigmoid(2 * x) if self.is_last else torch.sin(self.omega_0 * x)  # 在最后一层用sigmoid激活函数，在前面还是用sin激活函数
        return torch.sin(self.omega_0 * self.linear(input))


# 小波变换里面的模型，暂时不太考虑这个，这个复小波激活函数可以用来直接学习复数的应用中去
# 应用不可行
class ComplexGaborLayer2D(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        self.in_features = in_features
        # self.i = 1

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
        # Second Gaussian window
        self.scale_orth = nn.Linear(in_features,
                                    out_features,
                                    bias=bias,
                                    dtype=dtype)

    def forward(self, input):
        if self.is_first:
            input = input.to(torch.float)
        else:
            input = input.to(torch.cfloat)
        lin = self.linear(input)
        scale_x = lin
        scale_y = self.scale_orth(input)
        freq_term = torch.exp(1j * self.omega_0 * lin)
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0 * self.scale_0 * arg)
        return freq_term * gauss_term


def y_func(x, omega):
    return (1 - np.exp(-omega * x)) / (1 + np.exp(-omega * x))


# 现在用的MLP，在这个的基础上面改，修改编码方式，激活函数，网络参数以及损失函数
class MLP(nn.Module):
    def __init__(self,data_kargs={'ic': 2, 'oc': 1}, net_kargs={}):
        super(MLP, self).__init__()
        self.data_kargs = data_kargs
        self.ffm = net_kargs.get('ffm', 'linear')
        self.encoder_layer_num = net_kargs.get('encoder_layer_num',8)  # 4
        self.decoder_layer_num = net_kargs.get('decoder_layer_num',1)  # 1
        self.feature_num = net_kargs.get('feature_num',512)
        self.coordinates_size = net_kargs.get('ic',2)
        self.out_dim = net_kargs.get('oc',1)


        # NIK损失函数超参数设置
        self.sigma = 1
        self.eps = 1e-2
        self.factor = 10
        self.weight_coef = 1

        # 哈希编码参数设置
        # # 添加哈希编码，编码器，这里通过encoder的输出得到在网络中串联之后的维度
        # self.encoder = encoder
        # self.in_dim = encoder.output_dim
        # # 这里为了方便设置为了超参数
        # self.bound = 1  # 哈希编码接受输入的范围

        self.L = net_kargs.get('L', 256)  # 傅里叶编码


        # # 调整后的超参数
        # self.sigma = 1
        # # self.eps = 1e-2
        # self.eps = 1e-8
        # # self.sigma = 50
        # self.factor = 10
        # self.lambda_factor = 1
        # self.gamma = 0.1


        self.b_scale = 4
        self.B = torch.normal(0, 1, (self.coordinates_size, int(self.feature_num/2)))# (mean, std, size)
        self.B = self.B.cuda()
        # dtype = torch.cfloat
        # B = None
        # # 定义高斯编码器，高斯编码的方式时用到
        # if B is None:
        #     self.B = torch.randn((self.coordinates_size, self.feature_num // 2), dtype=torch.float32)
        #     # self.B = torch.normal(0, 1, (self.coordinates_size, self.feature_num // 2))
        # else:
        #     self.B = B
        # self.B = self.B.cuda()

        # self.mu = 255
        # self.height = 256
        # # self.sigma = 100
        # self.B = self.generate_distribution()
        # self.B = torch.tensor(self.B, dtype=torch.float32)
        # # # 把 self.B 放到和模型参数相同的设备上
        # # device = next(self.parameters()).device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.B = self.B.to(device)

        # input_size = self.data_kargs['ic'] * 2 * self.L  # 这里是设置傅里叶编码的方式
        # hidden_features = int(self.feature_num / 2) # 这里暂时是用不到的，这个是设置复数层时需要设置的隐藏层数量

        self.net = [nn.Linear(self.feature_num, self.feature_num)]  # 哈希编码的

        # self.net = [nn.Linear(2 * self.feature_num, self.feature_num)] # MLP的
        # self.net = [nn.Linear(input_size, hidden_features)]
        # self.net = [SineLayer(input_size, self.feature_num, is_first=True, omega_0=30)] # siren的
        # self.net = [SineLayer(self.coordinates_size, self.feature_num, is_first=True, omega_0=30)] # siren不加编码的
        for layer in range(self.encoder_layer_num - 1):
            # siren的
            # self.net.append(SineLayer(self.feature_num, self.feature_num, is_first=False, is_last=False, omega_0=30))
            # 线性的模型
            self.net.append(nn.Linear(self.feature_num, self.feature_num))
            self.net.append(nn.ReLU())
            # 小波变换的
            # self.hidlayer = ComplexGaborLayer2D
            # self.net.append(self.hidlayer(self.feature_num,
            #                             self.feature_num,
            #                             omega0=10.,
            #                             sigma0=10.0))

        final_linear = nn.Linear(self.feature_num, self.out_dim * 2)  # 线性的MLP的
        # final_linear = nn.Linear(self.feature_num, self.out_dim*2,dtype = dtype)  # siren的
        # self.net.append(SineLayer(self.feature_num, self.out_dim * 2, is_last=True))

        # final_linear = nn.Linear(hidden_features, self.out_dim * 2, dtype=dtype)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / self.feature_num) / 30,
                                         np.sqrt(6 / self.feature_num) / 30)  # MLP时会用到的，一个比较常用的初始化权重的方式

        # MLP的时候用的
        self.net.append(final_linear)
        # tanh加不加作用不大
        # self.net.append(nn.Tanh())
        self.net = nn.Sequential(*self.net)


# ======================用函数模拟生成非线性的频率位置编码，暂时pass，无效
# def normal_distribution(self, x):
#     return self.height * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))
#     # return x
#
# def generate_distribution(self):
#     indices = np.arange(self.feature_num)
#     # 生成前256个数据
#     first_half = self.normal_distribution(indices[:256])
#     # 生成后256个数据，通过前256个数据关于 x 轴对称
#     # second_half = -first_half[::-1]
#     # 合并数据
#     # data = np.concatenate((first_half, second_half))
#     # 重复两次以形成 (2, feature_num) 的数组
#     # repeated_data = np.tile(data, (self.coordinates_size, 1))
#     return first_half
# ============================end=================================


    def pre_process(self, x):
        # # # 傅里叶编码
        # freq_embedding = []
        # for l in range(self.L):
        #     if self.ffm == 'linear':
        #         # 设置为单位频率进行编码
        #         sin_term = torch.sin((l + 1) * np.pi * x)
        #         cos_term = torch.cos((l + 1) * np.pi * x)
        #     elif self.ffm == 'loglinear':
        #         sin_term = torch.sin(2 ** l * np.pi * x)
        #         cos_term = torch.cos(2 ** l * np.pi * x)
        #     freq_embedding.append(sin_term)
        #     freq_embedding.append(cos_term)
        # freq_embedding = torch.cat(freq_embedding, dim=-1)
        # in_node = freq_embedding

        # # 不加编码
        # in_node = x

        # # 高斯编码
        # embedding =  x @ self.B
        # embedding = torch.cat([torch.sin(2 * np.pi * embedding), torch.cos(2 * np.pi * embedding)], dim=-1)
        # in_node = embedding

        # freq_embedding = []
        # for l in self.B:
        #     if self.ffm == 'linear':
        #         sin_term = torch.sin(l * 2 * np.pi * x)
        #         cos_term = torch.cos(l * 2 * np.pi * x)
        #     elif self.ffm == 'loglinear':
        #         sin_term = torch.sin(2 ** l * np.pi * x)
        #         cos_term = torch.cos(2 ** l * np.pi * x)
        #     freq_embedding.append(sin_term)
        #     freq_embedding.append(cos_term)
        # freq_embedding = torch.cat(freq_embedding, dim=-1)
        # in_node = freq_embedding

        # # 哈希编码
        # in_node = self.encoder(x, self.bound)

        scaled = self.coordinates_size * x  # [batch_size, in_features]
        d = torch.norm(x, dim=-1, keepdim=True)  # [batch_size, 1]
        proj = scaled @ (self.B * self.b_scale)  # [batch_size, num_freq]
        proj = proj + d

        sin_feat = (2 ** 0.5) * torch.sin(proj) * self.weight_coef
        cos_feat = (2 ** 0.5) * torch.cos(proj) * self.weight_coef
        in_node = torch.cat((sin_feat, cos_feat), dim=-1)
        return in_node


    def post_process(self, outputs):
        # output = torch.complex(outputs[..., 0:self.out_dim].real, outputs[..., self.out_dim:].real) # 实验只学实部的代码，后面几乎用不到
        output = torch.complex(outputs[..., 0:self.out_dim], outputs[..., self.out_dim:])
        return output


    def forward(self, inputs):
        output = self.post_process(self.net(self.pre_process(inputs)))
        # # 将可逆归一化函数作为激活函数应用到输出上
        # omega = 10  # 可以根据需要调整 omega 的值
        # output = torch.tensor(y_func(output.cpu().detach().numpy(), omega), dtype=torch.complex64, device=output.device)
        return output


    # 以下代码在进行频域的缩放，利用复数的数学形式转换,但是这个好像只能针对整体的频域数据
    # 这种散乱的点没有办法运用上来，后面舍弃这种缩放放的方式
    # def log_extension(self, F, K):
    #     # 假设 F 是一个张量
    #     # 将张量从 CUDA 设备移到 CPU 上
    #     F = F.detach().cpu()
    #     magnitude = np.abs(F)
    #     phase = np.angle(F)
    #     magnitude = torch.from_numpy(magnitude) if isinstance(magnitude, np.ndarray) else magnitude
    #     # 将 phase 转换为 torch.Tensor
    #     phase = torch.from_numpy(phase) if isinstance(phase, np.ndarray) else phase
    #     extended_magnitude = K * torch.log(1 + magnitude)
    #     return extended_magnitude * torch.exp(1j * phase)


    # 均方误差损失函数
    def loss_train(self, x_hat, y):
        # 实部和虚部均方误差
        real_loss = torch.mean((x_hat.real - y.real) ** 2)
        imag_loss = torch.mean((x_hat.imag - y.imag) ** 2)
        # 模值的均方误差
        mag_loss = torch.mean((torch.abs(x_hat) - torch.abs(y)) ** 2)
        # if real_loss > 40:
        #     real_loss = real_loss/2
        # 结合损失
        loss = mag_loss + real_loss + imag_loss
        # L1 正则化项
        # l1_reg = 0
        # for param in self.parameters():
        #     l1_reg += torch.norm(param, 1)
        signal_power = torch.mean(torch.abs(x_hat) ** 2)
        # 计算噪声功率
        noise_power = torch.mean(torch.abs(x_hat - y) ** 2)
        # 计算 SNR
        snr = 10 * torch.log10(signal_power / noise_power)
        return loss, snr


    # 结合上面的幅值缩放进行的，后面舍弃了这种方式
    # def loss_train0(self, x_hat, y):
    #     K = 10
    #     magnitude_x = torch.abs(x_hat)
    #     phase_x = torch.angle(x_hat)
    #     extended_magnitude_x = K * torch.log(1 + magnitude_x)
    #     x_hat1 = extended_magnitude_x * torch.exp(1j * phase_x)
    #     magnitude_y = torch.abs(y)
    #     phase_y = torch.angle(y)
    #     extended_magnitude_y = K * torch.log(1 + magnitude_y)
    #     y1 = extended_magnitude_y * torch.exp(1j * phase_y)
    #     real_loss = torch.mean((x_hat1.real - y1.real) ** 2)
    #     imag_loss = torch.mean((x_hat1.imag - y1.imag) ** 2)
    #     # 模值的均方误差
    #     mag_loss = torch.mean((torch.abs(x_hat1) - torch.abs(y1)) ** 2)
    #     # 结合损失
    #     loss = mag_loss + real_loss + imag_loss
    #     # 计算信号功率
    #     signal_power = torch.mean(torch.abs(x_hat) ** 2)
    #     # 计算噪声功率
    #     noise_power = torch.mean(torch.abs(x_hat-y) ** 2)
    #     # 计算 SNR
    #     snr = 10 * torch.log10(signal_power / noise_power )
    #     return loss, snr


    # # 现在用的方法
    # # 把中心设置为坐标原点，以距离坐标原点的距离来设置一个滤波，作用是降噪
    # def loss_train1(self, x_hat, y, cor_index):
    #     # 计算复数损失
    #     center = torch.tensor([0, 0], dtype=torch.float32, device=cor_index.device)  # 确保 device 相同
    #     dist_to_center2 = torch.sum((cor_index - center) ** 2, dim=1)
    #     # 创建滤波器值
    #     filter_value = torch.exp(-dist_to_center2 / (2 * self.sigma ** 2)).unsqueeze(-1)
    #     if x_hat.dtype == torch.float:
    #         x_hat = torch.view_as_complex(x_hat)  # * filter_value
    #     if y.dtype == torch.float:
    #         y = torch.view_as_complex(y)
    #     assert x_hat.shape == y.shape
    #     error = x_hat - y
    #     # error = error * filter_value
    #     loss = (error.abs() / (x_hat.detach().abs() + self.eps)) ** 2
    #     reg_error = x_hat - x_hat * filter_value
    #     reg = self.factor * (reg_error.abs() / (x_hat.detach().abs() + self.eps)) ** 2
    #     # reg = torch.matmul(torch.conj(reg).t(), reg)
    #     # reg = reg.abs() * self.factor
    #     # reg = torch.zeros([1]).mean()
    #     m = loss.mean()
    #     n = reg.mean()
    #     loss = m + n
    #
    #     signal_power = torch.mean(torch.abs(x_hat) ** 2)
    #     # 计算噪声功率
    #     noise_power = torch.mean(torch.abs(x_hat - y) ** 2)
    #     # 计算 SNR
    #     snr = 10 * torch.log10(signal_power / noise_power)
    #     return loss, snr


    # # def loss_train1(self, x_hat, y, cor_index):
    # #     # 计算复数损失
    # #     center = torch.tensor([0, 0], dtype=torch.float32, device=cor_index.device)  # 确保 device 相同
    # #     dist_to_center2 = torch.sum((cor_index - center) ** 2, dim=1)
    # #     self.sigma = torch.std(dist_to_center2).item()
    # #     self.c = torch.mean(dist_to_center2).item()
    # #     # 计算权重：远离中心的点贡献更大的损失
    # #     weight = 1 + self.lambda_factor * (dist_to_center2 / (dist_to_center2 + self.c))
    # #     filter_value = torch.exp(-dist_to_center2 / (2 * self.sigma ** 2)).unsqueeze(-1)
    # #
    # #     # 确保输入数据是复数
    # #     if x_hat.dtype == torch.float:
    # #         x_hat = torch.view_as_complex(x_hat)
    # #     if y.dtype == torch.float:
    # #         y = torch.view_as_complex(y)
    # #
    # #     assert x_hat.shape == y.shape
    # #     error = x_hat - y
    # #
    # #     # 计算损失，远离中心的点贡献更大的损失
    # #     loss = weight.unsqueeze(-1) * (error.abs() / (x_hat.detach().abs() + self.eps)) ** 2
    # #
    # #     # 计算正则化项，防止靠近中心的点过拟合
    # #     reg_weight = self.gamma * (1 - weight)
    # #     reg_error = x_hat - x_hat * filter_value
    # #     reg = reg_weight.unsqueeze(-1) * (reg_error.abs() / (x_hat.detach().abs() + self.eps)) ** 2
    # #
    # #     # 计算最终损失
    # #     loss = loss.mean() + reg.mean()
    # #
    # #     # 计算 SNR
    # #     signal_power = torch.mean(torch.abs(x_hat) ** 2)
    # #     noise_power = torch.mean(torch.abs(x_hat - y) ** 2)
    # #     snr = 10 * torch.log10(signal_power / (noise_power + self.eps))
    #
    #     return loss, snr

    # def loss_train1(self, x_hat, y, cor_index):
    #     # 计算每个点到中心的平方距离
    #     center = torch.tensor([0, 0], dtype=torch.float32, device=cor_index.device)
    #     dist2 = torch.sum((cor_index - center) ** 2, dim=1)  # [N]
    #
    #     # 定义两个权重函数
    #     # 基本误差权重：离中心越远，权重越大
    #     weight_err = torch.exp(-dist2 / (2 * self.sigma ** 2))  # [N]
    #     weight_err = weight_err.unsqueeze(-1)  # 调整形状以便与数据匹配
    #
    #     # 正则化项使用原先的衰减因子（可视需求调整形式）
    #     decay_factor = torch.exp(-dist2 / (2 * self.sigma ** 2)).unsqueeze(-1)
    #
    #     # 如果 x_hat, y 是 float，则转换为复数数据
    #     if x_hat.dtype == torch.float:
    #         x_hat = torch.view_as_complex(x_hat)
    #     if y.dtype == torch.float:
    #         y = torch.view_as_complex(y)
    #     assert x_hat.shape == y.shape
    #
    #     # 计算相对误差（这里使用分母来稳定数值）
    #     error = (x_hat - y) / (x_hat.detach().abs() + self.eps)
    #     loss_err = weight_err * (error.abs() ** 2)
    #
    #     # 正则化项设计：鼓励预测值符合衰减模式
    #     reg_error = (x_hat - x_hat * decay_factor) / (x_hat.detach().abs() + self.eps)
    #     loss_reg = self.factor * (reg_error.abs() ** 2)
    #
    #     # 平均所有样本的损失
    #     loss_total = loss_err.mean() + loss_reg.mean()
    #
    #     # 计算信噪比（SNR），仅作为辅助评估
    #     signal_power = torch.mean(torch.abs(x_hat) ** 2)
    #     noise_power = torch.mean(torch.abs(x_hat - y) ** 2)
    #     snr = 10 * torch.log10(signal_power / noise_power + self.eps)
    #
    #     return loss_total, snr
        # 把中心设置为坐标原点，以距离坐标原点的距离来设置一个滤波，作用是降噪
    def loss_train1(self, x_hat, y, cor_index):
        # 计算复数损失
        center = torch.tensor([0, 0], dtype=torch.float32, device=cor_index.device)  # 确保 device 相同
        dist_to_center2 = torch.sum((cor_index - center) ** 2, dim=1)
        # 创建滤波器值
        filter_value = torch.exp(-dist_to_center2 / (2 * self.sigma ** 2)).unsqueeze(-1)
        if x_hat.dtype == torch.float:
            x_hat = torch.view_as_complex(x_hat)  # * filter_value
        if y.dtype == torch.float:
            y = torch.view_as_complex(y)
        assert x_hat.shape == y.shape
        error = x_hat - y
        # error = error * filter_value
        loss = (error.abs() / (x_hat.detach().abs() + self.eps)) ** 2
        reg_error = x_hat - x_hat * filter_value
        reg = self.factor * (reg_error.abs() / (x_hat.detach().abs() + self.eps)) ** 2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()
        m = loss.mean()
        n = reg.mean()
        loss = m + n

        signal_power = torch.mean(torch.abs(x_hat) ** 2)
        # 计算噪声功率
        noise_power = torch.mean(torch.abs(x_hat - y) ** 2)
        # 计算 SNR
        snr = 10 * torch.log10(signal_power / noise_power)
        return loss, snr