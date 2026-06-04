class FourierFeatureMap(nn.Module):
    def __init__(self, in_features, out_features, coordinate_scales, b_scale, distribution, distribution_scale):
        super(FourierFeatureMap, self).__init__()
        assert out_features % 2 == 0, "Fourier Features not even number!"
        self.num_freq = out_features // 2
        self.out_features = out_features
        self.use_modulation = True

        # 固定坐标缩放参数（这里不需要梯度）
        self.coordinate_scales = nn.Parameter(
            torch.tensor(coordinate_scales, dtype=torch.float32).unsqueeze(0),
            requires_grad=False
        )

        # 根据 distribution 参数生成随机矩阵 B，作为频率基底
        if distribution == "normal":
            B_init = torch.normal(0, distribution_scale, (in_features, self.num_freq))
        elif distribution == "uniform":
            B_init = torch.empty(in_features, self.num_freq).uniform_(-math.sqrt(distribution_scale),
                                                                      math.sqrt(distribution_scale))
        elif distribution == "beta":
            B_init = torch.distributions.Beta(2.0, distribution_scale).sample((in_features, self.num_freq)) * (
                    2 * torch.randint(0, 2, (in_features, self.num_freq)).float() - 1)
        else:
            raise ValueError("Unsupported distribution")

        self.B = nn.Parameter(B_init, requires_grad=False)
        self.register_buffer("b_scale", torch.tensor(b_scale, dtype=torch.float32))

        init = 1.0 / (torch.arange(1, self.num_freq + 1, dtype=torch.float32))
        self.register_buffer("initial_weight_coef", init)
        self.register_buffer("weight_coef", init.clone())

        # 如果启用调制，则增加一个简单的全连接网络获得调制量
        if self.use_modulation:
            self.modulation = nn.Sequential(
                nn.Linear(in_features, self.num_freq),
                nn.Tanh()
            )

    def forward(self, x):
        scaled = self.coordinate_scales * x  # [batch_size, in_features]
        d = torch.norm(x, dim=-1, keepdim=True)  # [batch_size, 1]
        proj = scaled @ (self.B * self.b_scale)  # [batch_size, num_freq]
        proj = proj + d
        if self.use_modulation:
            mod = self.modulation(x)  # [batch_size, num_freq]
            proj = proj + mod
        sin_feat = (2 ** 0.5) * torch.sin(proj) * self.weight_coef
        cos_feat = (2 ** 0.5) * torch.cos(proj) * self.weight_coef

        return torch.cat((sin_feat, cos_feat), dim=-1)