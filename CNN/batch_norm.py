import torch
from torch import nn, Tensor


class BatchNorm(nn.Module):
    def __init__(self, features: int, epsilon=1e-5, momentum=0.9):
        """
        批量归一化层

        通过每个小批量的均值 running_mean 和方差 running_var 归一化数据，并使用拉伸幅度 γ (gamma) 和偏移参数 β (beta) 学习并
        恢复数据的分布特性。

        :param features: 特征数。即，全连接层的特征数，或卷积层的通道数
        :param epsilon: 小常数。保证在计算标准差时，方差不为零，避免除零错误
        :param momentum: 动量系数。每个小批量的均值和方差更新时，依赖于历史统计量的程度，取值范围为 [0, 1]
        """
        assert 0 <= momentum <= 1, '动量系数的取值范围为 [0, 1]'

        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(features))  # 注册可学习参数 gamma：拉伸幅度
        self.beta = nn.Parameter(torch.zeros(features))  # 注册可学习参数 beta：偏移

        self.register_buffer('running_mean', torch.zeros(features))  # 注册模型状态参数全局均值 running_mean 到缓冲区
        self.register_buffer('running_var', torch.ones(features))  # 注册模型状态参数全局方差 running_var 到缓冲区

    def forward(self, x: Tensor) -> Tensor:
        """根据输入数据的维度，进行批量归一化"""
        if x.dim() == 2:  # 输入数据来自全连接层的输出 (batch_size, features)
            return self._batch_norm_fc(x)
        elif x.dim() == 4:  # 输入数据来自卷积层的输出 (batch_size, channels, height, width)
            return self._batch_norm_conv(x)
        else:
            raise ValueError(f'暂不支持的输入形状：{x.shape}')

    def _batch_norm_fc(self, x: Tensor) -> Tensor:
        """
        对来自全连接层的输入进行批量归一化

        计算各个特征的均值与方差（有偏估计，除以 n 而不是 n-1）
        """
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True, unbiased=False)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        out = self.gamma * x_normalized + self.beta
        return out

    def _batch_norm_conv(self, x: Tensor) -> Tensor:
        """
        对来自卷积层的输入进行批量归一化

        计算各个通道的均值与方差（有偏估计，除以 n 而不是 n-1）
        """
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean[None, :, None, None]
            var = self.running_var[None, :, None, None]

        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        out = self.gamma[None, :, None, None] * x_normalized + self.beta[None, :, None, None]
        return out
