"""
多输入多输出通道
"""
from typing import Literal, Tuple

import torch
from torch import nn


def corr2d(input2d: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """二维的互相关运算"""

    h_input, w_input = input2d.shape
    h_kernel, w_kernel = kernel.shape

    # 不使用填充 (padding) 时的输出尺寸
    h_output = h_input - h_kernel + 1
    w_output = w_input - w_kernel + 1

    output = torch.empty(h_output, w_output)

    # 互相关运算
    for i in range(h_output):  # 遍历输出张量的每一行
        for j in range(w_output):  # 遍历输出张量的每一列
            output[i, j] = (input2d[i:i + h_kernel, j:j + w_kernel] * kernel).sum()  # 输入的局部区域与卷积核的逐元素相乘，并求和

    return output


def pool2d(input2d: torch.Tensor, window: Tuple[int, int], mode: Literal['max', 'avg']) -> torch.Tensor:
    """二维的池化运算"""

    h_input, w_input = input2d.shape
    h_window, w_window = window

    # 不使用填充 (padding) 时的输出尺寸
    h_output = h_input - h_window + 1
    w_output = w_input - w_window + 1

    output = torch.empty(h_output, w_output)

    # 池化运算
    for h in range(h_output):
        for w in range(w_output):
            if mode == 'max':
                output[h, w] = input2d[h:h + h_window, w:w + w_window].max()
            elif mode == 'avg':
                output[h, w] = input2d[h:h + h_window, w:w + w_window].mean()
            else:
                raise NotImplementedError("只实现了 'max' 和 'avg' 池化")

    return output


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) * self.bias


if __name__ == '__main__':
    # i = torch.tensor([[[0., 1., 2.],  # (2, 3, 3)
    #                    [3., 4., 5.],
    #                    [6., 7., 8.]],
    #                   [[1., 2., 3.],
    #                    [4., 5., 6.],
    #                    [7., 8., 9.]]])
    # k0 = torch.tensor([[[0., 1.],  # (2, 2, 2)
    #                     [2., 3.]],
    #                    [[1., 2.],
    #                     [3., 4.]]])
    # k1 = k0 + 1
    # k2 = k0 + 2
    # k_stack = torch.stack((k0, k1, k2))  # (3, 2, 2, 2)
    #
    # container = []
    # for k in k_stack:
    #     temp = torch.stack([corr2d(c_i, c_k) for c_i, c_k in zip(i, k)])
    #     container.append(temp.sum(dim=0))
    #
    # result = torch.stack(container)
    #
    # print(f'{result = }')
    # print(f'{result.shape = }')

    i = torch.tensor([[0.0, 1.0, 2.0],
                      [3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0]])
    result = pool2d(i, window=(2, 2), mode='max')
    print(f'{result = }')
