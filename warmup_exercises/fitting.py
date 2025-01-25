import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn, optim

LR = 0.01


def polynomial_gen(polynomial_coefficients: list, size, noise_std=0.1) -> tuple[Tensor, Tensor]:
    """
    生成 n 阶多项式数据
    .. math:: y = \\sum_{i=0}^{n} \\frac{a_i}{i!} x^i + \\epsilon
    :param polynomial_coefficients: a_i
    :param size: 生成的数据集大小
    :param noise_std: 多项式的噪声项
    :return: 特征张量与标签张量的元组
    """
    polynomial_coefficients = np.array(polynomial_coefficients)
    orders = np.arange(len(polynomial_coefficients))

    x = np.random.normal(size=(size, len(polynomial_coefficients)))
    features = np.power(x, orders) / np.vectorize(math.factorial)(orders)
    labels = features @ polynomial_coefficients
    labels += np.random.normal(scale=noise_std, size=labels.shape)
    return torch.from_numpy(features).to(torch.float32), torch.from_numpy(labels).to(torch.float32)


def resize_features(features: torch.Tensor, to_columns: int) -> torch.Tensor:
    """
    为每一列生成新的随机数据并扩展特征张量

    :param features: 原始特征张量
    :param to_columns: 新特征张量的目标列数（编号从 1 开始）
    :return: 扩展后的特征张量
    """
    num_samples, num_features = features.shape
    if to_columns > num_features:
        new_data = torch.randn(num_samples, to_columns - num_features)  # 生成新的随机数据
        expanded_features = torch.cat((features, new_data), dim=1)
        return expanded_features
    else:
        return features[:, :to_columns]


def split_dataset(features: torch.Tensor, labels: torch.Tensor, train_ratio: float = 0.8) -> tuple:
    """
    将数据集划分为训练集和验证集

    :param features: 特征张量
    :param labels: 标签张量
    :param train_ratio: 训练集的比例，默认值为 0.8
    :return: 训练集和验证集的特征及标签
    """
    num_samples = features.size(0)
    num_train = int(num_samples * train_ratio)

    indices = torch.randperm(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]

    return train_features, train_labels, val_features, val_labels


def calculate_accuracy(outputs: Tensor, labels: Tensor) -> float:
    """
    计算模型的精度

    :param outputs: 模型输出
    :param labels: 真实标签
    :return: 精度
    """
    with torch.no_grad():
        # 计算均方误差（MSE）
        mse = nn.MSELoss()(outputs, labels)
        return mse.item()  # 返回损失值


def train_test(train_features, train_labels, test_features, test_labels, epochs=1000):
    net = nn.Sequential(nn.Linear(in_features=train_features.size(1), out_features=1, bias=False))
    net.apply(lambda m: nn.init.normal_(m.weight, mean=0, std=0.01) if isinstance(m, nn.Linear) else None)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), LR)

    # 用于保存损失和 MSE 的列表
    train_losses = []
    val_losses = []
    train_mses = []
    val_mses = []

    for epoch in range(epochs):
        # 训练阶段
        net.train()
        optimizer.zero_grad()
        train_outputs = net(train_features).squeeze()
        train_loss = criterion(train_outputs, train_labels)
        train_loss.backward()
        optimizer.step()

        # 验证阶段
        net.eval()
        with torch.no_grad():
            val_outputs = net(test_features).squeeze()
            val_loss = criterion(val_outputs, test_labels)

        # 记录损失和 MSE
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_mses.append(calculate_accuracy(train_outputs, train_labels))
        val_mses.append(calculate_accuracy(val_outputs, test_labels))

        # 输出训练和验证的损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1:4}/{epochs}]:'
                  f'Train Loss: {train_loss.item():.4f}, Train MSE: {train_mses[-1]:.4f}, '
                  f'Validation Loss: {val_loss.item():.4f}, Validation MSE: {val_mses[-1]:.4f}')

    param = net[0].weight.data.numpy().flatten()
    return param, train_losses, val_losses, train_mses, val_mses


def plot_training_history(train_losses, val_losses, train_mses, val_mses):
    """
    绘制训练损失、验证损失、训练 MSE 和验证 MSE 的变化过程

    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表
    :param train_mses: 训练 MSE 列表
    :param val_mses: 验证 MSE 列表
    """

    # 绘制损失变化
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='#81BBF8')
    plt.plot(val_losses, label='Validation Loss', color='#0C68CA')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.yscale('log')  # 设置纵坐标为对数坐标
    plt.legend()

    # 绘制 MSE 变化
    plt.subplot(1, 2, 2)
    plt.plot(train_mses, label='Train MSE', color='#F1A2AB')
    plt.plot(val_mses, label='Validation MSE', color='#AD1A2B')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.yscale('log')  # 设置纵坐标为对数坐标
    plt.legend()

    plt.tight_layout()  # 自动调整子图间距
    plt.show()  # 展示图表


if __name__ == '__main__':
    coeffs = [5.0, 1.2, -3.4, 5.6]  # 三阶多项式，共 4 项系数
    SIZE = 1000
    EPOCHS = 1000

    features, labels = polynomial_gen(coeffs, SIZE)
    features = resize_features(features, to_columns=4)

    train_f, train_l, val_f, val_l = split_dataset(features, labels)

    learned_coeffs, *record = train_test(train_f, train_l, val_f, val_l, epochs=EPOCHS)
    print(f'\n真实的多项式系数：{coeffs}')
    print(f'习得的多项式系数：{learned_coeffs}')
    plot_training_history(*record)
