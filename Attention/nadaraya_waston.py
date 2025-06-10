import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def ground_truth_func(x: Tensor) -> Tensor:
    r"""基础非线性函数 y = 2 \sin(x) + x^{0.8}"""
    return 2 * torch.sin(x) + pow(x, 0.8)


def gaussian_noise(size: int, mean: float = 0, std: float = 0.5) -> Tensor:
    r"""
    生成高斯噪声 \epsilon \sim N(mean, std^2)

    :param size: 样本数
    :param mean: 噪声的均值
    :param std: 噪声的标准差
    """
    return torch.normal(mean=mean, std=std, size=(size,))


def generate_train_data(size: int, start: float = 0, end: float = 5, noise_mean: float = 0, noise_std: float = 0.5
                        ) -> tuple[Tensor, Tensor, Tensor]:
    """
    生成训练数据

    :param size: 样本数
    :param start: 特征的起始范围，在 [start, end) 区间内均匀采样
    :param end: 特征的结束范围，在 [start, end) 区间内均匀采样
    :param noise_mean: 高斯噪声的均值
    :param noise_std: 高斯噪声的标准差
    :return: 特征、无噪声标签和带噪声标签的元组
    """
    features = torch.rand(size) * (end - start) + start
    features, _ = features.sort()  # 升序，以便注意力权重的可视化

    clean_targets = ground_truth_func(features)  # 计算无噪声标签
    noisy_targets = clean_targets + gaussian_noise(size, mean=noise_mean, std=noise_std)  # 添加噪声

    return features, clean_targets, noisy_targets


def generate_test_data(size: int, start: float = 0, end: float = 5) -> Tensor:
    """生成均匀分布的特征"""
    features = torch.linspace(start, end, size)
    return features


def plot_regression_comparison(x_train: Tensor, y_truth: Tensor, y_train: Tensor, x_test: Tensor, y_pred: Tensor):
    """绘制回归对比图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, alpha=0.5, color='#FAC21E', edgecolor='none')  # 训练数据散点
    plt.plot(x_train, y_truth, alpha=0.85, linewidth=3, color='#E43D30', label='Truth')  # 真实数据曲线
    plt.plot(x_test, y_pred, alpha=0.85, linewidth=3, color='#269745', label='Pred', linestyle='-.')  # 预测数据曲线
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()


def nadaraya_watson_weights(queries: Tensor, keys: Tensor, h: float = 1.0) -> Tensor:
    r"""
    Nadaraya-Watson 核回归中的注意力权重 w_i(x, x_i) = Softmax(-\frac{(x - x_i)^2}{2h^2})

    :param queries: 查询 x
    :param keys: 键 x_i
    :param h: 带宽参数，控制注意力分布的聚焦程度
    :return: 注意力权重矩阵，形状为 (n_queries, n_keys)
    """
    # (n_queries * n_keys) -> (n_queries, n_keys)
    queries = queries.repeat_interleave(keys.shape[0]).reshape(-1, keys.shape[0])
    weights = F.softmax(-((queries - keys) ** 2) / (2 * h ** 2), dim=1)

    return weights


class ParameterizedNadarayaWatsonKernelRegression(nn.Module):
    def __init__(self, feature_dim: int = 1, h: float = 1.0):
        """
        :param feature_dim: 键或查询的特征维度
        :param h: 带宽参数
        """
        super().__init__()
        self.w = nn.Parameter(torch.rand(size=(feature_dim,)))  # 形状随后转换为 (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, KEYS_DIM)
        self.h = h
        self.attn_weights = None  # 注意力权重，形状为 (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor) -> Tensor:
        """
        :param keys: 键，形状为 (BATCH_SIZE, KEYS_NUM, KEYS_DIM)
        :param values: 值，形状为 (BATCH_SIZE, KEYS_NUM, VALUES_DIM)
        :param queries: 查询，形状为 (BATCH_SIZE, QUERIES_NUM, KEYS_DIM)
        :return: 预测值，形状为 (BATCH_SIZE, QUERIES_NUM, VALUES_DIM)
        """
        q_expanded = queries.unsqueeze(2)  # (BATCH_SIZE, QUERIES_NUM, 1, KEYS_DIM)
        k_expanded = keys.unsqueeze(1)  # (BATCH_SIZE, 1, KEYS_NUM, KEYS_DIM)

        diff_w_sq = ((((q_expanded - k_expanded) * self.w) ** 2)  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, KEYS_DIM)
                     .squeeze(dim=-1))  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)
        self.attn_weights = F.softmax(-diff_w_sq / (2 * self.h ** 2), dim=-1)  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)

        # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM) @ (BATCH_SIZE, KEYS_NUM, VALUES_DIM) -> (BATCH_SIZE, QUERIES_NUM, VALUES_DIM)
        return self.attn_weights.bmm(values)  # 等价于 `@` 运算符


if __name__ == '__main__':
    from utils import plot_attention_heatmap

    TRAIN_SAMPLE_NUM = 50
    TEST_SAMPLE_NUM = 30
    x_train, y_truth, y_train = generate_train_data(size=TRAIN_SAMPLE_NUM)
    x_test = generate_test_data(size=TEST_SAMPLE_NUM)

    x_train = x_train.reshape(1, -1, 1)  # (TRAIN_SAMPLE_NUM,) -> (1, TRAIN_SAMPLE_NUM, 1)
    y_train = y_train.reshape(1, -1, 1)  # (TRAIN_SAMPLE_NUM,) -> (1, TRAIN_SAMPLE_NUM, 1)
    x_test = x_test.reshape(1, -1, 1)  # (TEST_SAMPLE_NUM,) -> (1, TEST_SAMPLE_NUM, 1)

    model = ParameterizedNadarayaWatsonKernelRegression()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for epoch in range(5):  # 模型训练
        optimizer.zero_grad()
        loss = criterion(model(keys=x_train, values=y_train, queries=x_train), y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch + 1}, Loss: {loss:.6f}')

    with torch.no_grad():  # 模型预测
        y_pred = model(keys=x_train, values=y_train, queries=x_test).detach()  # (1, TEST_SAMPLE_NUM, 1)
        attention_weights = model.attn_weights.detach()  # (1, TEST_SAMPLE_NUM, TRAIN_SAMPLE_NUM)

    plot_regression_comparison(x_train.squeeze(), y_truth, y_train.squeeze(), x_test.squeeze(), y_pred.squeeze())
    plot_attention_heatmap(
        # (1, TEST_SAMPLE_NUM, TRAIN_SAMPLE_NUM) -> (1, 1, TEST_SAMPLE_NUM, TRAIN_SAMPLE_NUM)
        weights=attention_weights.unsqueeze(dim=0),
        x_label='K: x train', y_label='Q: x test', figsize=(5, 5)
    )
