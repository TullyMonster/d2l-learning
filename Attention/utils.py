from typing import Optional, List, Union, Tuple

from matplotlib.axes import Axes
from matplotlib.pyplot import tight_layout, show, subplots
from numpy import ndarray
from seaborn import heatmap
from torch import Tensor


def plot_attention_heatmap(
        weights: Union[ndarray, Tensor],
        x_label: str,
        y_label: str,
        titles: Optional[Union[List[str], str]] = 'Attention Weight',
        figsize: Tuple[float, float] = (2.5, 2.5)
):
    """
    绘制注意力权重热图

    :param weights: 形状为 [subplot_rows, subplot_cols, seq_len_q, seq_len_k] 的注意力权重
    :param x_label: 横轴名称
    :param y_label: 纵轴名称
    :param titles: 各子图标题
    :param figsize: 子图的基准大小
    """
    weights = weights.detach().cpu().numpy() if isinstance(weights, Tensor) else weights
    subplot_rows, subplot_cols, *_ = weights.shape
    total_figsize = (figsize[0] * subplot_cols, figsize[1] * subplot_rows)
    _, axes = subplots(subplot_rows, subplot_cols, figsize=total_figsize, squeeze=False)

    for i in range(subplot_rows):
        for j in range(subplot_cols):
            title = titles if isinstance(titles, str) else titles[i * subplot_cols + j]
            ax: Axes = axes[i, j]  # type: ignore
            heatmap(weights[i, j], cmap='YlOrRd', square=True, xticklabels=5, yticklabels=5, ax=ax,
                    cbar_kws={'shrink': 0.6})
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

    tight_layout()
    show()


if __name__ == '__main__':
    import torch
    from numpy import random

    weight1 = random.rand(1, 1, 12, 23)
    weight2 = random.rand(2, 4, 12, 23)
    titles2 = [f'Head {i + 1}' for i in range(weight2.shape[1])] * weight2.shape[0]
    weight3 = torch.eye(10).reshape(1, 1, 10, 10)

    plot_attention_heatmap(weight1, x_label='Keys', y_label='Queries', figsize=(5, 5))
    plot_attention_heatmap(weight2, x_label='Keys', y_label='Queries', titles=titles2)
    plot_attention_heatmap(weight3, x_label='Keys', y_label='Queries')
