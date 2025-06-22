from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_softmax(atten_logits: Tensor, valid_len: Tensor) -> Tensor:
    """
    根据 valid_len 有效长度，在原始注意力分数 atten_logits 的 KEYS_NUM 维度上执行 Softmax 归一化

    :param atten_logits: 原始注意力分数，形状为：(BATCH_SIZE, QUERIES_NUM, KEYS_NUM)
    :param valid_len: 有效长度，形状为：(BATCH_SIZE,) 或 (BATCH_SIZE, QUERIES_NUM)

    :return: 注意力权重分布，形状与原始注意力分数相同
    """
    batch_size, queries_num, keys_num = atten_logits.shape
    # 转换形状为 (BATCH_SIZE * QUERIES_NUM,)
    valid_len = torch.repeat_interleave(valid_len, queries_num) if valid_len.dim() == 1 else valid_len.flatten()
    # 转换形状为 (BATCH_SIZE * QUERIES_NUM, KEYS_NUM)
    atten_logits = atten_logits.reshape(-1, keys_num)

    # 创建掩码，形状为 (BATCH_SIZE * QUERIES_NUM, KEYS_NUM)
    mask = torch.arange(0, keys_num, device=atten_logits.device).unsqueeze(dim=0) < valid_len.unsqueeze(dim=1)
    # 使用大负数（-1e6）为非目标位置掩码，使对应位置经 Softmax 后的权重趋近 0
    masked_atten_logits = torch.where(mask, atten_logits, torch.tensor(-1e6, device=atten_logits.device))

    return F.softmax(masked_atten_logits.reshape(batch_size, queries_num, keys_num), dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, hidden_num: int, dropout: float):
        super().__init__()
        self.hidden_layer = nn.Linear(query_dim + key_dim, hidden_num, bias=False)
        self.score_layer = nn.Linear(hidden_num, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.atten_weights: Optional[Tensor] = None  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor, valid_len: Tensor) -> Tensor:
        """
        :param keys: 键，形状为 (BATCH_SIZE, KEYS_NUM, KEYS_DIM)
        :param values: 值，形状为 (BATCH_SIZE, KEYS_NUM, VALUES_DIM)
        :param queries: 查询，形状为 (BATCH_SIZE, QUERIES_NUM, QUERIES_DIM)
        :param valid_len: 有效长度，形状为 (BATCH_SIZE,) 或 (BATCH_SIZE, QUERIES_NUM)

        :return: 加性注意力输出，形状为 (BATCH_SIZE, QUERIES_NUM, VALUES_DIM)
        """
        batch_size, keys_num, _ = keys.shape
        _, queries_num, _ = queries.shape

        keys = (keys.unsqueeze(1)  # (BATCH_SIZE, 1, KEYS_NUM, KEYS_DIM)
                .expand(batch_size, queries_num, keys_num, -1))  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, KEYS_DIM)
        queries = (queries.unsqueeze(2)  # (BATCH_SIZE, QUERIES_NUM, 1, QUERIES_DIM)
                   .expand(batch_size, queries_num, keys_num, -1))  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, QUERIES_DIM)
        concat_qk = torch.cat([queries, keys], dim=-1)  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, QUERIES_DIM+KEYS_DIM)

        hidden = self.hidden_layer(concat_qk).tanh()  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, num_hiddens)
        logits = (self.score_layer(hidden)  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM, 1)
                  .squeeze(-1))  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)

        self.atten_weights = masked_softmax(logits, valid_len)  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)

        return torch.bmm(self.dropout(self.atten_weights), values)  # (BATCH_SIZE, QUERIES_NUM, VALUES_DIM)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.atten_weights: Optional[Tensor] = None  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)

    def forward(self, keys: Tensor, values: Tensor, queries: Tensor, valid_len: Tensor) -> Tensor:
        """
        :param keys: 键，形状为 (BATCH_SIZE, KEYS_NUM, KEYS_DIM)
        :param values: 值，形状为 (BATCH_SIZE, KEYS_NUM, VALUES_DIM)
        :param queries: 查询，形状为 (BATCH_SIZE, QUERIES_NUM, QUERIES_DIM)，其中 QUERIES_DIM = KEYS_DIM
        :param valid_len: 有效长度，形状为 (BATCH_SIZE,) 或 (BATCH_SIZE, QUERIES_NUM)

        :return: 缩放点积注意力输出，形状为 (BATCH_SIZE, QUERIES_NUM, VALUES_DIM)
        """
        _, _, q_k_dim = keys.shape

        logits = torch.bmm(  # (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)
            queries,  # (BATCH_SIZE, QUERIES_NUM, q_k_dim)
            keys.transpose(1, 2)  # (BATCH_SIZE, KEYS_NUM, q_k_dim) -> (BATCH_SIZE, q_k_dim, KEYS_NUM)
        ) / (q_k_dim ** 0.5)

        self.atten_weights = masked_softmax(logits, valid_len)

        return torch.bmm(self.dropout(self.atten_weights), values)


if __name__ == '__main__':
    from utils import plot_attention_heatmap

    k = torch.ones(2, 10, 2)  # (2, 10, 2)
    v = torch.arange(40, dtype=torch.float).reshape(1, 10, 4).repeat(2, 1, 1)  # (2, 10, 4)
    q = torch.normal(mean=0, std=1, size=(2, 1, 2))  # (2, 1, 2)

    scaled_dot_product_attention = ScaledDotProductAttention(dropout=0.5)
    scaled_dot_product_attention.eval()
    result = scaled_dot_product_attention(k, v, q, valid_len=torch.tensor([2, 6]))
    weights = scaled_dot_product_attention.atten_weights

    print(f'{result}\n\n{result.shape  = }')
    print(f'{weights.shape = }')

    plot_attention_heatmap(weights.reshape((1, 1, 2, 10)), x_label='Keys', y_label='Queries', figsize=(4, 4))
