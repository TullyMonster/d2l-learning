from typing import Optional

import torch
from torch import Tensor, nn

from Attention.atten_scoring_func import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False,
                 kdim: Optional[int] = None, vdim: Optional[int] = None):
        """
        :param embed_dim: 模型的特征维度
        :param num_heads: 注意力头的数量（必须能整除 embed_dim）
        :param dropout: Dropout 概率
        :param bias: 是否在线性层中使用偏置项
        :param kdim: 键的维度
        :param vdim: 值的维度
        """
        super().__init__()

        if embed_dim <= 0 or num_heads <= 0: raise ValueError(f'{embed_dim=} 和 {num_heads=} 必须均大于 0')
        if embed_dim % num_heads != 0: raise ValueError(f'{embed_dim=} 必须能被 {num_heads=} 整除')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 查询的线性投影层
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)  # 键的线性投影层
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)  # 值的线性投影层
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出的线性投影层

        self.atten_score = ScaledDotProductAttention(dropout)  # 使用缩放点积注意力
        self.atten_weights: Optional[Tensor] = None  # 用于可视化的注意力权重

    def forward(self, query: Tensor, key: Tensor, value: Tensor, valid_len: Tensor) -> Tensor:
        """
        :param query: 查询张量，形状为 (BATCH_SIZE, QUERIES_NUM, embed_dim)
        :param key: 键张量，形状为 (BATCH_SIZE, KEYS_NUM, kdim or embed_dim)
        :param value: 值张量，形状为 (BATCH_SIZE, KEYS_NUM, vdim or embed_dim)
        :param valid_len: 有效长度，形状为 (BATCH_SIZE,) 或 (BATCH_SIZE, QUERIES_NUM)
        :return: 输出张量，形状为 (BATCH_SIZE, QUERIES_NUM, embed_dim)
        """
        self.atten_weights = None

        q_proj = self.q_proj(query)  # (BATCH_SIZE, QUERIES_NUM, embed_dim)
        k_proj = self.k_proj(key)  # (BATCH_SIZE, KEYS_NUM, embed_dim)
        v_proj = self.v_proj(value)  # (BATCH_SIZE, KEYS_NUM, embed_dim)

        k_heads = self.__transpose_qkv(k_proj)  # (BATCH_SIZE * num_heads, KEYS_NUM, head_dim)
        v_heads = self.__transpose_qkv(v_proj)  # (BATCH_SIZE * num_heads, KEYS_NUM, head_dim)
        q_heads = self.__transpose_qkv(q_proj)  # (BATCH_SIZE * num_heads, QUERIES_NUM, head_dim)

        # (BATCH_SIZE,) -> (BATCH_SIZE * num_heads,) 或 (BATCH_SIZE, QUERIES_NUM) -> (BATCH_SIZE * num_heads, QUERIES_NUM)
        valid_len = torch.repeat_interleave(valid_len, repeats=self.num_heads, dim=0)

        attn_output = self.atten_score(k_heads, v_heads, q_heads, valid_len)

        # (BATCH_SIZE * num_heads, QUERIES_NUM, head_dim) -> (BATCH_SIZE, QUERIES_NUM, embed_dim)
        output_concat = self.__transpose_o(attn_output)

        if not self.training:
            # (BATCH_SIZE, num_heads, QUERIES_NUM, KEYS_NUM)
            self.atten_weights = (self.atten_score.atten_weights  # (BATCH_SIZE * num_heads, QUERIES_NUM, KEYS_NUM)
                                  .reshape(query.shape[0], self.num_heads, query.shape[1], key.shape[1]))

        return self.o_proj(output_concat)  # (BATCH_SIZE, QUERIES_NUM, embed_dim)

    def __transpose_qkv(self, x: Tensor) -> Tensor:
        """
        :param x: (BATCH_SIZE, QKV_NUM, embed_dim)
        :return: (BATCH_SIZE * num_heads, QKV_NUM, head_dim)
        """
        batch_size, qkv_num, _ = x.shape
        x = (x.reshape(batch_size, qkv_num, self.num_heads, self.head_dim)  # (BATCH_SIZE, QKV_NUM, num_heads, head_dim)
             .permute(0, 2, 1, 3)  # (BATCH_SIZE, num_heads, QKV_NUM, head_dim)
             .reshape(-1, qkv_num, self.head_dim))  # (batch_size * num_heads, qkv_num, head_dim)
        return x

    def __transpose_o(self, x: Tensor) -> Tensor:
        """
        :param x: (BATCH_SIZE * num_heads, QKV_NUM, head_dim)
        :return: (BATCH_SIZE, QKV_NUM, embed_dim)
        """
        batch_times_heads, qkv_num, _ = x.shape
        batch_size = batch_times_heads // self.num_heads
        x = (x.reshape(batch_size, self.num_heads, qkv_num, self.head_dim)  # (BATCH_SIZE, num_heads, QKV_NUM, head_dim)
             .permute(0, 2, 1, 3)  # (BATCH_SIZE, QKV_NUM, num_heads, head_dim)
             .reshape(batch_size, qkv_num, self.embed_dim))  # (BATCH_SIZE, QKV_NUM, embed_dim)
        return x


if __name__ == '__main__':
    from Attention.utils import plot_attention_heatmap

    embed_dim = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    valid_lens = torch.tensor([4, 7])

    m_h_atten = MultiHeadAttention(embed_dim, num_heads, dropout=0.1)
    m_h_atten.eval()

    o = m_h_atten(q, k, v, valid_lens)

    print(f'输入形状: {list(q.shape)}')
    print(f'输出形状: {list(o.shape)}')
    print(f'注意力权重形状: {list(m_h_atten.atten_weights.shape)}')

    plot_attention_heatmap(m_h_atten.atten_weights, x_label='Keys', y_label='Queries',
                           titles=[f'Batch {i + 1}, Head {j + 1}' for i in range(batch_size) for j in range(num_heads)])
