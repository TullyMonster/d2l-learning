import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.0, max_len: int = 1000):
        """
        :param embed_dim: 模型的特征维度，必须为偶数
        :param dropout: Dropout 概率
        :param max_len: 支持的最大序列长度
        :note: 生成位置编码 pe 张量的形状为 (1, max_len, embed_dim)
        """
        assert embed_dim % 2 == 0, f'模型的特征维度 ({embed_dim}) 应为偶数'

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('log_10000', torch.log(torch.tensor(10000)))

        indices_even = torch.arange(start=0, end=embed_dim, step=2)  # 偶数维度索引（2i），形状为 (embed_dim/2,)
        indices_odd = torch.arange(start=1, end=embed_dim, step=2)  # 奇数维度索引（2i + 1），形状为 (embed_dim/2,)
        freq_scale = torch.exp(indices_even.float() * (-self.log_10000 / embed_dim))  # (embed_dim/2,)
        position = torch.arange(start=0, end=max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        pe = torch.zeros(max_len, embed_dim)  # 位置编码矩阵，形状为 (max_len, embed_dim)
        pe[:, indices_even] = torch.sin(freq_scale * position)  # (max_len, embed_dim/2)
        pe[:, indices_odd] = torch.cos(freq_scale * position)  # (max_len, embed_dim/2)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)  # 将 pe 注册为不参与训练更新的、固定的非可学习参数

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: 输入张量，形状为 (BATCH_SIZE, SEQ_LENGTH, embed_dim)
        :return: 执行位置编码后的张量，形状为 (BATCH_SIZE, SEQ_LENGTH, embed_dim)
        """
        _, max_len, _ = x.shape
        x = x + self.pe[:, :max_len, :]
        x = self.dropout(x)
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from Attention.multihead_atten import MultiHeadAttention

    EMBED_DIM = 100
    NUM_HEADS = 5
    BATCH_SIZE = 2
    SEQ_LEN = 4

    m_h_atten = MultiHeadAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=0.5)
    m_h_atten.eval()

    q_k_v = torch.ones(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    valid_lens = torch.tensor([3, 2])

    output = m_h_atten(q_k_v, q_k_v, q_k_v, valid_lens)

    print(m_h_atten)
    print(f'q_k_v.shape  = {list(q_k_v.shape)}')
    print(f'output.shape = {list(output.shape)}')

    EMBED_DIM = 26
    SEQ_LEN = 26

    positional_encoding = PositionalEncoding(embed_dim=EMBED_DIM)
    positional_encoding.eval()

    _ = positional_encoding(torch.zeros(1, SEQ_LEN, EMBED_DIM))
    PE = positional_encoding.pe[0, :SEQ_LEN, :]  # (SEQ_LEN, EMBED_DIM)

    plt.figure(figsize=(5, 5))
    plt.imshow(PE.T, aspect='equal', cmap='Blues', origin='lower')
    plt.colorbar(fraction=0.04)
    plt.xlabel('Position')
    plt.ylabel('Embedding Dimension')
    plt.title('Positional Encoding')
    plt.tight_layout()
    plt.show()
