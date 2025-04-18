from typing import Tuple, Optional, Iterable

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoder_decoder import AbstractEncoder, AbstractDecoder


class Seq2SeqEncoder(AbstractEncoder[Tuple[Tensor, Tensor]]):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_num: int, num_layers: int, dropout: float = 0):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_num, num_layers, dropout=dropout)

    def forward(self, input_seq: Tensor, valid_lengths: Optional[Tensor] = None, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        将输入序列编码为中间表示

        :param input_seq: 输入序列张量，由词元索引组成，形状为：(BATCH_SIZE, SEQ_LENGTH)
        :param valid_lengths: 各序列的有效长度，形状为：(BATCH_SIZE,)。None 表示所有序列的有效长度相同
        :return: 编码器输出和最终的隐状态元组，形状为：((SEQ_LENGTH, BATCH_SIZE, HIDDEN_NUM), (NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM))
        """

        if input_seq.dim() != 2: raise ValueError(f'input_seq 应为二维张量！')
        if input_seq.dtype != torch.long: input_seq = input_seq.long()  # 将词元索引转换为 LongTensor（或 IntTensor） 用于 nn.Embedding

        # (BATCH_SIZE, SEQ_LENGTH) -> (BATCH_SIZE, SEQ_LENGTH, EMBED_DIM) -> (SEQ_LENGTH, BATCH_SIZE, EMBED_DIM)
        embedded = self.embedding_layer(input_seq).permute(1, 0, 2).contiguous()  # 将词元索引张量词嵌入后，重排维度，并保证内存连续

        if valid_lengths is None:
            output, state = self.rnn(embedded)  # 未显式地提供初始隐状态，PyTorch 将自动创建全零张量
        else:
            packed = pack_padded_sequence(  # 序列打包，“压缩”为无填充的紧密格式
                input=embedded,
                lengths=valid_lengths.cpu(),  # 确保 valid_lengths 在 CPU 上
                enforce_sorted=False
            )
            output, state = self.rnn(packed)  # 更高效的 RNN 处理
            output, _ = pad_packed_sequence(output)  # 序列解包，转换为填充格式

        return output, state


class Seq2SeqDecoder(AbstractDecoder[Tensor]):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_num: int, num_layers: int, dropout: float = 0):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_num, hidden_num, num_layers, dropout=dropout)  # 输入维度需要增加hidden_num
        self.output_layer = nn.Linear(hidden_num, vocab_size)
        self.hidden_num = hidden_num

    def init_state(self, enc_output: Tuple[Tensor, Tensor], **kwargs) -> Tensor:
        """
        从编码器输出中返回上下文向量，作为解码器的初始隐状态

        :param enc_output: 编码器输出
        :return: 解码器的初始隐状态，形状为：(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
        """
        return enc_output[1]  # 编码器的完整隐状态

    def forward(self, input_seq: Tensor, state: Tensor) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """
        执行序列解码

        :param input_seq: 输入序列张量，由词元索引组成，形状为：(BATCH_SIZE, SEQ_LENGTH)
        :param state: 解码器的隐状态，形状为：(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
        :return: 解码器输出和更新后的隐状态元组
        """
        if input_seq.dim() != 2: raise ValueError(f'input_seq 应为二维张量！')
        if input_seq.dtype != torch.long: input_seq = input_seq.long()

        # (BATCH_SIZE, SEQ_LENGTH) -> (BATCH_SIZE, SEQ_LENGTH, EMBED_DIM) -> (SEQ_LENGTH, BATCH_SIZE, EMBED_DIM)
        embedded = self.embedding_layer(input_seq).permute(1, 0, 2).contiguous()  # 将词元索引张量词嵌入后，重排维度，并保证内存连续

        # (NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM) -> (1, BATCH_SIZE, HIDDEN_NUM) -> (SEQ_LENGTH, BATCH_SIZE, HIDDEN_NUM)
        context = state[-1:].expand(embedded.shape[0], -1, -1)  # 使上下文向量与词嵌入向量形状匹配
        rnn_input = torch.cat([embedded, context], dim=2)  # (SEQ_LENGTH, BATCH_SIZE, EMBED_DIM + HIDDEN_NUM)

        output, state = self.rnn(rnn_input, state)  # RNN 前向传播
        output = self.output_layer(output)  # 将 RNN 输出映射到词表空间

        return (output,), state


class SequenceLengthCrossEntropyLoss(nn.Module):
    """基于序列有效长度的交叉熵损失函数，用于忽略序列填充部分的损失计算"""

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, label_smoothing: float = 0.0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight,
                                                 size_average=size_average,
                                                 ignore_index=-100,  # 使用 PyTorch 默认值
                                                 reduce=reduce,
                                                 reduction='none',  # 设置为 'none' 以便后续手动应用掩码
                                                 label_smoothing=label_smoothing)

    def forward(self, inputs: Tensor, targets: Tensor, valid_lengths: Tensor) -> Tensor:
        """基于序列有效长度计算交叉熵损失

        在序列预测任务下，nn.CrossEntropyLoss 的预测值形状为：(BATCH_SIZE, VOCAB_SIZE, SEQ_LENGTH)
                                                目标值形状为：(BATCH_SIZE, SEQ_LENGTH)
                                                reduction='none' 时的各样本损失值的形状与目标值的一致

        :param inputs: 模型预测的输出，形状为：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        :param targets: 目标标签，形状为：(BATCH_SIZE, SEQ_LENGTH)
        :param valid_lengths: 各序列的有效长度，形状为：(BATCH_SIZE,)
        :return 掩码后损失的平均值
        """
        inputs = inputs.permute(1, 2, 0)  # (SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE) -> (BATCH_SIZE, VOCAB_SIZE, SEQ_LENGTH)

        seq_length = targets.shape[1]
        mask = torch.arange(seq_length, device=targets.device).unsqueeze(0) < valid_lengths.unsqueeze(1)

        losses = self.cross_entropy(inputs, targets)  # 计算交叉熵损失，形状为：(BATCH_SIZE, SEQ_LENGTH)
        masked_mean_losses = (losses * mask.float()).mean(dim=1)

        return masked_mean_losses


class MultiIgnoreIndicesCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self,
                 ignore_indices: Iterable,
                 weight: Optional[Tensor] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super().__init__(
            weight=weight,
            ignore_index=-100,  # 使用 PyTorch 默认值
            size_average=size_average,
            reduce=reduce,
            reduction='none',  # 设置对多个样本的损失值聚合的方式为 'none'，以便应用掩码
            label_smoothing=label_smoothing
        )
        self.ignore_indices = set(ignore_indices)
        self.reduction = reduction

    def forward(self, inputs, targets):
        mask = torch.ones_like(targets, dtype=torch.bool)  # 初始化掩码张量（全为 True）
        for idx in self.ignore_indices:
            mask = mask & (targets != idx)

        losses = super().forward(inputs, targets)  # 首先计算每个位置的损失
        masked_losses = losses * mask.float()  # 掩码后的损失值

        if self.reduction == 'sum':
            return masked_losses.sum()
        elif self.reduction == 'mean':
            return masked_losses.sum() / mask.sum().float().clamp(min=1.0)  # 防止极端情况下的除零错误
        else:
            return masked_losses  # 'none'
