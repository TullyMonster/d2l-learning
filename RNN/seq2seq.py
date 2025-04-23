from collections import Counter
from dataclasses import dataclass
from time import time
from typing import Tuple, Optional, Iterable

import math
import torch
from torch import nn, Tensor, optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoder_decoder import AbstractEncoder, AbstractDecoder, EncoderDecoder
from text_preprocessing import Vocabulary, ST


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


def train_one_epoch(
        module: EncoderDecoder,
        data_iter: Iterable[Tuple[Tensor, ...]],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        tgt_vocab: Vocabulary,
        device: torch.device
) -> Tuple[float, float]:
    """
    一个迭代周期内 Seq2Seq 模型的训练

    :param module: 序列到序列模型
    :param data_iter: 数据集加载器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param tgt_vocab: 目标语言词表
    :param device: 计算设备
    :return: 平均损失, 训练速度 (tokens/sec)
    """
    sos_idx = tgt_vocab.get_index(ST.SOS)  # 目标语言词表中，序列开始标记词元的索引值
    total_loss = 0.0
    total_tokens = 0

    module.train()  # 每个迭代周期开始前，将模型设置为训练模式
    start_time = time()
    for src, src_valid_len, tgt, tgt_valid_len in data_iter:
        optimizer.zero_grad()  # 每个批次处理前，清除上一次迭代累积的梯度

        src = src.to(device)
        tgt = tgt.to(device)  # 形状为：(BATCH_SIZE, SEQ_LENGTH)
        # src_valid_len 将用于 pack_padded_sequence，不必在 device 上生成副本
        tgt_valid_len = tgt_valid_len.to(device)

        dec_input = torch.cat([  # 以强制教学的方式输入解码器
            torch.full((tgt.shape[0], 1), sos_idx, device=device),  # 形状为 (BATCH_SIZE, 1)、由 sos_idx 填充的张量
            tgt[..., :-1]  # (BATCH_SIZE, SEQ_LENGTH) -> (BATCH_SIZE, SEQ_LENGTH - 1)
        ], dim=1)

        tgt_pred = module(src, dec_input, valid_lengths=src_valid_len)  # 前向传播
        loss = criterion(inputs=tgt_pred[0], targets=tgt, valid_lengths=tgt_valid_len)  # 计算损失

        loss.sum().backward()  # 反向传播
        clip_grad_norm_(module.parameters(), max_norm=2)  # 梯度裁剪
        optimizer.step()  # 更新参数

        num_tokens = tgt_valid_len.sum().item()
        total_loss += loss.sum().item()
        total_tokens += num_tokens

    # 计算平均损失和训练速度
    avg_loss = total_loss / total_tokens
    tokens_per_sec = total_tokens / (time() - start_time)

    return avg_loss, tokens_per_sec


def forecast_autoregressive(
        module: EncoderDecoder,
        src_sentence: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        device: torch.device,
        max_length: int = 20,
        record_attn_weights: bool = False
) -> tuple[str, Tensor]:
    """
    以自回归方式实现序列预测（生成）

    :param module: 序列到序列模型
    :param src_sentence: 源语言句子
    :param src_vocab: 源语言词表
    :param tgt_vocab: 目标语言词表
    :param max_length: 生成序列的最大长度
    :param device: 计算设备
    :param record_attn_weights: 是否保存注意力权重

    :return: 生成的目标语言句子，必要时返回注意力权重
    """
    # 获取特殊词元索引
    pad_src_index: int = src_vocab.get_index(ST.PAD)
    sos_tgt_index: int = tgt_vocab.get_index(ST.SOS)
    eos_tgt_index: int = tgt_vocab.get_index(ST.EOS)

    # 输入预处理
    src_tokens: list[int] = src_vocab.encode([*src_sentence.lower().split(), ST.EOS])  # 大小写转换、分词、添加 EOS 词元
    src_tokens_pad_trunc: list[int] = [  # 截断或填充到指定长度
        *src_tokens[:max_length],
        *[pad_src_index] * (max_length - len(src_tokens))
    ]

    # 组织逐时间步的输出
    output_tokens: list[int] = []
    attn_weights: list[Tensor] = []

    module.eval()
    with torch.no_grad():
        src_input = torch.tensor([src_tokens_pad_trunc], dtype=torch.long, device=device)  # (BATCH_SIZE=1, SEQ_LENGTH)
        dec_input = torch.tensor(data=[[sos_tgt_index]], dtype=torch.long, device=device)  # (BATCH_SIZE=1, 1)
        src_valid_length = torch.tensor([len(src_tokens)], device=device)  # (BATCH_SIZE=1,)

        for _ in range(max_length):  # 执行预测
            output: tuple[Tensor, ...] = module(src_input, dec_input, valid_lengths=src_valid_length)
            next_token = output[0][-1].argmax(dim=-1).item()  # 最后一个时间步的预测结果（贪心搜索：每一步都选择概率最高的词元）

            if next_token == eos_tgt_index: break
            if record_attn_weights: attn_weights.append(output[1].squeeze(0))

            output_tokens.append(next_token)
            dec_input = torch.cat(tensors=[dec_input, torch.tensor([[next_token]], device=device)],
                                  dim=1)  # (BATCH_SIZE, 1) -> (BATCH_SIZE, 2) -> ...

    tgt_sentence = ' '.join(tgt_vocab.decode(output_tokens))
    stack_attn_weights = torch.stack(attn_weights) if record_attn_weights else torch.tensor([])
    return tgt_sentence, stack_attn_weights


def evaluate_bleu(
        candi_str: str,
        refer_strs: list[str],
        max_n_gram: int = 4,
        weights: Optional[list[float]] = None
) -> float:
    """
    计算候选句子与参考句子之间的 BLEU 分数（基于空格分词）

    :param candi_str: 候选翻译/生成的文本
    :param refer_strs: 一个或多个参考翻译/标准文本
    :param max_n_gram: 最大 n-gram
    :param weights: n-gram 权重列表，默认为均匀权重 [0.25, 0.25, 0.25, 0.25]

    :return: BLEU 得分，范围从 0~1
    """
    weights = [1.0 / max_n_gram] * max_n_gram if weights is None else weights  # 设置默认权重

    if len(weights) != max_n_gram:  # 确保权重列表与最大 n-gram 匹配
        raise ValueError(f'n-gram 权重列表 weights 的长度必须等于 {max_n_gram=}，当前为{len(weights)}')

    candi_tokens: list[str] = candi_str.strip().split()  # 候选文本分词
    candi_len = len(candi_tokens)  # 候选文本长度
    refer_tokens_list: list[list[str]] = [ref.strip().split() for ref in refer_strs]  # 参考文本列表分词
    refer_len = min([len(ref_tokens) for ref_tokens in refer_tokens_list], key=lambda x: abs(x - candi_len))  # 参考文本长度

    # 计算短文本惩罚因子 (Brevity Penalty)
    if candi_len == 0:
        return 0.0  # 避免除以零错误
    bp = 1.0 if candi_len > refer_len else math.exp(1 - refer_len / candi_len)

    precisions = []  # 各阶 n-gram 精度列表
    for n in range(1, max_n_gram + 1):
        candi_ngrams = [' '.join(candi_tokens[i:i + n]) for i in range(len(candi_tokens) - n + 1)]  # 从候选文本中提取 n-gram
        if len(candi_ngrams) == 0:  # 如果没有 n-gram，则精度为小值 1e-10，避免 log(0) 问题
            precisions.append(1e-10)
            continue

        candi_counter = Counter(candi_ngrams)  # 统计候选文本中每个 n-gram 的出现次数
        max_matches = Counter()  # 计算匹配的 n-gram 数量 (对每个参考文本统计，然后取最大值)
        for ref_tokens in refer_tokens_list:
            ref_ngrams = [' '.join(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)]  # 从参考文本中提取 n-gram
            ref_counter = Counter(ref_ngrams)

            # 对每个 n-gram，取候选文本和参考文本中出现次数的最小值
            # 然后与当前最大匹配次数比较
            for ngram, count in candi_counter.items():
                max_matches[ngram] = max(max_matches[ngram], min(count, ref_counter[ngram]))

        matches = sum(max_matches.values())  # 计算匹配的 n-gram 总数
        precision = matches / sum(candi_counter.values()) if sum(candi_counter.values()) > 0 else 1e-10  # 计算精度
        precision = max(precision, 1e-10)

        precisions.append(precision)

    log_avg = sum(w * math.log(p) for w, p in zip(weights, precisions))  # 加权几何平均
    bleu = bp * math.exp(log_avg)

    return bleu


@dataclass
class TestSentence:
    """用于测试的源语言与目标语言句子对"""
    src: list[str]
    tgt: list[list[str]]


if __name__ == '__main__':
    from translation_dataset_loader import nmt_eng_fra_dataloader

    BATCH_SIZE = 128
    SEQ_LENGTH = 20
    EMBED_DIM = 256
    HIDDEN_NUM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.0005
    EPOCHS_NUM = 50
    TEST_INTERVAL = 1
    TEST_SENTENCES = TestSentence(src=["I like apples .",
                                       "She reads books regularly .",
                                       "They play soccer together .",
                                       "We studied French yesterday .",
                                       "The weather is beautiful today ."],
                                  tgt=[["J'aime les pommes .", "J'adore les pommes .", "Les pommes me plaisent .",
                                        "Je raffole des pommes .", "J'apprécie les pommes ."],
                                       ["Elle lit des livres régulièrement .", "Elle lit des livres souvent .",
                                        "Elle lit des livres fréquemment .", "Elle lit régulièrement des ouvrages ."],
                                       ["Ils jouent au football ensemble .", "Ils jouent au foot ensemble .",
                                        "Ils pratiquent le football ensemble .", "Ensemble, ils jouent au football ."],
                                       ["Nous avons étudié le français hier .", "Hier, nous avons étudié le français .",
                                        "Nous avons appris le français hier .", "Nous avons fait du français hier ."],
                                       ["Le temps est magnifique aujourd'hui .", "Il fait beau aujourd'hui .",
                                        "Le temps est splendide aujourd'hui .", "La météo est belle aujourd'hui ."]])

    data_iter, eng_vocab, fra_vocab = nmt_eng_fra_dataloader(BATCH_SIZE, SEQ_LENGTH, num_workers=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nmt_model = EncoderDecoder(encoder=Seq2SeqEncoder(len(eng_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT),
                               decoder=Seq2SeqDecoder(len(fra_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT),
                               device=device)  # 使用默认的模型参数初始化方法，不手动初始化
    optimizer = optim.Adam(nmt_model.parameters(), lr=LEARNING_RATE)
    criterion = SequenceLengthCrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        loss, speed = train_one_epoch(nmt_model, data_iter, optimizer, criterion, fra_vocab, device)
        print(f'第 {epoch + 1:03} 轮：损失为 {loss:.3f}，速度为 {speed:.1f} tokens/sec')

        if (epoch + 1) % TEST_INTERVAL == 0:
            for eng, fra in zip(TEST_SENTENCES.src, TEST_SENTENCES.tgt):
                forecast_fra, _ = forecast_autoregressive(nmt_model, eng, eng_vocab, fra_vocab, device)
                print(f'INFO: '
                      f'{eng.ljust(max(map(len, TEST_SENTENCES.src)))} '
                      f'→ (BLEU={evaluate_bleu(forecast_fra, fra, max_n_gram=3):.2f}) {forecast_fra}')
