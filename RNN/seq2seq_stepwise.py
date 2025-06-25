from typing import Tuple

import torch
from torch import nn, Tensor, optim

from encoder_decoder import AbstractDecoder, EncoderDecoder
from text_preprocessing import Vocabulary, ST


class Seq2SeqDecoderStepwise(AbstractDecoder[Tuple[Tensor, Tensor]]):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_num: int, num_layers: int, dropout: float = 0):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_num, hidden_num, num_layers, dropout=dropout)  # 输入维度需要增加 HIDDEN_NUM
        self.output_layer = nn.Linear(hidden_num, vocab_size)

    def init_state(self, enc_output: Tuple[Tensor, Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        """
        从编码器输出中返回上下文向量，作为解码器的初始隐状态

        :param enc_output: 编码器输出
        :return: (enc_hidden, rnn_hidden) 元组
        """
        enc_hidden = enc_output[1]  # 固定的上下文向量，形状为：(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
        rnn_hidden = enc_hidden.clone()  # RNN 的起始隐状态，将随解码时间步更新
        return enc_hidden, rnn_hidden

    def forward(self, input_seq: Tensor, state: Tuple[Tensor, Tensor]
                ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, Tensor]]:
        """
        逐步的序列解码

        输入目标序列的单个词元和解码器隐状态元组，输出对目标序列中下一个词元概率分布的预测

        :param input_seq: 输入序列张量，由词元索引组成，形状为：(BATCH_SIZE, 1)
        :param state: 解码器的隐状态元组，包含 (enc_hidden, rnn_hidden)
                     - enc_hidden: 固定的上下文向量，形状为：(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
                     - rnn_hidden: 可更新的 RNN 隐状态，形状为：(NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
        :return: 解码器输出和更新后的隐状态元组 (output, (enc_hidden, new_rnn_hidden))
        """
        if input_seq.dim() != 2: raise ValueError(f'input_seq 应为二维张量！')
        if input_seq.dtype != torch.long: input_seq = input_seq.long()

        enc_hidden, rnn_hidden = state
        # (BATCH_SIZE, 1) -> (BATCH_SIZE, 1, EMBED_DIM) -> (1, BATCH_SIZE, EMBED_DIM)
        embedded = self.embedding_layer(input_seq).permute(1, 0, 2).contiguous()

        # (NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM) -> (1, BATCH_SIZE, HIDDEN_NUM) -> (BATCH_SIZE, HIDDEN_NUM)
        context = enc_hidden[-1:].expand(embedded.size(0), -1, -1)
        rnn_input = torch.cat([embedded, context], dim=2)

        output, new_rnn_hidden = self.rnn(rnn_input, rnn_hidden)
        output = self.output_layer(output)

        return (output,), (enc_hidden, new_rnn_hidden)  # 把 enc_hidden 原样返回


def forecast_greedy_search_stepwise(
        module: EncoderDecoder,
        src_sentence: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        device: torch.device,
        max_length: int = 20,
        record_attn_weights: bool = False
) -> tuple[str, Tensor | None]:
    """
    以贪心搜索的方式实现序列预测（生成），逐步预测

    :param module: 序列到序列模型
    :param src_sentence: 源语言句子
    :param src_vocab: 源语言词表
    :param tgt_vocab: 目标语言词表
    :param device: 计算设备
    :param max_length: 生成序列的最大长度
    :param record_attn_weights: 是否保存注意力权重

    :return: 生成的目标语言句子，必要时返回注意力权重
    """

    # 获取特殊词元索引
    pad_src_index: int = src_vocab.get_index(ST.PAD)
    sos_tgt_index: int = tgt_vocab.get_index(ST.SOS)
    eos_tgt_index: int = tgt_vocab.get_index(ST.EOS)

    # 输入预处理
    src_tokens: list[int] = src_vocab.encode([*src_sentence.lower().split(), ST.EOS])
    src_tokens_pad_trunc: list[int] = [
        *src_tokens[:max_length],
        *[pad_src_index] * (max_length - len(src_tokens))
    ]

    output_tokens: list[int] = []
    attn_weights: list[Tensor] = []

    module.eval()
    with torch.no_grad():
        src_input = torch.tensor([src_tokens_pad_trunc], dtype=torch.long, device=device)  # (BATCH_SIZE=1, SEQ_LENGTH)
        src_valid_length = torch.tensor([len(src_tokens)], device=device)

        enc_outputs = module.encoder(src_input, valid_lengths=src_valid_length)
        dec_state = module.decoder.init_state(enc_outputs, valid_lengths=src_valid_length)

        last_token = torch.tensor([[sos_tgt_index]], dtype=torch.long, device=device)  # (BATCH_SIZE=1, SEQ_LENGTH=1)

        for _ in range(max_length):
            dec_output, dec_state = module.decoder(last_token, dec_state)  # 仅输入最新词元
            next_token = dec_output[0][-1].argmax(dim=-1).item()

            if next_token == eos_tgt_index: break
            if record_attn_weights and len(dec_output) > 1: attn_weights.append(dec_output[1].squeeze(0))

            output_tokens.append(next_token)
            last_token = torch.tensor([[next_token]], dtype=torch.long, device=device)  # 准备下一轮输入

    tgt_sentence = ' '.join(tgt_vocab.decode(output_tokens))
    stack_attn_weights = torch.stack(attn_weights) if attn_weights else None
    return tgt_sentence, stack_attn_weights


if __name__ == '__main__':
    from translation_dataset_loader import nmt_eng_fra_dataloader
    from seq2seq import TestSentence, Seq2SeqEncoder, SequenceLengthCrossEntropyLoss, train_one_epoch, evaluate_bleu

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
                               decoder=Seq2SeqDecoderStepwise(len(fra_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS,
                                                              DROPOUT),
                               device=device)  # 使用默认的模型参数初始化方法，不手动初始化
    optimizer = optim.Adam(nmt_model.parameters(), lr=LEARNING_RATE)
    criterion = SequenceLengthCrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        loss, speed = train_one_epoch(nmt_model, data_iter, optimizer, criterion, fra_vocab, device)
        print(f'第 {epoch + 1:03} 轮：损失为 {loss:.3f}，速度为 {speed:.1f} tokens/sec')

        if (epoch + 1) % TEST_INTERVAL == 0:
            for eng, fra in zip(TEST_SENTENCES.src, TEST_SENTENCES.tgt):
                forecast_fra, _ = forecast_greedy_search_stepwise(nmt_model, eng, eng_vocab, fra_vocab, device)
                print(f'INFO: '
                      f'{eng.ljust(max(map(len, TEST_SENTENCES.src)))} '
                      f'→ (BLEU={evaluate_bleu(forecast_fra, fra, max_n_gram=3):.2f}) {forecast_fra}')
