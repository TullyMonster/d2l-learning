from typing import Tuple, List

import torch
from torch import nn, Tensor

from RNN.encoder_decoder import AbstractDecoder
from atten_scoring_func import AdditiveAttention


class Seq2SeqAttentionDecoder(AbstractDecoder):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_num: int, num_layers: int, dropout: float = 0):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_num, hidden_num, num_layers, dropout=dropout)
        self.output_layer = nn.Linear(hidden_num, vocab_size)

        self.atten_score = AdditiveAttention(hidden_num, hidden_num, hidden_num, dropout)
        self.__atten_weights: List[Tensor] = []  # 各元素即解码时间步上的注意力权重，形状为 (BATCH_SIZE, QUERIES_NUM=1, KEYS_NUM)

    def init_state(self, enc_output: Tuple[Tensor, Tensor], **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        根据编码器输出和最终的隐状态元组，用于带注意力的解码器初始化

        :param enc_output: 编码器输出
        :param kwargs: 额外参数，必须包含表示源序列有效长度的 'valid_lengths'
        :return: 三元元组：(编码器输出, 隐状态, 有效长度)
        """
        if 'valid_lengths' not in kwargs: raise ValueError("缺少 'valid_lengths' 参数")

        outputs, enc_state = enc_output
        valid_lengths = kwargs.get('valid_lengths')

        return (outputs.permute(1, 0, 2),  # (BATCH_SIZE，SEQ_LENGTH，HIDDEN_NUM) -> (SEQ_LENGTH, BATCH_SIZE, HIDDEN_NUM)
                enc_state,  # (NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
                valid_lengths)  # (BATCH_SIZE,)

    def forward(self, input_seq: Tensor, state: Tuple[Tensor, Tensor, Tensor]):
        """
        使用 Bahdanau 注意力机制的序列解码

        :param input_seq: 输入序列张量，由词元索引组成，形状为：(BATCH_SIZE, SEQ_LENGTH)
        :param state: 解码器状态：(编码器输出, 隐状态, 有效长度)
        :return: 二元元组：((解码器输出,), 新的解码器状态)
        """
        self.__atten_weights.clear()  # 清空注意力权重列表
        (enc_outputs,  # (SEQ_LENGTH, BATCH_SIZE, HIDDEN_NUM)
         dec_state,  # (NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM)
         valid_lengths  # (BATCH_SIZE,)
         ) = state
        # (BATCH_SIZE, SEQ_LENGTH) -> (BATCH_SIZE, SEQ_LENGTH, EMBED_DIM) -> (SEQ_LENGTH, BATCH_SIZE, EMBED_DIM)
        embedded = self.embedding_layer(input_seq).permute(1, 0, 2)
        outputs = []

        for step_embed in embedded:
            step_embed = step_embed.unsqueeze(dim=1)  # (BATCH_SIZE, EMBED_DIM) -> (BATCH_SIZE, 1, EMBED_DIM)
            atten_context = self.atten_score(  # (BATCH_SIZE, QUERIES_NUM=1, VALUES_DIM=HIDDEN_NUM)
                keys=enc_outputs,
                values=enc_outputs,
                # (NUM_LAYERS, BATCH_SIZE, HIDDEN_NUM) -> (BATCH_SIZE, HIDDEN_NUM) -> (BATCH_SIZE, QUERIES_NUM=1, VALUES_DIM=HIDDEN_NUM)
                queries=dec_state[-1].unsqueeze(dim=1),
                valid_len=valid_lengths
            )
            rnn_input = (
                torch.cat([step_embed, atten_context], dim=2)  # (BATCH_SIZE, SEQ_LENGTH=1, EMBED_DIM + HIDDEN_NUM)
                .permute(1, 0, 2)  # (SEQ_LENGTH=1, BATCH_SIZE, EMBED_DIM + HIDDEN_NUM)
            )
            output, dec_state = self.rnn(rnn_input, dec_state)

            outputs.append(output)  # (SEQ_LENGTH=1, BATCH_SIZE, HIDDEN_NUM)
            self.__atten_weights.append(self.atten_score.atten_weights)  # (BATCH_SIZE, QUERIES_NUM=1, KEYS_NUM)

        logits = self.output_layer(torch.cat(outputs))  # (SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)

        return (logits,), (enc_outputs, dec_state, valid_lengths)

    @property
    def atten_weights(self) -> Tensor:
        """
        获取并合并所有解码时间步的注意力权重

        :return: 注意力权重张量，形状为 (BATCH_SIZE, QUERIES_NUM, KEYS_NUM)
        """
        if not self.__atten_weights: raise RuntimeError('请先执行前向传播')
        return torch.cat(self.__atten_weights, dim=1)


if __name__ == '__main__':
    from torch import optim
    from utils import plot_attention_heatmap

    from RNN.encoder_decoder import EncoderDecoder
    from RNN.seq2seq import Seq2SeqEncoder, SequenceLengthCrossEntropyLoss, TestSentence
    from RNN.seq2seq import evaluate_bleu, forecast_greedy_search, train_one_epoch
    from RNN.translation_dataset_loader import nmt_eng_fra_dataloader

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

    encoder = Seq2SeqEncoder(len(eng_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT)
    decoder = Seq2SeqAttentionDecoder(len(fra_vocab), EMBED_DIM, HIDDEN_NUM, NUM_LAYERS, DROPOUT)
    nmt_model = EncoderDecoder(encoder, decoder, device=device)  # 使用默认的模型参数初始化方法，不手动初始化
    optimizer = optim.Adam(nmt_model.parameters(), lr=LEARNING_RATE)
    criterion = SequenceLengthCrossEntropyLoss()

    for epoch, _ in enumerate(range(EPOCHS_NUM), start=1):
        loss, speed = train_one_epoch(nmt_model, data_iter, optimizer, criterion, fra_vocab, device)
        print(f'第 {epoch:03} 轮：损失为 {loss:.3f}，速度为 {speed:.1f} tokens/sec')

        if epoch % TEST_INTERVAL == 0:
            for eng, fra in zip(TEST_SENTENCES.src, TEST_SENTENCES.tgt):
                forecast_fra, _ = forecast_greedy_search(nmt_model, eng, eng_vocab, fra_vocab, device)
                print(f'INFO: '
                      f'{eng.ljust(max(map(len, TEST_SENTENCES.src)))} '
                      f'→ (BLEU={evaluate_bleu(forecast_fra, fra, max_n_gram=3):.2f}) {forecast_fra}')

                if epoch == EPOCHS_NUM:
                    plot_attention_heatmap(decoder.atten_weights.reshape(1, 1, *decoder.atten_weights.shape[1:]),
                                           x_label='Eng', y_label='Fra', titles='Bahdanau Attention Weight')
