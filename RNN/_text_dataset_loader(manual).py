from typing import Iterator

from torch import Tensor, arange, randperm, tensor

from RNN.text_preprocessing import get_vocab_corpus_from_timemachine, Vocabulary


class LanguageModelDataGenerator:
    """为神经网络语言模型生成数据"""

    def __init__(self, corpus: list[int], seq_length: int, batch_size: int, shuffle: bool):
        """
        将语料库按指定序列长度生成具有特定批量大小的数据生成器。

        用 `samples_num` 表示特征-标签对的总数，即全部批量的样本总数。

        :param corpus: 经标记化的语料库列表
        :param seq_length: 生成的每个样本的序列长度
        :param batch_size: 每个批次中的样本数
        :param shuffle: 是否在迭代前打乱数据
        """
        self.corpus = corpus
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.samples_num = len(corpus) - seq_length

    def __len__(self):
        """数据生成器的批次数"""
        return self.samples_num // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = arange(self.samples_num)  # 全部样本索引
        if self.shuffle:
            indices = indices[randperm(self.samples_num)]

        for i in range(0, self.samples_num, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            features = [self.corpus[idx: idx + self.seq_length]
                        for idx in batch_indices]
            labels = [self.corpus[idx + 1: idx + self.seq_length + 1]
                      for idx in batch_indices]

            yield tensor(features), tensor(labels)


def timemachine_data_loader(
        batch_size: int, seq_length: int, shuffle=False, max_token_num=10_000
) -> tuple[LanguageModelDataGenerator, Vocabulary]:
    vocab, corpus = get_vocab_corpus_from_timemachine(token_type='char', max_token_num=max_token_num)
    vocab: Vocabulary
    corpus: list[int]

    data_iter = LanguageModelDataGenerator(corpus, seq_length, batch_size, shuffle)
    return data_iter, vocab


if __name__ == '__main__':
    data_iter, vocab = timemachine_data_loader(batch_size=5, seq_length=10)

    for f, l in data_iter:
        print(f'特征：{f.tolist()}')
        print(f'标签：{l.tolist()}')

        print(f'第 1 个特征解码：{"".join(vocab.decode(f[0].tolist()))!r}')
        break
    print(f'每个小批量中的样本数：{data_iter.batch_size}')
    print(f'批量总数：{len(data_iter)}')
