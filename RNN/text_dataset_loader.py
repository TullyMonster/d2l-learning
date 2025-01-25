import torch
from torch.utils.data import Dataset, DataLoader

from RNN.text_preprocessing import get_vocab_corpus_from_timemachine, Vocabulary


class TextDataset(Dataset):
    def __init__(self, corpus: list[int], seq_length: int):
        self.corpus = corpus
        self.seq_length = seq_length

    def __len__(self):
        return len(self.corpus) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.corpus[idx: idx + self.seq_length]),
                torch.tensor(self.corpus[idx + 1: idx + self.seq_length + 1]))


def timemachine_data_loader(
        batch_size: int, seq_length: int, shuffle=False, max_token_num=10_000
) -> tuple[DataLoader, Vocabulary]:
    vocab, corpus = get_vocab_corpus_from_timemachine(token_type='char', max_token_num=max_token_num)
    vocab: Vocabulary
    corpus: list[int]

    data_iter = DataLoader(TextDataset(corpus, seq_length), batch_size, shuffle)
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
