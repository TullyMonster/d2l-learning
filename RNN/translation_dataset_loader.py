import os
from typing import Optional, Literal

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from RNN.text_preprocessing import tokenize, Vocabulary, ST

UnicodeNormalForm = Optional[Literal['NFC', 'NFKC', 'NFD', 'NFKD']]


def process_eng_fra_dataset(normal_unicode: UnicodeNormalForm = None, lowercase=True,
                            max_sample_pair_num: Optional[int] = None, ) -> tuple[list[str], list[str]]:
    dataset = 'eng_fra.txt'
    encoding = 'UTF-8'
    source_tab_target: list[str] = []

    if not os.path.exists(dataset):
        import io, re, requests, zipfile, unicodedata
        temp_dataset = 'fra.txt'
        url = r'https://www.manythings.org/anki/fra-eng.zip'

        print(f'下载{url!r}……')
        response = requests.get(url, headers={'User-Agent': '...'}, timeout=30)

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extract(temp_dataset)

        with (open(temp_dataset, 'r', encoding=encoding) as temp_dataset_file):
            lines = temp_dataset_file.readlines()
            for line in lines:
                if lowercase: line = line.lower()
                if normal_unicode: line = unicodedata.normalize(normal_unicode, line)

                line = re.sub(r'([!.,;?"])', r' \1 ', line)
                line = line.replace('\u00A0', ' ') \
                    .replace('\u202F', ' ') \
                    .replace('’', "'") \
                    .replace(' \t', '\t')
                line = re.sub(r' +', r' ', line)

                eng, fra, *_ = line.strip().split(sep='\t')
                source_tab_target.append(f'{eng}\t{fra}')

        if os.path.exists(temp_dataset): os.remove(temp_dataset)

        with open(dataset, 'w', encoding=encoding) as dataset_file:
            dataset_file.write('\n'.join(source_tab_target))

    with open(dataset, 'r', encoding=encoding) as cachefile:
        if max_sample_pair_num:
            source_tab_target = cachefile.readlines()[:max_sample_pair_num]
        else:
            source_tab_target = cachefile.readlines()

    sources, targets = [], []
    for line in source_tab_target:
        source, target = line.strip().split(sep='\t')
        sources.append(source)
        targets.append(target)

    return sources, targets


def get_encoded_padded_tensor(lines: list[list[str]], vocab: Vocabulary, num_step: int) -> tuple[Tensor, Tensor]:
    encoded_pad = vocab.get_index(ST.PAD)
    lines_encoded: list[list[int]]
    lines_padded: list[list[int]]

    lines_encoded = [vocab.encode(tokens) + [vocab.get_index(ST.EOS)]
                     for tokens in lines]
    lines_padded = [encoded_line[:num_step] + [encoded_pad] * (num_step - len(encoded_line))
                    for encoded_line in lines_encoded]

    tensor_lines_padded = torch.tensor(lines_padded)
    length_without_padding = torch.sum(torch.ne(tensor_lines_padded, encoded_pad), dim=1)

    return tensor_lines_padded, length_without_padding


def nmt_eng_fra_dataloader(
        batch_size: int,
        seq_length: int,
        num_workers: int = 4,
        max_simple_pair_num: Optional[int] = None
) -> tuple[DataLoader, Vocabulary, Vocabulary]:
    eng_list, fra_list = process_eng_fra_dataset(normal_unicode='NFC', lowercase=True,
                                                 max_sample_pair_num=max_simple_pair_num)

    eng_tokenized: list[list[str]] = [tokenize(string, token_type='word') for string in eng_list]
    fra_tokenized: list[list[str]] = [tokenize(string, token_type='word') for string in fra_list]
    special_tokens_nmt = (ST.UNK, ST.PAD, ST.SOS, ST.EOS)

    eng_vocab = Vocabulary([word for string in eng_tokenized for word in string], special_tokens_nmt, min_freq=2)
    fra_vocab = Vocabulary([word for string in fra_tokenized for word in string], special_tokens_nmt, min_freq=2)

    eng_tensor, eng_valid_len = get_encoded_padded_tensor(eng_tokenized, eng_vocab, seq_length)
    fra_tensor, fra_valid_len = get_encoded_padded_tensor(fra_tokenized, fra_vocab, seq_length)

    dataset = TensorDataset(eng_tensor, eng_valid_len, fra_tensor, fra_valid_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader, eng_vocab, fra_vocab


if __name__ == '__main__':
    data_iter, eng_vocab, fra_vocab = nmt_eng_fra_dataloader(batch_size=2, seq_length=20)

    print(f'英文词表大小：{len(eng_vocab)}')
    print(f'法文词表大小：{len(fra_vocab)}\n')

    for eng_encoded_lines, eng_valid_len, fra_encoded_lines, fra_valid_len in data_iter:
        print(f'英文句子编码值：{eng_encoded_lines.tolist()}')
        print(f'英文句子有效长度：{eng_valid_len.tolist()}')
        print(f'英文句子解码（去除填充词元）：'
              f'{[" ".join(eng_vocab.decode(line[:length].tolist())) for line, length in zip(eng_encoded_lines, eng_valid_len)]}\n')

        print(f'法文句子编码值：{fra_encoded_lines.tolist()}')
        print(f'法文句子有效长度：{fra_valid_len.tolist()}')
        print(f'法文句子解码（去除填充词元）：'
              f'{[" ".join(fra_vocab.decode(line[:length].tolist())) for line, length in zip(fra_encoded_lines, fra_valid_len)]}')

        break
