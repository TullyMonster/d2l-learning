from typing import Tuple

import torch
import torch.nn.functional as F
from torch import optim, nn, Tensor

from rnn_from_scratch import train_one_epoch, forecast_chars


class GruScratch:
    def __init__(self, vocab_size: int, hidden_num: int, device: torch.device | str):
        self.__vocab_size = vocab_size
        self.__hidden_num = hidden_num

        self.params = self.__init_params(device)

    def __init_params(self, device):
        # 重置门参数
        W_xr = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hr = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_r = torch.zeros(self.__hidden_num, device=device)

        # 更新门参数
        W_xz = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hz = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_z = torch.zeros(self.__hidden_num, device=device)

        # 候选隐状态参数
        W_xh = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hh = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_h = torch.zeros(self.__hidden_num, device=device)

        # 输出层参数
        W_hq = torch.randn((self.__hidden_num, self.__vocab_size), device=device) * 0.01
        b_q = torch.zeros(self.__vocab_size, device=device)

        return (W_xr.requires_grad_(),
                W_hr.requires_grad_(),
                b_r.requires_grad_(),
                W_xz.requires_grad_(),
                W_hz.requires_grad_(),
                b_z.requires_grad_(),
                W_xh.requires_grad_(),
                W_hh.requires_grad_(),
                b_h.requires_grad_(),
                W_hq.requires_grad_(),
                b_q.requires_grad_())

    def init_hidden_states(self, batch_size: int, device: torch.device | str):
        """初始化隐状态，并用元组组织"""
        return (torch.zeros((batch_size, self.__hidden_num), device=device),)

    @staticmethod
    def __gru_step(inputs: Tensor, states: Tuple[Tensor, ...], params: Tuple[Tensor, ...]):
        W_xr, W_hr, b_r, W_xz, W_hz, b_z, W_xh, W_hh, b_h, W_hq, b_q = params
        state, = states
        outputs_temp = []

        for step in inputs:  # step：(BATCH_SIZE, VOCAB_SIZE)
            gate_reset = torch.sigmoid((step @ W_xr) + (state @ W_hr) + b_r)
            gate_update = torch.sigmoid((step @ W_xz) + (state @ W_hz) + b_z)
            hidden_candidate = torch.tanh((step @ W_xh) + ((gate_reset * state) @ W_hh) + b_h)
            state = gate_update * state + (1 - gate_update) * hidden_candidate  # state: (BATCH_SIZE, HIDDEN_NUM)
            output_layer = state @ W_hq + b_q
            outputs_temp.append(output_layer)

        outputs = torch.cat(outputs_temp, dim=0)
        out_states = (state,)

        return outputs, out_states

    def __call__(self, inputs, states) -> Tuple[Tensor, tuple]:
        inputs = F.one_hot(inputs.T, self.__vocab_size).type(torch.float32)
        return self.__gru_step(inputs, states=states, params=self.params)


if __name__ == '__main__':
    from text_dataset_loader import timemachine_data_loader

    BATCH_SIZE = 32
    SEQ_LENGTH = 35
    HIDDEN_NUM = 512
    EPOCHS_NUM = 50
    LEARNING_RATE = 0.7
    IS_SHUFFLE = False
    FORCAST_INTERVAL = 10
    PREFIX_STRING = 'time traveller'

    data_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH, IS_SHUFFLE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gru = GruScratch(vocab_size=len(vocab), hidden_num=HIDDEN_NUM, device=device)
    optimizer = optim.SGD(gru.params, lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        ppl, speed = train_one_epoch(gru, data_iter, loss_fn, optimizer, device, IS_SHUFFLE)
        print(f'第 {epoch + 1:02} 轮：困惑度为 {ppl:04.1f}，速度为 {speed:.1f} (tokens/sec)')

        if (epoch + 1) % FORCAST_INTERVAL == 0:
            with torch.no_grad():  # 评估模式
                prediction = forecast_chars(PREFIX_STRING, 50, gru, vocab, device)
                print(f'预测结果：{prediction!r}')
