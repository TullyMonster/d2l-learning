from typing import Tuple

import torch
import torch.nn.functional as F
from torch import optim, nn, Tensor

from rnn_from_scratch import train_one_epoch, forecast_chars


class LstmScratch:
    def __init__(self, vocab_size: int, hidden_num: int, device: torch.device | str):
        self.__vocab_size = vocab_size
        self.__hidden_num = hidden_num

        self.params = self.__init_params(device)

    def __init_params(self, device):
        # 遗忘门参数
        W_xf = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hf = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_f = torch.zeros(self.__hidden_num, device=device)

        # 输入门参数
        W_xi = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hi = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_i = torch.zeros(self.__hidden_num, device=device)

        # 输出门参数
        W_xo = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_ho = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_o = torch.zeros(self.__hidden_num, device=device)

        # 候选记忆元参数
        W_xc = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        W_hc = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_c = torch.zeros(self.__hidden_num, device=device)

        # 输出层参数
        W_hq = torch.randn((self.__hidden_num, self.__vocab_size), device=device) * 0.01
        b_q = torch.zeros(self.__vocab_size, device=device)

        return (W_xf.requires_grad_(),
                W_hf.requires_grad_(),
                b_f.requires_grad_(),
                W_xi.requires_grad_(),
                W_hi.requires_grad_(),
                b_i.requires_grad_(),
                W_xo.requires_grad_(),
                W_ho.requires_grad_(),
                b_o.requires_grad_(),
                W_xc.requires_grad_(),
                W_hc.requires_grad_(),
                b_c.requires_grad_(),
                W_hq.requires_grad_(),
                b_q.requires_grad_())

    def init_hidden_states(self, batch_size: int, device: torch.device | str):
        """初始化隐状态和记忆元，并用元组组织"""
        hidden_state = torch.zeros((batch_size, self.__hidden_num), device=device)
        memory_cell = torch.zeros((batch_size, self.__hidden_num), device=device)

        return hidden_state, memory_cell

    @staticmethod
    def __lstm_step(inputs: Tensor, states: Tuple[Tensor, ...], params: Tuple[Tensor, ...]):
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
        state, memory_cell = states
        outputs_temp = []

        for step in inputs:  # step：(BATCH_SIZE, VOCAB_SIZE)
            gate_forget = torch.sigmoid((step @ W_xf) + (state @ W_hf) + b_f)
            gate_input = torch.sigmoid((step @ W_xi) + (state @ W_hi) + b_i)
            gate_output = torch.sigmoid((step @ W_xo) + (state @ W_ho) + b_o)
            memory_cell_candidate = torch.tanh((step @ W_xc) + (state @ W_hc) + b_c)
            memory_cell = gate_forget * memory_cell + gate_input * memory_cell_candidate
            state = gate_output * torch.tanh(memory_cell)  # state: (BATCH_SIZE, HIDDEN_NUM)
            output_layer = state @ W_hq + b_q
            outputs_temp.append(output_layer)

        outputs = torch.cat(outputs_temp, dim=0)
        out_states = state, memory_cell

        return outputs, out_states

    def __call__(self, inputs, states) -> Tuple[Tensor, tuple]:
        inputs = F.one_hot(inputs.T, self.__vocab_size).type(torch.float32)
        return self.__lstm_step(inputs, states=states, params=self.params)


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
    lstm = LstmScratch(vocab_size=len(vocab), hidden_num=HIDDEN_NUM, device=device)
    optimizer = optim.SGD(lstm.params, lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        ppl, speed = train_one_epoch(lstm, data_iter, loss_fn, optimizer, device, IS_SHUFFLE)
        print(f'第 {epoch + 1:02} 轮：困惑度为 {ppl:04.1f}，速度为 {speed:.1f} (tokens/sec)')

        if (epoch + 1) % FORCAST_INTERVAL == 0:
            with torch.no_grad():  # 评估模式
                prediction = forecast_chars(PREFIX_STRING, 50, lstm, vocab, device)
                print(f'预测结果：{prediction!r}')
