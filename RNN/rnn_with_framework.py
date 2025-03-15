from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor


class RnnFramework(nn.Module):
    """支持普通 RNN、GRU、LSTM 及其双向网络的 RNN 框架"""

    def __init__(self, rnn_layer: nn.RNNBase, vocab_size: int):
        """
        :param rnn_layer: PyTorch 内置的 RNN 层（nn.RNN、nn.GRU、nn.LSTM）
        :param vocab_size: 词汇表大小（用于one-hot编码）
        """
        super().__init__()
        self.rnn_layer = rnn_layer

        self.rnn_direction_num = 2 if self.rnn_layer.bidirectional else 1
        self.rnn_hidden_num = self.rnn_layer.hidden_size
        self.fc = nn.Linear(in_features=self.rnn_hidden_num * self.rnn_direction_num, out_features=vocab_size)

        self.vocab_size = vocab_size

    def forward(self, inputs: Tensor, states: Tuple[Tensor, ...]) -> Tuple[Tensor, tuple]:
        """
        :param inputs: 原始输入，(BATCH_SIZE, SEQ_LENGTH)
        :param states: 以元组组织的隐状态。LSTM 的隐状态为 (h, c)，其他 RNN 均为 (h,)
        :return: 形状为 (SEQ_LENGTH * BATCH_SIZE, VOCAB_SIZE) 的 logits、与输入隐状态形状一致的隐状态更新
        """
        inputs = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)  # inputs：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        states = states[0] if not isinstance(self.rnn_layer, nn.LSTM) else states

        # outputs：(SEQ_LENGTH, BATCH_SIZE, DIRECTION_NUM * HIDDEN_NUM)
        outputs, new_states = self.rnn_layer(inputs, states)

        new_states = (new_states,) if not isinstance(self.rnn_layer, nn.LSTM) else new_states
        # outputs：... -> (SEQ_LENGTH * BATCH_SIZE, DIRECTION_NUM * HIDDEN_NUM)
        # logits：(SEQ_LENGTH * BATCH_SIZE, VOCAB_SIZE)
        logits = self.fc(outputs.reshape(-1, outputs.shape[-1]))

        return logits, new_states

    def init_hidden_states(self, batch_size: int, device: torch.device | str) -> Tuple[Tensor, ...]:
        """
        :param batch_size: 当前批次的样本数
        :param device: 计算设备
        :return: 符合对应 RNN 类型隐状态
        """
        shape = (self.rnn_layer.num_layers * self.rnn_direction_num, batch_size, self.rnn_hidden_num)
        hidden_state = torch.zeros(shape, device=device)
        if isinstance(self.rnn_layer, nn.LSTM):
            cell_state = hidden_state.clone()
            return hidden_state, cell_state
        else:
            return (hidden_state,)


if __name__ == '__main__':
    from text_dataset_loader import timemachine_data_loader
    from rnn_from_scratch import forecast_chars, train_one_epoch

    BATCH_SIZE = 32
    SEQ_LENGTH = 35
    HIDDEN_NUM = 512
    HIDDEN_LAYER_NUM = 1
    EPOCHS_NUM = 500
    LEARNING_RATE = 0.2
    IS_SHUFFLE = False
    FORCAST_INTERVAL = 10
    PREFIX_STRING = 'time traveller'

    data_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rnn_core = nn.RNN(input_size=len(vocab), hidden_size=HIDDEN_NUM, num_layers=HIDDEN_LAYER_NUM)
    rnn = RnnFramework(rnn_layer=rnn_core, vocab_size=len(vocab)).to(device)
    optimizer = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        ppl, speed = train_one_epoch(rnn, data_iter, loss_fn, optimizer, device, IS_SHUFFLE)
        print(f'第 {epoch + 1:02} 轮：困惑度为 {ppl:04.1f}，速度为 {speed:.1f} (tokens/sec)')

        if (epoch + 1) % FORCAST_INTERVAL == 0:
            with torch.no_grad():  # 评估模式
                prediction = forecast_chars(PREFIX_STRING, 50, rnn, vocab, device)
                print(f'预测结果：{prediction!r}')
