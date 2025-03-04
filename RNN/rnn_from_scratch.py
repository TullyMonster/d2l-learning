from time import time
from typing import Tuple, Protocol, Iterable, Callable, Optional

import torch
import torch.nn.functional as F
from math import exp
from torch import Tensor, nn, optim

from RNN.text_preprocessing import Vocabulary


class RnnProtocol(Protocol):
    params: Tuple[Tensor, ...]

    def init_hidden_states(self, batch_size: int, device: torch.device | str) -> Tuple[Tensor, ...]:
        ...


class RnnScratch:
    def __init__(self, vocab_size: int, hidden_num: int, device: torch.device | str):
        self.__vocab_size = vocab_size
        self.__hidden_num = hidden_num

        self.params = self.__init_params(device)

    def __init_params(self, device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # 初始化参数，并设置 requires_grad=True
        w_input2hidden = torch.randn((self.__vocab_size, self.__hidden_num), device=device) * 0.01
        w_hidden2hidden = torch.randn((self.__hidden_num, self.__hidden_num), device=device) * 0.01
        b_hidden = torch.zeros(self.__hidden_num, device=device)

        w_hidden2output = torch.randn((self.__hidden_num, self.__vocab_size), device=device) * 0.01
        b_output = torch.zeros(self.__vocab_size, device=device)

        return (w_input2hidden.requires_grad_(),
                w_hidden2hidden.requires_grad_(),
                b_hidden.requires_grad_(),
                w_hidden2output.requires_grad_(),
                b_output.requires_grad_())

    def init_hidden_states(self, batch_size: int, device: torch.device | str) -> Tuple[Tensor, ...]:
        """初始化隐状态，并用元组组织"""
        return torch.zeros((batch_size, self.__hidden_num), device=device),

    @staticmethod
    def __rnn_step(inputs: Tensor, states: Tuple[Tensor, ...], params: Tuple[Tensor, ...]) -> Tuple[Tensor, tuple]:
        """
        RNN 的一个时间步内的隐状态计算
        :param inputs: 形状为：(SEQ_LENGTH, BATCH_SIZE, VOCAB_SIZE)
        :param states: 隐状态元组，其中的隐状态形状为：(BATCH_SIZE, HIDDEN_NUM)
        :param params: 模型参数元组
        :return: 由计算结果与隐状态元组组成的元组
        """
        w_input2hidden, w_hidden2hidden, b_hidden, w_hidden2output, b_output = params
        state, = states
        outputs_temp = []

        for step in inputs:  # step：(BATCH_SIZE, VOCAB_SIZE)
            state = torch.tanh(
                step @ w_input2hidden + state @ w_hidden2hidden + b_hidden
            )  # 更新隐状态。state: (BATCH_SIZE, HIDDEN_NUM)
            output_layer = state @ w_hidden2output + b_output  # output_layer: (BATCH_SIZE, VOCAB_SIZE)
            outputs_temp.append(output_layer)

        outputs = torch.cat(outputs_temp, dim=0)  # outputs: (BATCH_SIZE * SEQ_LENGTH, VOCAB_SIZE)
        out_states = (state,)

        return outputs, out_states

    def __call__(self, inputs, states) -> Tuple[Tensor, tuple]:
        inputs = F.one_hot(inputs.T, self.__vocab_size).type(torch.float32)
        return self.__rnn_step(inputs, states=states, params=self.params)


def forecast_chars(prefix: str, num: int, model, vocab: Vocabulary, device: torch.device | str) -> str:
    states = model.init_hidden_states(batch_size=1, device=device)
    outputs = [vocab.get_index(prefix[0])]

    for y in prefix[1:]:  # 预热
        _, states = model(torch.tensor(outputs[-1:], device=device).unsqueeze(0), states)
        outputs.append(vocab.get_index(y))

    for _ in range(num):  # 预测
        y, states = model(torch.tensor(outputs[-1:], device=device).unsqueeze(0), states)
        outputs.append(torch.argmax(y, dim=1).item())
    return ''.join(vocab.decode(outputs))


def clip_gradients(model: nn.Module | RnnProtocol, max_norm: float):
    params: Iterable[Tensor]
    params = [p for p in model.parameters() if p.requires_grad] if isinstance(model, nn.Module) else model.params

    grad_l2_norm = torch.norm(torch.cat([p.grad.flatten() for p in params]), 2)
    if grad_l2_norm > max_norm:
        for p in params:
            p.grad.mul_(max_norm / grad_l2_norm)


def train_one_epoch(
        net: nn.Module | RnnProtocol,
        data_iter: Iterable[tuple[Tensor, Tensor]],
        loss_fn: nn.Module,
        updater: optim.Optimizer | Callable,
        device: torch.device,
        shuffle: bool = False
) -> Tuple[float, float]:
    """
    在一个迭代周期内训练模型

    :param net: RNN 实例
    :param data_iter: 数据集加载器
    :param loss_fn: 损失函数
    :param updater: 优化器或参数更新函数
    :param device: 计算设备
    :param shuffle: 是否随机采样
    :return: 困惑度, 训练速度
    """

    # 定义隐状态变量、使用词元和损失值计数，设置计时器
    states: Optional[Tuple[Tensor, ...]] = None
    total_tokens: int = 0
    total_loss: float = 0.0
    start_time: float = time()

    for features, labels in data_iter:
        # 处理标签维度并转移到设备
        features = features.to(device)  # features: (BATCH_SIZE, SEQ_LENGTH)
        labels = labels.T.flatten().to(device)  # labels: (SEQ_LENGTH * BATCH_SIZE)

        # 初始化或复用隐状态
        if shuffle or states is None:  # 初始化
            states = net.init_hidden_states(features.shape[0], device)
        else:  # 复用隐状态，安全分离隐状态
            states = tuple(s.detach() for s in states)

        # 前向传播
        output, states = net(features, states)

        # 计算损失  损失函数参数分别是 (N, C) 和 (N,)
        # output: (SEQ_LENGTH * BATCH_SIZE, VOCAB_SIZE)
        # labels: (SEQ_LENGTH * BATCH_SIZE)
        loss = loss_fn(output, labels)

        # 反向传播、梯度裁剪、参数更新
        if isinstance(updater, optim.Optimizer):
            updater.zero_grad()
            loss.backward()
            clip_gradients(net, max_norm=1)
            updater.step()
        else:
            # 假设自定义的优化函数内置了梯度清零操作
            loss.backward()
            clip_gradients(net, max_norm=1)
            updater(features.shape[0])

        # 统计指标
        batch_tokens = labels.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    # 计算困惑度和训练速度
    perplexity = exp(total_loss / total_tokens)
    tokens_per_sec = total_tokens / (time() - start_time)

    return perplexity, tokens_per_sec


if __name__ == '__main__':
    from RNN.text_dataset_loader import timemachine_data_loader

    BATCH_SIZE = 32
    SEQ_LENGTH = 35
    HIDDEN_NUM = 512
    EPOCHS_NUM = 500
    LEARNING_RATE = 0.7
    IS_SHUFFLE = False
    FORCAST_INTERVAL = 10
    PREFIX_STRING = 'time traveller'

    data_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH, IS_SHUFFLE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnn = RnnScratch(vocab_size=len(vocab), hidden_num=HIDDEN_NUM, device=device)
    optimizer = optim.SGD(rnn.params, lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS_NUM):
        ppl, speed = train_one_epoch(rnn, data_iter, loss_fn, optimizer, device, IS_SHUFFLE)
        print(f'第 {epoch + 1:02} 轮：困惑度为 {ppl:04.1f}，速度为 {speed:.1f} (tokens/sec)')

        if (epoch + 1) % FORCAST_INTERVAL == 0:
            with torch.no_grad():  # 评估模式
                prediction = forecast_chars(PREFIX_STRING, 50, rnn, vocab, device)
                print(f'预测结果：{prediction!r}')
