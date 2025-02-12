from text_dataset_loader import timemachine_data_loader
from torch.nn import functional as F
import torch


def get_params(vocab_size: int, hidden_num: int, device: torch.device | str):
    w_input2hidden = torch.randn(size=(vocab_size, hidden_num), device=device, requires_grad=True) * 0.01
    w_hidden2hidden = torch.randn(size=(hidden_num, hidden_num), device=device, requires_grad=True) * 0.01
    b_hidden = torch.zeros(hidden_num, device=device, requires_grad=True)

    w_hidden2output = torch.randn(size=(hidden_num, vocab_size), device=device, requires_grad=True) * 0.01
    b_output = torch.zeros(vocab_size, device=device, requires_grad=True)

    return w_input2hidden, w_hidden2hidden, b_hidden, w_hidden2output, b_output


def init_rnn_state(batch_size, num_hiddens, device: torch.device | str):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn_step(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


class RnnScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


if __name__ == '__main__':
    BATCH_SIZE = 32
    SEQ_LENGTH = 35
    train_iter, vocab = timemachine_data_loader(BATCH_SIZE, SEQ_LENGTH)
    print(f'{len(vocab.vocabulary) = }')

    print(init_rnn_state(4, 6, device='cpu'))
