import torch
from torch.utils import data


def lineal_gen(true_w, true_b, size, features_num, noise_mean=0, noise_std=0.01):
    X = torch.normal(mean=0, std=1, size=(size, features_num))  # [数据量, 维数]
    w = torch.full(size=(features_num, 1), fill_value=true_w)  # [维数, 单列]
    noise = torch.normal(noise_mean, noise_std, size=(size, 1))  # [数据量, 单列]
    y = X @ w + true_b + noise
    return X, y


def init_params(features_num):
    w = torch.normal(0, 1, size=(features_num, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def linreg(x: torch.Tensor, weighs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x @ weighs + bias


def mse_loss(y_predict: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    mse = pow((y_predict - y_real), 2) / 2
    return mse


def sgd(params: [torch.Tensor], lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_eval(predi_w, predi_b, net, loss, epochs, lr, data_iter, batch_size, lambd):
    for epoch in range(epochs):
        for X_i, y_i in data_iter:
            l = loss(net(X_i, predi_w, predi_b), y_i) + lambd * l2_penalty(predi_w)
            l.sum().backward()
            sgd([predi_w, predi_b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1:3}, trainLoss:{l.sum().item(): >10.5f}', end='')
            avg_loss = evaluate(predi_w, predi_b, net=linreg, loss=mse_loss, data_iter=iter_test)
            print(f', testLoss:{avg_loss: >10.5f}')

    print(f'参数 predi_w 的 L2 范数是：{torch.norm(predi_w).item():.5f}')


def evaluate(predi_w, predi_b, net, loss, data_iter):
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for X_i, y_i in data_iter:
            y_pred = net(X_i, predi_w, predi_b)
            total_loss += loss(y_pred, y_i).sum().item()
            num_samples += y_i.size(0)

    avg_loss = total_loss / num_samples
    return avg_loss


if __name__ == '__main__':
    TRUE_WEIGHT = 0.01
    TRUE_BIAS = 0.05
    FEATURES_NUM = 200

    TRAIN_SIZE = 20
    TEST_SIZE = 100
    BATCH_SIZE = 5

    LEARNING_RATE = 0.003
    EPOCHS_NUM = 100

    X, y = lineal_gen(TRUE_WEIGHT, TRUE_BIAS, TRAIN_SIZE + TEST_SIZE, FEATURES_NUM)
    iter_train = data.DataLoader(dataset=data.TensorDataset(X[:TRAIN_SIZE], y[:TRAIN_SIZE]),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
    iter_test = data.DataLoader(dataset=data.TensorDataset(X[TRAIN_SIZE:], y[TRAIN_SIZE:]),
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    w, b = init_params(FEATURES_NUM)
    train_eval(w, b, net=linreg, loss=mse_loss, epochs=EPOCHS_NUM, lr=LEARNING_RATE,
               data_iter=iter_train, batch_size=BATCH_SIZE, lambd=8)
