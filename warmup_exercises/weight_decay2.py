import torch
from torch import nn, optim
from torch.utils import data


def lineal_gen(true_w, true_b, size, features_num, noise_mean=0, noise_std=0.01):
    X = torch.normal(mean=0, std=1, size=(size, features_num))  # [数据量, 维数]
    w = torch.full(size=(features_num, 1), fill_value=true_w)  # [维数, 单列]
    noise = torch.normal(noise_mean, noise_std, size=(size, 1))  # [数据量, 单列]
    y = X @ w + true_b + noise
    return X, y


def evaluate(model, data_iter, loss_fn):
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for X_i, y_i in data_iter:
            y_pred = model(X_i)
            l = loss_fn(y_pred, y_i)
            total_loss += l.sum().item()
            num_samples += y_i.size(0)

    avg_loss = total_loss / num_samples
    return avg_loss


def train_eval(wd):
    net = nn.Sequential(nn.Linear(FEATURES_NUM, 1))  # 初始化模型参数
    for param in net.parameters():
        param.data.normal_()

    loss = nn.MSELoss(reduction='none')  # 指定损失函数

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=wd)  # 定义优化器

    for epoch in range(EPOCHS_NUM):  # 训练与评估
        for X_i, y_i in iter_train:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1:3}, trainLoss:{l.sum().item(): >10.5f}', end='')
            avg_loss = evaluate(net, iter_train, loss)
            print(f', testLoss:{avg_loss: >10.5f}')
    print(f'参数 predi_w 的 L2 范数是：{net[0].weight.norm().item():.5f}')


if __name__ == '__main__':
    TRUE_WEIGHT = 0.01
    TRUE_BIAS = 0.05
    FEATURES_NUM = 200

    TRAIN_SIZE = 20
    TEST_SIZE = 100
    BATCH_SIZE = 5

    LEARNING_RATE = 0.003
    EPOCHS_NUM = 100

    WEIGHT_DECAY = 8  # 权重衰减系数

    # 生成并加载数据
    X, y = lineal_gen(TRUE_WEIGHT, TRUE_BIAS, TRAIN_SIZE + TEST_SIZE, FEATURES_NUM)
    iter_train = data.DataLoader(dataset=data.TensorDataset(X[:TRAIN_SIZE], y[:TRAIN_SIZE]),
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)
    iter_test = data.DataLoader(dataset=data.TensorDataset(X[TRAIN_SIZE:], y[TRAIN_SIZE:]),
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    train_eval(WEIGHT_DECAY)
