import torch
from torch import nn

from dropout1 import train_test, validate
from softmax1 import fashionMNIST_loader


def net_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 显式地将偏置初始化为零


if __name__ == '__main__':
    IN = 28 * 28
    HIDDEN1 = HIDDEN2 = 256
    OUT = 10

    DROPOUT1 = 0.2
    DROPOUT2 = 0.5

    EPOCHS = 10
    LEARNING_RATE = 0.5
    BATCH_SIZE = 256

    net = nn.Sequential(
        nn.Flatten(),  # 输入层
        nn.Linear(IN, 256), nn.ReLU(), nn.Dropout(DROPOUT1),  # 隐藏层 1
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(DROPOUT2),  # 隐藏层 2
        nn.Linear(256, OUT)  # 输出层
    )
    net.apply(net_init)

    loss = nn.CrossEntropyLoss(reduction='none')  # 设置交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)  # 设置优化器

    train, test = fashionMNIST_loader(BATCH_SIZE)

    for epoch in range(EPOCHS):
        train_loss, train_accu = train_test(net, optimizer, loss, train)
        print(f'Epoch:{epoch + 1: >2}/{EPOCHS}, TrainLoss: {train_loss:.4f}, TrainAccu: {train_accu:.2f}%', end='')
        test_accu = validate(net, test)
        print(f', TestAccu: {test_accu:.2f}%')
