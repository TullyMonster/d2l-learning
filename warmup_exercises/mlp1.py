import torch
from torch import nn, optim

from softmax1 import fashionMNIST_loader

BATCH_SIZE = 256
NUM_IN = 28 * 28
NUM_OUT = 10
NUM_HIDDEN = 256
NUM_EPOCH = 10
LR = 0.1

W1 = nn.Parameter(torch.randn(NUM_IN, NUM_HIDDEN, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(NUM_HIDDEN, requires_grad=True))
W2 = nn.Parameter(torch.randn(NUM_HIDDEN, NUM_OUT, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(NUM_OUT, requires_grad=True))
PARAMS = [W1, b1, W2, b2]


def net(data_input: torch.Tensor):
    data_flatten = data_input.flatten(start_dim=1)  # 输入层：(batch_size, 28, 28) -> (batch_size, 784)
    h1 = relu(data_flatten @ W1 + b1)  # 隐藏层：(batch_size, 784) @ (784, 256) + (256) -> (batch_size, 256)
    o = h1 @ W2 + b2  # 输出层：(batch_size, 256) @ (256, 10) + (10) -> (batch_size, 10)
    return o


def relu(x):
    a = torch.zeros_like(x)
    return torch.max(x, a)


def train_model(train_loader):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data, target in train_loader:
        optimizer.zero_grad()  # 清零梯度

        output = net(data)  # 前向传播
        loss = loss_fn(output, target)  # 计算损失
        loss.mean().backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.mean().item()  # 累加损失

        # 计算训练精度
        predicted: torch.Tensor
        _, predicted = torch.max(output, 1)  # 获取预测结果
        correct_predictions += (predicted == target).sum().item()  # 计算正确预测的数量
        total_samples += target.size(0)  # 计算总样本数

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def evaluate_model(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            output = net(data)  # 前向传播
            predicted: torch.Tensor
            _, predicted = torch.max(output, 1)  # 获取预测结果
            total += target.size(0)  # 计算总样本数
            correct += (predicted == target).sum().item()  # 计算正确预测的数量
    return correct / total


loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(PARAMS, lr=LR)

if __name__ == '__main__':
    train, test = fashionMNIST_loader(BATCH_SIZE)
    for epoch in range(NUM_EPOCH):
        train_loss, train_accuracy = train_model(train)  # 训练模型
        print(f'Epoch [{epoch + 1}/{NUM_EPOCH}], trainLoss: {train_loss:.4f}, trainAccu: {train_accuracy:.4f}', end='')
        test_accuracy = evaluate_model(test)  # 评估模型
        print(f', testAccu: {test_accuracy:.4f}')
