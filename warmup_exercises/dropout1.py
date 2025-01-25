import torch
import torch.nn as nn
import torch.nn.functional as nn_fun

from softmax1 import fashionMNIST_loader


def dropout(x, prob):
    """
    用于层的暂退 (dropout)。
    :param x: 层的输出矩阵
    :param prob: 神经元被置为零的概率
    :return: 经过暂退处理后的矩阵
    """
    assert 0 <= prob <= 1, f'神经元被置为零的概率应为 [0, 1]'
    if prob == 1:
        return torch.zeros_like(x)  # 丢弃全部元素
    elif prob == 0:
        return x  # 保留全部元素
    else:
        mask = torch.rand(x.shape) > prob  # 同样形状的均匀分布 -> 根据概率生成掩码
        return mask.float() * x / (1.0 - prob)  # 执行运算，输出


class MLP(nn.Module):
    def __init__(self, in_size, hidden1_size, hidden2_size, out_size, *dropout_probs):
        super().__init__()
        self.__in_size = in_size
        self.dropout_probs = dropout_probs
        self.input = nn.Linear(in_size, hidden1_size)  # 输入层
        self.hidden1 = nn.Linear(hidden1_size, hidden2_size)  # 隐藏层 1
        self.hidden2 = nn.Linear(hidden2_size, out_size)  # 隐藏层 2

    def forward(self, x):
        reshaped_x = x.reshape(-1, self.__in_size)
        h1 = dropout(nn_fun.relu(self.input(reshaped_x)), self.dropout_probs[0]) if self.training \
            else nn_fun.relu(self.input(reshaped_x))
        h2 = dropout(nn_fun.relu(self.hidden1(h1)), self.dropout_probs[1]) if self.training \
            else nn_fun.relu(self.hidden1(h1))
        out = self.hidden2(h2)
        return out


def train_test(net, optimizer, loss_fn, train_loader):
    net.train()  # 设置模型为训练模式
    total_loss = 0.0
    correct = 0
    total = 0

    # 训练过程
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 清零梯度
        outputs = net(inputs)  # 前向传播
        loss = loss_fn(outputs, labels)  # 计算损失
        loss.mean().backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.mean().item() * inputs.size(0)  # 累加损失
        predicted: torch.Tensor
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)  # 累加样本数量
        correct += (predicted == labels).sum().item()  # 累加正确预测的数量

    train_loss = total_loss / total  # 计算平均损失
    train_accuracy = correct / total * 100  # 计算训练集准确率

    return train_loss, train_accuracy


def validate(net, test_loader):
    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        for inputs, labels in test_loader:
            outputs = net(inputs)  # 前向传播
            predicted: torch.Tensor
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累加样本数量
            correct += (predicted == labels).sum().item()  # 累加正确预测的数量

    accuracy = correct / total * 100  # 计算验证集准确率
    return accuracy


if __name__ == '__main__':
    IN = 28 * 28
    HIDDEN1 = HIDDEN2 = 256
    OUT = 10

    DROPOUT1 = 0.2
    DROPOUT2 = 0.5

    EPOCHS = 10
    LEARNING_RATE = 0.5
    BATCH_SIZE = 256

    net = MLP(IN, HIDDEN1, HIDDEN2, OUT, DROPOUT1, DROPOUT2)  # 定义模型并初始化
    loss = nn.CrossEntropyLoss(reduction='none')  # 设置交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)  # 设置优化器

    train, test = fashionMNIST_loader(BATCH_SIZE)

    for epoch in range(EPOCHS):
        train_loss, train_accu = train_test(net, optimizer, loss, train)
        print(f'Epoch:{epoch + 1: >2}/{EPOCHS}, TrainLoss: {train_loss:.4f}, TrainAccu: {train_accu:.2f}%', end='')
        test_accu = validate(net, test)
        print(f', TestAccu: {test_accu:.2f}%')
