import torch
from torch import nn, optim

from softmax1 import fashionMNIST_loader

BATCH_SIZE = 256


def initialize_nn(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_one_epoch(net, train_loader, optimizer, loss_fn):
    net.train()  # 将模型设置为训练模式
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = net(images)  # 前向传播
        loss_value = loss_fn(outputs, labels)  # 计算损失
        loss_value.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss_value.item()  # 累计损失

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(net, test_loader):
    net.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = net(images)
            predicted: torch.Tensor
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    train, test = fashionMNIST_loader(BATCH_SIZE)
    net = nn.Sequential(nn.Flatten(), nn.Linear(in_features=28 * 28, out_features=10))  # 构建网络
    net.apply(initialize_nn)  # 初始化
    loss = nn.CrossEntropyLoss(reduction='mean')  # 定义损失函数
    trainer = optim.SGD(net.parameters(), lr=0.1)  # 定义优化算法

    EPOCHS = 10  # 设置训练的轮数
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(net, train, trainer, loss)
        print(f'Epoch: {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}', end='')
        accuracy = validate(net, test)
        print(f', Test Accu: {accuracy:.2f}%')
