import torch
from torch import nn, optim

from softmax1 import fashionMNIST_loader

BATCH_SIZE = 256
NUM_EPOCH = 10
LR = 0.1


def init_nn(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_model(train_loader):
    net.train()  # 将模型设为训练模式
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        optimizer.zero_grad()  # 清空之前的梯度
        output = net(data)  # 前向传播
        loss = loss_fn(output, target)  # 计算损失
        loss.mean().backward()  # 反向传播
        optimizer.step()  # 更新权重

        total_loss += loss.sum().item()  # 累加损失
        predicted: torch.Tensor
        _, predicted = torch.max(output.data, 1)  # 获取预测结果
        total += target.size(0)  # 总样本数
        correct += (predicted == target).sum().item()  # 正确预测数

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(test_loader):
    net.eval()  # 将模型设为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            output = net(data)  # 前向传播
            predicted: torch.Tensor
            _, predicted = torch.max(output.data, 1)  # 获取预测结果
            total += target.size(0)  # 总样本数
            correct += (predicted == target).sum().item()  # 正确预测数

    accuracy = correct / total
    return accuracy


net = nn.Sequential(nn.Flatten(),  # 输入层
                    nn.Linear(in_features=784, out_features=256), nn.ReLU(),  # 隐藏层
                    nn.Linear(in_features=256, out_features=10))  # 输出层
net.apply(init_nn)

loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=LR)

if __name__ == '__main__':
    train, test = fashionMNIST_loader(BATCH_SIZE)
    for epoch in range(NUM_EPOCH):
        train_loss, train_accuracy = train_model(train)  # 训练模型
        print(f'Epoch [{epoch + 1}/{NUM_EPOCH}], trainLoss: {train_loss:.4f}, trainAccu: {train_accuracy:.4f}', end='')
        test_accuracy = evaluate_model(test)  # 评估模型
        print(f', testAccu: {test_accuracy:.4f}')
