import torch
from dataset_loader import fashionMNIST_loader
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard


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


def net_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 显式地将偏置初始化为零


if __name__ == '__main__':
    # 模型参数
    IN = 28 * 28
    HIDDEN1 = HIDDEN2 = 256
    OUT = 10
    DROPOUT1 = 0.2
    DROPOUT2 = 0.5

    # 训练参数
    EPOCHS = 10
    LEARNING_RATE = 0.5
    BATCH_SIZE = 256

    # 初始化网络
    net = nn.Sequential(
        nn.Flatten(),  # 输入层
        nn.Linear(IN, 256), nn.ReLU(), nn.Dropout(DROPOUT1),  # 隐藏层 1
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(DROPOUT2),  # 隐藏层 2
        nn.Linear(256, OUT)  # 输出层
    )
    net.apply(net_init)

    # 定义损失函数和优化器
    loss = nn.CrossEntropyLoss(reduction='none')  # 设置交叉熵损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)  # 设置优化器

    # 加载数据
    train, test = fashionMNIST_loader(BATCH_SIZE)

    # 初始化 TensorBoard 记录器
    writer = SummaryWriter('runs/fashionMNIST_experiment')

    # 开始训练
    for epoch in range(EPOCHS):
        # 训练并获取指标
        train_loss, train_accu = train_test(net, optimizer, loss, train)
        test_accu = validate(net, test)

        # 打印日志
        print(f'Epoch:{epoch + 1: >2}/{EPOCHS}, TrainLoss: {train_loss:.4f}, TrainAccu: {train_accu:.2f}%', end='')
        print(f', TestAccu: {test_accu:.2f}%')

        # 将指标记录到 TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accu, epoch)
        writer.add_scalar('Accuracy/Test', test_accu, epoch)

    # 关闭 TensorBoard 记录器
    writer.close()
