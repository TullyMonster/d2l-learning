import torch
import torchvision
from torch.utils import data
from torchvision import transforms

DATASETS_PATH = r'D:\Coding\DATASETS'
BATCH_SIZE = 256  # 小批量的大小
PROCESS_NUM = 4  # 读取数据所用的进程数
LEARNING_RATE = 0.01  # 学习率
NUM_EPOCHS = 20  # 训练轮数


# 数据加载器
def fashionMNIST_loader(batch_size=BATCH_SIZE, *, resize=None) -> (data.DataLoader, data.DataLoader):
    t = [transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor，并归一化到 [0, 1] 之间
         transforms.Normalize((0.5,), (0.5,))]  # 标准化操作，将数据均值和标准差设为 0.5
    if resize:
        t.insert(0, transforms.Resize(size=resize))  # 调整图像尺寸。整数，按比例将短边调整到指定的大小；元组，指定宽高
    trans_pipe = transforms.Compose(t)  # 转换操作仅在数据被访问时动态应用
    dataset_train = torchvision.datasets.FashionMNIST(
        root=DATASETS_PATH, train=True, transform=trans_pipe, download=True)
    dataset_test = torchvision.datasets.FashionMNIST(
        root=DATASETS_PATH, train=False, transform=trans_pipe, download=True)
    iter_train = data.DataLoader(dataset_train, batch_size, shuffle=True, num_workers=PROCESS_NUM)
    iter_test = data.DataLoader(dataset_test, batch_size, shuffle=True, num_workers=PROCESS_NUM)
    return iter_train, iter_test


# Softmax函数
def softmax(x: torch.Tensor):
    x_exp = torch.exp(x)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition


# Softmax 线性回归模型
def linreg(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    return softmax(x.reshape(-1, w.shape[0]) @ w + b)


# 交叉熵损失函数
def cross_entropy_loss(y_hat, y):
    # y 是类别索引而非 one-hot 编码
    return -torch.mean(torch.log(y_hat[range(len(y_hat)), y]))


# 分类精度
def evaluate_accuracy(data_iter, w, b):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            predicted_labels = torch.argmax(linreg(X, w, b), dim=1)  # 获取预测的类别索引
            acc_sum += (predicted_labels == y).float().sum().item()  # 预测正确的数量
            n += y.shape[0]
    return acc_sum / n


# 训练模型
def train_model(train_iter, test_iter, w, b, lr, num_epochs):
    for epoch in range(num_epochs):
        loss = None
        for X, y in train_iter:
            y_hat = linreg(X, w, b)
            loss = cross_entropy_loss(y_hat, y)
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
        train_acc = evaluate_accuracy(train_iter, w, b)
        test_acc = evaluate_accuracy(test_iter, w, b)

        print(f'epoch {epoch + 1}, loss {loss.item():.4f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')


# 可视化预测结果
def visualize_predictions(test_iter, w, b, num_samples=10):
    import matplotlib.pyplot as plt  # 导入可视化库
    import numpy as np  # 导入 numpy 以处理数据
    # 随机选择样本
    data_iter = iter(test_iter)
    images, labels = next(data_iter)

    # 获取预测
    with torch.no_grad():
        y_hat = linreg(images, w, b)
        predicted_labels = torch.argmax(y_hat, dim=1)

    # 随机选择10个样本进行展示
    indices = np.random.choice(len(images), num_samples, replace=False)
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.title(f'True: {labels[idx].item()}\nPred: {predicted_labels[idx].item()}')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    train, test = fashionMNIST_loader()

    # 初始化权重和偏置
    w = torch.normal(0, 0.01, size=(28 * 28, 10), requires_grad=True)
    b = torch.zeros(10, requires_grad=True)

    # 训练模型
    train_model(train, test, w, b, LEARNING_RATE, NUM_EPOCHS)

    # 可视化预测结果
    visualize_predictions(test, w, b)
