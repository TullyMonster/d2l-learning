import torch
from torch import Tensor
from torch import nn
from torch import optim

from training_tools import fashionMNIST_loader, Trainer


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 每张图片为 28×28
            # 特征提取器 = 卷积层 + 激活函数 + 池化层
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),  # (BatchSize, 6, 28, 28)
            nn.AvgPool2d(kernel_size=2, stride=2),  # (BatchSize, 6, 14, 14)

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.Sigmoid(),  # (BatchSize, 16, 10, 10)
            nn.AvgPool2d(kernel_size=2, stride=2),  # (BatchSize, 16, 5, 5)

            # 分类器（展平后依次进入三个全连接层）
            nn.Flatten(),  # (BatchSize, 16×5×5)
            nn.Linear(in_features=16 * 5 * 5, out_features=120), nn.Sigmoid(),  # (BatchSize, 120)
            nn.Linear(in_features=120, out_features=84), nn.Sigmoid(),  # (BatchSize, 84)
            nn.Linear(in_features=84, out_features=10)  # (BatchSize, 10)
        )

    def forward(self, x) -> Tensor:
        return self.model(x)


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS_NUM = 200
    LEARNING_RATE = 0.3

    model = LeNet()
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
