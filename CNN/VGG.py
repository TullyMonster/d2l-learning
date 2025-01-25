import torch
from torch import Tensor, nn, optim

from training_tools import fashionMNIST_loader, Trainer


class VGGBlock(nn.Module):
    """
    VGG 块

    由两层使用 ReLU 激活函数的 3x3 卷积层，后接一个 2x2 步幅为 2 的最大池化层组成
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x) -> Tensor:
        return self.block(x)


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.extractor = nn.Sequential(
            # 输出通道数由 16 逐层翻倍至 128
            VGGBlock(1, 16),  # 灰度图
            VGGBlock(16, 32),
            VGGBlock(32, 64),
            VGGBlock(64, 128),
            VGGBlock(128, 128),
        )

        self.classifier = nn.Sequential(
            # 输入图像为 224x224，经过 5 个 VGGBlock 后尺寸变为 7x7
            nn.Linear(7 * 7 * 128, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x) -> Tensor:
        x = self.extractor(x)
        x = torch.flatten(x, 1)  # 展平，准备进入全连接层
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    BATCH_SIZE = 128
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.05

    model = VGG11(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=224)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
