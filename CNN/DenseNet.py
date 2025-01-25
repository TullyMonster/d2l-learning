import torch
from torch import Tensor, nn, optim

from training_tools import fashionMNIST_loader, Trainer


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, layers_num: int, growth_rate: int):
        """
        稠密块

        由多组 “BatchNorm → ReLU → Conv” 结构（稠密层，DenseLayer）组成，每循环一次这样的结构，通道数增长 k。输出结果在通道上完成拼接。

        :param in_channels: 输入特征图的通道数
        :param layers_num: 堆叠的稠密层 (DenseLayer) 数量
        :param growth_rate: 增长率 (k)。每个稠密层输出的新特征图通道数，将与先前层的拼接
        """
        super().__init__()

        self.growth_rate = growth_rate

        # 根据需要堆叠的稠密层数，动态创建
        self.dense_layers = nn.ModuleList([
            self._get_dense_layer(in_channels + growth_rate * i)
            for i in range(layers_num)
        ])

    def _get_dense_layer(self, connected_channels: int) -> nn.Sequential:
        """
        返回单个稠密层实例
        BatchNorm → ReLU → Conv(3x3)
        :param connected_channels: 该稠密层的输入通道数，等于初始输入的通道数加上之前所有层的增长率累积
        """
        dense_layer = nn.Sequential(
            nn.BatchNorm2d(connected_channels),
            nn.ReLU(),
            nn.Conv2d(connected_channels, self.growth_rate, kernel_size=3, padding=1, bias=False)
        )
        return dense_layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.dense_layers:
            out = layer(x)
            x = torch.cat((x, out), dim=1)  # 在通道维度拼接
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        过渡层

        通过 1x1 卷积减少通道数，并通过 2x2 平均池化层下采样。

        :param in_channels: 输入特征图的通道数
        :param out_channels: 输出特征图的通道数
        """
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1 卷积，减少通道数
            nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2 平均池化，下采样
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 组 1
            DenseBlock(in_channels=64, layers_num=4, growth_rate=32),
            TransitionLayer(in_channels=192, out_channels=96),

            # 组 2
            DenseBlock(in_channels=96, layers_num=4, growth_rate=32),
            TransitionLayer(in_channels=224, out_channels=112),

            # 组 3
            DenseBlock(in_channels=112, layers_num=4, growth_rate=32),
            TransitionLayer(in_channels=240, out_channels=120),

            # 组 4
            DenseBlock(in_channels=120, layers_num=4, growth_rate=32),

            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_features=248, out_features=num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x) -> Tensor:
        return self.model(x)


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.1

    model = DenseNet(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
