from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor


class ResNetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, *,
            stride: int = 1,
            bottleneck_ratio: int = 4,
            downsample: Optional[nn.Module] = None):
        """
        后激活的残差块

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步长
        :param bottleneck_ratio: 瓶颈比率，控制中间层宽度
        :param downsample: 下采样层，用于维度匹配
        """
        super().__init__()

        # 计算中间层通道数
        assert bottleneck_ratio >= 1, f'瓶颈比率 ({bottleneck_ratio}) 应大于或等于 1'
        mid_channels = out_channels // bottleneck_ratio if bottleneck_ratio > 1 else out_channels

        if bottleneck_ratio > 1:
            # 1x1 Conv -> 3x3 Conv -> 1x1 Conv
            assert mid_channels > 0, f'中间通道数 ({mid_channels}) 必须大于 0'

            self.main_path = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),

                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),

                nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # 3x3 Conv -> 3x3 Conv
            self.main_path = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),

                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 匹配维度用于跳跃连接
        if downsample is not None:
            self.shortcut = downsample
        elif stride != 1 or in_channels != out_channels:  # 空间尺寸、通道数发生变化时，再调整维度
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()  # 恒等映射

    def forward(self, x: Tensor) -> Tensor:
        out = self.main_path(x) + self.shortcut(x)  # 主路径与跳跃连接相加
        out = F.relu(out)  # 激活加和后的结果
        return out


class ResNeXtBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int, *,
            stride: int = 1,
            groups: int = 1,
            bottleneck_ratio: int = 4,
            downsample: Optional[nn.Module] = None):
        """
        预激活的分组残差块

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param stride: 卷积步长
        :param groups: 分组卷积的组数
        :param bottleneck_ratio: 瓶颈比率，控制中间层宽度
        :param downsample: 下采样层，用于维度匹配
        """
        super().__init__()

        # 计算中间层通道数
        assert bottleneck_ratio >= 1, f'瓶颈比率 ({bottleneck_ratio}) 应大于或等于 1'
        mid_channels = out_channels // bottleneck_ratio if bottleneck_ratio > 1 else out_channels

        if bottleneck_ratio > 1:
            # 1x1 Conv -> 3x3 Conv -> 1x1 Conv
            assert mid_channels % groups == 0, f'中间通道数 ({mid_channels}) 必须能被卷积分组数 ({groups}) 整除'
            assert mid_channels > 0, f'中间通道数 ({mid_channels}) 必须大于 0'

            self.main_path = nn.Sequential(
                nn.BatchNorm2d(in_channels), nn.ReLU(),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),

                nn.BatchNorm2d(mid_channels), nn.ReLU(),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=groups,
                          bias=False),

                nn.BatchNorm2d(mid_channels), nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            # 3x3 Conv -> 3x3 Conv
            assert out_channels % groups == 0, f'输出通道数 ({out_channels}) 必须能被卷积分组数 ({groups}) 整除'

            self.main_path = nn.Sequential(
                nn.BatchNorm2d(in_channels), nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups,
                          bias=False),

                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
            )

        # 匹配维度用于跳跃连接
        if downsample is not None:
            self.shortcut = downsample
        elif stride != 1 or in_channels != out_channels:  # 空间尺寸、通道数发生变化时，再调整维度
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.main_path(x) + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResNetBlock(64, 64, bottleneck_ratio=1),
            ResNetBlock(64, 64, bottleneck_ratio=1),

            # 组 1
            ResNetBlock(64, 128, stride=2, bottleneck_ratio=1),
            ResNetBlock(128, 128, bottleneck_ratio=4),

            # 组 2
            ResNetBlock(128, 256, stride=2, bottleneck_ratio=1),
            ResNetBlock(256, 256, bottleneck_ratio=4),

            # 组 3
            ResNetBlock(256, 512, stride=2, bottleneck_ratio=1),
            ResNetBlock(512, 512, bottleneck_ratio=4),

            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes),
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


class ResNeXt(nn.Module):
    def __init__(self, num_classes: int, cardinality: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ResNeXtBlock(64, 64, groups=1, bottleneck_ratio=1),
            ResNeXtBlock(64, 64, groups=1, bottleneck_ratio=1),

            # 组 1
            ResNeXtBlock(64, 128, stride=2, groups=cardinality, bottleneck_ratio=4),
            ResNeXtBlock(128, 128, groups=cardinality, bottleneck_ratio=4),

            # 组 2
            ResNeXtBlock(128, 256, stride=2, groups=cardinality, bottleneck_ratio=4),
            ResNeXtBlock(256, 256, groups=cardinality, bottleneck_ratio=4),

            # 组 3
            ResNeXtBlock(256, 512, stride=2, groups=cardinality, bottleneck_ratio=4),
            ResNeXtBlock(512, 512, groups=cardinality, bottleneck_ratio=4),

            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes),
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
    import torch
    from torch import optim

    from training_tools import fashionMNIST_loader, Trainer

    BATCH_SIZE = 256
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.01
    USE_RESNEXT = True

    model = ResNeXt(num_classes=10, cardinality=16) if USE_RESNEXT else ResNet(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
