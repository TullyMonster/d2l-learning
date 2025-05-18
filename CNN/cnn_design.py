from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from ResNet import ResNeXtBlock


@dataclass
class StageConfig:
    """网络主体中，每个 Stage 的配置"""
    depth: int  # 块数量
    groups: int  # 分组卷积块的组数
    out_channels: int  # 输出通道数
    bottleneck_ratio: int  # 瓶颈比率


class AnyNet(nn.Module):
    def __init__(self, in_channels: int, stem_out_channels: int, num_classes: int, arch_params: list[StageConfig]):
        """
        :param in_channels: 输入数据的通道数
        :param stem_out_channels: 网络 stem 部分的输出通道数
        :param num_classes: 分类任务的类别数
        :param arch_params: 网络主体的架构参数列表
        """
        super().__init__()

        self.stem = self._create_stem(in_channels, stem_out_channels)  # 创建 Stem

        self.body = nn.Sequential()  # 创建 Body
        next_stage_in_channels = stem_out_channels

        for idx, stage_config in enumerate(arch_params):
            self.body.append(
                self._create_stage(next_stage_in_channels, stage_config.out_channels, depth=stage_config.depth,
                                   groups=stage_config.groups, bottleneck_ratio=stage_config.bottleneck_ratio)
            )
            next_stage_in_channels = stage_config.out_channels

        self.head = self._create_head(next_stage_in_channels, num_classes)  # 创建 Head（分类头）

        self._initialize_weights()

    @classmethod
    def _create_stem(cls, in_channels: int, out_channels: int) -> nn.Sequential:
        stem = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        return stem

    @classmethod
    def _create_stage(cls, in_channels: int, out_channels: int, depth: int, groups: int,
                      bottleneck_ratio: int) -> nn.Sequential:
        """
        创建由多个 ResNeXtBlock 块构成的 Stage

        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param depth: 当前 Stage 的 ResNeXtBlock 块数
        :param groups: 分组卷积的组数
        :param bottleneck_ratio: 瓶颈比率
        """
        stage = nn.Sequential()

        stage.append(  # 第一个块
            ResNeXtBlock(in_channels, out_channels, stride=2, groups=groups, bottleneck_ratio=bottleneck_ratio)
        )

        for _ in range(1, depth):
            stage.append(  # 剩余的块
                ResNeXtBlock(out_channels, out_channels, stride=1, groups=groups, bottleneck_ratio=bottleneck_ratio)
            )

        return stage

    @classmethod
    def _create_head(cls, in_channels: int, num_classes: int) -> nn.Sequential:
        head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(in_channels, num_classes))
        return head

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: 形状为 [B, C, H, W]
        :return: 形状为 [B, num_classes]
        """
        x = self.stem(x)  # Stem
        x = self.body(x)  # Body
        x = self.head(x)  # Head

        return x


class RegNetX32(AnyNet):
    def __init__(self, in_channels, num_classes: int):
        super().__init__(
            in_channels=in_channels, stem_out_channels=32, num_classes=num_classes,
            arch_params=[
                StageConfig(depth=4, groups=16, out_channels=32, bottleneck_ratio=1),
                StageConfig(depth=6, groups=16, out_channels=80, bottleneck_ratio=1)
            ]
        )


if __name__ == "__main__":
    import torch
    from torch import optim

    from training_tools import fashionMNIST_loader, Trainer

    BATCH_SIZE = 128
    EPOCHS_NUM = 10
    LEARNING_RATE = 0.05

    model = RegNetX32(in_channels=1, num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
