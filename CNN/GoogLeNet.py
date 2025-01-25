from typing import Tuple

import torch
from torch import nn, Tensor, optim

from training_tools import fashionMNIST_loader, Trainer


class IncepBlock(nn.Module):
    def __init__(self, in_channels: int, c1_out: int, c2_out: Tuple[int, int], c3_out: Tuple[int, int], c4_out: int):
        super().__init__()

        self.channel1 = nn.Sequential(  # 路径一: 1×1 卷积
            nn.Conv2d(in_channels, c1_out, kernel_size=1), nn.ReLU()
        )

        self.channel2 = nn.Sequential(  # 路径二: 1×1 卷积 -> 3×3 卷积
            nn.Conv2d(in_channels, c2_out[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2_out[0], c2_out[1], kernel_size=3, padding=1), nn.ReLU()
        )

        self.channel3 = nn.Sequential(  # 路径三: 1×1 卷积 -> 5×5 卷积
            nn.Conv2d(in_channels, c3_out[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3_out[0], c3_out[1], kernel_size=5, padding=2), nn.ReLU()
        )

        self.channel4 = nn.Sequential(  # 路径四: 3×3 最大汇聚 -> 1×1 卷积
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4_out, kernel_size=1), nn.ReLU()
        )

    def forward(self, x) -> Tensor:
        output1 = self.channel1(x)
        output2 = self.channel2(x)
        output3 = self.channel3(x)
        output4 = self.channel4(x)
        output = torch.cat([output1, output2, output3, output4], dim=1)
        return output


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            IncepBlock(in_channels=192, c1_out=64, c2_out=(96, 128), c3_out=(16, 32), c4_out=32),
            IncepBlock(in_channels=256, c1_out=128, c2_out=(128, 192), c3_out=(32, 96), c4_out=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            IncepBlock(in_channels=480, c1_out=192, c2_out=(96, 208), c3_out=(16, 48), c4_out=64),
            IncepBlock(in_channels=512, c1_out=160, c2_out=(112, 224), c3_out=(24, 64), c4_out=64),
            IncepBlock(in_channels=512, c1_out=128, c2_out=(128, 256), c3_out=(24, 64), c4_out=64),
            IncepBlock(in_channels=512, c1_out=112, c2_out=(144, 288), c3_out=(32, 64), c4_out=64),
            IncepBlock(in_channels=528, c1_out=256, c2_out=(160, 320), c3_out=(32, 128), c4_out=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            IncepBlock(in_channels=832, c1_out=256, c2_out=(160, 320), c3_out=(32, 128), c4_out=128),
            IncepBlock(in_channels=832, c1_out=384, c2_out=(192, 384), c3_out=(48, 128), c4_out=128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_features=1024, out_features=num_classes)
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
    BATCH_SIZE = 128
    EPOCHS_NUM = 30
    LEARNING_RATE = 0.005

    model = GoogLeNet(num_classes=10)
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE, resize=96)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
