import torch
from torch import nn, optim, Tensor

from torch.nn import BatchNorm1d, BatchNorm2d
from training_tools import fashionMNIST_loader, Trainer


class LeNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), BatchNorm2d(6, momentum=0.9), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), BatchNorm2d(16, momentum=0.9), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120), BatchNorm1d(120, momentum=0.9), nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84), BatchNorm1d(84, momentum=0.9), nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x) -> Tensor:
        return self.model(x)


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCHS_NUM = 12
    LEARNING_RATE = 0.9

    model = LeNetNorm()
    train_loader, test_loader = fashionMNIST_loader(BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), LEARNING_RATE)
    platform = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with Trainer(model, train_loader, test_loader, criterion, optimizer, platform) as trainer:
        trainer.train(EPOCHS_NUM)
