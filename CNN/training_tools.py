import os
from typing import Literal
from typing import Tuple

import torch
import torchvision
from torch import Tensor, device, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

DATASETS_PATH = os.path.expanduser('~/DataSets')


def fashionMNIST_loader(batch_size, *, resize=None) -> Tuple[
    DataLoader[list[Tensor, Tensor]], DataLoader[list[Tensor, Tensor]]]:
    t = [transforms.ToTensor(),  # 将 PIL 图像或 numpy.ndarray 转换为 Tensor，并归一化到 [0, 1] 之间
         transforms.Normalize((0.5,), (0.5,))]  # 标准化操作，将数据均值和标准差设为 0.5
    if resize:
        t.insert(0, transforms.Resize(size=resize))  # 调整图像尺寸。整数，按比例将短边调整到指定的大小；元组，指定宽高
    trans_pipe = transforms.Compose(t)  # 转换操作仅在数据被访问时动态应用
    dataset_train = torchvision.datasets.FashionMNIST(
        root=DATASETS_PATH, train=True, transform=trans_pipe, download=True)
    dataset_test = torchvision.datasets.FashionMNIST(
        root=DATASETS_PATH, train=False, transform=trans_pipe, download=True)
    iter_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=4)
    iter_test = DataLoader(dataset_test, batch_size, shuffle=True, num_workers=4)
    return iter_train, iter_test


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader[list[Tensor, Tensor]],
                 test_loader: DataLoader[list[Tensor, Tensor]],
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 platform: device,
                 log_dir: str = './runs'):
        self.model: nn.Module = model.to(platform)
        self.train_loader: DataLoader[list[Tensor, Tensor]] = train_loader
        self.test_loader: DataLoader[list[Tensor, Tensor]] = test_loader
        self.criterion: nn.Module = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.platform: device = platform
        self.writer = SummaryWriter(log_dir)

    def _run_epoch(self, data_loader: DataLoader[list[Tensor, Tensor]], mode: Literal['train', 'eval']):
        assert mode in ['train', 'eval'], "mode must be either 'train' or 'eval'"
        total_loss = correct_count = total_count = 0.0
        is_train: bool = mode == 'train'

        self.model.train() if is_train else self.model.eval()

        with torch.set_grad_enabled(is_train):
            for features, labels in data_loader:
                features: Tensor = features.to(self.platform)
                labels: Tensor = labels.to(self.platform)

                outputs: Tensor = self.model(features)  # 前向传播
                loss: Tensor = self.criterion(outputs, labels)

                if is_train:  # 反向传播与优化（只有在训练时）
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                _, predicted = outputs.max(dim=1)
                total_count += labels.size(0)
                correct_count += (predicted == labels).sum().item()

        accuracy = 100 * correct_count / total_count
        loss = total_loss / len(data_loader)
        return accuracy, loss

    def train(self, epochs_num: int):
        for epoch in range(epochs_num):
            accuracy_train, loss_train = self._run_epoch(self.train_loader, mode='train')
            accuracy_test, loss_test = self.evaluate()  # 调用评估方法

            # TODO: 记录训练损失和精度。可以考虑使用装饰器输出或 TensorBoard 数据
            self.writer.add_scalar('Accuracy/train', accuracy_train, global_step=epoch + 1)
            self.writer.add_scalar('Loss/train', loss_train, global_step=epoch + 1)
            self.writer.add_scalar('Accuracy/test', accuracy_test, global_step=epoch + 1)
            self.writer.add_scalar('Loss/test', loss_test, global_step=epoch + 1)

            print(f'第 {epoch + 1:03}/{epochs_num} 轮，'
                  f'训练损失：{loss_train:.4f}，训练精度：{accuracy_train:05.2f}%，'
                  f'测试损失：{loss_test:.4f}，测试精度：{accuracy_test:05.2f}%')

    def evaluate(self):
        accuracy_test, loss_test = self._run_epoch(self.test_loader, mode='eval')
        return accuracy_test, loss_test

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
