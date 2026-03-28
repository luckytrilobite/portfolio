import torch
import torch.nn as nn
import torch.nn.functional as F

# 殘差塊
class block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(block, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 調整尺寸 讓運算後的尺寸跟xㄧ樣
        if self.downsample is not None:
            identity = self.downsample(x)
        # F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 訊號混合
        out += identity
        # H(x)
        out = F.relu(out)
        return out
#
def make_layer(in_ch, out_ch, num_blocks, stride=1):
    layers = []
    downsample = None  # ← 用 None, 不要 False

    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    layers.append(block(in_ch, out_ch, stride=stride, downsample=downsample))
    for _ in range(1, num_blocks):
        layers.append(block(out_ch, out_ch))

    return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 初始層
        # ****************************
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # ****************************
        # 殘差層
        # ****************************
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # ****************************
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x