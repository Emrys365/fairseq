import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchaudio.transforms as trans


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, m_channels=32, feat_dim=40, emb_dim=128, feat_type='fbank',
                 sr=16000):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.emb_dim = emb_dim
        self.feat_type = feat_type

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == 'fbank':
            self.feature_extract = trans.MelSpectrogram(sample_rate=sr, n_fft=512, win_length=win_len,
                                                        hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                                                        pad=0, n_mels=feat_dim)
        else:
            melkwargs = {
                'n_fft': 512,
                'win_length': win_len,
                'hop_length': hop_len,
                'f_min': 0.0,
                'f_max': sr // 2,
                'pad': 0
            }
            self.feature_extract = trans.MFCC(sample_rate=sr, n_mfcc=feat_dim, log_mels=False,
                                              melkwargs=melkwargs)
        self.instance_norm = nn.InstanceNorm1d(feat_dim)

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels*8, num_blocks[3], stride=2)

        self.embedding = nn.Linear(math.ceil(feat_dim/8) * m_channels * 16 * block.expansion, emb_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_feat(self, x):
        with torch.no_grad():
            x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
            if self.feat_type == 'fbank':
                x = x.log()
            x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        pooling_mean = torch.mean(out, dim=-1)
        pooling_std = torch.sqrt(torch.var(out, dim=-1) + 1e-10)
        out = torch.cat((torch.flatten(pooling_mean, start_dim=1),
                         torch.flatten(pooling_std, start_dim=1)), 1)

        embedding = self.embedding(out)
        return embedding


def ResNet34(feat_dim, emb_dim, m_channels=32, feat_type='fbank', sr=16000):
    return ResNet(BasicBlock, [3,4,6,3], feat_dim=feat_dim, emb_dim=emb_dim, m_channels=m_channels,
                  feat_type=feat_type, sr=sr)


if __name__ == '__main__':
    net = ResNet34(40, 256)
    x = torch.randn(2, 16000)
    out = net(x)
    print(out.shape)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)



