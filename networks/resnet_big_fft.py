"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class FrequencyFeatureExtractor(nn.Module):
    def __init__(self, in_channels, is_last_layer=False):
        super(FrequencyFeatureExtractor, self).__init__()
        self.is_last_layer = is_last_layer
        
        # Bộ lọc Conv 3x3 để "ngửi" hình dạng FFT (vạch, tia, vòng tròn)
        # Giữ nguyên số channel để xử lý song song từng kênh tần số
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        if not self.is_last_layer:
            # Nhánh điều biến (Modulation) cho các tầng giữa
            # Downsample không gian xuống 1 nửa để nén thông tin
            self.downsample = nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=2, padding=1)

            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels),
                nn.Sigmoid()
            )
        else:
            # Nhánh tạo Embedding cho tầng cuối
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            # Kết quả trả về sẽ là vector embedding [B, in_channels]

    def forward(self, x, mag_down=None):
        B, C, H, W = x.shape
        
        # 1. Chuyển sang miền tần số
        # rfft2 cho kết quả [B, C, H, W//2 + 1]
        ffted = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(ffted) 
        
        # 2. Học hình dáng từ Magnitude qua Conv 3x3
        # FFT Magnitude có thể coi như một "ảnh" đặc trưng tần số
        if mag_down is not None:
            mag_features = self.freq_conv(mag) + mag_down
        else:
            mag_features = self.freq_conv(mag)
            
        if not self.is_last_layer:
            # --- TRƯỜNG HỢP TẦNG GIỮA ---
            # Downsample đặc trưng tần số
            mag_down = self.downsample(mag_features)

            # Tạo trọng số điều biến (Channel Attention dựa trên tần số)
            weights = self.gate(mag_features).view(B, C, 1, 1)
            
            # Nhân để điều biến Spatial Features (tránh loạn thông tin)
            return x *(1 + weights), mag_down
        
        else:
            # --- TRƯỜNG HỢP TẦNG CUỐI ---
            # Nén toàn bộ thông tin hình học tần số thành 1 vector
            embedding = self.global_pool(mag_features)
            return self.flatten(embedding)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    
    
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.fft_layer1 = FrequencyFeatureExtractor(64)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.fft_layer2 = FrequencyFeatureExtractor(128)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.fft_layer3 = FrequencyFeatureExtractor(256)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fft_layer4 = FrequencyFeatureExtractor(512, is_last_layer=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # print(out.shape)
        out, mag_down = self.fft_layer1(out)
        out = self.layer2(out)
        out, mag_down = self.fft_layer2(out, mag_down)
        out = self.layer3(out)
        out, mag_down = self.fft_layer3(out, mag_down)
        out = self.layer4(out)
        mag_down = self.fft_layer4(out, mag_down)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        print(mag_down.shape, out.shape)
        vec = torch.cat([out, mag_down], dim=1)
        return vec


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 1024],
    'resnet34': [resnet34, 1024],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    

class MulSupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(MulSupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

if __name__ == "__main__":
    batch_size = 2
    input_channels = 3
    height, width = 224, 224
    dummy_input = torch.randn(batch_size, input_channels, height, width)
    
    model = SupConResNet('resnet18')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Forward
    output = model(dummy_input)
    loss = loss_fn(output, torch.randn_like(output)) # Giả lập loss

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Kiểm tra gradient từng lớp
    print(f"{'Layer Name':<50} | {'Grad Mean':<10} | {'Grad Max':<10}")
    print("-" * 75)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_abs_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"{name:<50} | {grad_abs_mean:.10f} | {grad_max:.10f}")
        else:
            print(f"{name:<50} | {'NO GRADIENT':<10}")