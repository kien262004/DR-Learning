"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        # Tính mean và var trên chiều Channel
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # Áp dụng weight và bias (reshape để broadcast đúng channel)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


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
        

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        
        # 1. Depthwise Convolution: dùng groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = LayerNorm2d(in_channels)
        
        # 2. Pointwise Convolution: dùng kernel 1x1
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn2 = LayerNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Bước 1: Trích xuất không gian (Spatial features)
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Bước 2: Kết hợp kênh (Channel features)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class LowBranch(nn.Module):
    def __init__(self, planes):
        super(LowBranch, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = LayerNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln2 = LayerNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += x
        return out
    
class HighBranch(nn.Module):
    def __init__(self, planes):
        super(HighBranch, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = LayerNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln2 = LayerNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += x
        return out
    
class Block(nn.Module):
    
    def __init__(self, planes, is_first=False, is_last=False):
        super(Block, self).__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.low_branch = LowBranch(planes)
        if not is_first:
            self.high_branch = HighBranch(planes)
        
        if is_last:
            self.conv1 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=1, padding=1, bias=False)
            self.ln1 = LayerNorm2d(planes*2)
            self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=1, padding=1, bias=False)
            self.ln2 = LayerNorm2d(planes*2)


        
    def forward(self, x, x_high=None):
        out = self.low_branch(x)
        if not self.is_first:
            out_high = self.high_branch(x_high)
            if self.is_last:
                out = F.relu(self.ln1(self.conv1(torch.concat([out, out_high], dim=1))))
                out = self.ln2(self.conv2(out))
                return out
            return out, out_high
        return out
    
class DWTDown(nn.Module):
    def __init__(self, planes):
        super(DWTDown, self).__init__()
        self.xfm = DWTForward(J=1, mode='zero', wave='haar')
        self.fusion_high = DWConv(planes*3, planes)
    
    def forward(self, x):
        x_low, x_high = self.xfm(x)
        x_high = x_high[0]
        b, c, n, h, w = x_high.shape
        x_high = x_high.view(b, c*n, h, w)
        x_high = self.fusion_high(x_high)
        return x_low, x_high
    
    
class SFNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3):
        super(SFNet, self).__init__()
        
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.ln1 = LayerNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], is_first=True)
        self.layer2 = self._make_layer(block, 64, num_blocks[1])
        self.layer3 = self._make_layer(block, 128, num_blocks[2])
        self.layer4 = self._make_layer(block, 256, num_blocks[3])
        self.down1 = DWTDown(64)
        self.down2 = DWTDown(128)
        self.down3 = DWTDown(256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, num_blocks, is_first=False):
        layers = []
        for i in range(num_blocks):
            is_last = True if (not is_first and i == num_blocks - 1) else False
            layers.append(block(planes, is_first=is_first, is_last=is_last))
        return nn.ModuleList(layers)
    
    def run_layer(self, layer, x, x_high=None):
        for idx, block in enumerate(layer):
            if x_high is None:
                x = block(x)
            else:
                if idx != len(layer) - 1:
                    x, x_high = block(x, x_high)
                else:
                    x = block(x, x_high)
        return x
    
    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.run_layer(self.layer1, out)
        out, out_high = self.down1(out)
        out = self.run_layer(self.layer2, out, out_high)
        out, out_high = self.down2(out)
        out = self.run_layer(self.layer3, out, out_high)
        out, out_high = self.down3(out)
        out = self.run_layer(self.layer4, out, out_high)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def sfnet(**kwargs):
    return SFNet(Block, [2, 2, 2, 2], **kwargs)


model_dict = {
    'sfnet': [sfnet, 512],
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
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
    def __init__(self, name='sfnet', head='mlp', feat_dim=128):
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
    
    model = SupConResNet()
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