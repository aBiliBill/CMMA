import torch.nn as nn
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Mutil_module(nn.Module):
    def __init__(self, channel, reduction=16,spatial_kernel_size=7,flag=False):
        super(Mutil_module, self).__init__()
        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.bn = nn.BatchNorm2d(channel // 4)



    def forward(self, x):
        branches = getattr(self, 'active_branches', 3)
        if branches == 3:
            x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        elif branches == 2:
            x1 = (self.FC11(x) + self.FC12(x))/2
        elif branches == 1:
            x1 = self.FC11(x)
        else:
            raise ValueError('active_branches must be 1, 2 or 3')
        x1 = self.bn(x1)
        x1 = F.relu(x1)

        return x1
