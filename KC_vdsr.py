import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self,channel_s,kernel_s):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=channel_s, out_channels=kernel_s, kernel_size=3, stride=1,padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class KC_Net(nn.Module):
    def __init__(self,weights_sequence):
        super(KC_Net, self).__init__()
        layers=[]
        for lay in range(0,18):
            layer=weights_sequence[lay]
            layers.append(Conv_ReLU_Block(layer.shape[1],layer.shape[0]))
        self.residual_layer = nn.Sequential(*layers)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1,bias=False)
        self.output = nn.Conv2d(in_channels=weights_sequence[17].shape[0], out_channels=1, kernel_size=3, stride=1,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        count=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                temp_tensor=torch.tensor(weights_sequence[count],dtype=torch.float32)
                m.weight.data=temp_tensor
                count+=1
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
