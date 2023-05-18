import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import clip
import scipy.io as sio
from einops import rearrange, repeat
from torch.nn import PixelUnshuffle
import numpy as np

## Channel Attention (CA) Layer
class CA_Block(nn.Module):
    def __init__(self,in_channels,reduction=16):
        super(CA_Block,self).__init__()
        self.se_module=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//reduction,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction,in_channels,kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=x*self.se_module(x)
        return x
    
## Residual Channel Attention Block (RCAB)
class RCAB_Block(nn.Module):
    def __init__(self,in_channels,reduction):
        super(RCAB_Block,self).__init__()
        self.rcab=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            CA_Block(in_channels, reduction)
        )
    def forward(self,x):
        return x+self.rcab(x)
    
## Double Res Block
class DR_Block(nn.Module):
    def __init__(self, in_channels, reduction = 16, num_crab = 2):
        super(DR_Block,self).__init__()
        self.rg_block=[RCAB_Block(in_channels,reduction) for _ in range(num_crab)]
        self.rg_block.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.rg_block=nn.Sequential(*self.rg_block)
    def forward(self,x):
        return x+self.rg_block(x)

class CSNET(nn.Module):
    def __init__(self,
                 sr, 
                 block_size=32,
                 in_channels=3,
                 out_channels=5,
                 bias=False):
        super(CSNET,self).__init__()
        self.sr = sr
        self.block_size = block_size
        self.in_channels = in_channels
        self.base = 64

        self.xdim = int(self.block_size*self.block_size*self.in_channels)
        ydim = int(sr*self.xdim)

        Phi_init = np.random.normal(0.0, (1 / self.xdim) ** 0.5, size=(ydim, self.xdim))
        self.Phi = nn.Parameter(torch.from_numpy(Phi_init).float(), requires_grad=True)
        self.Phi_T = nn.Parameter(torch.from_numpy(np.transpose(Phi_init)).float(), requires_grad=True)
        self.conv_in = nn.Conv2d(in_channels, self.base, kernel_size=3, padding=1)
        self.drb1 = DR_Block(self.base)
        self.drb2 = DR_Block(self.base)
        self.drb3 = DR_Block(self.base)
        self.conv_out = nn.Conv2d(self.base, out_channels, kernel_size=3, padding=1)
        
    def coding(self,inputs):
        out = self.conv_in(inputs)
        out = self.drb1(out)
        out = torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.drb2(out)
        out = torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.drb3(out)
        out = self.conv_out(out)
        return out

    def forward(self,x):
        batch_size = x.size()[0]
        self.image_size = x.size()[3]
        y = self.sampling(x)
        output = self.initial(y,batch_size)
        output = self.coding(output)
        return output
    
    def sampling(self, inputs):
        x = self.vectorize(inputs)
        y = torch.matmul(self.Phi, x)
        return y
    
    def initial(self, y, batch_size):
        x = torch.matmul(self.Phi_T, y)
        out = self.devectorize(x, batch_size)
        return out

    def vectorize(self,inputs):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=self.block_size, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=self.block_size, dim=2), dim=0)
        inputs = torch.reshape(inputs, [-1, self.xdim])
        inputs = torch.transpose(inputs, 0, 1)
        return inputs

    def devectorize(self, inputs, batch_size):
        rows = self.image_size//self.block_size
        recon = torch.reshape(torch.transpose(inputs, 0, 1), [-1, self.in_channels, self.block_size, self.block_size])
        recon = torch.cat(torch.split(recon, split_size_or_sections=rows * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        return recon
    
    def encode(self, x):
        return self(x)

class GCSNET(nn.Module):
    def __init__(self,
                 sr, 
                 block_size=32,
                 in_channels=3,
                 out_channels=5,
                 bias=False):
        super(GCSNET,self).__init__()
        self.sr = sr
        self.block_size = block_size
        self.in_channels = in_channels
        self.base = 64

        self.xdim = int(self.block_size*self.block_size*self.in_channels)
        ydim = int(sr*self.xdim)
        mpath = './sampling_matrix/' + str(int(sr*100)) + '.mat'
        Phi_data = sio.loadmat(mpath)
        Phi_input = Phi_data['Phi']
        self.Phi = torch.from_numpy(Phi_input).to("cuda:0",torch.float32)


        self.Phi_T = nn.Parameter(torch.from_numpy(np.transpose(Phi_input)).float(), requires_grad=True)
        self.conv_in = nn.Conv2d(in_channels, self.base, kernel_size=3, padding=1)
        self.drb1 = DR_Block(self.base)
        self.drb2 = DR_Block(self.base)
        self.drb3 = DR_Block(self.base)
        self.conv_out = nn.Conv2d(self.base, out_channels, kernel_size=3, padding=1)
        
    def coding(self,inputs):
        out = self.conv_in(inputs)
        out = self.drb1(out)
        out = torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.drb2(out)
        out = torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        out = self.drb3(out)
        out = self.conv_out(out)
        return out

    def forward(self,x):
        batch_size = x.size()[0]
        self.image_size = x.size()[3]
        y = self.sampling(x)
        output = self.initial(y,batch_size)
        output = self.coding(output)
        return output
    
    def sampling(self, inputs):
        x = self.vectorize(inputs)
        y = torch.matmul(self.Phi, x)
        return y
    
    def initial(self, y, batch_size):
        x = torch.matmul(self.Phi_T, y)
        out = self.devectorize(x, batch_size)
        return out

    def vectorize(self,inputs):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=self.block_size, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=self.block_size, dim=2), dim=0)
        inputs = torch.reshape(inputs, [-1, self.xdim])
        inputs = torch.transpose(inputs, 0, 1)
        return inputs

    def devectorize(self, inputs, batch_size):
        rows = self.image_size//self.block_size
        recon = torch.reshape(torch.transpose(inputs, 0, 1), [-1, self.in_channels, self.block_size, self.block_size])
        recon = torch.cat(torch.split(recon, split_size_or_sections=rows * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        return recon
    
    def encode(self, x):
        return self(x)