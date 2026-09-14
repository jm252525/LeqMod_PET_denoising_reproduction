import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Unet(nn.Module):
    def __init__(self, inshape, nb_features=None):
        super().__init__()
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        self.enc_nf, self.dec_nf = nb_features

        # configure encoder (down-sampling path)
        prev_nf = 1
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvNormActi_Double(ndims, prev_nf, nf, stride=2))
            prev_nf = nf
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvNormActi_Double(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 1
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvActi(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # encoder
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))
        # decoder
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)
        # extra convs at output
        for layer in self.extras:
            x = layer(x)

        return x

class ConvActi(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = getattr(nn, 'Conv%dd' % ndims)(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out

class ConvNormActi(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = getattr(nn, 'Conv%dd' % ndims)(in_channels, out_channels, 3, stride, 1)
        self.norm = getattr(nn, 'InstanceNorm%dd' % ndims)(out_channels, affine=False)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.activation(self.norm(self.conv(x)))
        return out

class ConvNormActi_Double(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = getattr(nn, 'Conv%dd' % ndims)(in_channels, out_channels, 3, stride, 1)
        self.norm = getattr(nn, 'InstanceNorm%dd' % ndims)(out_channels, affine=False)
        self.activation = nn.LeakyReLU(0.2)

        self.conv1 = getattr(nn, 'Conv%dd' % ndims)(out_channels, out_channels, 3, 1, 1)
        self.norm1 = getattr(nn, 'InstanceNorm%dd' % ndims)(out_channels, affine=False)
        self.activation1 = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.activation(self.norm(self.conv(x)))
        out = self.activation1(self.norm1(self.conv1(out)))
        return out

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, nf=32, norm_layer='IN'):
        super(Discriminator, self).__init__()
        model = []
        model += [LeakyReLUConv3d(in_channels, nf, kernel_size=4, stride=2, padding=1)]
        model += [LeakyReLUConv3d(nf, nf * 2, kernel_size=4, stride=2, padding=1, norm=norm_layer)]
        model += [nn.Dropout(p=0.5)]
        model += [LeakyReLUConv3d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, norm=norm_layer)]
        model += [LeakyReLUConv3d(nf * 4, nf * 8, kernel_size=4, stride=1, norm=norm_layer)]
        model += [nn.Conv3d(nf * 8, 1, kernel_size=1, stride=1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out

class LeakyReLUConv3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv3d, self).__init__()
        model = []
        model += [nn.ReplicationPad3d(padding)]
        if sn:
            pass
        else:
            model += [spectral_norm(nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True))]
        if norm == 'IN':
            model += [nn.InstanceNorm3d(n_out, affine=False)]
        elif norm == 'BN':
            model += [nn.BatchNorm3d(n_out)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)