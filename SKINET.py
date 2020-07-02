from block import *
import cv2,torch
import numpy as np
from blue_noise_generation import*

class SKINET(nn.Module):
    def __init__(self, in_channels, out_channels, nf, gc=32, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 res_scale=0.2, act_type='leakyrelu', last_act=None, pad_type='reflection', norm_type=None,
                 negative_slope=0.2, n_prelu=1, inplace=False, scale_factor=1, mode='nearest', n_basic_block=3):
        super(SKINET, self).__init__()

        basic_block_layer = []
        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc, kernel_size, stride, dilation, groups,
                                                              bias, res_scale, act_type, last_act, pad_type, norm_type,
                                                              negative_slope, n_prelu, inplace)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                       norm_type, negative_slope, n_prelu, inplace)

        self.upsample = upsample_block(nf, nf, scale_factor=scale_factor)
        self.conv3 = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                       norm_type, negative_slope, n_prelu, inplace)

        self.conv4 = conv_block(nf, out_channels, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                                norm_type, negative_slope, n_prelu, inplace)
        self.conv5 = conv_block(nf, out_channels, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                                norm_type, negative_slope, n_prelu, inplace)
       ###

        self.conv1_1 = conv_block(in_channels, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                            norm_type, negative_slope, n_prelu, inplace)
        self.conv2_2 = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                                norm_type, negative_slope, n_prelu, inplace)
        self.conv3_3 = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                                norm_type, negative_slope, n_prelu, inplace)
        self.conv4_4 = conv_block(nf, out_channels, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                                norm_type, negative_slope, n_prelu, inplace)

        # weight_init
        self.conv_gabor = nn.Conv2d(nf, nf, kernel_size, stride, 1, dilation, groups, bias=False)
        self.conv_gabor2 = nn.Conv2d(nf, nf, kernel_size, stride, 1, dilation, groups, bias=False)
        self.conv_gabor3 = nn.Conv2d(nf, nf, kernel_size, stride, 1, dilation, groups, bias=False)

        self.conv1 = conv_block(in_channels, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                            norm_type, negative_slope, n_prelu, inplace)
        self.weight_noise = nn.Conv2d(nf, nf, 1, stride, 0, dilation, groups, bias)
        self.weight_noise2 = nn.Conv2d(nf, nf, 1, stride, 0, dilation, groups, bias)
        self.weight_noise3 = nn.Conv2d(nf, nf, 1, stride, 0, dilation, groups, bias)
        self.weight_gabor = nn.Conv2d(nf, nf, 1, stride, 0, dilation, groups, bias)
        self.weight_gabor2 = nn.Conv2d(nf, nf, 1, stride, 0, dilation, groups, bias)
        self.weight_gabor3 = nn.Conv2d(nf, nf, 1, stride, 0, dilation, groups, bias)
        self.gabor_conv = conv_block(nf, nf, 15, stride, dilation, groups, bias, act_type, pad_type,
                            norm_type, negative_slope, n_prelu, inplace)
        self.gabor_conv2 = conv_block(nf, nf, 9, stride, dilation, groups, bias, act_type, pad_type,
                                     norm_type, negative_slope, n_prelu, inplace)
        self.gabor_conv3 = conv_block(nf, nf, 9, stride, dilation, groups, bias, act_type, pad_type,
                                      norm_type, negative_slope, n_prelu, inplace)
        self.relu = nn.ReLU(inplace=inplace)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def weight_init(self):
        for m in self._modules:
            if m == "conv_gabor":
                gabor__init(self._modules[m])
            if m == "conv_gabor2":
                gabor__init(self._modules[m])
            if m == "conv_gabor3":
                gabor__init(self._modules[m] )

    def addnoise(self, x):
        noise = blue_noise(x.size()).to(self.device)
        noise = self.weight_noise(noise) #1x1 conv
        addednoise = torch.mul(noise, x)
        return addednoise

    def addnoise2(self, x):
        noise = blue_noise(x.size()).to(self.device)
        noise = self.weight_noise2(noise) #1x1 conv
        addednoise = torch.mul(noise, x)
        return addednoise

    def addnoise3(self, x):
        noise = blue_noise(x.size()).to(self.device)
        noise = self.weight_noise3(noise) #1x1 conv
        addednoise = torch.mul(noise, x)
        return addednoise

    def forward(self, x):

        # 1.Identity Content Enhancement Branch
        x1 = self.conv1(x)
        x2 = self.basic_block(x1)
        x3 = self.conv2(x2)
        x4 = self.basic_block(x3)
        x_branch1 = self.conv3(x1+x4)

        # 2.Texture Detail Generation Branch
        x1_1 = self.conv1_1(x)
        # Blue-noise Gabor Module 1
        x_noise = self.addnoise(x1_1)
        x_gabor = self.conv_gabor(x_noise)
        x_module = self.weight_gabor(x_gabor)
        x_out1= self.gabor_conv(x_module)
        # Module 2
        x_noise2 = self.addnoise2(x1_1)
        x_module2 = self.weight_gabor2(x_noise2)
        x_out2 = self.gabor_conv2(x_module2)
        # Module 3
        x_noise3 = self.addnoise3(x1_1)
        x_gabor3 = self.conv_gabor3(x_noise3)
        x_module3 = self.weight_gabor3(x_gabor3)
        x_out3 = self.gabor_conv3(x_module3)
        # concate
        x_out = torch.cat((x_out1, x_out2, x_out3),1)
        x_branch2 = self.relu(self.conv5(x_out))

        # 3.fuse
        x6 = self.conv4(x_branch1 + x_branch2)
        x7 = self.relu(self.conv5(x6))
        return x7

def gabor__init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        value = gabor_generate((m.kernel_size[0], m.kernel_size[0], m.in_channels,m.out_channels) )
        value = torch.from_numpy(value).float()
        value = value.permute(3,2,0,1)
        m.weight.data = value
        print(value.size())

def gabor_generate(shape, dtype=None):
    kernel_size_1, kernel_size_2, ch, num_filters = shape

    if kernel_size_1 % 2 == 1:
        kernel_size = kernel_size_1
    elif kernel_size_2 % 2 == 1:
        kernel_size = kernel_size_2
    else:
        kernel_size = kernel_size_1 + 1

    # evaluate parameter interval based on number of convolution filters
    range_sigma = np.arange(5, ((kernel_size / 2) + 1), (((kernel_size / 2) + 1) - 5) / (num_filters * 1.))
    range_lambda = np.array([kernel_size - 2] * num_filters)
    range_theta = np.arange(0, 360 + 1, (360 + 1) / (num_filters * 1.))
    range_gamma = np.arange(100, 300 + 1, ((300 + 1) - 100) / (num_filters * 1.))
    range_psi = np.arange(90, 360 + 1, ((360 + 1) - 90) / (num_filters * 1.))

    kernels = []
    for i in range(num_filters):
        g_sigma = range_sigma[i]
        g_lambda = range_lambda[i] + 2
        g_theta = range_theta[i] * np.pi / 180.
        g_gamma = range_gamma[i] / 100.
        g_psi = (range_psi[i] - 180) * np.pi / 180

        print
        'kern_size=' + str(kernel_size) + ', sigma=' + str(g_sigma) + ', theta=' + str(g_theta) + ', lambda=' + str(
            g_lambda) + ', gamma=' + str(g_gamma) + ', psi=' + str(g_psi)
        kernel = cv2.getGaborKernel((kernel_size, kernel_size),
                                    g_sigma,
                                    g_theta,
                                    g_lambda,
                                    g_gamma,
                                    g_psi)
        kernels = kernels + kernel.ravel().tolist() * ch

    kernels = np.array(kernels)

    return kernels.reshape(shape)

def blue_noise(size):
    noise = blue_noise_generation(size)
    return noise