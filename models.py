"""
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
torch.nn.InstanceNorm2d()
"""
import torch
import torch.nn as nn
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])
    def forward(self, img):
        return self.vgg19_54(img)

class ResizeBlock(nn.Module):
    def __init__(self,input,scale):
        self.num = input.shape[0]
        self.scale = scale
        self.x = input
    def forward(self):
        output = nn.Upsample(input=self.x,scale_factor=self.scale,mode='bicubic')
        return output

#feature extraction blocks
class SRBlock(nn.Module):
    def __init__(self,input_channels,output_channels,num):
        super(SRBlock,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,kernel_size=3,padding=1,padding_mode='reflect',stride=1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.block = nn.Sequential(
            nn.Conv2d(output_channels,output_channels,kernel_size=3,padding=1,padding_mode='reflect',stride=1),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.num = num
    def forward(self,x):
        bs = []
        x = self.block1(x)
        for i in range(1,self.num):
            x = self.block(x)
            if i != self.num:
                bs.append(x)
        for layer in bs:
            x += layer
        return x

#upsample network or reconstruction network(version: g)
class UpsampleNetwork(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpsampleNetwork,self).__init__()
        self.a1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding_mode='reflect',padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(out_channels*2,out_channels,kernel_size=3,stride=1,padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=4,stride=1,padding='same'),nn.PixelShuffle(upscale_factor=4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_channels//16,3,kernel_size=1,stride=1)
        )


    def forward(self,x):
        x1 = self.a1(x)
        x2 = self.b1(x)
        x2 = self.b2(x2)
        y1 = torch.concat([x1,x2],axis=1)
        y1 = self.c1(y1)
        y1 = self.c2(y1)
        return y1

class OutputNetwork(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutputNetwork,self).__init__()
        self.a1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding_mode='reflect', padding=0),
            nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=1, padding='same'),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.outputnet = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(), nn.Conv2d(out_channels // 16, 3, kernel_size=3, stride=1, padding=1,padding_mode='reflect')
        )

    def forward(self,x):
        x1 = self.a1(x)
        x2 = self.b1(x)
        x2 = self.b2(x2)
        y1 = torch.concat([x1, x2], axis=1)
        # print("size of y1")
        # print(y1.size())
        y1 = self.c1(y1)
        y1 = self.c2(y1)
        # print("size of y1")
        # print(y1.size())
        y1 = self.outputnet(y1)
        return y1



class Generator(nn.Module):
    def __init__(self,mode,in_channels=3,channels=64,out_channels=3,n_blocks=12):
        super(Generator,self).__init__()

        self.mode = mode
        self.hinput = nn.Sequential(
            nn.Conv2d(in_channels,channels,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channels,channels,kernel_size=4,stride=2,padding=1,padding_mode='reflect')
        )
        self.linput = nn.Sequential(
            nn.Conv2d(in_channels,channels,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        )
        # feature extraction network
        self.srblock = SRBlock(channels,channels,10)
        self.upsamplenet = UpsampleNetwork(64,64)
        self.outputnet = OutputNetwork(64,64)

    def forward(self,x):
        if self.mode == 'hl':
            x = self.hinput(x)
        else:
            x = self.linput(x)
        x = self.srblock(x)
        if self.mode == 'lh':
            x = self.upsamplenet(x)
        else:
            x = self.outputnet(x)
        return x




class Discriminator(nn.Module):
    def __init__(self,mode,input_channels=3,channels=64):
        super(Discriminator,self).__init__()
        self.mode = mode
        self.output_shape = (1,8,8)
        self.inputh = nn.Sequential(
            nn.Conv2d(input_channels,channels,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels,channels,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU()
        )

        self.inputl = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU()
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=4,padding='same',stride=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(channels,channels*2,kernel_size=4,padding=1,padding_mode='reflect',stride=2),
            nn.InstanceNorm2d(channels*2),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(channels*2,channels*2,kernel_size=4,padding='same',stride=1),
            nn.InstanceNorm2d(channels*2),
            nn.LeakyReLU(negative_slope=0.1)
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(channels*2,channels*4,kernel_size=4,padding=1,stride=2),
            nn.InstanceNorm2d(channels*4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.d5 = nn.Sequential(
            nn.Conv2d(channels*4, channels*4, kernel_size=4, padding='same', stride=1),
            nn.InstanceNorm2d(channels * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.d6 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 8, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(channels * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.d7 = nn.Sequential(
            nn.Conv2d(channels * 8, channels * 8, kernel_size=4, padding='same', stride=1),
            nn.InstanceNorm2d(channels * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.d8 = nn.Sequential(
            nn.Conv2d(channels * 8, channels*8, kernel_size=4, padding='same', stride=1),
            nn.InstanceNorm2d(channels * 8),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.fcn = nn.Sequential(
            nn.Conv2d(channels*8,channels,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(channels,1,kernel_size=1,stride=1,padding=0)
        )



        self.discriminatorl = nn.Sequential(
            self.inputl,self.d1,self.d2,self.d3,self.d4,self.d5,self.d6,self.d7,self.d8,self.fcn
        )
        self.discriminatorh = nn.Sequential(
            self.inputh,self.d1,self.d2,self.d3,self.d4,self.d5,self.d6,self.d7,self.d8,self.fcn
        )

    def forward(self,x):
        # if self.mode=='h':
        #     x = self.discriminatorh(x)
        # else:
        #     x = self.discriminatorl(x)
        if self.mode=='h':
            x = self.inputh(x)
            x = self.d1(x)
            x = self.d2(x)
            x = self.d3(x)
            x = self.d4(x)
            x = self.d5(x)
            x = self.d6(x)
            x = self.d7(x)
            # print("size of x")
            # print(x.size())
            x = self.d8(x)
            x = self.fcn(x)
        else:
            x = self.inputl(x)
            x = self.d1(x)
            x = self.d2(x)
            x = self.d3(x)
            x = self.d4(x)
            x = self.d5(x)
            x = self.d6(x)
            x = self.d7(x)
            x = self.d8(x)
            x = self.fcn(x)
        return x

class SRGANDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(SRGANDiscriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

class ESRGANDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(ESRGANDiscriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


###   srgan  generator/discriminator
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    #in_channels:RGB图像:out_channels
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        #卷积是方形,只需要一个整数边长kernel_size 步长stride=1 padding=4相当于等宽卷积
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks

        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        # print("out1:")
        # print(out1.size())
        # print("\n")
        # print("out2:")
        # print(out2.size())
        # print("\n")
        # print("out:")
        # print(out.size())
        # print("\n")
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
