import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            #nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect', groups=in_features),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            #nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3, padding=1, padding_mode='reflect', groups=in_features),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class Generator(nn.Module):
    #? NOTE: The Generator model is a U-Net architecture with skip connections;
    def __init__(self, input_nc: int, output_nc: int, n_residual_blocks: int = 9):
        super(Generator, self).__init__()
        # Downsampling parameters
        in_features = 64
        out_features = in_features*2
        resid_in_features = 4*in_features
        upsampling_out_features = 2*in_features # half of the above
        self.model = nn.Sequential(
            # Initial convolution block
            #nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, in_features, kernel_size=7, padding=3, padding_mode='reflect', groups=input_nc),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            # downsampling blocks with output dims being successively doubled
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1, groups=in_features),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_features, 2*out_features, 3, stride=2, padding=1, groups=out_features),
            nn.InstanceNorm2d(2*out_features),
            nn.ReLU(inplace=True),
            # residual blocks
            *[ResidualBlock(resid_in_features) for _ in range(n_residual_blocks)],
            # upsampling blocks with output dims being successively halved
            nn.ConvTranspose2d(resid_in_features, upsampling_out_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(upsampling_out_features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(upsampling_out_features, upsampling_out_features//2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(upsampling_out_features//2),
            nn.ReLU(inplace=True),
            # output layers
            #nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, output_nc, 7, padding=3, padding_mode='reflect', groups=in_features),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, mask=None):
        generated = x.clone()
        generated = self.model(generated)
        # for layer in self.model:
        #     print("layer: ", layer)
        #     generated = layer(generated)
        #print("generated shape: ", generated.shape)
        #print("x shape: ", x.shape)
        if mask is not None:
            #print("mask shape: ", mask.shape)
            # Apply the mask to restrict changes only to the masked regions
                #~ maybe replace with boolean masking later
            return mask * generated + (1 - mask) * x
        return generated


class Discriminator(nn.Module):
    def __init__(self, input_nc: int):
        super(Discriminator, self).__init__()
        # A bunch of convolutions one after another
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1, groups=input_nc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # FCN classification layer
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x: torch.Tensor):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)