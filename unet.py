import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(*[
            nn.Conv2d(input_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.65),
            nn.Conv2d(output_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.65),
        ])

    def forward(self, x):
        return self.layers(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3,
    encoder_f=[64, 128, 256, 512], decoder_f=[1024, 512, 256], out_channels=64):
        super(UNet, self).__init__()

        self.module_1 = nn.Sequential(*[
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.65),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.65),
        ])


        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = nn.ModuleList([ConvBlock(channels, 2*channels) for channels in encoder_f])

        out_sizes = [int(x/2) for x in decoder_f]

        self.decoder_transpose = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder_f])
        
        self.decoder = nn.ModuleList(
            [ConvBlock(3*channels_out, channels_out) for channels_out in out_sizes])
        
        self.final_decoder_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

        self.module_2 = nn.Sequential(*[
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.65),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.65),

            nn.Conv2d(64, out_channels, 1),
            nn.ReLU(),
        ])


    def forward(self, x):

        act = [self.module_1(x)]

        for module in self.encoder:
            act.append(module(self.pool(act[-1])))
        
        x1 = act.pop(-1)
        for conv, conv_t in zip(self.decoder, self.decoder_transpose):
            skip_connection = act.pop(-1)
            x1 = conv(
                torch.cat((skip_connection, conv_t(x1)), 1)
            )
        seg = self.module_2(
            torch.cat((act[-1], self.final_decoder_layer(x1)), 1)
        )
        return seg

