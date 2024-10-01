import torch.nn as nn
from unet import UNet

class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()
        self.encoder = UNet(in_channels=3, encoder_f=[64, 128, 256, 512],
                                    decoder_f=[1024,512, 256], out_channels=64)
        self.decoder = UNet(in_channels=64, encoder_f=[64, 128, 256, 512],
                                    decoder_f=[1024,512, 256], out_channels=3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()

    def forward_encoder(self, x):
        x = self.encoder(x)
        return self.softmax(x)

    def forward_decoder(self, seg):
        x = self.decoder(seg)
        return self.sigmoid(x)

    def forward(self, x):
        seg = self.forward_encoder(x)
        x = self.forward_decoder(seg)
        return seg, x