import torch
import torch.nn as nn
from mcdropout import MCDropout3D, MCDropout2D
from monai.networks.nets import UNet
from monai.networks.layers import Conv
from monai.networks.blocks import ResidualUnit

class UNet3D(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(UNet3D, self).__init__()

        self.enc1 = self._encoder_block(1, 32, dropout_prob)
        self.enc2 = self._encoder_block(32, 64, dropout_prob)
        self.enc3 = self._encoder_block(64, 128, dropout_prob)
        self.enc4 = self._encoder_block(128, 256, dropout_prob)

        self.dec1 = self._decoder_block(256, 128, dropout_prob)
        self.dec2 = self._decoder_block(256, 64, dropout_prob)
        self.dec3 = self._decoder_block(128, 32, dropout_prob)
        self.dec4 = self._decoder_block(64, 2, dropout_prob, final_layer=True)

    def forward(self, x):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        dec1_out = self.dec1(enc4_out)
        dec1_out = torch.cat([dec1_out, enc3_out], dim=1)
        dec2_out = self.dec2(dec1_out)
        dec2_out = torch.cat([dec2_out, enc2_out], dim=1)
        dec3_out = self.dec3(dec2_out)
        dec3_out = torch.cat([dec3_out, enc1_out], dim=1)
        out = self.dec4(dec3_out)
        
        return out

    @staticmethod
    def _encoder_block(in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            MCDropout3D(dropout_prob)
        )

    @staticmethod
    def _decoder_block(in_channels, out_channels, dropout_prob, final_layer=False):
        layers = [
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()         
        ]
        if final_layer:
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=1))
            layers.append(nn.Softmax())
        
        return nn.Sequential(*layers)

class DeepUNet2D(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(DeepUNet2D, self).__init__()

        self.conv_in = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.dropout_prob = dropout_prob
        self.enc1 = self._encoder_block(32, 64, dropout_prob)
        self.enc2 = self._encoder_block(64, 128, dropout_prob)
        self.enc3 = self._encoder_block(128, 256, dropout_prob)
        self.enc4 = self._encoder_block(256, 512, dropout_prob)
        self.enc5 = self._encoder_block(512, 1024, dropout_prob)
        self.enc6 = self._encoder_block(1024, 2048, dropout_prob) 
        self.enc7 = self._encoder_block(2048, 4096, dropout_prob)  

        self.dec1 = self._decoder_block(4096, 2048, dropout_prob) 
        self.dec2 = self._decoder_block(4096, 1024, dropout_prob)
        self.dec3 = self._decoder_block(2048, 512, dropout_prob)
        self.dec4 = self._decoder_block(1024, 256, dropout_prob)
        self.dec5 = self._decoder_block(512, 128, dropout_prob)
        self.dec6 = self._decoder_block(256, 64, dropout_prob)
        self.dec7 = self._decoder_block(128, 64, dropout_prob)

        self.conv_out = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 2, kernel_size=1),
                                      nn.Softmax(dim=1))
        self.relu = nn.ReLU()          

    def forward(self, x):
        conv_in_out = self.conv_in(x)
        relu = self.relu(conv_in_out)

        enc1_out = self.enc1(relu)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        enc6_out = self.enc6(enc5_out)
        enc7_out = self.enc7(enc6_out)

        enc_out = MCDropout2D(self.dropout_prob)(enc7_out)

        dec1_out = self.dec1(enc_out)
        dec1_out = torch.cat([dec1_out, enc6_out], dim=1)

        dec2_out = self.dec2(dec1_out)
        dec2_out = torch.cat([dec2_out, enc5_out], dim=1)

        dec3_out = self.dec3(dec2_out)
        dec3_out = torch.cat([dec3_out, enc4_out], dim=1)

        dec4_out = self.dec4(dec3_out)
        dec4_out = torch.cat([dec4_out, enc3_out], dim=1)

        dec5_out = self.dec5(dec4_out)
        dec5_out = torch.cat([dec5_out, enc2_out], dim=1)

        dec6_out = self.dec6(dec5_out)
        dec6_out = torch.cat([dec6_out, enc1_out], dim=1)

        dec7_out = self.dec7(dec6_out)

        dec_out = MCDropout2D(self.dropout_prob)(dec7_out)

        out = self.conv_out(dec_out)

        return out

    @staticmethod
    def _encoder_block(in_channels, out_channels, dropout_prob):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    @staticmethod
    def _decoder_block(in_channels, out_channels, dropout_prob):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        return nn.Sequential(*layers)
