import torch
import torch.nn as nn
import torch.nn.functional as F

# Corrected Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.avg_pool(x)  # Compress spatial dimensions to 1x1
        scale = self.fc(scale)   # Channel attention weights
        return x * scale         # Broadcast weights back to match input size



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_BN=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        self.n_classes = n_classes # Number of output channels (e.g., 1 for binary segmentation, more for multi-class)
        self.bilinear = bilinear # Whether to use bilinear upsampling (less parameters) or transposed convolutions
        self.use_BN = use_BN # Whether to use Batch Normalization in convolutional layers
        self.sigmoid = nn.Sigmoid()
        factor = 2 if bilinear else 1

        # Downsampling path
        self.conv1 = self.conv_block(n_channels, 64, 64)
        self.conv2 = self.conv_block(64, 128, 128)
        self.conv3 = self.conv_block(128, 256, 256)
        self.conv4 = self.conv_block(256, 512, 512)
        self.conv5 = self.conv_block(512, 1024 // factor, 1024 // factor)

        self.pool1 = nn.MaxPool2d(2) # Max Pooling is for spatial down-sampling
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        # Upsampling path
        if bilinear:
            self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up8 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up9 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv6 = self.conv_block(1024, 512, 512 // factor)
        self.conv7 = self.conv_block(512, 256, 256 // factor)
        self.conv8 = self.conv_block(256, 128, 128 // factor)
        self.conv9 = self.conv_block(128, 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) # Last Conv, produce the final output

    # Combination of two convolutional layers with optional Batch Normalization, activation (ReLU or LeakyReLU), and a Squeeze-and-Excitation (SE) block for channel-wise attention.
    def conv_block(self, in_channels, mid_channels, out_channels):
        if self.use_BN:
            layers = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                SEBlock(out_channels)  # Use the corrected SEBlock
            ]
        else:
            layers = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                SEBlock(out_channels)  # Use the corrected SEBlock
            ]
        return nn.Sequential(*layers)

    # Squeeze-and-Excitation (Block Highlights important channels & Suppresses irrelevant features)
#    def se_block(self, channels, reduction=16):
#        return nn.Sequential(
#            nn.AdaptiveAvgPool2d(1),
#            nn.Conv2d(channels, channels // reduction, kernel_size=1),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(channels // reduction, channels, kernel_size=1),
#            nn.Sigmoid()
#        )

    # Concatenation leads to better performance for Segmentation as it keeps more features and details SEE TABLE ABOVE
    def pad_and_concat(self, enc_feature, dec_feature):
        diffY = enc_feature.size()[2] - dec_feature.size()[2]
        diffX = enc_feature.size()[3] - dec_feature.size()[3]
        dec_feature = F.pad(dec_feature, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        return torch.cat([enc_feature, dec_feature], dim=1)

    # NOT USED FOR NOW SEE TABLE ABOVE
    def pad_and_add(enc_feature, dec_feature):
        diffY = enc_feature.size()[2] - dec_feature.size()[2]
        diffX = enc_feature.size()[3] - dec_feature.size()[3]
        dec_feature = F.pad(dec_feature, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        return enc_feature + dec_feature  # Element-wise addition

    def forward(self, x):
        assert x.size(2) >= 32 and x.size(3) >= 32, "Input size must be at least 32x32 for this UNet"

        # Downsampling
        h1 = self.conv1(x)
        h2 = self.conv2(self.pool1(h1))
        h3 = self.conv3(self.pool2(h2))
        h4 = self.conv4(self.pool3(h3))
        h5 = self.conv5(self.pool4(h4))

        # Upsampling
        h6 = self.conv6(self.pad_and_concat(h4, self.up6(h5)))
        h7 = self.conv7(self.pad_and_concat(h3, self.up7(h6)))
        h8 = self.conv8(self.pad_and_concat(h2, self.up8(h7)))
        h9 = self.conv9(self.pad_and_concat(h1, self.up9(h8)))

        logits = self.sigmoid(self.outc(h9))



        return logits