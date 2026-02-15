import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetFire(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetFire, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024)
        
        self.up1 = DoubleConv(1536, 512) # 1024 + 512
        self.up2 = DoubleConv(768, 256)  # 512 + 256
        self.up3 = DoubleConv(384, 128)  # 256 + 128
        self.up4 = DoubleConv(192, 64)   # 128 + 64
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(F.max_pool2d(x1, 2))
        x3 = self.down2(F.max_pool2d(x2, 2))
        x4 = self.down3(F.max_pool2d(x3, 2))
        x5 = self.down4(F.max_pool2d(x4, 2))
        
        # Decoder
        # Upsample and concatenate
        x = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)
        x = self.up1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.up3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.up4(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)

# Helper to build model compatible with previous interface
def build_unet_model(input_shape=(12, 128, 128), num_classes=1):
    """
    Build U-Net Model (PyTorch).
    input_shape is (Channels, Height, Width) for PyTorch.
    """
    c_in = input_shape[0] if len(input_shape) == 3 else input_shape[2] # Handle HWC vs CHW
    return UNetFire(n_channels=c_in, n_classes=num_classes)
