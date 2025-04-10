import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out
    
class DndCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DndCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Deformable_CBAM(features, reduction=16))
        for _ in range(num_of_layers-3):
            layers.append(Deformable_conv(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dndcnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dndcnn(x)
        return out
    
class Deformable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Deformable_conv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        print(type(in_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Calculate number of offset parameters (2 per kernel position)
        self.offset_channels = 2 * kernel_size * kernel_size
        self.mask_channels = kernel_size * kernel_size
        
        self.offset_conv = nn.Conv2d(in_channels, self.offset_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mask_conv = nn.Conv2d(in_channels, self.mask_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Initialize to zero so we start with regular convolution
        self.offset_conv.weight.data.zero_()
        self.offset_conv.bias.data.zero_()
        self.mask_conv.weight.data.zero_()
        self.mask_conv.bias.data.zero_()

    def forward(self, x):
        # Get output of regular convolution
        conv_out = self.conv(x)
        
        # For 1x1 convolutions, just return the regular output
        # This prevents dimension errors with the deformable part
        if self.kernel_size == 1:
            return conv_out
        
        # For other kernel sizes, apply the deformable operations
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        k_h, k_w = self.kernel_size, self.kernel_size
        b, _, h, w = x.size()
        
        # Reshape tensors for proper multiplication
        offset = offset.view(b, 2, k_h*k_w, h, w)
        mask = mask.view(b, 1, k_h*k_w, h, w)
        
        # Apply deformable component
        return conv_out + (offset * mask).sum(dim=2)
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        b, c, _, _ = x.size()
        channel_att = self.channel_attention(x).expand_as(x)
        x = x * channel_att

        # Spatial attention
        spatial_att = self.spatial_attention(x).expand_as(x)
        x = x * spatial_att

        return x
    
class Deformable_CBAM(nn.Module):
    class channel_attention(nn.Module):
        def __init__(self, in_channels, reduction=16):
            super(Deformable_CBAM.channel_attention, self).__init__()
            self.maxpool = nn.AdaptiveMaxPool2d(1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                Deformable_conv(in_channels, in_channels // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                Deformable_conv(in_channels // reduction, in_channels, kernel_size=1),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.avgpool(x)
            avg_out = self.fc(avg_out)
            max_out = self.maxpool(x)
            max_out = self.fc(max_out)
            out = avg_out + max_out
            out = self.sigmoid(out)
            return out
        
    class spatial_attention(nn.Module):
        def __init__(self, kernel_size=7):
            super(Deformable_CBAM.spatial_attention, self).__init__()
            self.conv1 = Deformable_conv(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)

    def __init__(self, in_channels, reduction=16):
        super(Deformable_CBAM, self).__init__()
        self.channel_attention = Deformable_CBAM.channel_attention(in_channels, reduction)
        self.spatial_attention = Deformable_CBAM.spatial_attention(kernel_size=7)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        channel_att = self.channel_attention(x)
        print(f"Channel attention shape: {channel_att.shape}")
        x = x * channel_att
        
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        return x