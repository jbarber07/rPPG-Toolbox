import torch.nn as nn

class PhysNet_Light(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_Light, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 3, [1, 5, 5], stride=1, padding=[0, 2, 2], groups=3),  # Depthwise separable convolution
            nn.Conv3d(3, 8, [1, 1, 1], stride=1),  # Pointwise convolution
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(8, 8, [3, 3, 3], stride=1, padding=1, groups=8),  # Depthwise separable convolution
            nn.Conv3d(8, 16, [1, 1, 1], stride=1),  # Pointwise convolution
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(16, 16, [3, 3, 3], stride=1, padding=1, groups=16),  # Depthwise separable convolution
            nn.Conv3d(16, 32, [1, 1, 1], stride=1),  # Pointwise convolution
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1, groups=32),  # Depthwise separable convolution
            nn.Conv3d(32, 32, [1, 1, 1], stride=1),  # Pointwise convolution
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1, groups=32),  # Depthwise separable convolution
            nn.Conv3d(32, 32, [1, 1, 1], stride=1),  # Pointwise convolution
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(32),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(32),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(32, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [8, T, 64,64]

        x = self.ConvBlock2(x)  # x [16, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)  # x [32, T/2, 32,32]

        x = self.ConvBlock4(x)  # x [32, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [32, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [32, T/4, 16,16]

        x = self.upsample(x)  # x [32, T/2, 8, 8]
        x_visual1616 = self.upsample2(x)  # x [32, T, 8, 8]

        x = self.poolspa(x)  # x [32, T, 1,1]
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)

        return rPPG, x_visual, x_visual3232, x_visual1616
