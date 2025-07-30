# DenseNet121 Model 
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        inter_channels = 4 * growth_rate
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False) #1x1 convolution reduces the number of input feature maps to the bottleneck layer
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)#  #3x3 convolution extracts features from the input feature maps
        self.bn2 = nn.BatchNorm2d(growth_rate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU(inplace=True)(out)

        return torch.cat([x, out], 1)  # output of the bottleneck layer is obtained by concatenating the input feature maps with the output of the second convolutional layer

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = nn.ReLU(inplace=True)(out)
        return self.pool(out)  # Transition layer reduces the number of feature maps and downsamples the spatial dimensions
    



class DenseNet121(nn.Module):
    def __init__(self, num_classes=1000):
        super(DenseNet121, self).__init__()
        growth_rate = 32 # # The inner_channel is set to 4 * growth_rate, where growth_rate is a hyperparameter that determines the number of feature maps that each layer in a dense block should produce.
        num_init_features = 64

        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Track number of channels
        num_channels = num_init_features

        # Block 1
        self.block1 = self._make_dense_block(num_channels, growth_rate, 6)
        num_channels = num_channels + 6 * growth_rate
        self.trans1 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # Block 2
        self.block2 = self._make_dense_block(num_channels, growth_rate, 12)
        num_channels = num_channels + 12 * growth_rate
        self.trans2 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # Block 3
        self.block3 = self._make_dense_block(num_channels, growth_rate, 24)
        num_channels = num_channels + 24 * growth_rate
        self.trans3 = Transition(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        # Block 4
        self.block4 = self._make_dense_block(num_channels, growth_rate, 16)
        num_channels = num_channels + 16 * growth_rate

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(Bottleneck(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.trans3(x)

        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    model = DenseNet121(num_classes=1000)
    print(model)
    # Test with a random input
    x = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size
    output = model(x)
    print(output.shape)  # Should be [1, 1000] for num_classes=1000



