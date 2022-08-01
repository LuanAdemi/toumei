import torch.nn as nn

from toumei.models.helper_layers import RedirectedReluLayer


class SimpleCNN(nn.Module):
    def __init__(self, in_chn, out_chn, redirected_relu=False):
        super(SimpleCNN, self).__init__()

        if redirected_relu:
            act = RedirectedReluLayer
        else:
            act = nn.ReLU

        self.conv1 = nn.Conv2d(in_channels=in_chn, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = act()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=0, bias=True)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = act()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(in_features=512, out_features=32, bias=True)
        self.relu_fc1 = act()

        self.fc2 = nn.Linear(in_features=32, out_features=16, bias=True)
        self.relu_fc2 = act()

        self.fc3 = nn.Linear(in_features=16, out_features=8, bias=True)
        self.relu_fc3 = act()

        self.fc4 = nn.Linear(in_features=8, out_features=out_chn, bias=True)


    def forward(self, x):
        # perform the usual forward pass
        x = self.relu1(self.pool1(self.conv1(x)))

        x = self.relu2(self.pool2(self.conv2(x)))

        x = self.adaptive_pool(x)

        x = x.view(-1, 512)

        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        x = self.relu_fc3(self.fc3(x))

        return self.fc4(x)

