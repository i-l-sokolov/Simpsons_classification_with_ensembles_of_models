from torch import nn



class CNN3(nn.Module):

    def __init__(self, first=4):
        super(CNN3, self).__init__()
        self.conv1 = self.conv_block2(3, first)  # 224 -> 112
        self.conv2 = self.conv_block2(first, first * 2)  # 112 -> 56
        self.conv3 = self.conv_block4(first * 2, first * 4)  # 28
        self.conv4 = self.conv_block4(first * 4, first * 8)  # 14
        self.conv5 = self.conv_block4(first * 8, first * 8)  # 7
        self.fc = self.final_block(first * 8 * 49, 500)

    def conv_block2(self, in_chan, out_chan):
        block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        return block

    def conv_block4(self, in_chan, out_chan):
        block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        return block

    def final_block(self, in_size, middle_size):
        block = nn.Sequential(
            nn.Linear(in_size, middle_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(middle_size, 42),
        )
        return block

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x