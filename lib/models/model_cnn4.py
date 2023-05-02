from torch import nn


class CNN4(nn.Module):

    def __init__(self, first=4):
        super(CNN4, self).__init__()

        self.conv1_1 = self.conv_block3(3, first)
        self.conv1_2 = nn.Conv2d(3, first, kernel_size=(1, 1))
        self.pool1 = self.pooling_conv(2 * first, 2 * first)  # 224 -> 112

        self.conv2 = self.conv_block3(2 * first, 2 * first)
        self.pool2 = self.pooling_conv(4 * first, 4 * first)  # 112 -> 56

        self.conv3 = self.conv_block4(4 * first, 4 * first)
        self.pool3 = self.pooling_conv(8 * first, 8 * first)  # 56 -> 28

        self.conv4 = self.conv_block4(8 * first, 8 * first)
        self.pool4 = self.pooling_conv(16 * first, 16 * first)  # 28 -> 14

        self.conv5 = self.conv_block4(16 * first, 16 * first)
        self.pool5 = self.pooling_conv(32 * first, 32 * first)

        self.fc = self.final_block(first * 32 * 49, 5000, 1000)

    def pooling_conv(self, in_chan, out_chan):
        block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), stride=2, padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True))
        return block

    def conv_block3(self, in_chan, out_chan):
        block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_chan))
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
            nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=1))
        return block

    def final_block(self, in_size, middle_size1, middle_size2):
        block = nn.Sequential(
            nn.Linear(in_size, middle_size1),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(middle_size1, middle_size2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(middle_size2, 42))
        return block

    def forward(self, img):
        x = self.conv1_1(img)
        x1 = self.conv1_2(img)
        x = self.pool1(torch.cat([x, x1], dim=1))

        x1 = self.conv2(x)
        x = self.pool2(torch.cat([x, x1], dim=1))

        x1 = self.conv3(x)
        x = self.pool3(torch.cat([x, x1], dim=1))

        x1 = self.conv4(x)
        x = self.pool4(torch.cat([x, x1], dim=1))

        x1 = self.conv5(x)
        x = self.pool5(torch.cat([x, x1], dim=1))

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x