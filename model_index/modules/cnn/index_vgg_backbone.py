#-*-coding:utf-8-*-
import torch


# 取代maxpool
class indexBlock(torch.nn.Module):
    def __init__(self, inchannel, c1=4,c2=0.4):
        super().__init__()
        self.maxPool = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.out_channel1 = inchannel*c1
        self.out_channel2 = max(4, int(inchannel*c2))
        self.index_conv1_1= torch.nn.Sequential(
            torch.nn.Conv2d(inchannel*2, self.out_channel1, kernel_size=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.out_channel1, self.out_channel2, kernel_size=1, stride=1),
            torch.nn.ReLU() # TODO 加不加？
        )
        self.merge_conv1_1 = torch.nn.Conv2d(inchannel+self.out_channel2, inchannel, kernel_size=1, stride=1)


    def forward(self, x):
        # 采样
        feature, index = self.maxPool(x)
        # 采样特征
        _, width = x.shape[-2:]
        index_x = (index // width).squeeze(-1) % 2.
        index_y = (index %  width).squeeze(-1) % 2.
        index_xy = torch.cat([torch.cat([index_x[:, c:c + 1, :, :], index_y[:, c:c + 1, :, :]], dim=1) for c in range(x.size(1))], dim=1)
        index_xy = self.index_conv1_1(index_xy)
        # 融合特征
        feature = torch.cat((feature, index_xy), dim=1)
        feature = self.merge_conv1_1(feature)
        return feature



class VGGBackbone(torch.nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """
    def __init__(self, config, input_channel=1, device='cpu'):
        super(VGGBackbone, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            indexBlock(channels[1]),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            indexBlock(channels[3]),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            indexBlock(channels[5]),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
            # block 3
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )


    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)
        return feat_map



class VGGBackboneBN(torch.nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """
    def __init__(self, config, input_channel=1, device='cpu'):
        super(VGGBackboneBN, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[0],eps=1e-3 ),
        )

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[1],eps=1e-3 ),
            indexBlock(channels[1]),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[2],eps=1e-3 ),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[3],eps=1e-3 ),
            indexBlock(channels[3]),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[4],eps=1e-3 ),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[5],eps=1e-3 ),
            indexBlock(channels[5]),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
            # block 3
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[6],eps=1e-3 ),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[7],eps=1e-3 ),
        )

    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)

        return feat_map
