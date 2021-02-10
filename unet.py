import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

    def forward(self, image):
        # Batch_size, c, h, w
        # Encoder part
        x1 = self.down_conv_1(image) #
        x2 = self.max_pool_2x2(x1)
        print(x1.size())
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        print(x3.size())
        x5 = self.down_conv_3(x4)#
        x6 = self.max_pool_2x2(x5)
        print(x5.size())
        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool_2x2(x7)
        print(x7.size())
        x9 = self.down_conv_5(x8)
        print(x9.size())
        x = self.up_trans_1(x9)
        print(x.size())
        #Now concatenate x with x7
        #The size of x and x7 varies



if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = UNet()