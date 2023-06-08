import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class fusion(nn.Module):
    def __init__(self,in_ch = 3 ,bottom = 512):
        super(fusion, self).__init__()

        self.conv = nn.Conv2d(2*in_ch,in_ch,1,stride=1,bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

        self.conv_2 = nn.Conv2d(bottom, in_ch, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
    def forward(self,mid_layer,bottom_layer):
        in_batch, inchannel, in_h, in_w = mid_layer.size()
        bottom_layer = self.conv_2(bottom_layer)
        bottom_layer = self.bn1(self.relu1(bottom_layer))
        bottom_layer = bottom_layer.expand(in_batch, inchannel, in_h, in_w)
        cat  = torch.cat((mid_layer,bottom_layer),1)
        out = self.bn(self.relu(self.conv(cat)))

        return out
"""
if __name__ == '__main__':
    in_batch, inchannel, in_h, in_w = 4, 256, 14, 14

    x = torch.randn(in_batch, inchannel, in_h, in_w)
    x2 = torch.randn(in_batch,512,1,1)
    net = fusion(in_ch=256,bottom=512)
    out = net(x,x2)
    print(out.shape)
"""





class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class GFE(nn.Module):
    def __init__(self,in_ch=3,out_ch=3):
        super(GFE, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, 64, dirate=1)

        self.rebnconv1 = REBNCONV(64, 128, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=4, ceil_mode=True)

        self.rebnconv2 = REBNCONV(128, 256, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=4, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cat1 = fusion(in_ch=64, bottom=256)
        self.cat2 = fusion(in_ch=128, bottom=256)
        self.cat3 = fusion(in_ch=256, bottom=256)

        self.rebnconv3d = REBNCONV(256*2, 128, dirate=1)
        self.rebnconv2d = REBNCONV(128 * 2, 64, dirate=1)
        self.rebnconv1d = REBNCONV(64 * 2, out_ch, dirate=1)

    def forward(self,x):
        hin = x
        hx1 = self.rebnconvin(x)
        print(hx1.size())
        hx = self.pool1(hx1)

        hx2 = self.rebnconv1(hx)
        print(hx2.size())
        hx = self.pool2(hx2)


        hx3 = self.rebnconv2(hx)
        print(hx3.size())
        hx = self.avgpool(hx3)

        hx1f = self.cat1(hx1,hx)
        hx2f = self.cat2(hx2,hx)
        hx3f = self.cat3(hx3,hx)

        hx7 = _upsample_like(hx, hx3)
        #print(hx7.size())

        hx6d = self.rebnconv3d(torch.cat((hx7, hx3f), 1))
        hx6dup = _upsample_like(hx6d, hx2)

        hx5d = self.rebnconv2d(torch.cat((hx6dup, hx2f), 1))
        hx5dup = _upsample_like(hx5d, hx1)
        #print(hx5dup.size())
        # print(hx4f.size())
        hx4d = self.rebnconv1d(torch.cat((hx5dup, hx1f), 1))
        #print(hx4d.size())

        #out = hin+hx4d
        #print(out.size())
        return hin+hx4d




class MCRUNet(nn.Module):
    def __init__(self,in_ch = 3,out_ch = 2):
        super(MCRUNet, self).__init__()


        self.RGB_block = GFE(in_ch=in_ch,out_ch=3)
        import os
        from u2net import U2NET_G, U2NET

        net = U2NET(3, 1)
        weights_path = ".//saved_models//u2net.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        net.load_state_dict(torch.load(weights_path), strict=False)
        pretrained_dict = net.state_dict()

        model = U2NET_G(in_ch=3, out_ch=2)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        model.load_state_dict(model_dict, strict=False)
        net = model

        """
        net = models.resnet50(pretrained=True)
        # print(net)
        # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, out_ch)
        """
        self.resnet =net

    def forward(self,x):
        #x = self.RGB_block(x)
        #print(x.size())
        x = self.resnet(x)

        return x
