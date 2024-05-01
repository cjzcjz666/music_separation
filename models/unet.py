import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.conv_bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv_bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv_bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv_bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv_bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
        self.conv_bn6 = nn.BatchNorm2d(512)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout2d(0.5)

        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.5)

        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout2d(0.5)

        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn4 = nn.BatchNorm2d(32)

        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn5 = nn.BatchNorm2d(16)

        self.deconv6 = nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = torch.log(x + EPS)

        h1 = F.leaky_relu(self.conv_bn1(self.conv1(x)), 0.2)
        h2 = F.leaky_relu(self.conv_bn2(self.conv2(h1)), 0.2)
        h3 = F.leaky_relu(self.conv_bn3(self.conv3(h2)), 0.2)
        h4 = F.leaky_relu(self.conv_bn4(self.conv4(h3)), 0.2)
        h5 = F.leaky_relu(self.conv_bn5(self.conv5(h4)), 0.2)
        h = F.leaky_relu(self.conv_bn6(self.conv6(h5)), 0.2)

        h = self.dropout1(F.relu(self.deconv_bn1(self.deconv1(h))))
        h = torch.cat((h, h5), dim=1)

        h = self.dropout2(F.relu(self.deconv_bn2(self.deconv2(h))))
        h = torch.cat((h, h4), dim=1)

        h = self.dropout3(F.relu(self.deconv_bn3(self.deconv3(h))))
        h = torch.cat((h, h3), dim=1)

        h = F.relu(self.deconv_bn4(self.deconv4(h)))
        h = torch.cat((h, h2), dim=1)

        h = F.relu(self.deconv_bn5(self.deconv5(h)))
        h = torch.cat((h, h1), dim=1)

        h = F.softmax(self.deconv6(h), dim=1)

        return h


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# EPS = 1e-8

# # select dynamic output
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, segment_size):
#         super(UNet, self).__init__()
        
#         self.segment_size = segment_size
        
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
#         self.conv_bn1 = nn.BatchNorm2d(16)
        
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
#         self.conv_bn2 = nn.BatchNorm2d(32)
        
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
#         self.conv_bn3 = nn.BatchNorm2d(64)
        
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
#         self.conv_bn4 = nn.BatchNorm2d(128)
        
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
#         self.conv_bn5 = nn.BatchNorm2d(256)
        
#         self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2)
#         self.conv_bn6 = nn.BatchNorm2d(512)
        
#         self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv_bn1 = nn.BatchNorm2d(256)
#         self.dropout1 = nn.Dropout2d(0.5)
        
#         self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv_bn2 = nn.BatchNorm2d(128)
#         self.dropout2 = nn.Dropout2d(0.5)
        
#         self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv_bn3 = nn.BatchNorm2d(64)
#         self.dropout3 = nn.Dropout2d(0.5)
        
#         self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv_bn4 = nn.BatchNorm2d(32)
        
#         self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
#         self.deconv_bn5 = nn.BatchNorm2d(16)
        
#         self.deconv6 = nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        
#         self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
#     def pad_to_match(self, tensor1, tensor2):
#         # Get the size differences
#         diff_height = tensor1.size(2) - tensor2.size(2)
#         diff_width = tensor1.size(3) - tensor2.size(3)
        
#         # Pad tensor2 on the right and bottom to match the size of tensor1
#         tensor2_padded = F.pad(tensor2, (0, max(0, diff_width), 0, max(0, diff_height)))
        
#         return tensor2_padded
    
#     def forward(self, x):
#         x = torch.log(x + EPS)
        
#         h1 = F.leaky_relu(self.conv_bn1(self.conv1(x)), 0.2)
#         h2 = F.leaky_relu(self.conv_bn2(self.conv2(h1)), 0.2)
#         h3 = F.leaky_relu(self.conv_bn3(self.conv3(h2)), 0.2)
#         h4 = F.leaky_relu(self.conv_bn4(self.conv4(h3)), 0.2)
#         h5 = F.leaky_relu(self.conv_bn5(self.conv5(h4)), 0.2)
#         h = F.leaky_relu(self.conv_bn6(self.conv6(h5)), 0.2)
        
#         # Up-sampling and concatenating with skip connections
#         h = self.dropout1(F.relu(self.deconv_bn1(self.deconv1(h))))
#         h5 = self.pad_to_match(h, h5)
#         h = torch.cat((h, h5), dim=1)
        
#         # Repeat for other layers
#         h = self.dropout2(F.relu(self.deconv_bn2(self.deconv2(h))))
#         h4 = self.pad_to_match(h, h4)
#         h = torch.cat((h, h4), dim=1)
        
#         h = self.dropout3(F.relu(self.deconv_bn3(self.deconv3(h))))
#         h3 = self.pad_to_match(h, h3)
#         h = torch.cat((h, h3), dim=1)
        
#         h = F.relu(self.deconv_bn4(self.deconv4(h)))
#         h2 = self.pad_to_match(h, h2)
#         h = torch.cat((h, h2), dim=1)
        
#         h = F.relu(self.deconv_bn5(self.deconv5(h)))
#         h1 = self.pad_to_match(h, h1)
#         h = torch.cat((h, h1), dim=1)
        
#         h = self.deconv6(h)
        
#         # Resize the output to match the segment_size
#         h = self.final_conv(h)
#         # print("size", self.segment_size)
#         # print("111", x.shape)
#         # h = F.interpolate(h, size=(self.segment_size, self.segment_size), mode='bilinear', align_corners=False)
#         h = F.interpolate(h, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
#         h = F.softmax(h, dim=1)
        
#         return h