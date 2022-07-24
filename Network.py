import math
import numpy as np
import PIL.Image
import torch
from Config import params
# Network definition

class Network(torch.nn.Module):
  def __init__(self):
    super().__init__()

    def Basic(InChannel, OutChannel):  # 3 consecutive convolution blocks with ReLU activation
      return torch.nn.Sequential(
          torch.nn.Conv2d(in_channels = InChannel, out_channels = OutChannel, kernel_size = 3, stride = 1, padding = 1 ),
          torch.nn.ReLU(),
          torch.nn.Conv2d(in_channels = OutChannel, out_channels = OutChannel, kernel_size = 3, stride = 1, padding = 1 ),
          torch.nn.ReLU(),
          torch.nn.Conv2d(in_channels = OutChannel, out_channels = OutChannel, kernel_size = 3, stride = 1, padding = 1 ),
          torch.nn.ReLU()
      )
    #end

    def Upsample(OutChannel): # bilinear upsampling followed by a convolutional layer with ReLU activation
      return torch.nn.Sequential(
          torch.nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
          torch.nn.Conv2d(in_channels = OutChannel, out_channels = OutChannel, kernel_size = 3, stride = 1, padding =1 ),
          torch.nn.ReLU(),
      )
    #end

    def Subnet(kernel_length): #subnetworks for generating kernels
      return torch.nn.Sequential(
          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding =1 ),
          torch.nn.ReLU(),
          torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding =1 ),
          torch.nn.ReLU(),
          torch.nn.Conv2d(in_channels = 64, out_channels = kernel_length, kernel_size = 3, stride = 1, padding =1 ),
          torch.nn.ReLU(),
          torch.nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners = True),
          torch.nn.Conv2d(in_channels = kernel_length, out_channels = kernel_length, kernel_size = 3, stride = 1, padding =1 ),
      )
    #end

    #convolutional neural networks used with Average pooling for downsampling
    self.ConvNet1 = Basic(6,32)
    self.ConvNet2 = Basic(32,64)
    self.ConvNet3 = Basic(64,128)
    self.ConvNet4 = Basic(128,256)
    self.ConvNet5 = Basic(256,512)
    self.ConvNet6 = Basic(512,512)

    #convolutional neural networks used with bilinear upsampling for downsamplingfor upsampling
    self.ConvNet7 = Basic(512,256)
    self.ConvNet8 = Basic(256,128)
    self.ConvNet9 = Basic(128,64)

    #upsampling layers
    self.UpsampleNet1 = Upsample(512)
    self.UpsampleNet2 = Upsample(256)
    self.UpsampleNet3 = Upsample(128)
    self.UpsampleNet4 = Upsample(64)

    #subnetworks
    kernel_length = params["kernel_size"]
    self.NetVertical1 = Subnet(kernel_length)
    self.NetVertical2 = Subnet(kernel_length)
    self.NetHorizontal1 = Subnet(kernel_length)
    self.NetHorizontal2 = Subnet(kernel_length)

  def forward(self, ten1, ten2):
    #downsampling
    tenConv1 = self.ConvNet1(torch.cat([ten1, ten2],dim=1))
    tenConv2 = self.ConvNet2(torch.nn.functional.avg_pool2d(input = tenConv1, kernel_size = 2, stride =2 ,count_include_pad = False))
    tenConv3 = self.ConvNet3(torch.nn.functional.avg_pool2d(input = tenConv2, kernel_size = 2, stride =2 ,count_include_pad = False))
    tenConv4 = self.ConvNet4(torch.nn.functional.avg_pool2d(input = tenConv3, kernel_size = 2, stride =2 ,count_include_pad = False))
    tenConv5 = self.ConvNet5(torch.nn.functional.avg_pool2d(input = tenConv4, kernel_size = 2, stride =2 ,count_include_pad = False))
    tenConv6 = self.ConvNet6(torch.nn.functional.avg_pool2d(input = tenConv5, kernel_size = 2, stride =2 ,count_include_pad = False))

    #upsampling
    tenConv7 = self.ConvNet7(self.UpsampleNet1(tenConv6)+tenConv5)
    tenConv8 = self.ConvNet8(self.UpsampleNet2(tenConv7)+tenConv4)
    tenConv9 = self.ConvNet9(self.UpsampleNet3(tenConv8)+tenConv3)
    tenConv10 = self.UpsampleNet4(tenConv9)

    #going through the subnetworks
    tenVertical1 = self.NetVertical1(tenConv10+tenConv2)
    tenVertical2 = self.NetVertical2(tenConv10+tenConv2)
    tenHorizontal1 = self.NetHorizontal1(tenConv10+tenConv2)
    tenHorizontal2 = self.NetHorizontal2(tenConv10+tenConv2)

    #apply kernels to the images
    batch_size = ten1.shape[0]
    width = ten1.shape[3]
    height = ten1.shape[2]
    resolution = width * height
    kernel_length = params["kernel_size"]
    pad_num = int(kernel_length//2)
    ten_padded1 = torch.nn.functional.pad(input = ten1,pad = [pad_num,pad_num,pad_num,pad_num], mode = "replicate" )
    ten_padded2 = torch.nn.functional.pad(input = ten2,pad = [pad_num,pad_num,pad_num,pad_num], mode = "replicate" )
    tenVertical1 = tenVertical1.view(-1, kernel_length, resolution, 1)
    tenVertical2 = tenVertical2.view(-1, kernel_length, resolution, 1)
    tenHorizontal1 = tenHorizontal1.view(-1, kernel_length, 1, resolution)
    tenHorizontal2 = tenHorizontal2.view(-1, kernel_length, 1, resolution)
    tenVertical1 = tenVertical1.transpose(1,2)
    tenVertical2 = tenVertical2.transpose(1,2)
    tenHorizontal1 = tenHorizontal1.transpose(1,3)
    tenHorizontal2 = tenHorizontal2.transpose(1,3)
    """
    below loop part should be implemented via cuda programmingto use GPU accerlation.
    However, I don't know how to use cuda programming."""
    ten_padded1 = torch.transpose(ten_padded1, 0, 1)  # [batch_size, 3, 51, 51]->[3, batch_size, 51, 51]. same kernel applied to all color channels.
    ten_padded2 = torch.transpose(ten_padded2, 0, 1)
    tenVertical1 = torch.transpose(tenVertical1, 0, 1) # [batch_size, resolution, 51, 1]->[resolution, batch_size, 51, 1]
    tenVertical2 = torch.transpose(tenVertical2, 0, 1)
    tenHorizontal1 = torch.transpose(tenHorizontal1, 0, 1)
    tenHorizontal2 = torch.transpose(tenHorizontal2, 0, 1)
    tenOut = torch.zeros_like(ten1)
    for i in range(ten1.shape[2]): #loop through y axis
      for j in range(ten1.shape[3]): #loop through x axis
        t1 = ten_padded1[:,:,i:i+kernel_length,j:j+kernel_length]
        t2 = ten_padded2[:,:,i:i+kernel_length,j:j+kernel_length]
        tv1 = tenVertical1[(i*width+j):(i*width+j+1),:,:,:]
        tv2 = tenVertical2[(i*width+j):(i*width+j+1),:,:,:]
        th1 = tenHorizontal1[(i*width+j):(i*width+j+1),:,:,:]
        th2 = tenHorizontal2[(i*width+j):(i*width+j+1),:,:,:]
        tenOut1 = torch.multiply(t1, tv1.view(batch_size,kernel_length,1))
        tenOut1 = torch.multiply(tenOut1, th1.view(batch_size,1,kernel_length))
        tenOut2 = torch.multiply(t2, tv2.view(batch_size,kernel_length,1))
        tenOut2 = torch.multiply(tenOut2, th2.view(batch_size,1,kernel_length))
        tenSum = tenOut1 + tenOut2
        tenSum = torch.transpose(tenSum, 0, 1)
        tenOut[:,:,i,j] = torch.sum(input = tenSum, dim = (2,3)) # get one pixel of the interpolated frame
    return tenOut
  #end
