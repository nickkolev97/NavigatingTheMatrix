import torch
import torch.nn as nn # nn class our model inherits from
from matplotlib import pyplot as plt

#########
## modified unet for bright features. 
## Did this when trying to find why the bright one wasn't converting well to torch
## need to just retrain all the UNets into one architecture.
##
##########
class DoubleConv_m(nn.Module):
    # a double convolution that does not change the resolution

    def __init__(self, n_channels1, n_channels2, n_channels3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(n_channels1, n_channels2, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_channels2, eps=0.001, momentum = 0.99),
            nn.Conv2d(n_channels2, n_channels3, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(n_channels3, eps=0.001, momentum = 0.99))

    def forward(self, x):
        return self.double_conv(x)

class downsample_m(nn.Module):
  # a double convolution followed by a maxpooling layer to downsample
  def __init__(self, n_channels1, n_channels2, n_channels3):
      super().__init__()
      self.doubleconvDown = DoubleConv_m(n_channels1, n_channels2, n_channels3)


  def forward(self, x):
      skip = self.doubleconvDown(x)
      y = nn.MaxPool2d(2)(skip)
      y = nn.Dropout(0.3)(y)
      return skip, y

class upsample_m(nn.Module):
  # a traspose convolution (to upsample) followed by a double convolution
  def __init__(self, n_channels1, n_channels2, n_channels3):
    super().__init__()
    self.convTranspose = nn.Sequential(nn.ConvTranspose2d(n_channels1, n_channels2, 2 , stride=2))#,
                                      # nn.BatchNorm2d(n_channels2, eps=0.001),
                                      # nn.ReLU(inplace=True))

    self.doubleConv = DoubleConv_m(n_channels1, n_channels2, n_channels3) #n_channels2 is first because it's concatenated with skip

  def forward(self, skip, x):
    # upsample
    x = self.convTranspose(x)
    # concatenate with the skip
    x = torch.cat([x, skip], axis=1)
    # dropout
    x = nn.Dropout(0.3)(x)
    # double convolution
    x = self.doubleConv(x)
    return x


class UNet_m(nn.Module):

    def __init__(self):
        super().__init__()
        # downsample 1
        self.downsample1 = downsample_m(1, 32, 32)
        # downsample 2
        self.downsample2 = downsample_m(32, 64, 64)
        # downsample 3
        self.downsample3 = downsample_m(64, 128, 128)

        # bottleneck
        self.bottleneck = DoubleConv_m(128,256,256)
    
        # upsample 1
        self.upsample1 = upsample_m(256, 128, 128)
        # upsample 2
        self.upsample2 = upsample_m(128, 64, 64)
        # upsample 3
        self.upsample3 = upsample_m(64, 32, 32)

        # final layer
        self.output = nn.Sequential(nn.Conv2d(32,2, kernel_size=1, padding="same"))#,
                                  #nn.Softmax2d())

    def forward(self, x):
        # downsample 1
        skip1, x1 = self.downsample1(x)
        # downsample 2
        skip2, x2 = self.downsample2(x1)
        # downsample 3
        skip3, x3 = self.downsample3(x2)

        # bottleneck
        bottleneck = self.bottleneck(x3)

        # upsample 1
        x4 = self.upsample1(skip3, bottleneck)
        # upsample 2
        x5 = self.upsample2(skip2, x4)
        # upsample 3
        x6 = self.upsample3(skip1, x5)

        outputs = self.output(x6)

        return outputs


'''
class DoubleConv(nn.Module):
    # a double convolution that does not change the resolution

    def __init__(self, n_channels1, n_channels2, n_channels3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(n_channels1, n_channels2, kernel_size=3, padding='same'),
            nn.BatchNorm2d(n_channels2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels2, n_channels3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(n_channels3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class downsample(nn.Module):
  # a double convolution followed by a maxpooling layer to downsample
  def __init__(self, n_channels1, n_channels2, n_channels3):
      super().__init__()
      self.doubleconvDown = DoubleConv(n_channels1, n_channels2, n_channels3)


  def forward(self, x):
      skip = self.doubleconvDown(x)
      y = nn.MaxPool2d(2)(skip)
      y = nn.Dropout(0.3)(y)
      return skip, y

class upsample(nn.Module):
  # a traspose convolution (to upsample) followed by a double convolution
  def __init__(self, n_channels1, n_channels2, n_channels3):
    super().__init__()
    self.convTranspose = nn.Sequential(nn.ConvTranspose2d(n_channels1, n_channels2, 2 , stride=2),
                                       nn.BatchNorm2d(n_channels2),
                                       nn.ReLU(inplace=True))

    self.doubleConv = DoubleConv(n_channels1, n_channels2, n_channels3) #n_channels2 is first because it's concatenated with skip


  def forward(self, skip, x):
    # upsample
    x = self.convTranspose(x)
    # concatenate with the skip
    x = torch.cat([x, skip], axis=1)
    # double convolution
    x = self.doubleConv(x)
    return x


class UNet(nn.Module):
    # binary fully convolutional network 
    def __init__(self):
      super().__init__()
      # downsample 1
      self.downsample1 = downsample(1, 32, 32)
      # downsample 2
      self.downsample2 = downsample(32, 64, 64)
      # downsample 3
      self.downsample3 = downsample(64, 128, 128)

      # bottleneck
      self.bottleneck = DoubleConv(128,256,256)

      # upsample 1
      self.upsample1 = upsample(256, 128, 128)
      # upsample 2
      self.upsample2 = upsample(128, 64, 64)
      # upsample 3
      self.upsample3 = upsample(64, 32, 32)

      # final layer
      self.output = nn.Sequential(nn.Conv2d(32,3, kernel_size=1, padding="same"),
                                  nn.Softmax2d())



    def forward(self, x):
      # downsample 1
      skip1, x1 = self.downsample1(x)
      # downsample 2
      skip2, x2 = self.downsample2(x1)
      # downsample 3
      skip3, x3 = self.downsample3(x2)

      # bottleneck
      bottleneck = self.bottleneck(x3)

      # upsample 1
      x4 = self.upsample1(skip3, bottleneck)
      # upsample 2
      x5 = self.upsample2(skip2, x4)
      # upsample 3
      x6 = self.upsample3(skip1, x5)

      outputs = self.output(x6)

      return outputs

'''

class DoubleConv(nn.Module):
    # a double convolution that does not change the resolution

    def __init__(self, n_channels1, n_channels2, n_channels3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(n_channels1, n_channels2, kernel_size=3, padding='same'),
            nn.BatchNorm2d(n_channels2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels2, n_channels3, kernel_size=3, padding='same'),
            nn.BatchNorm2d(n_channels3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class downsample(nn.Module):
  # a double convolution followed by a maxpooling layer to downsample
  def __init__(self, n_channels1, n_channels2, n_channels3):
      super().__init__()
      self.doubleconvDown = DoubleConv(n_channels1, n_channels2, n_channels3)


  def forward(self, x):
      skip = self.doubleconvDown(x)
      y = nn.MaxPool2d(2)(skip)
      y = nn.Dropout(0.3)(y)
      return skip, y

class upsample(nn.Module):
  # a traspose convolution (to upsample) followed by a double convolution
  def __init__(self, n_channels1, n_channels2, n_channels3):
    super().__init__()
    self.convTranspose = nn.Sequential(nn.ConvTranspose2d(n_channels1, n_channels2, 2 , stride=2),
                                       nn.BatchNorm2d(n_channels2),
                                       nn.ReLU(inplace=True))

    self.doubleConv = DoubleConv(n_channels1, n_channels2, n_channels3) #n_channels2 is first because it's concatenated with skip


  def forward(self, skip, x):
    # upsample
    x = self.convTranspose(x)
    # concatenate with the skip
    x = torch.cat([x, skip], axis=1)
    # double convolution
    x = self.doubleConv(x)
    return x

class UNet(nn.Module):

    def __init__(self):
      super().__init__()
      # downsample 1
      self.downsample1 = downsample(1, 32, 32)
      # downsample 2
      self.downsample2 = downsample(32, 64, 64)
      # downsample 3
      self.downsample3 = downsample(64, 128, 128)

      # bottleneck
      self.bottleneck = DoubleConv(128,256,256)

      # upsample 1
      self.upsample1 = upsample(256, 128, 128)
      # upsample 2
      self.upsample2 = upsample(128, 64, 64)
      # upsample 3
      self.upsample3 = upsample(64, 32, 32)

      # final layer
      self.output = nn.Sequential(nn.Conv2d(32,2, kernel_size=1, padding="same"),
                                  nn.Softmax2d())



    def forward(self, x):
      # downsample 1
      skip1, x1 = self.downsample1(x)
      # downsample 2
      skip2, x2 = self.downsample2(x1)
      # downsample 3
      skip3, x3 = self.downsample3(x2)

      # bottleneck
      bottleneck = self.bottleneck(x3)

      # upsample 1
      x4 = self.upsample1(skip3, bottleneck)
      # upsample 2
      x5 = self.upsample2(skip2, x4)
      # upsample 3
      x6 = self.upsample3(skip1, x5)

      outputs = self.output(x6)

      return outputs

class Classifier(nn.Module):
    # classiifier network

    def __init__(self, channels=2, crop_size=11, n_outputs=7, fc_layers=2, fc_nodes=100, dropout=0.2):
        super().__init__()
        self.fc_layers = fc_layers
        
        self.convolutional_relu_stack = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = (crop_size-4)**2 *64, out_features=fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(100),
        )
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(fc_nodes, fc_nodes),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(fc_nodes)
        )
        
        self.linear_relu_stack_last = nn.Sequential(
            nn.Linear(fc_nodes, n_outputs)
        )
        

    def forward(self, x, training = True):
        x = self.convolutional_relu_stack(x)
        for i in range(self.fc_layers-1):
            x = self.linear_relu_stack(x)
        if training == True:
            x = self.linear_relu_stack_last(x)
            logits = torch.nn.functional.softmax(x,dim=1)
            return logits
        else: 
            x = torch.nn.functional.normalize(x)
            return x