#!/usr/bin/python3

### generate Conv1d model using Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelGeneratorUtils import make_parser, model_name, export_onnx, write_reference_output


class Net(nn.Module):

    def __init__(self, nc = 1, ng = 1, nl = 4, use_bn = False, use_maxpool = False, use_avgpool = False):
        super(Net, self).__init__()

        self.nc = nc
        self.ng = ng
        self.nl = nl
        self.use_bn = use_bn
        self.use_maxpool = use_maxpool
        self.use_avgpool = use_avgpool

        self.conv0 = nn.Conv1d(in_channels=self.nc, out_channels=4, kernel_size=2, groups=1, stride=1, padding=1)
        if (self.use_bn): self.bn1 = nn.BatchNorm1d(4)
        if (self.use_maxpool): self.pool1 = nn.MaxPool1d(2)
        if (self.use_avgpool): self.pool1 = nn.AvgPool1d(2)
        if (self.nl > 1):
           # output is 4x4 with optionally using group convolution
           self.conv1  = nn.Conv1d(in_channels=4,   out_channels=8, groups = self.ng,   kernel_size=3, stride=1, padding=1)
           #output is same 4x4


    def forward(self, x):
      x = self.conv0(x)
      x = F.relu(x)

      if (self.use_bn):
         x = self.bn1(x)
      if (self.use_maxpool or self.use_avgpool):
         x = self.pool1(x)
      if (self.nl == 1) : return x
      x = self.conv1(x)
      x = F.relu(x)

      return x


def main():

   parser = make_parser('parameters for the Conv network : batchSize , inputChannels, inputImageSize, nGroups, nLayers ',
                        pooling=True)
   args = parser.parse_args()

   if (len(args.params) < 5) : exit()
   bsize = args.params[0]
   nc = args.params[1]
   d = args.params[2]
   ngroups = args.params[3]
   nlayers = args.params[4]
   use_bn = args.bn
   use_maxpool = args.maxpool
   use_avgpool = args.avgpool

   print ("using batch-size =",bsize,"nchannels =",nc,"dim =",d,"ngroups =",ngroups,"nlayers =",nlayers)
   if (use_bn): print("using batch normalization layer")
   if (use_maxpool): print("using maxpool  layer")

   for ib in range(0,bsize):
      xa = torch.ones([1, 1, d]) * (ib+1)
      if (nc > 1) :
         xb = xa.neg()
         xc = torch.cat((xa,xb),1)  # concatenate tensors
         if (nc > 2) :
            xd = torch.zeros([1,nc-2,d])
            xc = torch.cat((xa,xb,xd),1)
      else:
         xc = xa

      #concatenate tensors
      if (ib == 0) :
         xinput = xc
      else :
         xinput = torch.cat((xinput,xc),0)

   print("input data",xinput.shape)
   print(xinput)

   name = model_name("Conv1dModel", bsize, use_bn, use_maxpool, use_avgpool)

   model = Net(nc,ngroups,nlayers, use_bn, use_maxpool, use_avgpool)
   print(model)

   model(xinput)

   export_onnx(model, xinput, name, use_bn)

   write_reference_output(model, xinput, name)


if __name__ == '__main__':
    main()
