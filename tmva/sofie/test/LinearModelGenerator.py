#!/usr/bin/python3

### generate Linear model using Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelGeneratorUtils import make_parser, model_name, export_onnx, write_reference_output


class Net(nn.Module):

    def __init__(self, nd = 1, nc = 1, nl = 4, use_bn = False):
        super(Net, self).__init__()

        self.nc = nc
        self.nl = nl
        self.use_bn = use_bn

        self.out0 = nn.Linear(in_features=nd, out_features=50)
        if (self.use_bn): self.bn1 = nn.BatchNorm1d(50)
        self.out1 = nn.Linear(in_features=50, out_features=100)
        self.out2 = nn.Linear(in_features=100, out_features=100)
        self.out3 = nn.Linear(in_features = 100, out_features = nc)

    def forward(self, x):

      x = self.out0(x)
      x = F.relu(x)
      #add bn layer
      if (self.use_bn):
         x = self.bn1(x)
      if (self.nl == 1) : return x
      x = self.out1(x)
      x = F.relu(x)
      x = self.out2(x)
      x = F.relu(x)
      x = self.out3(x)

      return x


def main():

   parser = make_parser('parameters for the Dense network : batchSize , inputChannels, nlayers ')
   args = parser.parse_args()

   nlayers = 4
   noutput = 4

   if (len(args.params) < 2) : exit()
   bsize = args.params[0]
   d = args.params[1]
   if (len(args.params) > 2) : nlayers = args.params[2]

   print ("using batch-size =",bsize,"input dim =",d,"nlayers =",nlayers)

   use_bn = args.bn
   if (use_bn) : print("using batch normalization layer")

   for ib in range(0,bsize):
      xa = torch.ones([1,d]) * (ib+1)
      #concatenate tensors
      if (ib == 0) :
         xinput = xa
      else :
         xinput = torch.cat((xinput,xa),0)

   xinput_test = xinput
   #in case of batch normalization generate different data for training
   if (use_bn):
       for id in range(0,d):
           xa = torch.randn([bsize,1]) * (id+1) + id * torch.ones([bsize,1])
           #concatenate tensors
           if (id == 0) :
               xinput = xa
           else :
               xinput = torch.cat((xinput,xa),1)

   print("input data",xinput.shape)
   print(xinput)

   name = model_name("LinearModel", bsize, use_bn)

   model = Net(d,noutput,nlayers,use_bn)

   model(xinput)

   export_onnx(model, xinput, name, use_bn)

   # the expected output is evaluated on the test input, which differs from the
   # training input when batch normalization is used
   write_reference_output(model, xinput_test, name)


if __name__ == '__main__':
    main()
