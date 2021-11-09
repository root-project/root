#!/usr/bin/python3

### generate COnv2d model using Pytorch

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


result = []

class Net(nn.Module):
    
    def __init__(self, nc = 1, ng = 1, nl = 4):
        super(Net, self).__init__()

        self.nc = nc
        self.ng = ng
        self.nl = nl
        
        self.conv0  = nn.Conv2d(in_channels=self.nc,   out_channels=4, kernel_size=2, groups=1, stride=1, padding=1)
        # output is 4x4 with optionally using group convolution
        self.conv1  = nn.Conv2d(in_channels=4,   out_channels=8, groups = self.ng,   kernel_size=3, stride=1, padding=1)
        #output is same 4x4
        self.conv2  = nn.Conv2d(in_channels=8,   out_channels=4, kernel_size=3, stride=1, padding=1)
        #use stride last layer 
        self.conv3 =  nn.Conv2d(in_channels=4,   out_channels=1,   kernel_size=2, stride=2, padding=0)


    def forward(self, x):
      x = self.conv0(x)
      x = F.relu(x)
      if (self.nl == 1) : return x
      x = self.conv1(x)
      x = F.relu(x)
      #print(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.conv3(x)

      return x

def main():

   #print(arguments)
   parser = argparse.ArgumentParser(description='PyTorch model generator')
   parser.add_argument('params', type=int, nargs='+',
                    help='parameters for the Conv network : batchSize , inputChannels, inputImageSize, nGroups, nLayers ')
   
   #  parser.add_argument('--save-onnx', action='store_true', default=False,
   #                      help='For Saving the current Model in the ONNX format')
   #  parser.add_argument('--load', default=False,
   #                      help='For Loading the saved Model from PyTorch format')
   

   args = parser.parse_args()
  
   #args.params = (4,2,4,1,4)

   np = len(args.params)
   if (np < 5) : exit()
   bsize = args.params[0]
   nc = args.params[1] 
   d = args.params[2]
   ngroups = args.params[3]
   nlayers = args.params[4] 

   print ("using batch-size =",bsize,"nchannels =",nc,"dim =",d,"ngroups =",ngroups,"nlayers =",nlayers)

    #sample = torch.zeros([2,1,5,5])
   input  = torch.zeros([])
   for ib in range(0,bsize):
      xa = torch.ones([1, 1, d, d]) * (ib+1)
      if (nc > 1) : 
         xb = xa.neg()
         xc = torch.cat((xa,xb),1)  # concatenate tensors
         if (nc > 2) :
            xd = torch.zeros([1,nc-2,d,d])
            xc = torch.cat((xa,xb,xd),1)
        
      #concatenate tensors 
      if (ib == 0) : 
         xinput = xc
      else :
         xinput = torch.cat((xinput,xc),0) 

   print("input data",xinput.shape)
   print(xinput)
   
   name = "Conv2dModel"

   saveOnnx=True
   loadModel=False
   savePtModel = False

    
   model = Net(nc,ngroups,nlayers)
    
   model(xinput)

   if loadModel :
        print('Loading model from file....')
        checkpoint = torch.load(name + ".pt")
        model.load_state_dict(checkpoint['model_state_dict'])

  
   y = model.forward(xinput)

   print("output data : shape, ",y.shape)
   print(y)

   if savePtModel :
      torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

   if saveOnnx:
        torch.onnx.export(
                model,
                xinput,
                name + ".onnx",
                export_params=True
        )

   outSize = y.nelement()
   yvec = y.reshape([outSize])
   # for i in range(0,outSize):
   #      print(float(yvec[i]))

   f = open(name + ".out", "w")
   for i in range(0,outSize):
        f.write(str(float(yvec[i]))+" ")
        
        
    

if __name__ == '__main__':
    main()
