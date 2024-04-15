#!/usr/bin/python3

### generate COnv2d model using Pytorch

from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


result = []
verbose=False

class Net(nn.Module):
    
   def __init__(self, type, input_size, hidden_size, num_layers=1, output_size=2):
      super(Net, self).__init__()
        

      if (type == "LSTM") :
            self.rc = nn.LSTM(input_size, hidden_size, num_layers,  batch_first=True)
      if (type == "GRU"):
            self.rc = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
      if (type == "RNN"):
            self.rc = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # FC layer 
      self.fc = nn.Linear(hidden_size, output_size)

      self.hidden_dim = hidden_size
      self.n_layers = num_layers

   def forward(self, x):

      batch_size = x.size(0)
        
      #Initializing hidden state for first input using method defined below
      #hidden = self.init_hidden(batch_size)

      # Passing in the input and hidden state into the model and obtaining outputs
      rc_out,self.hidden_cell  = self.rc(x)
        
      # Reshaping the outputs such that it can be fit into the fully connected layer
      #out = lstm_out.view(self.hidden_dim,-1)

      if (verbose):
         print("recurrent output", rc_out)
         print("hidden out ",self.hidden_cell)

      out = rc_out[:,-1,:]
      out = self.fc(out)
        
      return out

   def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
      hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))
      return hidden
        
def main():

   #print(arguments)
   parser = argparse.ArgumentParser(description='PyTorch model generator')
   parser.add_argument('params', type=int, nargs='+',
                    help='parameters for the Recurrent network : batchSize , inputSize, seqSize, hiddenSize, nLayers ')
   
   parser.add_argument('--lstm', action='store_true', default=False,
                         help='For using LSTM  layer')
   parser.add_argument('--gru', action='store_true', default=False,
                         help='For using GRU layer')
   # parser.add_argument('--avgpool', action='store_true', default=False,
   #                      help='For using average pool layer')
   parser.add_argument('--v', action='store_true', default=False,
                        help='For verbose mode')


   args = parser.parse_args()
  
   ##args.params = (1,3,10,4,1)

   np = len(args.params)
   if (np < 5) : exit()
   bsize = args.params[0]
   nd = args.params[1]
   nt = args.params[2] 
   nh = args.params[3]
   nl = args.params[4]


   type = "RNN"
   if (args.lstm): type = "LSTM"
   if (args.gru): type = "GRU"

   if (args.v) : verbose=True

   print ("using batch-size =",bsize,"inputSize=",nd,"sequence length",nt,"hidden Size",nh,"nlayers",nl)

   xinput = torch.zeros([])
   xb  = torch.zeros([])
   for ib in range(0, bsize):
      for it in range(0,nt) :
         xa = torch.ones([1, 1, nd]) * (it + 1) * pow(-1, ib + 2)
      #concatenate tensors 
         if (it == 0):
            xb = xa
         else:
            xb = torch.cat((xb,xa),1) 
      if (ib == 0) : 
         xinput = xb
      else :
         xinput = torch.cat((xinput,xb),0) 

   print("input data",xinput.shape)
   print(xinput)
  
   name = type + "Model"
   name += "_B" + str(bsize)

   saveOnnx=True
   loadModel=False
   savePtModel = False

    
   model = Net(type, nd, nh, nl)
   print(model)

   model(xinput)
 
   model.forward(xinput)

   if savePtModel :
      torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

   if saveOnnx:
        torch.onnx.export(
                model,
                xinput,
                name + ".onnx",
                export_params=True
        )

   if loadModel :
        print('Loading model from file....')
        checkpoint = torch.load(name + ".pt")
        model.load_state_dict(checkpoint['model_state_dict'])

   # evaluate model in test mode
   model.eval()
   y = model.forward(xinput)

   print("output data : shape, ",y.shape)
   print(y)

   outSize = y.nelement()
   yvec = y.reshape([outSize])

   f = open(name + ".out", "w")
   for i in range(0,outSize):
      f.write(str(float(yvec[i]))+" ")
        
        
    

if __name__ == '__main__':
    main()
