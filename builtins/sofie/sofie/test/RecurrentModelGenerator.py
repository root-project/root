#!/usr/bin/python3

### generate recurrent (RNN/LSTM/GRU) model using Pytorch

import torch
import torch.nn as nn

from ModelGeneratorUtils import make_parser, model_name, export_onnx, write_reference_output

verbose = False


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

      # Passing in the input and hidden state into the model and obtaining outputs
      rc_out,self.hidden_cell  = self.rc(x)

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

   global verbose

   parser = make_parser('parameters for the Recurrent network : batchSize , inputSize, seqSize, hiddenSize, nLayers ',
                        recurrent=True)
   args = parser.parse_args()

   if (len(args.params) < 5) : exit()
   bsize = args.params[0]
   nd = args.params[1]
   nt = args.params[2]
   nh = args.params[3]
   nl = args.params[4]

   type = "RNN"
   if (args.lstm): type = "LSTM"
   if (args.gru): type = "GRU"

   verbose = args.v

   print ("using batch-size =",bsize,"inputSize=",nd,"sequence length",nt,"hidden Size",nh,"nlayers",nl)

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

   name = model_name(type + "Model", bsize)

   model = Net(type, nd, nh, nl)
   print(model)

   model(xinput)

   # the new exporter does not work for recurrent models
   export_onnx(model, xinput, name, dynamo=False)

   write_reference_output(model, xinput, name)


if __name__ == '__main__':
    main()
