### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### This macro provides a simple example for the parsing of PyTorch .pt file
### into RModel object and further generating the .hxx header files for inference.
###
### \macro_code
### \macro_output
### \author Sanjiban Sengupta

import sys

import ROOT
import torch
import torch.nn as nn

SOFIE = ROOT.TMVA.Experimental.SOFIE

# Python and C++ write to separate stdout buffers; flush both on every line so
# that the Python prints and the RModel printouts appear in order
sys.stdout.reconfigure(line_buffering=True)
ROOT.gInterpreter.ProcessLine("std::cout << std::unitbuf;")

# ------------------------------------------------------------------------------
# Step 1: Create, train and save a simple PyTorch model
# ------------------------------------------------------------------------------

model = nn.Sequential(
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(2, 32)
y = torch.randn(2, 8)

for i in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
m = torch.jit.script(model)
torch.jit.save(m, "PyTorchModel.pt")

# ------------------------------------------------------------------------------
# Step 2: Parse the saved PyTorch .pt file with TMVA::SOFIE
# ------------------------------------------------------------------------------

# Parsing a PyTorch model requires the shape and data-type of input tensor
# Data-type of input tensor defaults to Float if not specified
input_shapes = ROOT.std.vector["std::vector<std::size_t>"]()
input_shapes.push_back([2, 32])

# Parsing the saved PyTorch .pt file into RModel object
model = SOFIE.PyTorch.Parse("PyTorchModel.pt", input_shapes)

# Generating inference code
model.Generate()
model.OutputGenerated("PyTorchModel.hxx")

# Printing required input tensors
print("\n")
model.PrintRequiredInputTensors()

# Printing initialized tensors (weights)
print("\n")
model.PrintInitializedTensors()

# Printing intermediate tensors
print("\n")
model.PrintIntermediateTensors()

# Checking if tensor already exist in model
tensor_exists = bool(model.CheckIfTensorAlreadyExist("0weight"))
print(f'\n\nTensor "0weight" already exist: {str(tensor_exists).lower()}\n')

tensor_shape = model.GetTensorShape("0weight")
print('Shape of tensor "0weight": ' + ",".join(str(dim) for dim in tensor_shape) + ",")

tensor_type = model.GetTensorType("0weight")
print(f'\nData type of tensor "0weight": {SOFIE.ConvertTypeToString(tensor_type)}')

# Printing generated inference code
print()
model.PrintGenerated()
