## \file
## \ingroup tutorial_ml
## \notebook -nodraw
## This macro provides a simple example for:
##  - creating a model with Pytorch and export to ONNX
##  - parsing the ONNX file with SOFIE and generate C++ code
##  - compiling the model using ROOT Cling
##  - run the code and optionally compare with ONNXRuntime
##
##
## \macro_code
## \macro_output
## \author Lorenzo Moneta


import torch
import torch.nn as nn
import ROOT
import numpy as np

def CreateAndTrainModel(modelName):

   model = nn.Sequential(
           nn.Linear(32,16),
           nn.ReLU(),
           nn.Linear(16,8),
           nn.ReLU(),
           nn.Linear(8,2),
           nn.Softmax(dim=1)
           )

   criterion = nn.MSELoss()
   optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


   #train model with the random data
   for i in range(500):
      x=torch.randn(2,32)
      y=torch.randn(2,2)
      y_pred = model(x)
      loss = criterion(y_pred,y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

   model.eval()
   #export the model to ONNX
   modelFile = modelName + ".onnx"
   dummy_x = torch.randn(1,32)
   model(dummy_x)
   torch.onnx.export(model, dummy_x, modelFile,  export_params=True,
                     dynamo = True,   # this is for new PyTorch exporter from version 2.5
                     external_data=False,  # this important to avoid weights saved in a different onnx.data file
                     input_names=["input"],
                     output_names=["output"])
   print("model exported to ONNX as",modelFile)
   return modelFile



def ParseModel(modelFile, verbose=False):

   parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
   model = parser.Parse(modelFile,verbose)
   #
   #print model weights
   if (verbose):
      model.PrintInitializedTensors()
      data = model.GetTensorData['float']('0weight')
      print("0weight",data)
      data = model.GetTensorData['float']('2weight')
      print("2weight",data)

   # Generating inference code
   model.Generate();
   #generate header file (and .dat file) with modelName+.hxx
   model.OutputGenerated();
   if (verbose) :
       model.PrintGenerated()

   modelCode = modelFile.replace(".onnx",".hxx")
   print("Generated model header file ",modelCode)
   return modelCode

###################################################################
## Step 1 : Create and Train model
###################################################################

#use an arbitrary modelName
modelName = "LinearModel"
modelFile = CreateAndTrainModel(modelName)


###################################################################
## Step 2 : Parse model and generate inference code with SOFIE
###################################################################

modelCode = ParseModel(modelFile, False)

###################################################################
## Step 3 : Compile the generated C++ model code
###################################################################

ROOT.gInterpreter.Declare('#include "' + modelCode + '"')

###################################################################
## Step 4: Evaluate the model
###################################################################

#get first the SOFIE session namespace
sofie = getattr(ROOT, 'TMVA_SOFIE_' + modelName)
session = sofie.Session()

x = np.random.normal(0,1,(1,32)).astype(np.float32)
print("\n************************************************************")
print("Running inference with SOFIE ")
print("\ninput to model is ",x)
y = session.infer(x)
# output shape is (1,2)
y_sofie = np.asarray(y.data())
print("-> output using SOFIE = ", y_sofie)

#check inference with onnx
try:
   import onnxruntime as ort
    # Load model
   print("Running inference with ONNXRuntime ")
   ort_session = ort.InferenceSession(modelFile)

   # Run inference
   outputs = ort_session.run(None, {"input": x})
   y_ort = outputs[0]
   print("-> output using ORT =", y_ort)

   testFailed =  abs(y_sofie-y_ort) > 0.01
   if (np.any(testFailed)):
      raiseError('Result is different between SOFIE and ONNXRT')
   else :
      print("OK")

except ImportError:
   print("Missing ONNXRuntime: skipping comparison test")
