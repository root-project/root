### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### This macro provides a simple example for the parsing of Keras .keras file
### into RModel object and further generating the .hxx header files for inference.
###
### \macro_code
### \macro_output
### \author Sanjiban Sengupta and Lorenzo Moneta



import ROOT

# Enable ROOT in batch mode (same effect as -nodraw)
ROOT.gROOT.SetBatch(True)

# -----------------------------------------------------------------------------
# Step 1: Create and train a simple Keras model (via embedded Python)
# -----------------------------------------------------------------------------

import numpy as np
from tensorflow.keras.layers import Activation, Dense, Input, Softmax
from tensorflow.keras.models import Model

input=Input(shape=(4,),batch_size=2)
x=Dense(32)(input)
x=Activation('relu')(x)
x=Dense(16,activation='relu')(x)
x=Dense(8,activation='relu')(x)
x=Dense(2)(x)
output=Softmax()(x)
model=Model(inputs=input,outputs=output)

randomGenerator=np.random.RandomState(0)
x_train=randomGenerator.rand(4,4)
y_train=randomGenerator.rand(4,2)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3, batch_size=2)
model.save('KerasModel.keras')
model.summary()

# -----------------------------------------------------------------------------
# Step 2: Use TMVA::SOFIE to parse the ONNX model
# -----------------------------------------------------------------------------

import ROOT

# Parse the ONNX model

model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse("KerasModel.keras")

# Generate inference code
model.Generate()
model.OutputGenerated()
#print generated code
print("\n**************************************************")
print("           Generated code")
print("**************************************************\n")
model.PrintGenerated()
print("**************************************************\n\n\n")

# Compile the generated code
ROOT.gInterpreter.Declare('#include "KerasModel.hxx"')


# -----------------------------------------------------------------------------
# Step 3: Run inference
# -----------------------------------------------------------------------------

#instantiate SOFIE session class
session = ROOT.TMVA_SOFIE_KerasModel.Session()

# Input tensor (same shape as training input)
x = np.array([[0.1, 0.2, 0.3, 0.4],[0.5, 0.6, 0.7, 0.8]], dtype=np.float32)

# Run inference
y = session.infer(x)

print("Inference output:", y)

