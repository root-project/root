### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### This macro run the SOFIE parser on the Keras model
### obtaining running TMVA_Higgs_Classification.C
### You need to run that macro before this one
###
### \author Lorenzo Moneta


import ROOT
from os.path import exists
import numpy as np
from keras import models, layers
from sklearn.model_selection import train_test_split

def CreateModel(nlayers = 4, nunits = 64):
   input = layers.Input(shape=(7,))
   x = input
   for i in range(1,nlayers) :
      y = layers.Dense(nunits, activation='relu')(x)
      x = y

   output = layers.Dense(1, activation='sigmoid')(x)
   model = models.Model(input, output)
   model.compile(loss = 'binary_crossentropy', optimizer = 'adam', weighted_metrics = ['accuracy'])
   model.summary()
   return model

def PrepareData() :
   #get the input data
   inputFile = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"

   df1 = ROOT.RDataFrame("sig_tree", inputFile)
   sigData = df1.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
   #print(sigData)

   # stack all the 7 numpy array in a single array (nevents x nvars)
   xsig = np.column_stack(list(sigData.values()))
   data_sig_size = xsig.shape[0]
   print("size of data", data_sig_size)

   # make SOFIE inference on background data
   df2 = ROOT.RDataFrame("bkg_tree", inputFile)
   bkgData = df2.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
   xbkg = np.column_stack(list(bkgData.values()))
   data_bkg_size = xbkg.shape[0]

   ysig = np.ones(data_sig_size)
   ybkg = np.zeros(data_bkg_size)
   inputs_data = np.concatenate((xsig,xbkg),axis=0)
   inputs_targets = np.concatenate((ysig,ybkg),axis=0)

   #split data in training and test data

   x_train, x_test, y_train, y_test = train_test_split(
        inputs_data, inputs_targets, test_size=0.50, random_state=1234)

   return x_train, y_train, x_test, y_test

def TrainModel(model, x, y, name) :
   model.fit(x,y,epochs=5,batch_size=50)
   modelFile = name + '.keras'
   model.save(modelFile)
   return modelFile


def  GenerateCode(modelFile = "model.keras") :

   #check if the input file exists
   if not exists(modelFile):
      raise FileNotFoundError("INput model file not existing. You need to run TMVA_Higgs_Classification.C to generate the Keras trained model")


   #parse the input Keras model into RModel object (force batch size to be 1)
   model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(modelFile)

   #Generating inference code
   model.Generate()
   model.OutputGenerated()

   modelName = modelFile.replace(".keras","")
   return modelName

###################################################################
## Step 1 : Create and Train model
###################################################################

x_train, y_train, x_test, y_test = PrepareData()
#create dense model with 3 layers of 64 units
model = CreateModel(3,64)
modelFile = TrainModel(model,x_train, y_train, 'HiggsModel')

###################################################################
## Step 2 : Parse model and generate inference code with SOFIE
###################################################################

modelName = GenerateCode(modelFile)
modelHeaderFile = modelName + ".hxx"

###################################################################
## Step 3 : Compile the generated C++ model code
###################################################################

ROOT.gInterpreter.Declare('#include "' + modelHeaderFile + '"')

###################################################################
## Step 4: Evaluate the model
###################################################################

#get first the SOFIE session namespace
sofie = getattr(ROOT, 'TMVA_SOFIE_' + modelName)
session = sofie.Session()

x = np.random.normal(0,1,7).astype(np.float32)
y = session.infer(x)
ykeras = model(x.reshape(1,7)).numpy()

print("input to model is ",x, "\n\t -> output using SOFIE = ", y[0], " using Keras = ", ykeras[0])

if (abs(y[0]-ykeras[0]) > 0.01) :
   raiseError('Result is different between SOFIE and Keras')

print("OK")



