### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
### Example of inference with SOFIE using a set of models trained with Keras.
### This tutorial shows how to store several models in a single header file and
### the weights in a ROOT binary file.
### The models are then evaluated using the RDataFrame
### First, generate the input model by running `TMVA_Higgs_Classification.C`.
###
### This tutorial parses the input model and runs the inference using ROOT's JITing capability.
###
### \macro_code
### \macro_output
### \author Lorenzo Moneta

import ROOT
from os.path import exists

ROOT.TMVA.PyMethodBase.PyInitialize()


## generate and train Keras models with different architectures

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

def CreateModel(nlayers = 4, nunits = 64):
   model = Sequential()
   model.add(Dense(nunits, activation='relu',input_dim=7))
   for i in range(1,nlayers) :
      model.add(Dense(nunits, activation='relu'))

   model.add(Dense(1, activation='sigmoid'))
   model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), weighted_metrics = ['accuracy'])
   model.summary()
   return model

def PrepareData() :
   #get the input data
   inputFileName = "Higgs_data.root"
   inputFile = "http://root.cern.ch/files/" + inputFileName

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
   model.fit(x,y,epochs=10,batch_size=50)
   modelFile = name + '.h5'
   model.save(modelFile)
   return modelFile

### run the models

x_train, y_train, x_test, y_test = PrepareData()

## create models and train them

model1 = TrainModel(CreateModel(4,64),x_train, y_train, 'Higgs_Model_4L_50')
model2 = TrainModel(CreateModel(4,64),x_train, y_train, 'Higgs_Model_4L_200')
model3 = TrainModel(CreateModel(4,64),x_train, y_train, 'Higgs_Model_2L_500')

#evaluate with SOFIE the 3 trained models


def GenerateModelCode(modelFile, generatedHeaderFile):
   model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(modelFile)

   print("Generating inference code for the Keras model from ",modelFile,"in the header ", generatedHeaderFile)
   #Generating inference code using a ROOT binary file
   model.Generate(ROOT.TMVA.Experimental.SOFIE.Options.kRootBinaryWeightFile)
   # add option to append to the same file the generated headers (pass True for append flag)
   model.OutputGenerated(generatedHeaderFile, True)
   #model.PrintGenerated()
   return generatedHeaderFile


generatedHeaderFile = "Higgs_Model.hxx"
#need to remove existing header file since we are appending on same one
import os
if (os.path.exists(generatedHeaderFile)):
   weightFile = "Higgs_Model.root"
   print("removing existing files", generatedHeaderFile,weightFile)
   os.remove(generatedHeaderFile)
   os.remove(weightFile)

GenerateModelCode(model1, generatedHeaderFile)
GenerateModelCode(model2, generatedHeaderFile)
GenerateModelCode(model3, generatedHeaderFile)

#compile the generated code

ROOT.gInterpreter.Declare('#include "' + generatedHeaderFile + '"')


#run the inference on the test data
session1 = ROOT.TMVA_SOFIE_Higgs_Model_4L_50.Session("Higgs_Model.root")
session2 = ROOT.TMVA_SOFIE_Higgs_Model_4L_200.Session("Higgs_Model.root")
session3 = ROOT.TMVA_SOFIE_Higgs_Model_2L_500.Session("Higgs_Model.root")

hs1 = ROOT.TH1D("hs1","Signal result 4L 50",100,0,1)
hs2 = ROOT.TH1D("hs2","Signal result 4L 200",100,0,1)
hs3 = ROOT.TH1D("hs3","Signal result 2L 500",100,0,1)

hb1 = ROOT.TH1D("hb1","Background result 4L 50",100,0,1)
hb2 = ROOT.TH1D("hb2","Background result 4L 200",100,0,1)
hb3 = ROOT.TH1D("hb3","Background result 2L 500",100,0,1)

def EvalModel(session, x) :
   result = session.infer(x)
   return result[0]

for i in range(0,x_test.shape[0]):
   result1 = EvalModel(session1, x_test[i,:])
   result2 = EvalModel(session2, x_test[i,:])
   result3 = EvalModel(session3, x_test[i,:])
   if (y_test[i] == 1) :
      hs1.Fill(result1)
      hs2.Fill(result2)
      hs3.Fill(result3)
   else:
      hb1.Fill(result1)
      hb2.Fill(result2)
      hb3.Fill(result3)

def PlotHistos(hs,hb):
   hs.SetLineColor(ROOT.kRed)
   hb.SetLineColor(ROOT.kBlue)
   hs.Draw()
   hb.Draw("same")

c1 = ROOT.TCanvas()
c1.Divide(1,3)
c1.cd(1)
PlotHistos(hs1,hb1)
c1.cd(2)
PlotHistos(hs2,hb2)
c1.cd(3)
PlotHistos(hs3,hb3)
c1.Draw()

## draw also ROC curves

def GetContent(h) :
   n = h.GetNbinsX()
   x = ROOT.std.vector['float'](n)
   w = ROOT.std.vector['float'](n)
   for  i in range(0,n):
      x[i] = h.GetBinCenter(i+1)
      w[i] = h.GetBinContent(i+1)
   return x,w

def MakeROCCurve(hs, hb) :
   xs,ws = GetContent(hs)
   xb,wb = GetContent(hb)
   roc = ROOT.TMVA.ROCCurve(xs,xb,ws,wb)
   print("ROC integral for ",hs.GetName(), roc.GetROCIntegral())
   curve = roc.GetROCCurve()
   curve.SetName(hs.GetName())
   return roc,curve

c2 = ROOT.TCanvas()

r1,curve1 = MakeROCCurve(hs1,hb1)
curve1.SetLineColor(ROOT.kRed)
curve1.Draw("AC")

r2,curve2 = MakeROCCurve(hs2,hb2)
curve2.SetLineColor(ROOT.kBlue)
curve2.Draw("C")

r3,curve3 = MakeROCCurve(hs3,hb3)
curve3.SetLineColor(ROOT.kGreen)
curve3.Draw("C")

c2.Draw()
