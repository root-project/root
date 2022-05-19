### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
### This macro provides an example of using a trained model with Keras
### and make inference using SOFIE directly from Numpy 
### This macro uses as input a Keras model generated with the
### TMVA_Higgs_Classification.C tutorial
### You need to run that macro before this one.
### In this case we are parsing the input file and then run the inference in the same
### macro making use of the ROOT JITing capability
###
###
### \macro_code
### \macro_output
### \author Lorenzo Moneta

import ROOT
import numpy as np


ROOT.TMVA.PyMethodBase.PyInitialize()


# check if the input file exists
modelFile = "Higgs_trained_model.h5"
if (ROOT.gSystem.AccessPathName(modelFile)) :
    ROOT.Info("TMVA_SOFIE_RDataFrame","You need to run TMVA_Higgs_Classification.C to generate the Keras trained model")
    exit()


# parse the input Keras model into RModel object
model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(modelFile)

generatedHeaderFile = modelFile.replace(".h5",".hxx")
print("Generating inference code for the Keras model from ",modelFile,"in the header ", generatedHeaderFile)
#Generating inference code
model.Generate()
model.OutputGenerated(generatedHeaderFile)
model.PrintGenerated()

# now compile using ROOT JIT trained model
modelName = modelFile.replace(".h5","")
print("compiling SOFIE model ", modelName)
ROOT.gInterpreter.Declare('#include "' + generatedHeaderFile + '"')


generatedHeaderFile = modelFile.replace(".h5",".hxx")
print("Generating inference code for the Keras model from ",modelFile,"in the header ", generatedHeaderFile)
#Generating inference

inputFileName = "Higgs_data.root"
inputFile = "http://root.cern.ch/files/" + inputFileName





# make SOFIE inference on signal data

df1 = ROOT.RDataFrame("sig_tree", inputFile)
sigData = df1.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
#print(sigData)

# stack all the 7 numpy array in a single array (nevents x nvars)
xsig = np.column_stack(list(sigData.values()))
dataset_size = xsig.shape[0]
print("size of data", dataset_size)

#instantiate SOFIE session class 
session = ROOT.TMVA_SOFIE_Higgs_trained_model.Session()

hs = ROOT.TH1D("hs","Signal result",100,0,1)
for i in range(0,dataset_size):
    result = session.infer(xsig[i,:])
    hs.Fill(result[0])


# make SOFIE inference on background data
df2 = ROOT.RDataFrame("bkg_tree", inputFile)
bkgData = df2.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])

xbkg = np.column_stack(list(bkgData.values()))
dataset_size = xbkg.shape[0]

hb = ROOT.TH1D("hb","Background result",100,0,1)
for i in range(0,dataset_size):
    result = session.infer(xbkg[i,:])
    hb.Fill(result[0])


c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)
hs.SetLineColor(ROOT.kRed)
hs.Draw()
hb.SetLineColor(ROOT.kBlue)
hb.Draw("SAME")
c1.BuildLegend()
c1.Draw()


print("Number of signal entries",hs.GetEntries())
print("Number of background entries",hb.GetEntries())

