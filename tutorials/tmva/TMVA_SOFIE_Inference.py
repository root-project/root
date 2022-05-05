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

#generatedFile = modelFile
generatedFile = modelFile.replace(".h5",".hxx")
print(modelFile, generatedFile)
#Generating inference code
model.Generate()
model.OutputGenerated(generatedFile)
model.PrintGenerated()

# now compile using ROOT JIT trained model

inputFileName = "Higgs_data.root"
inputFile = "http://root.cern.ch/files/" + inputFileName


print("compiling SOFIE model and functor....")

modelName = "Higgs_trained_model"
ROOT.gInterpreter.Declare('#include "' + generatedFile + '"')



# make SOFIE inference on signal data

df1 = ROOT.RDataFrame("sig_tree", inputFile)
sigData = df1.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
#print(sigData)

dataset_size = len(list(sigData.values())[0])

session = ROOT.TMVA_SOFIE_Higgs_trained_model.Session()

hs = ROOT.TH1D("hs","Signal result",100,0,1)
for i in range(0,dataset_size):
    xsig = np.array([sigData[x][i] for x in sigData.keys()])
    result = session.infer(xsig)
    hs.Fill(result[0])


# make SOFIE inference on background data
df2 = ROOT.RDataFrame("bkg_tree", inputFile)
bkgData = df2.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])

hb = ROOT.TH1D("hb","Background result",100,0,1)
dataset_size = len(list(bkgData.values())[0])
for i in range(0,dataset_size):
    xbkg = np.array([bkgData[x][i] for x in bkgData.keys()])
    result = session.infer(xbkg)
    hb.Fill(result[0])


c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)
hs.SetLineColor(ROOT.kRed)
hs.Draw()
hb.SetLineColor(ROOT.kBlue)
hb.Draw("SAME")


print("Number of signal entries",hs.GetEntries())
print("Number of background entries",hb.GetEntries())

