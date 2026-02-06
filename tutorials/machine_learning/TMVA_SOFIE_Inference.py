### \file
### \ingroup tutorial_ml
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

from os.path import exists

import numpy as np
import ROOT

# check if the input file exists
modelFile = "HiggsModel.keras"

if not exists(modelFile):
    raise FileNotFoundError("You need to run TMVA_Higgs_Classification.C to generate the Keras trained model")


# parse the input Keras model into RModel object
model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(modelFile)

generatedHeaderFile = modelFile.replace(".keras",".hxx")
print("Generating inference code for the Keras model from ",modelFile,"in the header ", generatedHeaderFile)
#Generating inference code
model.Generate()
model.OutputGenerated(generatedHeaderFile)
model.PrintGenerated()

# now compile using ROOT JIT trained model
modelName = modelFile.replace(".keras","")
print("compiling SOFIE model ", modelName)
ROOT.gInterpreter.Declare('#include "' + generatedHeaderFile + '"')


generatedHeaderFile = modelFile.replace(".keras",".hxx")
print("Generating inference code for the Keras model from ",modelFile,"in the header ", generatedHeaderFile)
#Generating inference

inputFileName = "Higgs_data.root"
inputFile = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/" + inputFileName





# make SOFIE inference on signal data

df1 = ROOT.RDataFrame("sig_tree", inputFile)
sigData = df1.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])
#print(sigData)

# stack all the 7 numpy array in a single array (nevents x nvars)
xsig = np.column_stack(list(sigData.values()))
dataset_size = xsig.shape[0]
print("size of signal data", dataset_size)

#instantiate SOFIE session class
#session = ROOT.TMVA_SOFIE_HiggsModel.Session()
#get the sofie session namespace
sofie = getattr(ROOT, 'TMVA_SOFIE_' + modelName)
session = sofie.Session()

print("Evaluating SOFIE models on signal data")
hs = ROOT.TH1D("hs","Signal result",100,0,1)
for i in range(0,dataset_size):
    result = session.infer(xsig[i,:])
    if (i % dataset_size/10 == 0) :
      print("result for signal event ",i,result[0])
    hs.Fill(result[0])

print("using RDsataFrame to extract input data in a numpy array")
# make SOFIE inference on background data
df2 = ROOT.RDataFrame("bkg_tree", inputFile)
bkgData = df2.AsNumpy(columns=['m_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'])

xbkg = np.column_stack(list(bkgData.values()))
dataset_size = xbkg.shape[0]
print("size of background data", dataset_size)

hb = ROOT.TH1D("hb","Background result",100,0,1)
for i in range(0,dataset_size):
    result = session.infer(xbkg[i,:])
    if (i % dataset_size/10 == 0) :
      print("result for background event ",i,result[0])

    hb.Fill(result[0])


c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)
hs.SetLineColor("kRed")
hs.Draw()
hb.SetLineColor("kBlue")
hb.Draw("SAME")
c1.BuildLegend()
c1.Draw()


print("Number of signal entries",hs.GetEntries())
print("Number of background entries",hb.GetEntries())

