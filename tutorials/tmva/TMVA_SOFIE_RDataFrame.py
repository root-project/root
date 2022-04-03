### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
### This macro provides an example of using a trained model with Keras
### and make inference using SOFIE and RDataFrame
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
ROOT.gInterpreter.Declare('auto sofie_functor = TMVA::Experimental::SofieFunctor<7,TMVA_SOFIE_'+modelName+'::Session>(0);')


df1 = ROOT.RDataFrame("sig_tree", inputFile)
h1 = df1.Define("DNN_Value", "sofie_functor(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)").Histo1D(("h_sig", "", 100, 0, 1),"DNN_Value")

df2 = ROOT.RDataFrame("bkg_tree", inputFile)
h2 = df2.Define("DNN_Value", "sofie_functor(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)").Histo1D(("h_bkg", "", 100, 0, 1),"DNN_Value")

print("Number of signal entries",h1.GetEntries())
print("Number of background entries",h2.GetEntries())

h1.SetLineColor(ROOT.kRed)
h2.SetLineColor(ROOT.kBlue)

c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)

h2.DrawClone()
h1.DrawClone("SAME")

