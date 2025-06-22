### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### Example of inference with SOFIE and RDataFrame, of a model trained with Keras.
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


# check if the input file exists
modelFile = "Higgs_trained_model.h5"
modelName = "Higgs_trained_model";

if not exists(modelFile):
    raise FileNotFoundError("You need to run TMVA_Higgs_Classification.C to generate the Keras trained model")

# parse the input Keras model into RModel object
model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(modelFile)

# generating inference code
model.Generate()
model.OutputGenerated("Higgs_trained_model_generated.hxx")
model.PrintGenerated()

# compile using ROOT JIT trained model
print("compiling SOFIE model and functor....")
ROOT.gInterpreter.Declare('#include "Higgs_trained_model_generated.hxx"')
ROOT.gInterpreter.Declare('auto sofie_functor = TMVA::Experimental::SofieFunctor<7,TMVA_SOFIE_'+modelName+'::Session>(0,"Higgs_trained_model_generated.dat");')

# run inference over input data
inputFile = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"
df1 = ROOT.RDataFrame("sig_tree", inputFile)
h1 = df1.Define("DNN_Value", "sofie_functor(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)").Histo1D(("h_sig", "", 100, 0, 1),"DNN_Value")

df2 = ROOT.RDataFrame("bkg_tree", inputFile)
h2 = df2.Define("DNN_Value", "sofie_functor(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)").Histo1D(("h_bkg", "", 100, 0, 1),"DNN_Value")

# run over the input data once, combining both RDataFrame graphs.
ROOT.RDF.RunGraphs([h1, h2]);

print("Number of signal entries",h1.GetEntries())
print("Number of background entries",h2.GetEntries())

h1.SetLineColor("kRed")
h2.SetLineColor("kBlue")

c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)

h2.DrawClone()
h1.DrawClone("SAME")
