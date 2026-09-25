### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### Example of inference with SOFIE and RDataFrame, of a model trained with PyTorch.
### First, generate the input ONNX model by running `TMVA_SOFIE_PyTorch_HiggsModel.py`.
###
### This tutorial parses the input model and runs the inference using ROOT's JITing capability.
###
### \macro_code
### \macro_output
### \author Lorenzo Moneta

from os.path import exists

import ROOT

# check if the input file exists
modelFile = "HiggsModel.onnx"
modelName = "HiggsModel"

if not exists(modelFile):
    raise FileNotFoundError("You need to run TMVA_SOFIE_PyTorch_HiggsModel.py to generate the ONNX trained model")

# parse the input ONNX model into RModel object
parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
model = parser.Parse(modelFile)

# generating inference code
model.Generate()
model.OutputGenerated("Higgs_trained_model_generated.hxx")
model.PrintGenerated()

# compile using ROOT JIT trained model
print("compiling SOFIE model and inference helper....")
ROOT.gInterpreter.Declare('#include "Higgs_trained_model_generated.hxx"\n#include <array>\n#include <vector>')

# A SOFIE Session holds the model weights and the intermediate buffers and is not
# thread-safe: create one Session per RDataFrame processing slot and dispatch on
# the slot number. This tutorial runs single-threaded, so a single Session is enough.
# The weights file name is passed explicitly because the generated header was
# written under a custom name.
ROOT.gInterpreter.Declare(
    'std::vector<TMVA_SOFIE_' + modelName + '::Session> sofie_sessions{TMVA_SOFIE_' + modelName +
    '::Session("Higgs_trained_model_generated.dat")};')

# Declare the inference function for RDataFrame: it assembles the model input
# tensor from the columns and evaluates the model. The column order must match
# the ordering of the model input tensor.
ROOT.gInterpreter.Declare("""
double sofie_eval(unsigned int slot, float m_jj, float m_jjj, float m_lv, float m_jlv, float m_bb, float m_wbb,
                  float m_wwbb)
{
   std::array<float, 7> input{m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb};
   return sofie_sessions[slot].infer(input.data())[0];
}
""")

# run inference over input data
inputFile = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"
df1 = ROOT.RDataFrame("sig_tree", inputFile)
h1 = df1.Define("DNN_Value", "sofie_eval(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)").Histo1D(("h_sig", "", 100, 0, 1),"DNN_Value")

df2 = ROOT.RDataFrame("bkg_tree", inputFile)
h2 = df2.Define("DNN_Value", "sofie_eval(rdfslot_,m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb)").Histo1D(("h_bkg", "", 100, 0, 1),"DNN_Value")

# run over the input data once, combining both RDataFrame graphs.
ROOT.RDF.RunGraphs([h1, h2])

print("Number of signal entries",h1.GetEntries())
print("Number of background entries",h2.GetEntries())

h1.SetLineColor("kRed")
h2.SetLineColor("kBlue")

c1 = ROOT.TCanvas()
ROOT.gStyle.SetOptStat(0)

h2.DrawClone()
h1.DrawClone("SAME")
