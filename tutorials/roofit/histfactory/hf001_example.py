## \file
## \ingroup tutorial_histfactory
## A ROOT script demonstrating  an example of writing and fitting a HistFactory model using Python.
##
## \macro_image
## \macro_code
## \macro_output
##
## \author George Lewis

import ROOT

"""
Create a HistFactory measurement from python
"""

InputFile = "./data/example.root"
if ROOT.gSystem.AccessPathName(InputFile):
    ROOT.Info("example.py", InputFile + " does not exist")
    ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
    if ROOT.gSystem.AccessPathName(InputFile):
        ROOT.Info("example.py", InputFile + " still does not exist. \n EXIT")
        exit()

# Create the measurement
meas = ROOT.RooStats.HistFactory.Measurement("meas", "meas")

meas.SetOutputFilePrefix("./results/example_UsingPy")
meas.SetPOI("SigXsecOverSM")
meas.AddConstantParam("Lumi")
meas.AddConstantParam("alpha_syst1")

meas.SetLumi(1.0)
meas.SetLumiRelErr(0.10)

# Create a channel

chan = ROOT.RooStats.HistFactory.Channel("channel1")
chan.SetData("data", InputFile)
chan.SetStatErrorConfig(0.05, "Poisson")

# Now, create some samples

# Create the signal sample
signal = ROOT.RooStats.HistFactory.Sample("signal", "signal", InputFile)
signal.AddOverallSys("syst1", 0.95, 1.05)
signal.AddNormFactor("SigXsecOverSM", 1, 0, 3)
chan.AddSample(signal)


# Background 1
background1 = ROOT.RooStats.HistFactory.Sample("background1", "background1", InputFile)
background1.ActivateStatError("background1_statUncert", InputFile)
background1.AddOverallSys("syst2", 0.95, 1.05)
chan.AddSample(background1)


# Background 1
background2 = ROOT.RooStats.HistFactory.Sample("background2", "background2", InputFile)
background2.ActivateStatError()
background2.AddOverallSys("syst3", 0.95, 1.05)
chan.AddSample(background2)


# Done with this channel
# Add it to the measurement:

meas.AddChannel(chan)

# Collect the histograms from their files,
# print some output,
meas.CollectHistograms()
meas.PrintTree()

# One can print XML code to an output directory:
# meas.PrintXML("xmlFromPyCode", meas.GetOutputFilePrefix())

# Now, do the measurement
ws = ROOT.RooStats.HistFactory.MakeModelAndMeasurementFast(meas)

# Retrieve the ModelConfig
modelConfig = ws["ModelConfig"]

# Extract the PDF and global observables
pdf = modelConfig.GetPdf()

# Perform the fit using Minos to get the correct asymmetric uncertainties
result = pdf.fitTo(
    ws["obsData"], Save=True, PrintLevel=-1, GlobalObservables=modelConfig.GetGlobalObservables(), Minos=True
)

# Getting list of Parameters of Interest and getting first from them
poi = modelConfig.GetParametersOfInterest().first()

nll = pdf.createNLL(ws["obsData"])
profile = nll.createProfile(poi)

# frame for future plot
frame = poi.frame()

frame.SetTitle("")
frame.GetYaxis().SetTitle("-log likelihood")
frame.GetXaxis().SetTitle(poi.GetTitle())

profileLikelihoodCanvas = ROOT.TCanvas("combined", "", 800, 600)

xmin = poi.getMin()
xmax = poi.getMax()

line = ROOT.TLine(xmin, 0.5, xmax, 0.5)
line.SetLineColor("kGreen")
line90 = ROOT.TLine(xmin, 2.71 / 2, xmax, 2.71 / 2)
line90.SetLineColor("kGreen")
line95 = ROOT.TLine(xmin, 3.84 / 2, xmax, 3.84 / 2)
line95.SetLineColor("kGreen")

frame.addObject(line)
frame.addObject(line90)
frame.addObject(line95)

nll.plotOn(frame, ShiftToZero=True, LineColor="r", LineStyle="--")
profile.plotOn(frame)

frame.SetMinimum(0)
frame.SetMaximum(2)

frame.Draw()

# Print fit results to console in verbose mode to see asymmetric uncertainties
result.Print("v")
