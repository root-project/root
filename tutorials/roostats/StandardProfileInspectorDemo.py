## \file
## \ingroup tutorial_roostats
## \notebook -js
## Standard demo of the ProfileInspector class
## StandardProfileInspectorDemo
##
## This is a standard demo that can be used with any ROOT file
## prepared in the standard way.  You specify:
##  - name for input ROOT file
##  - name of workspace inside ROOT file that holds model and data
##  - name of ModelConfig that specifies details for calculator tools
##  - name of dataset
##
## With the values provided below this script will attempt to run the
## standard hist2workspace example and read the ROOT file
## that it produces.
##
## The actual heart of the demo is only about 10 lines long.
##
## The ProfileInspector plots the conditional maximum likelihood estimate
## of each nuisance parameter in the model vs. the parameter of interest.
## (aka. profiled value of nuisance parameter vs. parameter of interest)
## (aka. best fit nuisance parameter with p.o.i fixed vs. parameter of interest)
##
## \macro_image
## \macro_output
## \macro_code
##
## \authors Akeem Hart, Kyle Cranmer (C++ Version)

import ROOT

# -------------------------------------------------------
# First part is just to access a user-defined file
# or create the standard example file if it doesn't exist

workspaceName = "combined"
modelConfigName = "ModelConfig"
dataName = "obsData"
filename = "results/example_combined_GaussExample_model.root"
# if file does not exists generate with histfactory
if ROOT.gSystem.AccessPathName(filename):
    # Normally this would be run on the command line
    print("will run standard hist2workspace example")
    ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
    ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
    print("\n\n---------------------")
    print("Done creating example input")
    print("---------------------\n\n")

file = ROOT.TFile.Open(filename)

# -------------------------------------------------------
# Tutorial starts here
# -------------------------------------------------------

# get the workspace out of the file

w = file.Get(workspaceName)

# get the modelConfig out of the file
mc = w[modelConfigName]

# get the modelConfig out of the file
data = w[dataName]

# -----------------------------
# now use the profile inspector
p = ROOT.RooStats.ProfileInspector()
profile_plots = p.GetListOfProfilePlots(data, mc)

# now make plots
c1 = ROOT.TCanvas("c1", "ProfileInspectorDemo", 800, 200)
n = len(profile_plots)
if n > 4:
    nx = int(n**0.5)
    ny = ROOT.TMath.CeilNint(n / nx)
    nx = ROOT.TMath.CeilNint(n**0.5)
    c1.Divide(ny, nx)
else:
    c1.Divide(n)
for i, plot in enumerate(profile_plots):
    c1.cd(i + 1)
    plot.Draw("al")

c1.Update()
