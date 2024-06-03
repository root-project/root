# \file
# \ingroup tutorial_roostats
# \notebook -js
# Standard demo of the ProfileInspector class
# StandardProfileInspectorDemo
#
# This is a standard demo that can be used with any ROOT file
# prepared in the standard way.  You specify:
#  - name for input ROOT file
#  - name of workspace inside ROOT file that holds model and data
#  - name of ModelConfig that specifies details for calculator tools
#  - name of dataset
#
# With default parameters the macro will attempt to run the
# standard hist2workspace example and read the ROOT file
# that it produces.
#
# The actual heart of the demo is only about 10 lines long.
#
# The ProfileInspector plots the conditional maximum likelihood estimate
# of each nuisance parameter in the model vs. the parameter of interest.
# (aka. profiled value of nuisance parameter vs. parameter of interest)
# (aka. best fit nuisance parameter with p.o.i fixed vs. parameter of interest)
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer


import ROOT

from ROOT import TFile, TROOT, TCanvas, TList, TMath, TSystem, RooWorkspace, RooAbsData

from ROOT import RooStats, RooFit

ModelConfig = 		RooStats.ModelConfig
ProfileInspector = 	RooStats.ProfileInspector



def StandardProfileInspectorDemo(infile = "", workspaceName = "combined", \
modelConfigName = "ModelConfig", dataName = "obsData"):

   
   # -------------------------------------------------------
   # First part is just to access a user-defined file
   # or create the standard example file if it doesn't exist
   
   filename = ""
   if (not ROOT.strcmp(infile, "")) :
      print("using... results/example...model.root file")
      filename = "results/example_combined_GaussExample_model.root"
      fileExist = not ROOT.gSystem.AccessPathName(filename) # note opposite return code
      print("does the .root file exists? : ", fileExist)
      # if file does not exists generate with histfactory
      if not fileExist:
         print("if file doesnt exist")
         # Normally this would be run on the command line
         print(f"will run standard hist2workspace example")
         ROOT.gROOT.ProcessLine(".!  prepareHistFactory .")
         ROOT.gROOT.ProcessLine(".!  hist2workspace config/example.xml")
         print(f"\n\n---------------------")
         print(f"Done creating example input")
         print(f"---------------------\n\n")
         
      
   else:
      filename = infile
   
   # Try to open the file
   print("filename", filename)
   file = TFile.Open(filename)
   
   # if input file was specified but not found, quit
   if not file:
      print(f"StandardRooStatsDemoMacro: Input file {filename} is not found")
      return
      
   
   # -------------------------------------------------------
   # Tutorial starts here
   # -------------------------------------------------------
   
   # get the workspace out of the file
   w = file.Get(workspaceName)
   if not w:
      print(f"workspace not found")
      return
      
   
   # get the modelConfig out of the file
   mc = w.obj(modelConfigName)
   
   # get the modelConfig out of the file
   data = w.data(dataName)
   
   # make sure ingredients are found
   if not data or not mc:
      w.Print()
      print(f"data or ModelConfig was not found")
      return
      
   
   # -----------------------------
   # now use the profile inspector
   p = ProfileInspector()
   root_list = p.GetListOfProfilePlots(data, mc)
   
   # now make plots
   c1 =  TCanvas("c1", "ProfileInspectorDemo", 800, 200)
   if root_list.GetSize() > 4:
      n = root_list.GetSize()
      nx = int(sqrt(n))
      ny = TMath.CeilNint(n / nx)
      nx = TMath.CeilNint(sqrt(n))
      c1.Divide(ny, nx)
   else:
      c1.Divide(root_list.GetSize())
      for i in range( root_list.GetSize() ):
         c1.cd(i + 1)
         root_list.At(i).Draw("al")
      
   print("\n")

   c1.Draw()
   c1.SaveAs("StandardProfileInspectorDemo.png")


StandardProfileInspectorDemo(infile = "", workspaceName = "combined", \
modelConfigName = "ModelConfig", dataName = "obsData")
