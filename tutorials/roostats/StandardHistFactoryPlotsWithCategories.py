# \file
# \ingroup tutorial_roostats
# \notebook -js
# StandardHistFactoryPlotsWithCategories
#
#  This is a standard demo that can be used with any ROOT file
#  prepared in the standard way.  You specify:
#  - name for input ROOT file
#  - name of workspace inside ROOT file that holds model and data
#  - name of ModelConfig that specifies details for calculator tools
#  - name of dataset
#
#  With default parameters the macro will attempt to run the
#  standard hist2workspace example and read the ROOT file
#  that it produces.
#
#  The macro will scan through all the categories in a simPdf find the corresponding
#  observable.  For each category, it will loop through each of the nuisance parameters
#  and plot
#  - the data
#  - the nominal model (blue)
#  - the +Nsigma (red)
#  - the -Nsigma (green)
#
#  You can specify how many sigma to vary by changing nSigmaToVary.
#  You can also change the signal rate by changing muVal.
#
#  The script produces a lot plots, you can merge them by doing:
# 
#  gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=merged.pdf `ls *pdf`
# ~~~
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer


import ROOT
from ROOT import RooStats, RooFit

TFile = 		 ROOT.TFile
TROOT = 		 ROOT.TROOT
TCanvas = 		 ROOT.TCanvas
TList = 		 ROOT.TList
TMath = 		 ROOT.TMath
TSystem = 		 ROOT.TSystem
RooWorkspace = 		 ROOT.RooWorkspace
RooAbsData = 		 ROOT.RooAbsData
RooRealVar = 		 ROOT.RooRealVar
RooPlot = 		 ROOT.RooPlot
RooSimultaneous = 		 ROOT.RooSimultaneous
RooCategory = 		 ROOT.RooCategory

ModelConfig = 		 RooStats.ModelConfig
ProfileInspector = 		 RooStats.ProfileInspector

TFile = ROOT.TFile
strcmp = ROOT.strcmp
Form = ROOT.Form
MarkerSize = RooFit.MarkerSize
Cut = RooFit.Cut
DataError = RooFit.DataError
RooArgSet = ROOT.RooArgSet
LineWidth = RooFit.LineWidth
Normalization = RooFit.Normalization
RooAbsReal = ROOT.RooAbsReal 
LineColor = RooFit.LineColor
kRed = ROOT.kRed
kDashed = ROOT.kDashed
LineStyle = RooFit.LineStyle
kGreen = ROOT.kGreen


def StandardHistFactoryPlotsWithCategories(infile = "", workspaceName = "combined",
                                            modelConfigName = "ModelConfig",
                                            dataName = "obsData"):

   
   nSigmaToVary = 5.
   muVal = 0
   doFit = False
   
   # -------------------------------------------------------
   # First part is just to access a user-defined file
   # or create the standard example file if it doesn't exist
   filename = ""
   if infile == "":
      filename = "results/example_combined_GaussExample_model.root"
      fileExist = not ROOT.gSystem.AccessPathName(filename) # note opposite return code
      # if file does not exists generate with histfactory
      if not fileExist:
         #ifdef _WIN32
         print(f"HistFactory file cannot be generated on Windows - exit")
         return
         #endif
         # Normally this would be run on the command line
         print(f"will run standard hist2workspace example")
         ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
         ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")
         print(f"\n\n---------------------")
         print(f"Done creating example input")
         print(f"---------------------\n\n")
         
      
   else:
      filename = infile
   
   # Try to open the file
   file = TFile.Open(filename)
   
   # if input file was specified byt not found, quit
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
      
   
   # -------------------------------------------------------
   # now use the profile inspector
   
   obs = mc.GetObservables().first()
   List =  TList()
   
   firstPOI = mc.GetParametersOfInterest().first()
   
   firstPOI.setVal(muVal)
   #  firstPOI.setConstant()
   if doFit:
      mc.GetPdf().fitTo(data)
      
   
   # -------------------------------------------------------
   
   mc.GetNuisanceParameters().Print("v")
   nPlotsMax = 1000
   print(f" check expectedData by category")
   simData = ROOT.kNone
   simPdf = ROOT.kNone
   if (strcmp(mc.GetPdf().ClassName(), "RooSimultaneous") == 0) :
      print(f"Is a simultaneous PDF")
      simPdf = mc.GetPdf()
   else:
      print(f"Is not a simultaneous PDF")
      
   
   if doFit:
      channelCat = simPdf.indexCat()
      catName = channelCat.begin().first
      pdftmp = (mc.GetPdf()).getPdf(str(catName))
      obstmppdftmp.getObservables(mc.GetObservables())
      obs = obstmp.first()
      frame = obs.frame()
      print(Form("%s==%s.%s", channelCat.GetName(), channelCat.GetName(), str(catName)))
      print(catName, " ", channelCat.getLabel())
      data.plotOn(frame, MarkerSize(1),
                   Cut(Form("%s==%s::%s", channelCat.GetName(), channelCat.GetName(), str(catName))),
                   DataError(getattr(RooAbsData, "None")))

      
      normCount = \
         data.sumEntries(Form("%s==%s::%s", channelCat.GetName(), channelCat.GetName(), str(catName)))

      
      pdftmp.plotOn(frame, LineWidth(2), Normalization(normCount, RooAbsReal.NumEvent))
      frame.Draw()
      print( "expected events = ", mc.GetPdf().expectedEvents(data.get()) )
      return;
   
   nPlots = 0
   if not simPdf:
      
      it = mc.GetNuisanceParameters().createIterator()
      var = ROOT.kNone
      while((var := it.Next()) != ROOT.kNone):
         frame = obs.frame()
         frame.SetYTitle(var.GetName())
         data.plotOn(frame, MarkerSize(1))
         var.setVal(0)
         mc.GetPdf().plotOn(frame, LineWidth(1))
         var.setVal(1)
         mc.GetPdf().plotOn(frame, LineColor(kRed), LineStyle(kDashed), LineWidth(1))
         var.setVal(-1)
         mc.GetPdf().plotOn(frame, LineColor(kGreen), LineStyle(kDashed), LineWidth(1))
         List.Add(frame)
         var.setVal(0)
         
      
   else:
      channelCat = simPdf.indexCat()
      for tt in channelCat:
         
         if (nPlots == nPlotsMax):
            break
            
         
         catName = tt.first
         
         print( "on type ", catName, " ")
         # Get pdf associated with state from simpdf
         pdftmp = simPdf.getPdf(str(catName))
         
         # Generate observables defined by the pdf associated with this state
         obstmp = pdftmp.getObservables(mc.GetObservables())
         #      obstmp.Print()
         
         obs = obstmp.first()
         
         it = mc.GetNuisanceParameters().createIterator()
         var = ROOT.kNone
         while (nPlots < nPlotsMax and (var := it.Next())) :
            c2 =  TCanvas("c2")
            frame = obs.frame()
            frame.SetName(Form("frame{:d}".format(nPlots)))
            frame.SetYTitle(var.GetName())
            
            print( Form("{}=={}.{}".format( channelCat.GetName(), channelCat.GetName(), str(catName))) )
            print(catName, " ", channelCat.getLabel())
            data.plotOn(frame, MarkerSize(1),
                   Cut(Form("{}=={}::{}".format( channelCat.GetName(), channelCat.GetName(), str(catName)))),
                   DataError(getattr(RooAbsData, "None")))
 
            normCount = \
               data.sumEntries(Form("{}=={}::{}".format( channelCat.GetName(), channelCat.GetName(), str(catName))))
            
            if (strcmp(var.GetName(), "Lumi") == 0) :
               print(f"working on lumi")
               var.setVal(w.var("nominalLumi").getVal())
               var.Print()
            else:
               var.setVal(0)
               
            # w.allVars().Print("v")
            # mc.GetNuisanceParameters().Print("v")
            # pdftmp.plotOn(frame,LineWidth(2))
            # mc.GetPdf().plotOn(frame,LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
            # pdftmp.plotOn(frame,LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
            global gobs
            gobs = obs
            global gpdftmp
            gpdftmp = pdftmp

            #notes: obs is RooRealVar / obstmp is RooArgSet
            #       pdftmp.expectedEvents receives RooArgSet as an argument
            #       in C++ automatic conversion is possible.
            #       in python the conversino is not possible.  
            #C-code : normCount = pdftmp->expectedEvents(*obs);
            #Python : normCount = pdftmp.expectedEvents(obs) #doesn't work properly 
            # RooArgSet(obs) doesn´t reproduce well the results
            # instead, we have to use obstmp
            #normCount = pdftmp.expectedEvents(RooArgSet(obstmp)) #doesn´t work properly 
            normCount = pdftmp.expectedEvents(obstmp)
            pdftmp.plotOn(frame, LineWidth(2), Normalization(normCount, RooAbsReal.NumEvent))
            
            if (strcmp(var.GetName(), "Lumi") == 0) :
               print(f"working on lumi")
               var.setVal(w.var("nominalLumi").getVal() + 0.05)
               var.Print()
            else:
               var.setVal(nSigmaToVary)
               
            # pdftmp.plotOn(frame,LineColor(kRed),LineStyle(kDashed),LineWidth(2))
            # mc.GetPdf().plotOn(frame,LineColor(kRed),LineStyle(kDashed),LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
            # pdftmp.plotOn(frame,LineColor(kRed),LineStyle(kDashed),LineWidth(2.),Slice(channelCat,catName.c_str()),ProjWData(data))
            normCount = pdftmp.expectedEvents(obstmp)
            pdftmp.plotOn(frame, LineWidth(2), LineColor(kRed), LineStyle(kDashed),
            Normalization(normCount, RooAbsReal.NumEvent))
            
            if (strcmp(var.GetName(), "Lumi") == 0) :
               print(f"working on lumi")
               var.setVal(w.var("nominalLumi").getVal() - 0.05)
               var.Print()
            else:
               var.setVal(-nSigmaToVary)
               
            # pdftmp.plotOn(frame,LineColor(kGreen),LineStyle(kDashed),LineWidth(2))
            # mc.GetPdf().plotOn(frame,LineColor(kGreen),LineStyle(kDashed),LineWidth(2),Slice(channelCat,catName.c_str()),ProjWData(data))
            # pdftmp.plotOn(frame,LineColor(kGreen),LineStyle(kDashed),LineWidth(2),Slice(channelCat,catName.c_str()),ProjWData(data))
            normCount = pdftmp.expectedEvents(obstmp)
            pdftmp.plotOn(frame, LineWidth(2), LineColor(kGreen), LineStyle(kDashed),
            Normalization(normCount, RooAbsReal.NumEvent))
            
            # set them back to normal
            if (strcmp(var.GetName(), "Lumi") == 0) :
               print(f"working on lumi")
               var.setVal(w.var("nominalLumi").getVal())
               var.Print()
            else:
               var.setVal(0)
               
            
            List.Add(frame)
            
            # quit making plots
            nPlots +=1
            
            frame.Draw()
            c2.Update()
            c2.Draw()
            c2.SaveAs(Form("StandardHistFactoryPlotsWithCategories.1.{}_{}_{}.png".format( str(catName), obs.GetName(), var.GetName())))
            del c2

            
         
      
   
   # -------------------------------------------------------
   
   # now make plots
   c1 =  TCanvas("c1", "ProfileInspectorDemo", 800, 200)
   if List.GetSize() > 4:
      n = List.GetSize()
      nx = int(sqrt(n))
      ny = TMath.CeilNint(n / nx)
      nx = TMath.CeilNint(sqrt(n))
      c1.Divide(ny, nx)
   else:
      c1.Divide(List.GetSize())
   for i in range( List.GetSize() ):
      c1.cd(i + 1)
      List.At(i).Draw()
      c1.Update()
   c1.Draw()
   c1.SaveAs("StandardHistFactoryPlotsWithCategories.2.pdf") 
StandardHistFactoryPlotsWithCategories(infile = "", workspaceName = "combined",
                                            modelConfigName = "ModelConfig",
                                            dataName = "obsData")

   
