# \file
# \ingroup roostats_python_tutorials
# \notebook
# StandardTestStatDistributionDemo.py
#
# This simple script plots the sampling distribution of the profile likelihood
# ratio test statistic based on the input Model File.  To do this one needs to
# specify the value of the parameter of interest that will be used for evaluating
# the test statistic and the value of the parameters used for generating the toy data.
# In this case, it uses the upper-limit estimated from the ProfileLikleihoodCalculator,
# which assumes the asymptotic chi-square distribution for -2 log profile likelihood ratio.
# Thus, the script is handy for checking to see if the asymptotic approximations are valid.
# To aid, that comparison, the script overlays a chi-square distribution as well.
# The most common parameter of interest is a parameter proportional to the signal rate,
# and often that has a lower-limit of 0, which breaks the standard chi-square distribution.
# Thus the script allows the parameter to be negative so that the overlay chi-square is
# the correct asymptotic distribution.
#
# \macro_image
# \macro_output
# \macro_code
#
# \author Kyle Cranmer
# \translator P. P.



import ROOT
from ROOT import RooStats, RooFit

TFile = 	 ROOT.TFile
TROOT = 	 ROOT.TROOT
TH1F = 		 ROOT.TH1F
TCanvas = 	 ROOT.TCanvas
TSystem = 	 ROOT.TSystem
TF1 = 		 ROOT.TF1
TSystem = 	 ROOT.TSystem

RooArgSet = ROOT.RooArgSet  
Form = ROOT.Form

ModelConfig = 		RooStats.ModelConfig
FeldmanCousins = 		RooStats.FeldmanCousins
ToyMCSampler = 		RooStats.ToyMCSampler
PointSetInterval = 		RooStats.PointSetInterval
ConfidenceBelt = 		RooStats.ConfidenceBelt

ProfileLikelihoodCalculator = 		RooStats.ProfileLikelihoodCalculator
LikelihoodInterval = 		RooStats.LikelihoodInterval
ProfileLikelihoodTestStat = 		RooStats.ProfileLikelihoodTestStat
SamplingDistribution = 		RooStats.SamplingDistribution
SamplingDistPlot = 		RooStats.SamplingDistPlot


useProof = False # flag to control whether to use Proof
nworkers = 0 # number of workers (default use all available cores)

# -------------------------------------------------------
# The actual macro


def StandardTestStatDistributionDemo(infile = "", workspaceName = "combined",
                                      modelConfigName = "ModelConfig", dataName = "obsData"):
   
   # the number of toy MC used to generate the distribution
   nToyMC = 1000
   # The parameter below is needed for asymptotic distribution to be chi-square,
   # but set to false if your model is not numerically stable if mu<0
   allowNegativeMu = True
   
   # -------------------------------------------------------
   # First part is just to access a user-defined file
   # or create the standard example file if it doesn't exist
   filename = ""
   if infile == "":
      filename = "results/example_combined_GaussExample_model.root"
      fileExist = not ROOT.gSystem.AccessPathName(filename) # note opposite return code
      print("does the .root file exists?: ", fileExist)
      # if file does not exists generate with histfactory
      if not fileExist:
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
   file = TFile.Open(filename)
   
   # if input file was specified byt not found, quit
   if not file:
      print(f"StandardRooStatsDemoMacro: Input file {filename} is not found")
      return
      
   
   # -------------------------------------------------------
   # Now get the data and workspace
   
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
      
   
   mc.Print()
   # -------------------------------------------------------
   # Now find the upper limit based on the asymptotic results
   firstPOI = mc.GetParametersOfInterest().first()
   plc = ProfileLikelihoodCalculator(data, mc)
   interval = plc.GetInterval()
   plcUpperLimit = interval.UpperLimit(firstPOI)
   del interval
   print(f"\n\n--------------------------------------")
   print("Will generate sampling distribution at ", firstPOI.GetName(), " = ", plcUpperLimit )
   nPOI = mc.GetParametersOfInterest().getSize()
   if nPOI > 1:
      print(f"not sure what to do with other parameters of interest, but here are their values")
      mc.GetParametersOfInterest().Print("v")
      
   
   # -------------------------------------------------------
   # create the test stat sampler
   ts = ProfileLikelihoodTestStat(mc.GetPdf())
   
   # to avoid effects from boundary and simplify asymptotic comparison, set min=-max
   if allowNegativeMu:
      firstPOI.setMin(-1 * firstPOI.getMax())
   
   # temporary RooArgSet
   poi = RooArgSet() 
   poi.add(mc.GetParametersOfInterest())
   
   # create and configure the ToyMCSampler
   sampler = ToyMCSampler(ts, nToyMC)
   sampler.SetPdf(mc.GetPdf())
   sampler.SetObservables(mc.GetObservables())
   sampler.SetGlobalObservables(mc.GetGlobalObservables())
   if (not mc.GetPdf().canBeExtended() and (data.numEntries() == 1)) :
      print(f"tell it to use 1 event")
      sampler.SetNEventsPerToy(1)
      
   firstPOI.setVal(plcUpperLimit)                                  # set POI value for generation
   sampler.SetParametersForTestStat(mc.GetParametersOfInterest()) # set POI value for evaluation
   
   if useProof:
      pc = ProofConfig(w, nworkers, "", False)
      sampler.SetProofConfig(pc) # enable proof
      
   
   firstPOI.setVal(plcUpperLimit)
   allParameters = RooArgSet()
   allParameters.add(mc.GetParametersOfInterest())
   allParameters.add(mc.GetNuisanceParameters())
   allParameters.Print("v")
   
   sampDist = sampler.GetSamplingDistribution(allParameters)
   plot = SamplingDistPlot() 
   plot.AddSamplingDistribution(sampDist)
   plot.GetTH1F(sampDist).GetYaxis().SetTitle( \
   Form("f(-log #lambda(#mu={:2f}) | #mu={:2f})".format( plcUpperLimit, plcUpperLimit)) ) 
   plot.SetAxisTitle(Form("-log #lambda(#mu={:2f})".format(plcUpperLimit)))
   
   c1 =  TCanvas("c1")
   c1.SetLogy()
   plot.Draw()
   MIN = plot.GetTH1F(sampDist).GetXaxis().GetXmin()
   MAX = plot.GetTH1F(sampDist).GetXaxis().GetXmax()
   
   tmp_Form =  Form("2*ROOT::Math::chisquared_pdf(2*x,{:f},0)".format(nPOI))
   f =  TF1("f", tmp_Form, MIN, MAX)  

   f.Draw("same")
   c1.Update()
   c1.Draw()
   c1.SaveAs("StandardTestStatDistributionDemo.png")
   
StandardTestStatDistributionDemo()
