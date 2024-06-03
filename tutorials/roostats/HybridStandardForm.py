# \file
# \ingroup tutorial_roostats
# \notebook -js
# A hypothesis testing example based on number counting  with background uncertainty.
#
# A hypothesis testing example based on number counting
# with background uncertainty.
#
# NOTE: This example is like HybridInstructional, but the model is more clearly
# generalizable to an analysis with shapes.  There is a lot of flexibility
# for how one models a problem in RooFit/RooStats.  Models come in a few
# common forms:
#   - standard form: extended PDF of some discriminating variable m:
#   eg: P(m) ~ S*fs(m) + B*fb(m), with S+B events expected
#   in this case the dataset has N rows corresponding to N events
#   and the extended term is Pois(N|S+B)
#
#   - fractional form: non-extended PDF of some discriminating variable m:
#   eg: P(m) ~ s*fs(m) + (1-s)*fb(m), where s is a signal fraction
#   in this case the dataset has N rows corresponding to N events
#   and there is no extended term
#
#   - number counting form: in which there is no discriminating variable
#   and the counts are modeled directly (see HybridInstructional)
#   eg: P(N) = Pois(N|S+B)
#   in this case the dataset has 1 row corresponding to N events
#   and the extended term is the PDF itself.
#
# Here we convert the number counting form into the standard form by
# introducing a dummy discriminating variable m with a uniform distribution.
#
# This example:
#  - demonstrates the usage of the HybridCalcultor (Part 4-6)
#  - demonstrates the numerical integration of RooFit (Part 2)
#  - validates the RooStats against an example with a known analytic answer
#  - demonstrates usage of different test statistics
#  - explains subtle choices in the prior used for hybrid methods
#  - demonstrates usage of different priors for the nuisance parameters
#  - demonstrates usage of PROOF
#
# The basic setup here is that a main measurement has observed x events with an
# expectation of s+b.  One can choose an ad hoc prior for the uncertainty on b,
# or try to base it on an auxiliary measurement.  In this case, the auxiliary
# measurement (aka control measurement, sideband) is another counting experiment
# with measurement y and expectation tau*b.  With an 'original prior' on b,
# called \f$ \eta(b) \f$ then one can obtain a posterior from the auxiliary measurement
# \f$ \pi(b) = \eta(b) * Pois(y|tau*b) \f$.  This is a principled choice for a prior
# on b in the main measurement of x, which can then be treated in a hybrid
# Bayesian/Frequentist way.  Additionally, one can try to treat the two
# measurements simultaneously, which is detailed in Part 6 of the tutorial.
#
# This tutorial is related to the FourBin.C tutorial in the modeling, but
# focuses on hypothesis testing instead of interval estimation.
#
# More background on this 'prototype problem' can be found in the
# following papers:
#
#  - Evaluation of three methods for calculating statistical significance
#    when incorporating a systematic uncertainty into a test of the
#    background-only hypothesis for a Poisson process
#    Authors: Robert D. Cousins, James T. Linnemann, Jordan Tucker
#    http:#arxiv.org/abs/physics/0702156
#    NIM  A 595 (2008) 480--501
#
#  - Statistical Challenges for Searches for New Physics at the LHC
#    Author: Kyle Cranmer
#    http:#arxiv.org/abs/physics/0511028
#
#  - Measures of Significance in HEP and Astrophysics
#    Author: J. T. Linnemann
#    http:#arxiv.org/abs/physics/0312059
#
# \macro_image
# \macro_output
# \macro_code
#
# \authors Kyle Cranmer, Wouter Verkerke, and Sven Kreiss


import ROOT 

from ROOT import RooFit, RooStats

TestStatistics = RooFit.TestStatistics
RooAbsData = ROOT.RooAbsData
ProofConfig = RooStats.ProofConfig
kFALSE = ROOT.kFALSE

ModelConfig = RooFit.ModelConfig
#RooGlobalFunc = 		 ROOT.RooGlobalFunc
RooArgSet = ROOT.RooArgSet
RooRealVar = 		 ROOT.RooRealVar
RooProdPdf = 		 ROOT.RooProdPdf
RooWorkspace = 		 ROOT.RooWorkspace
RooDataSet = 		 ROOT.RooDataSet
RooDataHist = 		 ROOT.RooDataHist
TCanvas = 		 ROOT.TCanvas
TStopwatch = 		 ROOT.TStopwatch
TH1 = 		 ROOT.TH1
RooPlot = 		 ROOT.RooPlot
RooMsgService = 		 ROOT.RooMsgService

NumberCountingUtils = 		 RooStats.NumberCountingUtils
HybridCalculator = 		 RooStats.HybridCalculator
ToyMCSampler = 		 RooStats.ToyMCSampler
HypoTestPlot = 		 RooStats.HypoTestPlot
NumEventsTestStat = 		 RooStats.NumEventsTestStat
ProfileLikelihoodTestStat = 		 RooStats.ProfileLikelihoodTestStat
SimpleLikelihoodRatioTestStat = 		 RooStats.SimpleLikelihoodRatioTestStat
RatioOfProfiledLikelihoodsTestStat = 		 RooStats.RatioOfProfiledLikelihoodsTestStat
MaxLikelihoodEstimateTestStat = 		 RooStats.MaxLikelihoodEstimateTestStat


#-------------------------------------------------------
# A New Test Statistic Class for this example.
# It simply returns the sum of the values in a particular
# column of a dataset.
# You can ignore this class and focus on the macro below
# # # 

class BinCountTestStat (TestStatistics) :
   
   fColumnName = str() 
   
   def __init__(self, columnName = "tmp"): 
      super().__init__()
      self.fColumnName = columnName

   def Evaluate(self, data, nullPOI = RooArgSet()):
      if data is not RooAbsData : 
         print("data is not a RooAbsData-object")
         return 
      # This is the main method in the interface
      value = ROOT.Double_t( 0.0)
      for i in range(data.numEntries()): 
         value += data.get(i).getRealValue(self.fColumnName)
         
      return value
      
   def GetVarName(self) : 
      return self.fColumnName
   
   def __init__(self, columnName = "tmp"): 
      super().__init__
      self.fColumnName = columnName
   

#-----------------------------
# The Actual Tutorial Macro
#-----------------------------

def HybridStandardForm():

   
   # This tutorial has 6 parts
   # Table of Contents
   # Setup
   #   1. Make the model for the 'prototype problem'
   # Special cases
   #   2. NOT RELEVANT HERE
   #   3. Use RooStats analytic solution for this problem
   # RooStats HybridCalculator -- can be generalized
   #   4. RooStats ToyMC version of 2. & 3.
   #   5. RooStats ToyMC with an equivalent test statistic
   #   6. RooStats ToyMC with simultaneous control & main measurement
   
   # Part 4 takes ~4 min without PROOF.
   # Part 5 takes about ~2 min with PROOF on 4 cores.
   # Of course, everything looks nicer with more toys, which takes longer.
   
   t = TStopwatch() 
   t.Start()
   c1 =  TCanvas("myc1", "myc1")
   
    
   #-----------------------------------------------------
   # P A R T   1  :  D I R E C T   I N T E G R A T I O N
   # ====================================================
   # Make model for prototype on/off problem
   # Pois(x | s+b) * Pois(y | tau b )
   # for Z_Gamma, use uniform prior on b.
   w =  RooWorkspace("w")
   
   # replace the pdf in 'number counting form'
   # w.factory("Poisson::px(x[150,0,500],sum::splusb(s[0,0,100],b[100,0,300]))")
   # with one in standard form.  Now x is encoded in event count
   w.factory("Uniform::f(m[0,1])") # m is a dummy discriminating variable
   w.factory("ExtendPdf::px(f,sum::splusb(s[0,0,100],b[100,0,300]))")
   w.factory("Poisson::py(y[100,0,500],prod::taub(tau[1.],b))")
   w.factory("PROD::model(px,py)")
   w.factory("Uniform::prior_b(b)")
   
   # We will control the output level in a few places to avoid
   # verbose progress messages.  We start by keeping track
   # of the current threshold on messages.
   msglevel = RooMsgService.instance().globalKillBelow()
   
   # Use PROOF-lite on multi-core machines
   pc = ROOT.kNone
   # uncomment below if you want to use PROOF
   #pc =  ProofConfig(w, 4, "workers=4", kFALSE) # machine with 4 cores
   #pc = ProofConfig(w, 2, "workers=2", kFALSE) # machine with 2 cores
   pc =  ProofConfig(w, 1, "workers=1", kFALSE) # machine with 1 core
   
   #-----------------------------------------------
   # P A R T   3  :  A N A L Y T I C   R E S U L T
   # ==============================================
   # In this special case, the integrals are known analytically
   # and they are implemented in RooStats::NumberCountingUtils
   
   # analytic Z_Bi
   p_Bi = NumberCountingUtils.BinomialWithTauObsP(150, 100, 1)
   Z_Bi = NumberCountingUtils.BinomialWithTauObsZ(150, 100, 1)
   print(f"-----------------------------------------")
   print(f"Part 3")
   print(f"Z_Bi p-value (analytic): ", p_Bi)
   print(f"Z_Bi significance (analytic): ", Z_Bi)
   t.Stop()
   t.Print()
   t.Reset()
   t.Start()
   
   #--------------------------------------------------------------
   # P A R T   4  :  U S I N G   H Y B R I D   C A L C U L A T O R
   # ==============================================================
   # Now we demonstrate the RooStats HybridCalculator.
   #
   # Like all RooStats calculators it needs the data and a ModelConfig
   # for the relevant hypotheses.  Since we are doing hypothesis testing
   # we need a ModelConfig for the null (background only) and the alternate
   # (signal+background) hypotheses.  We also need to specify the PDF,
   # the parameters of interest, and the observables.  Furthermore, since
   # the parameter of interest is floating, we need to specify which values
   # of the parameter corresponds to the null and alternate (eg. s=0 and s=50)
   #
   # define some sets of variables obs={x} and poi={s}
   # note here, x is the only observable in the main measurement
   # and y is treated as a separate measurement, which is used
   # to produce the prior that will be used in this calculation
   # to randomize the nuisance parameters.
   w.defineSet("obs", "m")
   w.defineSet("poi", "s")
   
   # create a toy dataset with the x=150
   #  data = RooDataSet("d", "d", w.set("obs"))
   #  data.add(w.set("obs"))
    
   data = w.pdf("px").generate(w.set("obs"), 150)
   
   # Part 3a : Setup ModelConfigs
   #-------------------------------------------------------
   # create the null (background-only) ModelConfig with s=0
   b_model = ModelConfig("B_model", w)
   b_model.SetPdf(w.pdf("px"))
   b_model.SetObservables(w.set("obs"))
   b_model.SetParametersOfInterest(w.set("poi"))
   w.var("s").setVal(50.0) # important!
   b_model.SetSnapshot(w.set("poi"))
   
   # create the alternate (signal+background) ModelConfig with s=50
   sb_model = ModelConfig("S+B_model", w)
   sb_model.SetPdf(w.pdf("px"))
   sb_model.SetObservables(w.set("obs"))
   sb_model.SetParametersOfInterest(w.set("poi"))
   w.var("s").setVal(150.0) # important!
   sb_model.SetSnapshot(w.set("poi"))
   
   # Part 3b : Choose Test Statistic
   #--------------------------------------------------------------
   # To make an equivalent calculation we need to use x as the test
   # statistic.  This is not a built-in test statistic in RooStats
   # so we define it above.  The new class inherits from the
   # RooStats::TestStatistic interface, and simply returns the value
   # of x in the dataset.
   
   eventCount = NumEventsTestStat(w.pdf("px"))
   
   # Part 3c : Define Prior used to randomize nuisance parameters
   #-------------------------------------------------------------
   #
   # The prior used for the hybrid calculator is the posterior
   # from the auxiliary measurement y.  The model for the aux.
   # measurement is Pois(y|tau*b), thus the likelihood function
   # is proportional to (has the form of) a Gamma distribution.
   # if the 'original prior' $\eta(b)$ is uniform, then from
   # Bayes's theorem we have the posterior:
   #  $\pi(b) = Pois(y|tau*b) * \eta(b)$
   # If $\eta(b)$ is flat, then we arrive at a Gamma distribution.
   # Since RooFit will normalize the PDF we can actually supply
   # py=Pois(y,tau*b) that will be equivalent to multiplying by a uniform.
   #
   # Alternatively, we could explicitly use a gamma distribution:
   #
   # `w.factory("Gamma::gamma(b,sum::temp(y,1),1,0)")`
   #
   # or we can use some other ad hoc prior that do not naturally
   # follow from the known form of the auxiliary measurement.
   # The common choice is the equivalent Gaussian:
   w.factory("Gaussian::gauss_prior(b,y, expr::sqrty('sqrt(y)',y))")
   # this corresponds to the "Z_N" calculation.
   #
   # or one could use the analogous log-normal prior
   w.factory("Lognormal::lognorm_prior(b,y, expr::kappa('1+1./sqrt(y)',y))")
   #
   # Ideally, the HybridCalculator would be able to inspect the full
   # model Pois(x | s+b) * Pois(y | tau b ) and be given the original
   # prior $\eta(b)$ to form $\pi(b) = Pois(y|tau*b) * \eta(b)$.
   # This is not yet implemented because in the general case
   # it is not easy to identify the terms in the PDF that correspond
   # to the auxiliary measurement.  So for now, it must be set
   # explicitly with:
   #  - ForcePriorNuisanceNull()
   #  - ForcePriorNuisanceAlt()
   # the name "ForcePriorNuisance" was chosen because we anticipate
   # this to be auto-detected, but will leave the option open
   # to force to a different prior for the nuisance parameters.
   
   # Part 3d : Construct and configure the HybridCalculator
   #-------------------------------------------------------
   hc1 = HybridCalculator(data, sb_model, b_model)
   toymcs1 = hc1.GetTestStatSampler()
   #  toymcs1.SetNEventsPerToy(1) # because the model is in number counting form
   toymcs1.SetTestStatistic(eventCount) # set the test statistic
   #  toymcs1.SetGenerateBinned()
   #hc1.SetToys(30000, 1000)
   #hc1.SetToys(3000, 100)
   hc1.SetToys(300, 100)
   hc1.ForcePriorNuisanceAlt(w.pdf("py"))
   hc1.ForcePriorNuisanceNull(w.pdf("py"))
   # if you wanted to use the ad hoc Gaussian prior instead
   # ~~~
   #  hc1.ForcePriorNuisanceAlt(w.pdf("gauss_prior"))
   #  hc1.ForcePriorNuisanceNull(w.pdf("gauss_prior"))
   # ~~~
   # if you wanted to use the ad hoc log-normal prior instead
   # ~~~
   #  hc1.ForcePriorNuisanceAlt(w.pdf("lognorm_prior"))
   #  hc1.ForcePriorNuisanceNull(w.pdf("lognorm_prior"))
   # ~~~
   
   # enable proof
   if(pc) :
      toymcs1.SetProofConfig(pc)
   
   # these lines save current msg level and thus kill any messages below ERROR
   RooMsgService.instance().setGlobalKillBelow(RooFit.ERROR)
   # Get the result
   r1 = hc1.GetHypoTest()
   RooMsgService.instance().setGlobalKillBelow(msglevel) # set it back
   print(f"-----------------------------------------")
   print(f"Part 4")
   r1.Print()
   t.Stop()
   t.Print()
   t.Reset()
   t.Start()
   
   c1 = TCanvas("myc1", "myc1")
   p1 =  HypoTestPlot(r1, 30) # 30 bins, TS is discrete
   p1.Draw()
   c1.Update()
   c1.Draw()
   c1.SaveAs("HybridStandardForm.1.png")
   #return
   # keep the running time sort by default
   print("\nPART 5 : Using Hybrid Calculator with an Alternative Test Statistic")
   #-------------------------------------------------------------------------
   # # P A R T   5  :  U S I N G   H Y B R I D   C A L C U L A T O R   W I T H
   # # A N   A L T E R N A T I V E   T E S T   S T A T I S T I C
   #
   # A likelihood ratio test statistics should be 1-to-1 with the count x
   # when the value of b is fixed in the likelihood.  This is implemented
   # by the SimpleLikelihoodRatioTestStat
   
   slrts = SimpleLikelihoodRatioTestStat(b_model.GetPdf(), sb_model.GetPdf())
   slrts.SetNullParameters(b_model.GetSnapshot())
   slrts.SetAltParameters(sb_model.GetSnapshot())
   
   # HYBRID CALCULATOR
   hc2 = HybridCalculator(data, sb_model, b_model)
   toymcs2 = hc2.GetTestStatSampler()
   #  toymcs2.SetNEventsPerToy(1)
   toymcs2.SetTestStatistic(slrts)
   #  toymcs2.SetGenerateBinned()
   #hc2.SetToys(30000, 1000)
   #hc1.SetToys(3000, 100)
   hc2.SetToys(300, 10)
   #hc2.ForcePriorNuisanceAlt(w.pdf("py"))
   #hc2.ForcePriorNuisanceNull(w.pdf("py"))
   # if you wanted to use the ad hoc Gaussian prior instead
   # ~~~
   hc2.ForcePriorNuisanceAlt(w.pdf("gauss_prior"))
   hc2.ForcePriorNuisanceNull(w.pdf("gauss_prior"))
   # ~~~
   # if you wanted to use the ad hoc log-normal prior instead
   # ~~~
   #  hc2.ForcePriorNuisanceAlt(w.pdf("lognorm_prior"))
   #  hc2.ForcePriorNuisanceNull(w.pdf("lognorm_prior"))
   # ~~~
   #
   # enable proof
   # proof not enabled for this test statistic
   #if pc: 
   #   toymcs2.SetProofConfig(pc)
   
   # these lines save current msg level and then kill any messages below ERROR
   RooMsgService.instance().setGlobalKillBelow(RooFit.ERROR)
   # Get the result
   r2 = hc2.GetHypoTest()
   print(f"-----------------------------------------")
   print(f"Part 5")
   r2.Print()
   t.Stop()
   t.Print()
   t.Reset()
   t.Start()
   RooMsgService.instance().setGlobalKillBelow(msglevel)
   
   #Final Plot ...
   c2 = TCanvas("myc2", "myc2")
   p2 =  HypoTestPlot(r2, 30) # 30 bins
   p2.Draw()
   
   c2.Update()
   c2.Draw()
   c2.SaveAs("HybridStandardForm.2.png") 
   return # Thus, such a standard tutorial runs faster

   
HybridStandardForm()
