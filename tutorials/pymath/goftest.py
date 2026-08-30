## \file
## \ingroup tutorial_math
## \notebook
## GoFTest tutorial macro
## 
## Goodness of Fit Test
##
## We are using Anderson-Darling and Kolmogorov-Smirnov goodness of fit for two-tests:
## first, 1-sample test is performed by comparing data with a log-normal distribution;
## after, a 2-sample test is done by comparing two gaussian data sets.
## Enjoy it very much.
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Bartolomeu Rabacal
## \translator P. P.


import ROOT
import ctypes

#cassert = ROOT.cassert #No need, assert is defined in python.

TCanvas = ROOT.TCanvas 
TPaveText = ROOT.TPaveText 
TH1 = ROOT.TH1 
TH1D = ROOT.TH1D
TF1 = ROOT.TF1 
TRandom3 = ROOT.TRandom3 

#
Error = ROOT.Error

#
TMath = ROOT.TMath

Math = ROOT.Math
GoFTest = Math.GoFTest 
Functor = Math.Functor 
#DistFunc = Math.DistFunc #No defined.

#types
Double_t = ROOT.Double_t
c_double = ctypes.c_double

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed

#globals
gRandom = ROOT.gRandom

#

#utils
def to_c(ls:list):
   return (c_double * len(ls))( * ls )

def sprintf(buffer, string, *args):
   buffer = string % args
   return buffer 

#flags
TEST_ERROR_MESSAGE = False
#Note: 
#     In goftest.C, TEST_ERROR_MESSAGE is declared for the  preprocessor using #def
#     If you do #undefine TEST_ERROR_MESSAge at the start of goftest.C, 
#     you'll get error. The same if you change TEST_ERROR_MESSAGE to True.
#     

# need to use Functor1D
# double
def landau(x: Double_t) :
   return ROOT.Math.landau_pdf(x)
   

# void
def goftest() :
   
   # ------------------------------------------------------------------------
   # Case 1: Create logNormal random sample
   
   
   nEvents1 = 1000
   
   #ROOT::Math::Random<ROOT::Math::GSLRngMT> r;
   f1 = TF1("logNormal","ROOT::Math::lognormal_pdf(x,[0],[1])",0,500)
   # set the lognormal parameters (m and s)
   f1.SetParameters(4.0,1.0)
   f1.SetNpx(1000)
   
   
   sample1 = [Double_t()]* nEvents1
   
   h1smp = TH1D("h1smp", "LogNormal distribution histogram", 100, 0, 500)
   h1smp.SetStats(False)
   
   #for (UInt_t i = 0; i < nEvents1; ++i) {
   for i in range(nEvents1): 
      #data = f1.GetRandom() # Double_t 
      data = gRandom.Gaus(4,1)
      data = TMath.Exp(data)
      sample1[i] = data
      h1smp.Fill(data)
      
   # normalize correctly the histogram using the entries inside
   h1smp.Scale( ROOT.Math.lognormal_cdf(500.,4.,1) / nEvents1, "width")
   
   c = TCanvas("c","1-Sample and 2-Samples GoF Tests")
   c.Divide(1, 2)
   pad = c.cd(1) # TPad
   h1smp.Draw()
   h1smp.SetLineColor(kBlue)
   pad.SetLogy()
   f1.SetNpx(100); # use same points as histo for drawing
   f1.SetLineColor(kRed)
   f1.Draw("SAME")
   
   # -----------------------------------------
   # Create GoFTest object
   
   
   c_sample1 = to_c(sample1)
   goftest_1 = ROOT.Math.GoFTest(nEvents1, c_sample1, ROOT.Math.GoFTest.kLogNormal)
   sample1 = list(c_sample1)
   #----------------------------------------------------
   # Possible calls for the Anderson-Darling-Test test
   # a) Returning the Anderson-Darling standardized test statistic
   A2_1 = goftest_1. AndersonDarlingTest("t")
   A2_2 = (goftest_1)(ROOT.Math.GoFTest.kAD, "t")
   assert(A2_1 == A2_2)
   
   # b) Returning the p-value for the Anderson-Darling test statistic
   # p-value is the default choice
   pvalueAD_1 = goftest_1. AndersonDarlingTest()
   # p-value and Anderson - Darling Test are the default choices
   pvalueAD_2 = (goftest_1)() 
   assert(pvalueAD_1 == pvalueAD_2)
   
   # Rebuild the test using the default 1-sample construtor
   del goftest_1
   # User must then input a distribution type option
   c_sample1 = to_c(sample1)
   goftest_1 = ROOT.Math.GoFTest(nEvents1, c_sample1 )
   sample1 = list(c_sample1)
   goftest_1.SetDistribution(ROOT.Math.GoFTest.kLogNormal)
   
   #--------------------------------------------------
   # Possible calls for the Kolmogorov - Smirnov test
   # a) Returning the Kolmogorov-Smirnov standardized test statistic
   Dn_1 = goftest_1. KolmogorovSmirnovTest("t")
   Dn_2 = (goftest_1)(ROOT.Math.GoFTest.kKS, "t")
   assert(Dn_1 == Dn_2)
   
   # b) Returning the p-value for the Kolmogorov-Smirnov test statistic
   pvalueKS_1 = goftest_1. KolmogorovSmirnovTest()
   pvalueKS_2 = (goftest_1)(ROOT.Math.GoFTest.kKS)
   assert(pvalueKS_1 == pvalueKS_2)
   
   # Valid but incorrect call for both samples
   # the 2-sample methods of the 1-sample constructed by goftest_1
   #ifdef TEST_ERROR_MESSAGE
   if TEST_ERROR_MESSAGE:
      A2 = (goftest_1)(ROOT.Math.GoFTest.kAD2s, "t"); # Issues error message
      pvalueKS = (goftest_1)(ROOT.Math.GoFTest.kKS2s); # Issues error message
      assert(A2 == pvalueKS)
   #endif
   
   pt1 = TPaveText(0.58, 0.6, 0.88, 0.80, "brNDC")
   str1 = " "*50 # Char_t
   sprintf(str1, "p-value for A-D 1-smp test: %f", pvalueAD_1)
   pt1.AddText(str1)
   pt1.SetFillColor(18)
   pt1.SetTextFont(20)
   pt1.SetTextColor(4)
   str2 = " "*50 # Char_t
   sprintf(str2, "p-value for K-S 1-smp test: %f", pvalueKS_1)
   pt1.AddText(str2)
   pt1.Draw()
   
   # ------------------------------------------------------------------------
   # Case 2: Create Gaussian random samples
   
   nEvents2 = 2000
   
   sample2 = [Double_t()]*nEvents2
   
   h2smps_1 = TH1D("h2smps_1", "Gaussian distribution histograms", 100, 0, 500)
   h2smps_1.SetStats(False)
   
   h2smps_2 = TH1D("h2smps_2", "Gaussian distribution histograms", 100, 0, 500)
   h2smps_2.SetStats(False)
   
   r = TRandom3()
   #for (UInt_t i = 0; i < nEvents1; ++i) {
   for i in range(nEvents1):
      data = r.Gaus(300, 50)
      sample1[i] = data
      h2smps_1.Fill(data)
      
   h2smps_1.Scale(1. / nEvents1, "width")
   c.cd(2)
   h2smps_1.Draw()
   h2smps_1.SetLineColor(kBlue)
   
   #for (UInt_t i = 0; i < nEvents2; ++i) {
   for i in range(nEvents2):
      data = r.Gaus(300, 50)
      sample2[i] = data
      h2smps_2.Fill(data)
      
   h2smps_2.Scale(1. / nEvents2, "width")
   h2smps_2.Draw("SAME")
   h2smps_2.SetLineColor(kRed)
   
   # -----------------------------------------
   # Create GoFTest object
   
   c_sample1 = to_c(sample1)
   c_sample2 = to_c(sample2)
   goftest_2 = ROOT.Math.GoFTest(nEvents1, c_sample1, nEvents2, c_sample2)
   sample1 = list(c_sample1)
   sample2 = list(c_sample2)
   
   #----------------------------------------------------
   # Possible calls for the Anderson - DarlingTest test
   # a) Returning the Anderson-Darling standardized test statistic
   A2_1 = goftest_2.AndersonDarling2SamplesTest("t")
   A2_2 = goftest_2.__call__(ROOT.Math.GoFTest.kAD2s, "t")
   assert(A2_1 == A2_2)
   
   # b) Returning the p-value for the Anderson-Darling test statistic
   # p-value is the default choice
   pvalueAD_1 = goftest_2. AndersonDarling2SamplesTest()
   # p-value is the default choices
   pvalueAD_2 = goftest_2.__call__(ROOT.Math.GoFTest.kAD2s)
   assert(pvalueAD_1 == pvalueAD_2)
   
   #--------------------------------------------------
   # Possible calls for the Kolmogorov - Smirnov test
   # a) Returning the Kolmogorov-Smirnov standardized test statistic
   Dn_1 = goftest_2. KolmogorovSmirnov2SamplesTest("t")
   Dn_2 = goftest_2.__call__(ROOT.Math.GoFTest.kKS2s, "t")
   assert(Dn_1 == Dn_2)
   
   # b) Returning the p-value for the Kolmogorov-Smirnov test statistic
   pvalueKS_1 = goftest_2. KolmogorovSmirnov2SamplesTest()
   pvalueKS_2 = (goftest_2).__call__(ROOT.Math.GoFTest.kKS2s)
   assert(pvalueKS_1 == pvalueKS_2)
   
   #ifdef TEST_ERROR_MESSAGE
   if TEST_ERROR_MESSAGE:
      ''' Valid but incorrect calls for the 1-sample methods of the 2-samples constucted goftest_2 '''
      A2 = (goftest_2).__call__(ROOT.Math.GoFTest.kAD, "t"); # Issues error message
      pvalueKS = (goftest_2).__call__(ROOT.Math.GoFTest.kKS); # Issues error message
      assert(A2 == pvalueKS)
   #endif
   
   pt2 = TPaveText(0.13, 0.6, 0.43, 0.8, "brNDC")
   sprintf(str1, "p-value for A-D 2-smps test: %f", pvalueAD_1)
   pt2.AddText(str1)
   pt2.SetFillColor(18)
   pt2.SetTextFont(20)
   pt2.SetTextColor(4)
   sprintf(str2, "p-value for K-S 2-smps test: %f", pvalueKS_1)
   pt2. AddText(str2)
   pt2. Draw()
   
   # ------------------------------------------------------------------------
   # Case 3: Create Landau random sample
   
   nEvents3 = 1000
   
   sample3 = [ Double_t() ] * nEvents3
   #for (UInt_t i = 0; i < nEvents3; ++i) {
   for i in range(nEvents3): 
      data = r.Landau()
      sample3[i] = data
      
   
   # ------------------------------------------
   # Create GoFTest objects
   #
   # Possible constructors for the user input distribution
   
   # a) User input PDF
   f = ROOT.Math.Functor1D( landau ) 
   
   c_sample3 = to_c(sample3)
   global gc_sample3
   gc_sample3 = c_sample3

   minimum = 3 * TMath.MinElement(nEvents3, c_sample3)
   maximum = 3 * TMath.MaxElement(nEvents3, c_sample3)
   # need to specify an interval
   goftest_3a = ROOT.Math.GoFTest(nEvents3, c_sample3, f, ROOT.Math.GoFTest.kPDF, minimum,maximum)
   # b) User input CDF
   Rf = ROOT.Math.Functor1D( TMath.LandauI )
   goftest_3b = ROOT.Math.GoFTest(nEvents3, c_sample3, Rf, ROOT.Math.GoFTest.kCDF,minimum,maximum)

   sample3 = list(c_sample3)
   
   # The next part take time!
   # Returning the p-value for the Anderson-Darling test statistic
   # p-value is the default choice
   #pvalueAD_1 = goftest_3a.AndersonDarlingTest()
   pvalueAD_1 = (goftest_3b).__call__()
   
   
   # p-value and Anderson - Darling Test are the default choices
   pvalueAD_2 = (goftest_3b).__call__()
   
   # Checking consistency between both tests
   print(f" \n\nTEST with LANDAU distribution:\t")
   if TMath.Abs(pvalueAD_1 - pvalueAD_2) > 1.E-1 * pvalueAD_2:
      print(f"FAILED ")
      Error("goftest","Error in comparing testing using Landau and Landau CDF")
      print(" pvalues are " ,  pvalueAD_1 ,  "  " ,  pvalueAD_2)
      
   else:
      print(f"OK ( pvalues = " , pvalueAD_2, "  )")
   


if __name__ == "__main__":
   goftest()
