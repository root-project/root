## \file
## \ingroup tutorial_math
## \notebook#
## 
## Data unfolding by using Singular Value Decomposition.
##
## A TSVDUnfold class example.
##
## More information on the next High Energy Physics Repository: 
##    https://arxiv.org/abs/hep-ph/9509307
##    Title:
##    Data unfolding using Singular Value Decomposition 
##    id: (hep-ph/9509307)
##
## Example distribution and smearing-model from Tim Adye (RAL)
##
## \macro_image
## \macro_code
##
## \authors Kerstin Tackmann, Andreas Hoecker, Heiko Lacker
## \translator P. P.


import ROOT
import ctypes

iostream = ROOT.iostream 
TROOT = ROOT.TROOT 
TSystem = ROOT.TSystem 
TStyle = ROOT.TStyle 
TRandom3 = ROOT.TRandom3 
TString = ROOT.TString 
TMath = ROOT.TMath 
TH1D = ROOT.TH1D 
TH2D = ROOT.TH2D 
TLegend = ROOT.TLegend 
TCanvas = ROOT.TCanvas 
TColor = ROOT.TColor 
TLine = ROOT.TLine 
TSVDUnfold = ROOT.TSVDUnfold 

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t

#globals
global gStyle
gStyle = ROOT.gStyle





# Double_t
def Reconstruct( xt : Double_t, R : TRandom3 ) :
   # apply some Gaussian smearing + bias and efficiency corrections to fake reconstruction
   cutdummy = -99999.0
   # efficiency
   xeff = 0.3 + (1.0 - 0.3)/20.0*(xt + 10.0)
   x = R.Rndm()
   if (x > xeff) : return cutdummy
   else:
      # bias and smear
      xsmear= R.Gaus(-2.5,0.2) # Double_t
      return xt+xsmear
      
   

# void
def TSVDUnfoldExample() :
   ROOT.gROOT.SetStyle("Plain")
   gStyle.SetOptStat(0)
   
   global R
   R = TRandom3()
   
   cutdummy= -99999.0 # Double_t 
   
   # --------------------------------------
   # Data/MC toy generation
   #
   # The MC input
   nbins = 40
   global xini, bini, Adet
   xini = TH1D("xini", "MC truth", nbins, -10.0, 10.0)
   bini = TH1D("bini", "MC reco", nbins, -10.0, 10.0)
   Adet = TH2D("Adet", "detector response", nbins, -10.0, 10.0, nbins, -10.0, 10.0)
   
   global data, dataTrue, statcov
   # Data
   data = TH1D("data", "data", nbins, -10.0, 10.0)
   # Data "truth" distribution to test the unfolding
   dataTrue = TH1D("datatrue", "data truth", nbins, -10.0, 10.0)
   # Statistical covariance matrix
   statcov = TH2D("statcov", "covariance matrix", nbins, -10.0, 10.0, nbins, -10.0, 10.0)
   
   # Fill the MC using a Breit-Wigner, mean 0.3 and width 2.5.
   #   for (Int_t i= 0; i<100000; i++) {
   for i in range(0, 100000, 1):
      xt = R.BreitWigner(0.3, 2.5)
      xini.Fill(xt)
      x = Reconstruct( xt, R )
      if (x != cutdummy):
         Adet.Fill(x, xt)
         bini.Fill(x)
         
      
   
   # Fill the "data" with a Gaussian, mean 0 and width 2.
   #   for (Int_t i=0; i<10000; i++) {
   for i in range(0, 10000, 1):
      xt = R.Gaus(0.0, 2.0)
      dataTrue.Fill(xt)
      x = Reconstruct( xt, R )
      if (x != cutdummy):
         data.Fill(x)
      
   
   print(f"Created toy distributions and errors for: ")
   print(f"... \"True MC\"   and \"reconstructed (smeared) MC\"")
   print(f"... \"True data\" and \"reconstructed (smeared) data\"")
   print(f"... the \"detector response matrix\"")
   
   # Fill the data covariance matrix
   #   for (int i=1; i<=data->GetNbinsX(); i++) {
   for i in range(1, data.GetNbinsX() + 1, 1):
      statcov.SetBinContent( i, i, data.GetBinError(i) * data.GetBinError(i) )
      
   
   # ----------------------------
   # Here starts the actual unfolding
   #
   # Create TSVDUnfold object and initialise
   global tsvdunf
   tsvdunf = TSVDUnfold( data, statcov, bini, xini, Adet )
   
   # It is possible to normalise unfolded spectrum to unit area
   tsvdunf.SetNormalize( False ); # no normalisation here
   
   # Perform the unfolding with regularisation parameter kreg = 13
   # - the larger kreg, the finer grained the unfolding, but the more fluctuations occur
   # - the smaller kreg, the stronger is the regularisation and the bias
   unfres = tsvdunf.Unfold( 13 )
   
   # Get the distribution of the d to cross check the regularization
   # - choose kreg to be the point where |d_i| stop being statistically significantly >>1
   ddist = tsvdunf.GetD()
   
   # Get the distribution of the singular values
   svdist = tsvdunf.GetSV()
   
   # Compute the error matrix for the unfolded spectrum using toy MC
   # using the measured covariance matrix as input to generate the toys
   # 100 toys should usually be enough
   # The same method can be used for different covariance matrices separately.
   ustatcov = tsvdunf.GetUnfoldCovMatrix( statcov, 100 )
   
   # Now compute the error matrix on the unfolded distribution originating
   # from the finite detector matrix statistics
   uadetcov = tsvdunf.GetAdetCovMatrix( 100 )
   
   # Sum up the two (they are uncorrelated)
   ustatcov.Add( uadetcov )
   
   #Get the computed regularized covariance matrix (always corresponding to total uncertainty passed in constructor) and add uncertainties from finite MC statistics.
   utaucov = tsvdunf.GetXtau()
   utaucov.Add( uadetcov )
   
   #Get the computed inverse of the covariance matrix
   uinvcov = tsvdunf.GetXinv()
   
   
   # ---------------------------------
   # Only plotting stuff below
   
   #   for (int i=1; i<=unfres->GetNbinsX(); i++) {
   for i in range(1, unfres.GetNbinsX() + 1, 1):
      unfres.SetBinError(i, TMath.Sqrt(utaucov.GetBinContent(i,i)))
      
   
   # Renormalize just to be able to plot on the same scale
   xini.Scale(0.7*dataTrue.Integral()/xini.Integral())
   
   global leg
   leg = TLegend(0.58,0.60,0.99,0.88)
   leg.SetBorderSize(0)
   leg.SetFillColor(0)
   leg.SetFillStyle(0)
   leg.AddEntry(unfres,"Unfolded Data","p")
   leg.AddEntry(dataTrue,"True Data","l")
   leg.AddEntry(data,"Reconstructed Data","l")
   leg.AddEntry(xini,"True MC","l")
   
   global c1
   c1 = TCanvas( "c1", "Unfolding toy example with TSVDUnfold", 1000, 900 )
   
   c1.Divide(1,2)
   c11 = c1.cd(1)
   
   global frame
   frame = TH1D( unfres )
   frame.SetTitle( "Unfolding toy example with TSVDUnfold" )
   frame.GetXaxis().SetTitle( "x variable" )
   frame.GetYaxis().SetTitle( "Events" )
   frame.GetXaxis().SetTitleOffset( 1.25 )
   frame.GetYaxis().SetTitleOffset( 1.29 )
   frame.Draw()
   
   data.SetLineStyle(2)
   data.SetLineColor(4)
   data.SetLineWidth(2)
   unfres.SetMarkerStyle(20)
   dataTrue.SetLineColor(2)
   dataTrue.SetLineWidth(2)
   xini.SetLineStyle(2)
   xini.SetLineColor(8)
   xini.SetLineWidth(2)
   # ------------------------------------------------------------
   
   # add histograms
   unfres.Draw("same")
   dataTrue.Draw("same")
   data.Draw("same")
   xini.Draw("same")
   
   leg.Draw()
   
   # covariance matrix
   c12 = c1.cd(2)
   c12.Divide(2,1)
   c2 = c12.cd(1)
   c2.SetRightMargin   ( 0.15         )
   
   global covframe
   covframe = TH2D( ustatcov )
   covframe.SetTitle( "TSVDUnfold covariance matrix" )
   covframe.GetXaxis().SetTitle( "x variable" )
   covframe.GetYaxis().SetTitle( "x variable" )
   covframe.GetXaxis().SetTitleOffset( 1.25 )
   covframe.GetYaxis().SetTitleOffset( 1.29 )
   covframe.Draw()
   
   ustatcov.SetLineWidth( 2 )
   ustatcov.Draw( "colzsame" )
   
   # distribution of the d quantity
   c3 = c12.cd(2)
   c3.SetLogy()
   
   global line
   line = TLine( 0.,1.,40.,1. )
   line.SetLineStyle(2)
   

   global dframe
   dframe = TH1D( ddist )
   dframe.SetTitle( "TSVDUnfold |d_{i}|" )
   dframe.GetXaxis().SetTitle( "i" )
   dframe.GetYaxis().SetTitle( "|d_{i}|" )
   dframe.GetXaxis().SetTitleOffset( 1.25 )
   dframe.GetYaxis().SetTitleOffset( 1.29 )
   dframe.SetMinimum( 0.001 )
   dframe.Draw()
   
   ddist.SetLineWidth( 2 )
   ddist.Draw( "same" )
   line.Draw()
   


if __name__ == "__main__":
   TSVDUnfoldExample()
