## \file
## \ingroup tutorial_math
## \notebook
##
##
## Script describing how to use the special mathematical functions of libmathcore.so
## taking full advantage of the precision and speed of the C99 compliant.
## environments. To execute the macro type in:
##
## ~~~{.py}
## IP[0] %run mathcoreSpecFunc.py
## ~~~
##
## This will create two canvases:
##
##   1. One with the representation of the tgamma, lgamma, erf and erfc functions.
##   2. One with the relative difference between the old ROOT versions and the
##      C99 implementation (on the obsolete platform+compiler combinations, which are
##      not C99 compliant, it will call the original ROOT implementations; hence
##      the difference will be 0).
##
## The naming and numbering of the functions is taken from:
## [Matt Austern, (Draft) Technical Report on Standard Library Extensions, N1687=04-0127, September 10, 2004]
## (A HREF="http:www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1687.pdf")
## <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1687.pdf>
##
## \macro_image
## \macro_code
##
## \author Andras Zsenei
## \translator P. P.


import ROOT
TF1 = ROOT.TF1 
TSystem = ROOT.TSystem 
TCanvas = ROOT.TCanvas 



#math
Math = ROOT.Math

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t

#constants
kWhite = ROOT.kWhite
kBlue = ROOT.kBlue



# void
def mathcoreSpecFunc() :
   
   global f1a, f1b, f2a, f2b, f3a, f3b, f4a, f4b
   f1a = TF1("f1a","ROOT::Math::tgamma(x)",0,20)
   f1b = TF1("f1b","abs((ROOT::Math::tgamma(x)-TMath::Gamma(x))/ROOT::Math::tgamma(x))",0,20)
   
   f2a = TF1("f2a","ROOT::Math::lgamma(x)",0,100)
   f2b = TF1("f2b","abs((ROOT::Math::lgamma(x)-TMath::LnGamma(x))/ROOT::Math::lgamma(x))",0,100)
   
   f3a = TF1("f3a","ROOT::Math::erf(x)",0,5)
   f3b = TF1("f3b","abs((ROOT::Math::erf(x)-TMath::Erf(x))/ROOT::Math::erf(x))",0,5)
   
   f4a = TF1("f4a","ROOT::Math::erfc(x)",0,5)
   f4b = TF1("f4b","abs((ROOT::Math::erfc(x)-TMath::Erfc(x))/ROOT::Math::erfc(x))",0,5)
   
   
   global c1
   c1 = TCanvas("c1","ROOT::Math functions",800,600)
   
   f1a.SetLineColor(kBlue)
   f1b.SetLineColor(kBlue)
   f2a.SetLineColor(kBlue)
   f2b.SetLineColor(kBlue)
   f3a.SetLineColor(kBlue)
   f3b.SetLineColor(kBlue)
   f4a.SetLineColor(kBlue)
   f4b.SetLineColor(kBlue)
   
   c1.Divide(2,2)
   
   c1.cd(1)
   f1a.Draw()
   c1.cd(2)
   f2a.Draw()
   c1.cd(3)
   f3a.Draw()
   c1.cd(4)
   f4a.Draw()
   
   
   # Relative Errors of ROOT::Math with respect to ROOT::TMath .
   global c2
   c2 = TCanvas("c2","Relative Errors of ROOT::Math with respect to ROOT::TMath. ",800,600)
   c2.SetTitle("Relative Error of ROOT::Math w.r.t ROOT::TMath")
   
   c2.Divide(2,2)
   
   c2.cd(1)
   f1b.Draw()
   c2.cd(2)
   f2b.Draw()
   c2.cd(3)
   f3b.Draw()
   c2.cd(4)
   f4b.Draw()
   
   


if __name__ == "__main__":
   mathcoreSpecFunc()
