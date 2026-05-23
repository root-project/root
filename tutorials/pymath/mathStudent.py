## \file
## \ingroup tutorial_math
## \notebook
## Tutorial illustrating the use of the Student and F-distributions.
##
## \macro_image
## \macro_code
##
## \author Anna Kreshuk
## \translator P. P.


import ROOT
import ctypes

TMath = ROOT.TMath 
TF1 = ROOT.TF1 
TCanvas = ROOT.TCanvas 
#Riostream = ROOT.Riostream 
TLegend = ROOT.TLegend 
TLegendEntry = ROOT.TLegendEntry 

#types
c_double = ctypes.c_double

#utils 
def to_c(ls):
   return (c_double * len(ls) )( * ls )

#globals
gPad = ROOT.gPad


# void
def mathStudent() :

   #drawing the set of student density functions
   #normal(0, 1) density drawn for comparison
   global DistCanvas
   DistCanvas = TCanvas("DistCanvas", "Distribution graphs", 10, 10, 800, 650)

   DistCanvas.SetFillColor(17)
   DistCanvas.Divide(2, 2)
   DistCanvas.cd(1)

   gPad.SetGrid()
   gPad.SetFrameFillColor(19)

   global leg
   leg = TLegend(0.6, 0.7, 0.89, 0.89)
   
   
   global fgaus
   fgaus = TF1("gaus", "TMath::Gaus(x, [0], [1], [2])", -5, 5)
 
   fgaus.SetTitle("Parametrized Student density functions")
   fgaus.SetLineStyle(2)
   fgaus.SetLineWidth(1)

   params = [ 0, 1, True ]
   c_params = to_c( params )
   fgaus.SetParameters( c_params )

   leg.AddEntry(fgaus.DrawCopy(), "Normal(0,1)", "l")
   
   global student
   student = TF1("student", "TMath::Student(x,[0])", -5, 5)

   student.SetTitle("Student density")
   student.SetLineWidth(1)
   student.SetParameter(0, 10)
   student.SetLineColor(4)

   leg.AddEntry(student.DrawCopy("lsame"), "10 degrees of freedom", "l")
   
   student.SetParameter(0, 3)
   student.SetLineColor(2)
   leg.AddEntry(student.DrawCopy("lsame"), "3 degrees of freedom", "l")
   
   student.SetParameter(0, 1)
   student.SetLineColor(1)
   leg.AddEntry(student.DrawCopy("lsame"), "1 degree of freedom", "l")
   
   leg.Draw()
   
   #Drawing the set of student's cumulative-probability functions.
   DistCanvas.cd(2)
   gPad.SetFrameFillColor(19)
   gPad.SetGrid()

   global studentI, leg2
   studentI = TF1("studentI", "TMath::StudentI(x, [0])", -5, 5)
   leg2 = TLegend(0.6, 0.4, 0.89, 0.6)

   studentI.SetTitle("Student cumulative dist.")
   studentI.SetLineWidth(1)
   
   studentI.SetParameter(0, 10)
   studentI.SetLineColor(4)
   leg2.AddEntry(studentI.DrawCopy(), "10 degrees of freedom", "l")
   
   studentI.SetParameter(0, 3)
   studentI.SetLineColor(2)
   leg2.AddEntry(studentI.DrawCopy("lsame"), "3 degrees of freedom", "l")
   
   studentI.SetParameter(0, 1)
   studentI.SetLineColor(1)
   leg2.AddEntry(studentI.DrawCopy("lsame"), "1 degree of freedom", "l")
   leg2.Draw()
   
   #Drawing the set of F-distribution-densities.
   global fDist, legF1
   fDist = TF1("fDist", "TMath::FDist(x, [0], [1])", 0, 2)
   legF1 = TLegend(0.7, 0.7, 0.89, 0.89)

   fDist.SetTitle("F-Dist. density")
   fDist.SetLineWidth(1)
   
   DistCanvas.cd(3)
   gPad.SetFrameFillColor(19)
   gPad.SetGrid()
   
   fDist.SetParameters(1, 1)
   fDist.SetLineColor(1)
   legF1.AddEntry(fDist.DrawCopy(), "N=1 M=1", "l")
   
   fDist.SetParameter(1, 10)
   fDist.SetLineColor(2)
   legF1.AddEntry(fDist.DrawCopy("lsame"), "N=1 M=10", "l")
   
   fDist.SetParameters(10, 1)
   fDist.SetLineColor(8)
   legF1.AddEntry(fDist.DrawCopy("lsame"), "N=10 M=1", "l")
   
   fDist.SetParameters(10, 10)
   fDist.SetLineColor(4)
   legF1.AddEntry(fDist.DrawCopy("lsame"), "N=10 M=10", "l")
   
   legF1.Draw()
   
   #Drawing the set of F-cumulative-distribution functions.
   global fDistI, legF2
   fDistI = TF1("fDist", "TMath::FDistI(x, [0], [1])", 0, 2)
   legF2 = TLegend(0.7, 0.3, 0.89, 0.5)

   fDistI.SetTitle("Cumulative dist. function for F")
   fDistI.SetLineWidth(1)
   
   DistCanvas.cd(4)
   gPad.SetFrameFillColor(19)
   gPad.SetGrid()
   fDistI.SetParameters(1, 1)
   fDistI.SetLineColor(1)
   legF2.AddEntry(fDistI.DrawCopy(), "N=1 M=1", "l")
   
   fDistI.SetParameters(1, 10)
   fDistI.SetLineColor(2)
   legF2.AddEntry(fDistI.DrawCopy("lsame"), "N=1 M=10", "l")
   
   fDistI.SetParameters(10, 1)
   fDistI.SetLineColor(8)
   legF2.AddEntry(fDistI.DrawCopy("lsame"), "N=10 M=1", "l")
   
   fDistI.SetParameters(10, 10)
   fDistI.SetLineColor(4)
   legF2.AddEntry(fDistI.DrawCopy("lsame"), "N=10 M=10", "l")
   
   legF2.Draw()
   DistCanvas.cd()
   


if __name__ == "__main__":
   mathStudent()
