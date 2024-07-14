## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script illustrates the danger of using asymmetric symbols.
##
## \macro_image
##
## Non-symmetric symbols should be used very carefully when plotting.
## These two graphs show how misleading can be a careless use of symbols.
## The two plots represent the same data sets but, because of a bad symbol
## choice, the two plots on the top appear further apart from the bottom
## example.
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TLatex = ROOT.TLatex
TGraphErrors = ROOT.TGraphErrors
TCanvas = ROOT.TCanvas
TPad = ROOT.TPad
TH1F = ROOT.TH1F

#types
c_double = ctypes.c_double

#utils
def to_c(ls):
   return (c_double * len(ls) )( * ls)

#globals
gStyle = ROOT.gStyle



# void
def markerwarning() :
   Nph = 14
   np_ph = [353.4,300.2,254.3,215.2,181.0,151.3,125.2,102.7,
      83.3, 66.7, 52.5, 40.2, 30.2, 22.0 ]
      
   nc_ph = [3.890,3.734,3.592,3.453,3.342,3.247,3.151,3.047,
      2.965,2.858,2.701,2.599,2.486,2.328 ]
      
   npe_ph = [10.068,9.004,8.086,7.304,6.620,6.026,5.504,5.054,
      4.666,4.334,4.050,3.804,3.604,3.440 ]
      
   nce_ph = [0.235,0.217,0.210,0.206,0.213,0.223,0.239,0.260,
      0.283,0.318,0.356,0.405,0.465,0.545 ]
      
   np_ph = to_c(np_ph)      
   nc_ph = to_c(nc_ph)      
   npe_ph = to_c(npe_ph)      
   nce_ph = to_c(nce_ph)   



   Nbr = 6
   np_br = [357.0,306.0,239.0,168.0,114.0, 73.0 ]
   nc_br = [3.501,3.275,3.155,3.060,3.053,3.014 ]
   npe_br = [8.000,11.000,10.000,9.000,9.000,8.000 ]
   nce_br = [0.318,0.311,0.306,0.319,0.370,0.429 ]
   np_br = to_c(np_br)
   nc_br = to_c(nc_br)
   npe_br = to_c(npe_br)
   nce_br = to_c(nce_br)
   
   global phUP, phDN, brUP, brDN
   phUP = TGraphErrors(Nph,np_ph,nc_ph,npe_ph,nce_ph)
   phDN = TGraphErrors(Nph,np_ph,nc_ph,npe_ph,nce_ph)
   brUP = TGraphErrors(Nbr,np_br,nc_br,npe_br,nce_br)
   brDN = TGraphErrors(Nbr,np_br,nc_br,npe_br,nce_br)
   
   Top_margin = 0.
   Left_margin = 0.025
   Right_margin = 0.005
   maxPlotPart = 395
   Marker_Size = 1.3
   Marker_Style = 8
   
   Et_200_Min = 0.71
   Et_200_Max = 3.80
   Et_130_Min = 1.21
   Et_130_Max = 3.29
   
   Nc_200_Min = 1.31
   Nc_200_Max = 4.30
   Nc_130_Min = 1.51
   Nc_130_Max = 3.89
   
   global canvasNc
   canvasNc = TCanvas("canvasNc", "Multiplicity",630,10,600,500)
   
   gStyle.SetOptStat(0)
   canvasNc.SetFillColor(10)
   canvasNc.SetBorderSize(0)
   
   # Primitives in Nc200 pad
   global padNcUP
   padNcUP = TPad("padNcUP","200 GeV",0.07,0.60,1.,1.00)
   padNcUP.Draw()
   padNcUP.cd()
   padNcUP.SetFillColor(10)
   padNcUP.SetFrameFillColor(10)
   padNcUP.SetBorderSize(0)
   padNcUP.SetLeftMargin(Left_margin)
   padNcUP.SetRightMargin(Right_margin)
   padNcUP.SetTopMargin(Top_margin+0.005)
   padNcUP.SetBottomMargin(0.00)
   
   global frameNcUP
   frameNcUP = TH1F("frameNcUP","",100,0,maxPlotPart)
   frameNcUP.GetYaxis().SetLabelOffset(0.005)
   frameNcUP.GetYaxis().SetLabelSize(0.10)
   frameNcUP.SetMinimum(Nc_200_Min)
   frameNcUP.SetMaximum(Nc_200_Max)
   frameNcUP.SetNdivisions(505,"Y")
   frameNcUP.SetNdivisions(505,"X")
   frameNcUP.Draw()
   
   brUP.SetMarkerStyle(22)
   brUP.SetMarkerSize (2.0)
   brUP.Draw("P")
   
   phDN.SetMarkerStyle(23)
   phDN.SetMarkerSize (2)
   phDN.Draw("P")
   
   canvasNc.cd()
   
   # Primitives in Nc130 pad
   global padNcDN
   padNcDN = TPad("padNcDN","130 GeV",0.07,0.02,1.,0.60)
   padNcDN.Draw()
   padNcDN.cd()
   padNcDN.SetFillColor(10)
   padNcDN.SetFrameFillColor(10)
   padNcDN.SetBorderSize(0)
   padNcDN.SetLeftMargin(Left_margin)
   padNcDN.SetRightMargin(Right_margin)
   padNcDN.SetTopMargin(Top_margin+0.005)
   padNcDN.SetBottomMargin(0.30)
   
   global frameNcDN
   frameNcDN = TH1F("frameNcDN","",100,0,maxPlotPart)
   frameNcDN.GetYaxis().SetLabelOffset(0.005)
   frameNcDN.GetYaxis().SetLabelSize(0.07)
   frameNcDN.GetXaxis().SetLabelOffset(0.005)
   frameNcDN.GetXaxis().SetLabelSize(0.07)
   frameNcDN.SetMinimum(Nc_200_Min)
   frameNcDN.SetMaximum(Nc_200_Max)
   frameNcDN.SetNdivisions(505,"Y")
   frameNcDN.SetNdivisions(505,"X")
   frameNcDN.Draw()
   
   brDN.SetMarkerStyle(23)
   brDN.SetMarkerSize (2.0)
   brDN.Draw("P")
   
   phUP.SetMarkerStyle(22)
   phUP.SetMarkerSize (2)
   phUP.Draw("P")
   
   global t1
   t1 = TLatex()
   t1.SetTextFont(12)
   t1.SetTextSize(0.0525)
   t1.DrawLatex(-5,0.6,"Non-symmetric symbols should be used carefully in plotting." +
   "These two graphs show how misleading")
   t1.DrawLatex(-5,0.4,"a careless use of symbols can be. The two plots represent "  +
   "the same data sets but because of a bad")
   t1.DrawLatex(-5,0.2,"symbol choice, the two plots on the top appear further apart " +
   "than for the bottom example.")
   
   canvasNc.cd()
   


if __name__ == "__main__":
   markerwarning()
