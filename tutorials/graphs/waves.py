## \file
## \ingroup tutorial_graphs
## \notebook
##
## Title: Spherical waves in PyROOT.
## Description:
##    The Double Slit Experiment is simulated 
##    using interference()-function and result()-functions 
##    which are the mathematical concepts of Quantum-Mechanics
##    in spherical waves.
## Enjoy!
##
## \macro_image
## \macro_code
##
## \author Otto Schaile
## \translator P. P.


import ROOT
import ctypes

TROOT = ROOT.TROOT 
TCanvas = ROOT.TCanvas 
TColor = ROOT.TColor 
TArc = ROOT.TArc 
TGraph = ROOT.TGraph 
TF2 = ROOT.TF2 
TLine = ROOT.TLine 
TLatex = ROOT.TLatex 
TMath = ROOT.TMath 
TStyle = ROOT.TStyle 

#classes
TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad

#maths
sin = ROOT.sin
cos = ROOT.cos
sqrt = ROOT.sqrt

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double
c_int = ctypes.c_int

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def to_c_int_ptr( ls ):
   return (c_int * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT




#______________________________________________________________
# Double_t
def interference( x : list[Double_t], par : list[Double_t]) :
   x_p2 = x[0] * x[0];
   d_2 = 0.5 * par[2];
   ym_p2 = (x[1] - d_2) * (x[1] - d_2);
   yp_p2 = (x[1] + d_2) * (x[1] + d_2);
   tpi_l = TMath.Pi() /  par[1];
   amplitude = par[0] * (cos(tpi_l  * sqrt(x_p2 + ym_p2)) \
                                  + par[3] * cos(tpi_l  * sqrt(x_p2 + yp_p2)));
   return amplitude * amplitude;


#_____________________________________________________________
# Double_t
def result( x : list[Double_t], par : list[Double_t]) :
   xint = [ Int_t() for _ in range(2) ];
   maxintens = 0; xcur = 14;
   dlambda = 0.1 * par[1];
   #for(Int_t i=0; i<10; i++) {
   for i in range(0, 10, 1):
       xint[0] = xcur;
       xint[1] = x[1];
       intens = interference(xint, par);
       if(intens > maxintens): maxintens = intens;
       xcur -= dlambda;

   return maxintens;



#_____________________________________________________________
# void
def waves( d : Double_t = 3, lmbd : Double_t = 1, amp : Double_t = 10) :

   global c1
   c1 = TCanvas("waves", "A double slit experiment", 300, 40, 1004, 759)
   c1.Range(0, -10,  30, 10)
   c1.SetFillColor(0)

   global pad
   pad = TPad("pr","pr", 0.5, 0, 1., 1)
   pad.Range(0, -10,  15, 10)
   pad.Draw()
   
   colNum = 30
   palette = [ Int_t() for _ in range(colNum) ]
   #for (Int_t i=0; i<colNum; i++) {
   for i in range(0, colNum, 1):
      level = 1.*i/colNum
      palette[i] = TColor.GetColor( TMath.Power(level,0.3),  TMath.Power(level,0.3),  0.5*level)
      # palette[i] = 1001+i;
      
   c_palette = to_c_int_ptr( palette )
   gStyle.SetPalette(colNum, c_palette)
   
   c1.cd()
   

   global f0
   f0 = TF2("ray_source",interference, 0.02, 15, -8, 8, 4)
   f0.SetParameters(amp, lmbd, 0, 0)
   f0.SetNpx(200)
   f0.SetNpy(200)
   f0.SetContour(colNum-2)
   f0.Draw("samecol")
   

   global title
   title = TLatex()
   title.DrawLatex(1.6, 8.5, "A double slit experiment")
   

   global graph, graph_list
   graph_list = []
   graph = TGraph(4)
   graph.SetFillColor(0)
   graph.SetFillStyle(1001)
   graph.SetLineWidth(0)
   graph.SetPoint(0, 0., 0.1)
   graph.SetPoint(1, 14.8, 8)
   graph.SetPoint(2, 0, 8)
   graph.SetPoint(3, 0, 0.1)
   graph.Draw("F")
   graph_list.append( graph )
   

   graph = TGraph(4)
   graph.SetFillColor(0)
   graph.SetFillStyle(1001)
   graph.SetLineWidth(0)
   graph.SetPoint(0, 0, -0.1)
   graph.SetPoint(1, 14.8, -8)
   graph.SetPoint(2, 0, -8)
   graph.SetPoint(3, 0, -0.1)
   graph.Draw("F")
   graph_list.append( graph )
   

   global line, line_list
   line_list = []
   line = TLine(15,-10, 15, 0 - 0.5*d -0.2)
   line.SetLineWidth(10)
   line.Draw()
   line_list.append( line )
   

   line = TLine(15, 0 - 0.5*d +0.2,15, 0 + 0.5*d -0.2)
   line.SetLineWidth(10)
   line.Draw()
   line_list.append( line )
   

   line = TLine(15,0 + 0.5*d + 0.2,15, 10)
   line.SetLineWidth(10)
   line.Draw()
   line_list.append( line )
   
   pad .cd()
   

   global finter
   finter = TF2("interference",interference, 0.01, 14, -10, 10, 4)
   finter.SetParameters(amp, lmbd, d, 1)
   finter.SetNpx(200)
   finter.SetNpy(200)
   finter.SetContour(colNum-2)
   finter.Draw("samecol")
   

   global arc
   arc = TArc()
   arc.SetFillStyle(0)
   arc.SetLineWidth(2)
   arc.SetLineColor(5)
   r = 0.5 * lmbd; dr = lmbd
   #for (Int_t i = 0; i < 16; i++) {
   for i in range(0, 16, 1):
      arc.DrawArc(0,  0.5*d, r, 0., 360., "only")
      arc.DrawArc(0, -0.5*d, r, 0., 360., "only")
      r += dr
      
   
   pad .cd()
   

   global fresult
   fresult = TF2("result",result, 14, 15, -10, 10, 4)
   fresult.SetParameters(amp, lmbd, d, 1)
   fresult.SetNpx(300)
   fresult.SetNpy(300)
   fresult.SetContour(colNum-2)
   fresult.Draw("samecol")
   

   line = TLine(13.8,-10, 14, 10)
   line.SetLineWidth(10)
   line.SetLineColor(0)
   line.Draw()
   line_list.append( line )

   c1.Modified(True)
   c1.Update()
   c1.SetEditable(True)
   


if __name__ == "__main__":
   waves()
