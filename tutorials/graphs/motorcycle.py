## \file
## \ingroup tutorial_graphs
## \notebook
## Macro to test scatterplot smoothers: ksmooth, lowess, supsmu
## as described in:
##
##      Modern Applied Statistics with S-Plus, 3rd Edition
##      W.N. Venables and B.D. Ripley
##      Chapter 9: Smooth Regression, Figure 9.1
##
## Example is a set of data on 133 observations of acceleration against time
## for a simulated motorcycle accident, taken from Silverman (1985).
##
## \macro_image
## \macro_code
##
## \author Christian Stratowa, Vienna, Austria
## \translator P. P.


import ROOT
import ctypes

#standard library
ifstream = ROOT.ifstream

TString = ROOT.TString 
TInterpreter = ROOT.TInterpreter 
fstream = ROOT.fstream 
TH1 = ROOT.TH1 
TGraphSmooth = ROOT.TGraphSmooth 
TCanvas = ROOT.TCanvas 
TSystem = ROOT.TSystem 

#classes
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TGraph = ROOT.TGraph
TLatex = ROOT.TLatex

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
char = ROOT.char
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
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

#system utils
Remove = gROOT.Remove



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#variables
vC1 = TCanvas()
grin, grout = [ TGraph() for _ in range(2) ]

# void
def DrawSmooth(pad : Int_t, title : char, xt : char, yt : char) :

   vC1.cd(pad)

   global vFrame
   vFrame = gPad.DrawFrame(0,-130,60,70)
   vFrame.SetTitle(title)
   vFrame.SetTitleSize(0.2)
   vFrame.SetXTitle(xt)
   vFrame.SetYTitle(yt)

   grin.Draw("P")

   grout.DrawClone("LPX")
   

# void
def motorcycle() :

   # data taken from R library MASS: mcycle.txt
   global Dir
   Dir = gROOT.GetTutorialDir()
   Dir.Append("/graphs/")
   Dir.ReplaceAll("/./","/")
   #Debug: print("Dir", Dir)
   
   # read file and add to fit object
   global x, y
   x = [ Double_t() for _ in range(133) ]
   y = [ Double_t() for _ in range(133) ]
   x = to_c( x )
   y = to_c( y )

   vX, vY = [ c_double() for _ in range(2) ]
   vNData = 0 # Int_t

   global vInput
   vInput = ifstream()
   vInput.open( "%smotorcycle.dat"% Dir.Data() )

   while (1) :
      vInput >> vX >> vY
      if ( not vInput.good()) : break
      x[vNData] = vX
      y[vNData] = vY
      vNData += 1
      #end-while
   vInput.close()
   #Debug: print("vndata", vNData)
   global grin
   grin = TGraph(vNData,x,y)
   
   # draw graph

   global vC1
   vC1 = TCanvas("vC1","Smooth Regression",200,10,900,700)
   vC1.Divide(2,3)
   
   # Kernel Smoother
   # create new kernel smoother and smooth data with bandwidth = 2.0

   global gs, grout
   gs = TGraphSmooth("normal")

   grout = gs.SmoothKern(grin,"normal",2.0)
   DrawSmooth(1,"Kernel Smoother: bandwidth = 2.0","times","accel")
   
   # redraw ksmooth with bandwidth = 5.0
   grout = gs.SmoothKern(grin,"normal",5.0)
   DrawSmooth(2,"Kernel Smoother: bandwidth = 5.0","","")
   
   # Lowess Smoother
   # create new lowess smoother and smooth data with fraction f = 2/3
   grout = gs.SmoothLowess(grin,"",0.67)
   DrawSmooth(3,"Lowess: f = 2/3","","")
   
   # redraw lowess with fraction f = 0.2
   grout = gs.SmoothLowess(grin,"",0.2)
   DrawSmooth(4,"Lowess: f = 0.2","","")
   
   # Super Smoother
   # create new super smoother and smooth data with default bass = 0 and span = 0
   grout = gs.SmoothSuper(grin,"",0,0)
   DrawSmooth(5,"Super Smoother: bass = 0","","")
   
   # redraw supsmu with bass = 3 (smoother curve)
   grout = gs.SmoothSuper(grin,"",3)
   DrawSmooth(6,"Super Smoother: bass = 3","","")
   
   # cleanup
   ##Remove( x )
   ##Remove( y )
   ##Remove( gs )
   del x
   del y
   del gs
   


if __name__ == "__main__":
   motorcycle()
