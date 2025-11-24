## \file
## \ingroup tutorial_graphics
## \notebook
##
## A “Perceptual” colormap explicitly identifies a fixed value in the data.
##
## On geographical plot, a fixed point can, for instance, be the "sea level". A perceptual
## colormap provides monotonic-luminance-variations above and below this fixed value.
## Unlike the rainbow colormap, this other colormap provides a faithful representation of the
## structures in the data.
##
## This macro demonstrates how to produce the perceptual colormap shown on the figure 2
## in [this article]
## < (https://root.cern/blog/rainbow-color-map/ > . 
##
## The function `Perceptual_Colormap` takes two parameters as its input:
##   1. `h`, the `TH2D` to be drawn
##   2. `val_cut`, the Z value defining the "sea level"
##
## Having these parameters this function defines two color maps: 
## one above `val_cut` and
## one below it.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TColor = ROOT.TColor
TH2D = ROOT.TH2D

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

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gROOT = ROOT.gROOT

gRandom = ROOT.gRandom



# void
def Perceptual_Colormap(h : TH2D, val_cut : Double_t) :

   Max = h.GetMaximum() # Histogram's maximum
   Min = h.GetMinimum() # Histogram's minimum
   per_cut = (val_cut-Min)/(Max-Min) # normalized value of val_cut
   eps = (Max-Min)*0.00001 # epsilon
   
   # Definition of the two palettes below and above val_cut
   Number = 4
   Red = [ 0.11, 0.19, 0.30, 0.89 ]
   Green = [ 0.03, 0.304, 0.60, 0.91 ]
   Blue = [ 0.18, 0.827, 0.50, 0.70 ]
   Stops = [ 0., per_cut, per_cut+eps, 1. ]
   #to C-types 
   c_Red = to_c( Red )
   c_Green = to_c( Green )
   c_Blue = to_c( Blue )
   c_Stops = to_c( Stops )

   nb = Int_t(256)
   h.SetContour(nb)
   
   TColor.CreateGradientColorTable(Number, c_Stops, c_Red, c_Green, c_Blue, nb)
   
   # Histogram drawing
   h.Draw("colz")
   

# void
def perceptualcolormap() :

   global h
   h = TH2D("h","Perceptual Colormap",200,-4,4,200,-4,4)
   h.SetStats(0)
   
   a = Double_t()
   b = Double_t()
   c_a = c_double( a )
   c_b = c_double( b )
   #for (Int_t i=0; i<1000000; i++) {
   for i in range(0, 1000000,1):

      gRandom.Rannor(c_a, c_b)
      a =  c_a.value
      b =  c_b.value
      
      h.Fill(a-1.5,b-1.5,0.1)
      h.Fill(a+2.,b-3.,0.07)
      h.Fill(a-3.,b+3.,0.05)

      gRandom.Rannor(c_a, c_b)
      h.Fill(a+1.5,b+1.5,-0.08)
      a =  c_a.value
      b =  c_b.value
      
   Perceptual_Colormap(h, 0.)
   


if __name__ == "__main__":
   perceptualcolormap()
