## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script demonstrates the use of color transparency.
##
## It is done by specifying the alpha value of a given color.
## For instance:
##
## ~~~
##    ellipse.SetFillColorAlpha(9, 0.571)
## ~~~
##
## changes the ellipse fill color to the index 9 with an alpha value of 0.571.
## 0. would be fully transparent (invisible) and 1. completely opaque (the default).
##
## The transparency is available on all platforms when the flag
## `OpenGL.CanvasPreferGL` is set to `1` in `$ROOTSYS/etc/system.rootrc`, or
## on Mac with the Cocoa backend. X11 does not support transparency. On the file
## output it is visible with PDF, PNG, Gif, JPEG, SVG, TeX ... but not PostScript.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TLatex = ROOT.TLatex
TArrow = ROOT.TArrow
TEllipse = ROOT.TEllipse
TGraph = ROOT.TGraph
TMarker = ROOT.TMarker

#types
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )


# void
def transparency() :

   global c1
   c1 = TCanvas("c1", "c1",224,330,700,527)
   c1.Range(-0.125,-0.125,1.125,1.125)
   

   global tex1
   tex1 = TLatex(0.06303724,0.0194223,"This text is opaque and this line is transparent")
   tex1.SetLineWidth(2)
   tex1.Draw()
   

   global arrow
   arrow = TArrow(0.5555158,0.07171314,0.8939828,0.6195219,0.05,"|>")
   arrow.SetLineWidth(4)
   arrow.SetAngle(30)
   arrow.Draw()
   
   # Draw a transparent graph.
   # Double_t
   x = [
      0.5232808, 0.8724928, 0.9280086, 0.7059456, 0.7399714,
      0.4659742, 0.8241404, 0.4838825, 0.7936963, 0.743553
   ]    
   # Double_t
   y = [
      0.7290837, 0.9631474, 0.4775896, 0.6494024, 0.3555777,
      0.622012, 0.7938247, 0.9482072, 0.3904382, 0.2410359
   ]
   
   #to C-types
   c_x = to_c( x )
   c_y = to_c( y )
   

   global graph
   graph = TGraph(10, c_x, c_y)
   graph.SetLineColorAlpha(46, 0.1)
   graph.SetLineWidth(7)
   graph.Draw("l")
   
   # Draw an ellipse with opaque colors.

   global ellipse1
   ellipse1 = TEllipse(0.1740688,0.8352632,0.1518625,0.1010526,0,360,0)
   ellipse1.SetFillColor(30)
   ellipse1.SetLineColor(51)
   ellipse1.SetLineWidth(3)
   ellipse1.Draw()
   
   # Draw an ellipse with transparent colors, above the previous one.

   global ellipse2
   ellipse2 = TEllipse(0.2985315,0.7092105,0.1566977,0.1868421,0,360,0)
   ellipse2.SetFillColorAlpha(9, 0.571)
   ellipse2.SetLineColorAlpha(8, 0.464)
   ellipse2.SetLineWidth(3)
   ellipse2.Draw()
   
   # Draw a transparent blue text.

   global tex2
   tex2 = TLatex(0.04871059,0.1837649,"This text is transparent")
   tex2.SetTextColorAlpha(9, 0.476)
   tex2.SetTextSize(0.125)
   tex2.SetTextAngle(26.0)
   tex2.Draw()
   
   # Draw two transparent markers

   global marker
   marker = TMarker(0.03080229,0.998008,20)
   marker.SetMarkerColorAlpha(2, .3)
   marker.SetMarkerStyle(20)
   marker.SetMarkerSize(1.7)
   marker.Draw()

   global marker2
   marker2 = TMarker(0.1239255,0.8635458,20)
   marker2.SetMarkerColorAlpha(2, .2)
   marker2.SetMarkerStyle(20)
   marker2.SetMarkerSize(1.7)
   marker2.Draw()
   
   # Draw an opaque marker

   global marker3
   marker3 = TMarker(0.3047994,0.6344622,20)
   marker3.SetMarkerColor(2)
   marker3.SetMarkerStyle(20)
   marker3.SetMarkerSize(1.7)
   marker3.Draw()
   
   


if __name__ == "__main__":
   transparency()
