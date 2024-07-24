## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script shows a 3-d polymarker.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.



import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TView = ROOT.TView
TPolyMarker3D = ROOT.TPolyMarker3D
TPaveText = ROOT.TPaveText

#maths
sin = ROOT.sin
cos = ROOT.cos

#types
Double_t = ROOT.Double_t
c_double = ctypes.c_double

#utils
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gROOT = ROOT.gROOT
gBenchmark = ROOT.gBenchmark



# void
def tornado() :

   gBenchmark.Start("tornado")
   
   PI = 3.141592653
   d = 16
   numberOfPoints=200
   numberOfCircles=40
   
   # create and open a canvas
   global sky
   sky = TCanvas( "sky", "Tornado", 300, 10, 700, 500 )
   sky.SetFillColor(14)
   
   # creating view
   global view
   view = TView.CreateView(1,0,0)
   Range = numberOfCircles*d
   view.SetRange( 0, 0, 0, 4.0*Range, 2.0*Range, Range )
   
   #for( int j = d; j < numberOfCirclesd; j += d ) {
   for j in range(d, numberOfCircles*d, d):
      
      # create a PolyMarker3D
      pm3d = TPolyMarker3D( numberOfPoints )
      
      x = y = z = Double_t() 
      
      # set points
      #for( int i = 1; i < numberOfPoints; i++ ) {
      for i in range(1, numberOfPoints, 1):
         csin = sin(2*PI / numberOfPoints  *i) + 1
         ccos = cos(2*PI / numberOfPoints  *i) + 1
         esin = sin(2*PI / (numberOfCircles*d) * j) + 1
         x = j * ( csin + esin )
         y = j * ccos
         z = j
         c_x = c_double(x)
         c_y = c_double(y)
         c_z = c_double(z)
         pm3d.SetPoint( i, x, y, z )
         
      
      # set marker size, color & style
      pm3d.SetMarkerSize( 1 )
      pm3d.SetMarkerColor( 2 + ( d == ( j & d ) ) )
      pm3d.SetMarkerStyle( 3 )
      
      #draw
      pm3d.Draw()
      
   
   timeStr = " "*60
   gBenchmark.Show("tornado")
   
   ct = gBenchmark.GetCpuTime("tornado")
   timeStr = sprintf( timeStr, "Execution time: %g sec.", ct)
   
   global text
   text = TPaveText( 0.1, 0.81, 0.9, 0.97 )
   text.SetFillColor( 42 )
   text.AddText("ROOT example: tornado.C")
   text.AddText(timeStr)
   text.Draw()

   sky.Update()
   


if __name__ == "__main__":
   tornado()
