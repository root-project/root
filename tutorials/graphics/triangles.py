## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script generates small triangles randomly in a canvas.
## Each triangle has a unique id-number and a random color from the color palette.
##
## ~~~{.py}
## IP[0]: %run triangles.py
## ~~~
##
## Do click on any triangle and a message will show the triangle id-number
## and its color will be printed.
## Enjoy!
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
TColor = ROOT.TColor
TPolyLine = ROOT.TPolyLine
TRandom = ROOT.TRandom

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
def printf(string, *args):
   print( string % args )

#constants

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gROOT = ROOT.gROOT



# void
def triangles(ntriangles : Int_t = 50) :

   global c1, r
   c1 = TCanvas("c1","triangles",10,10,700,700)
   r = TRandom() 

   dx = 0.2
   dy = 0.2
   ncolors = gStyle.GetNumberOfColors()
   x = [ Double_t() ] *4
   y = [ Double_t() ] *4
   #to C-types
   x = to_c( x )
   y = to_c( x )

   global pl_list
   pl_list = []
   #for (Int_t i=0; i<ntriangles; i++) {
   for i in range(0, ntriangles, 1):
      x[0] = r.Uniform(.05,.95)
      y[0] = r.Uniform(.05,.95)
      x[1] = x[0] + dx * r.Rndm()
      y[1] = y[0] + dy * r.Rndm()
      x[2] = x[1] - dx * r.Rndm()
      y[2] = y[1] - dy * r.Rndm()
      x[3] = x[0]
      y[3] = y[0]

      pl = TPolyLine(4,x,y)
      pl.SetUniqueID(i)

      ci = Int_t( ncolors * r.Rndm() )
      c = ROOT.gROOT.GetColor(TColor.GetColorPalette(ci))
      c.SetAlpha(r.Rndm())

      pl.SetFillColor(ci)
      pl.Draw("f")
      pl_list.append( pl )
      
   command = """
      TPython::Exec( \"TriangleClicked()\" );
   """
   c1.AddExec("ex", command)
   

# void
def TriangleClicked() :

   #This action function is called whenever you move the mouse over the canvas.
   #It just prints the id-number of the picked triangle by your cursor.
   #Let's say, you can add graphics actions instead to add complexity.
   global event
   event = gPad.GetEvent()

   #may be comment this line
   if (event != 11) : return 

   global select
   select = gPad.GetSelected()

   if (not select): return
   if select.InheritsFrom(TPolyLine.Class()):
      pl = select # TPolyLine
      printf("You have clicked triangle %d, color=%d\n",
         pl.GetUniqueID(),pl.GetFillColor()
      )
      
   


if __name__ == "__main__":
   triangles()
