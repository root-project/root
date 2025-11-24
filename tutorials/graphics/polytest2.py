## \file
## \notebook
## \ingroup tutorial_graphics
## This macro is testing the "compacting" algorithm in TPadPainter.
## It reduces the number of polygon's vertices using actual pixel coordinates.
##
## \macro_image
##
## This macro is testing new "compacting" algorithm in TPadPainter
## (it reduces the number of polygon's vertices using actual pixel coordinates).
## In principle, this test case is what our histograms (fringe cases) are:
## "saw-like" polygon (bins == teeth).
##
## \macro_code
##
## \author Timur Pocheptsov
## \translator P. P.


import ROOT
import ctypes
from enum import Enum

#classes
TRandom = ROOT.TRandom 
TCanvas = ROOT.TCanvas 
#Rtypes = ROOT.Rtypes 
TNamed = ROOT.TNamed 
TPolyLine = ROOT.TPolyLine
#cassert = ROOT.cassert  #not to use

#standard library
std = ROOT.std
vector = std.vector 

#constant color
kGreen = ROOT.kGreen
kOrange = ROOT.kOrange
kBlue = ROOT.kBlue
kMagenta = ROOT.kMagenta

#c-integration
ProcessLine = ROOT.gInterpreter.ProcessLine

#types
nullptr = ROOT.nullptr
Int_t = ROOT.Int_t
Float_t = ROOT.Float_t
Double_t = ROOT.Double_t
#Option_t = ROOT.Option_t #Not implemented yet.
ProcessLine("""
Option_t * Option ( const std::string & o = "" ){
   return new Option_t( o[0] ) ;
}
""")
Option_t = ROOT.Option
c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )

#globals
gPad = ROOT.gPad
gRandom = ROOT.gRandom


#Defining a MetaClass
ProcessLine("""
class MetaClass: public TNamed, public TAttLine, public TAttFill{};
//class MetaClass: public TNamed, public TPolyLine{};

//Not to use: class MetaClass: public TNamed, public TAttLine, public TAttFill, public TPad{};
""")
MetaClass = ROOT.MetaClass

#prototype 
#class PolyTest2(TNamed, TAttLine, TAttFill): 
class PolyTest2( MetaClass ): 
#class PolyTest2( ): 
   #public:
   def __init__(self, kNSawPoints : int = 10000):

      super(MetaClass, self).__init__()
      self.kNSawPoints = kNSawPoints 

      #Part 1.
      self.__fXs1__ = vector["Double_t"]()
      self.__fYs1__ = vector["Double_t"]()
      #Part 2.
   
      self.__fXs2__ = vector["Double_t"]()
      self.__fYs2__ = vector["Double_t"]()
      
      #Polygon 1.
      self.polygon1_line = TPolyLine()
      self.polygon1_fill = TPolyLine()

      #Polygon 2.
      self.polygon2_line = TPolyLine()
      self.polygon2_fill = TPolyLine()
      pass

   #void  
   def Paint(self, notUsed : Option_t = "notUsed"): 
      pass

   ##Deprecated.
   ##private:
   #class TestSize( Enum ):
   #   kNSawPoints = 10000
   #kNSawPoints = #10000
   
   
   #Part 1.
   __fXs1__ = vector["Double_t"]()
   __fYs1__ = vector["Double_t"]()
   #Part 2.

   __fXs2__ = vector["Double_t"]()
   __fYs2__ = vector["Double_t"]()

#_____________________________________________________________
def PolyTest2___init__(self, kNSawPoints: int = 10000):
   super(MetaClass, self).__init__()

   self.kNSawPoints = kNSawPoints 
   #Part 1.
   self.__fXs1__ = vector["Double_t"]()
   self.__fYs1__ = vector["Double_t"]()
   #Part 2.

   self.__fXs2__ = vector["Double_t"]()
   self.__fYs2__ = vector["Double_t"]()
   
   self.my_Named = TNamed("polygon_compression_test2", "polygon_compression_test2")

   #Polygon 1.
   self.polygon1_line = TPolyLine()
   self.polygon1_fill = TPolyLine()

   #Polygon 2.
   self.polygon2_line = TPolyLine()
   self.polygon2_fill = TPolyLine()
   #Polygon 1, n of points is 10003, after 'compression' : 1897
   #Polygon 2, n of points is 10003, after 'compression' : 2093
   
   #Some canvas must already exist by this point.
   assert( gPad != nullptr ), "PolyTest2, gPad is null"
   #We need a gRandom to exist.
   assert( gRandom != nullptr ), "PolyTest2, gRandom is null"
   
   self.xMin = c_double(0.)
   self.xMax = c_double(0.)
   self.yMin = c_double(0.)
   self.yMax = c_double(0.)

   gPad.GetRange(self.xMin, self.yMin, self.xMax, self.yMax)

   self.xMin = self.xMin.value
   self.xMax = self.xMax.value
   self.yMin = self.yMin.value
   self.yMax = self.yMax.value

   #Checking ranges. 
   assertion = (self.xMax - self.xMin > 0 and self.yMax - self.yMin > 0) # Bool_t
   assert assertion, "Polself.yTest2, invalid canvas' ranges"
   
   
   # Diagram of dependencies:
   #
   # .(0/the last)--------.(1)
   # |                      /
   # |                      \
   # |                      /
   # .(kNSawPoints + 1)--.(kNSawPoints)
   
   nVertices = 3 + self.kNSawPoints # Int_t
   
   #Polygon 1, "vertical saw":
   #code-block 
   if True:
      self.__fXs1__.resize(nVertices)
      self.__fYs1__.resize(nVertices)
      
      self.__fXs1__[0] = 0.
      self.__fYs1__[0] = 0.
      
      w1 = 0.2 * (self.xMax - self.xMin)
      saw1ToothSize = 0.1 * w1
      yStep = (self.yMax - self.yMin) / (self.kNSawPoints - 1)
      
      #for (unsigned i = 1; i <= kNSawPoints; ++i) {
      for i in range(1, self.kNSawPoints + 1, 1):
         self.__fXs1__[i] = w1 + gRandom.Rndm() * saw1ToothSize
         self.__fYs1__[i] = self.yMin + yStep * (i - 1)
         
      self.__fXs1__[nVertices - 2] = 0.
      self.__fYs1__[nVertices - 2] = self.yMax
      #Let's close it.
      self.__fXs1__[nVertices - 1] = self.__fXs1__[0]
      self.__fYs1__[nVertices - 1] = self.__fYs1__[0]
 
      
   
   #Polygon 2, "horizontal saw":
   #code block
   if True:
      x2Min = self.xMin + 0.25 * (self.xMax - self.xMin)
      h2 = 0.1 * (self.yMax - self.yMin)
      saw2ToothSize = 0.1 * h2
      xStep = (self.xMax - x2Min) / (self.kNSawPoints - 1)
      
      self.__fXs2__.resize(nVertices)
      self.__fYs2__.resize(nVertices)
      
      self.__fXs2__[0] = x2Min
      self.__fYs2__[0] = 0.
      
      #for (unsigned i = 1; i <= kNSawPoints; ++i) {
      for i in range(1, self.kNSawPoints + 1, 1):
         self.__fXs2__[i] = x2Min + xStep * i
         self.__fYs2__[i] = h2 + gRandom.Rndm() * saw2ToothSize
         
      self.__fXs2__[nVertices - 2] = self.xMax
      self.__fYs2__[nVertices - 2] = 0.
      self.__fXs2__[nVertices - 1] = self.__fXs2__[0]
      self.__fYs2__[nVertices - 1] = self.__fYs2__[0]
      
      
      
   

#_____________________________________________________________
#void
def PolyTest2_Paint(self, notUsed: Option_t = '''notUsed'''):

   assert(gPad != nullptr), "Paint, gPad is null"

   c_fXs1 = to_c( list(self.__fXs1__) )
   c_fYs1 = to_c( list(self.__fYs1__) )

   c_fXs2 = to_c( list(self.__fXs2__) )
   c_fYs2 = to_c( list(self.__fYs2__) )
   
   #self.SetFillColor(kGreen)
   #self.TAttFill.Modify()
   #gPad.PaintFillArea( Int_t( self.__fXs1__.size() ), self.__fXs1__[0], self.__fYs1__[0])
   self.polygon1_fill.SetPolyLine( Int_t( self.__fXs1__.size() ), c_fXs1, c_fYs1)
   self.polygon1_fill.SetLineColor(kGreen)
   self.polygon1_fill.SetLineWidth(2)
   self.polygon1_fill.Draw("")
   
   #self.SetLineColor(kBlue)
   #self.TAttLine.Modify()
   #gPad.PaintPolyLine( Int_t( self.__fXs11__.size() ), self.__fXs11__[0], self.__fYs11__[0])
   self.polygon1_line.SetPolyLine( Int_t( self.__fXs1__.size() ), c_fXs1, c_fYs1)
   self.polygon1_line.SetFillColor(kBlue)
   self.polygon1_line.Draw("f")
   
   #self.SetFillColor(kOrange)
   #self.TAttFill.Modify()
   #gPad.PaintFillArea( Int_t( self.__fXs2__.size() ), self.__fXs2__[0], self.__fYs2__[0])
   self.polygon2_fill.SetPolyLine( Int_t( self.__fXs2__.size() ), c_fXs2, c_fYs2)
   self.polygon2_fill.SetLineColor(kOrange)
   self.polygon2_fill.SetLineWidth(1)
   self.polygon2_fill.Draw("")
   
   #self.SetLineColor(kMagenta)
   #self.TAttLine.Modify()
   #gPad.PaintPolyLine( Int_t( self.__fXs2__.size() ), self.__fXs2__[0], self.__fYs2__[0])
   self.polygon2_line.SetPolyLine( Int_t( self.__fXs2__.size() ), c_fXs2, c_fYs2)
   self.polygon2_line.SetFillColor(kMagenta)
   self.polygon2_line.Draw("f")
   
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#Loading functions into PolyTest2-class.

PolyTest2.__init__ = PolyTest2___init__
PolyTest2.Paint = PolyTest2_Paint

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 




#void
def polytest2():

   global cnv
   cnv = TCanvas()
   cnv.cd()
   
   global polygon
   polygon = PolyTest2()
   #polygon = PolyTest2(kNSawPoints = 100000) #Ok.
   #polygon = PolyTest2(kNSawPoints = 1000)   #Ok.
   #polygon = PolyTest2(kNSawPoints = 10)     #Ok.

   #Painting.
   polygon.Paint()
   #Attach a polygon to a canvas.
   polygon.Draw()
   cnv.Update()
   


if __name__ == "__main__":
   polytest2()
