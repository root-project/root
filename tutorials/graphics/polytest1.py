## \file
## \ingroup tutorial_graphics
## \notebook
##
## This macro is testing the "compacting" algorithm in TPadPainter.
## It reduces the number of polygon's vertices using actual pixel coordinates.
##
## \macro_image
##
## It's not really useful, but just to test that the resulting polygon
## is still reasonable. Initial number of points is 1000000, after "compression"
## it's 523904 (with default canvas size, before you tried to resize it) - so almost half of
## vertices were removed but you can still see the reasonable shape. If you resize
## a canvas to a smaller size, the number of vertices after compression can be something like 5000 and even less.
## It's easy to 'fool' this algorithm though in this particular case (ellipse is a kind of fringe case,
## you can easily have a sequence of almost unique vertices (at a pixel level).
##
## \macro_code
##
## \author Timur Pocheptsov
## \translator P. P.


import ROOT
import ctypes

#standard library
std = ROOT.std
vector = std.vector

#classes
TAttFill = ROOT.TAttFill
TAttLine = ROOT.TAttLine
TPolyLine = ROOT.TPolyLine
TRandom = ROOT.TRandom 
TCanvas = ROOT.TCanvas 
TPad = ROOT.TPad
#TError = ROOT.TError 
TNamed = ROOT.TNamed 
TMath = ROOT.TMath 
#cassert = ROOT.cassert 
#vector = ROOT.vector 
#Rtypes = ROOT.Rtypes 

#c-integration
ProcessLine = ROOT.gInterpreter.ProcessLine

#types
UInt_t = ROOT.UInt_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double
#Option_t = ROOT.Option_t #Not implemented yet.
ProcessLine("""
Option_t * Option ( const std::string & o = "" ){
   return new Option_t( o[0] ) ;
}
""")
Option_t = ROOT.Option

nullptr = ROOT.nullptr

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )

#globals
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gApplication = ROOT.gApplication



#Note:
#      None of the below methods works for defininga triple derived class from:
#      TNamed, TAttLine, TAttFill.
#      C++ can do it, so, let's use it as simple as 
#      class MetaClass: public TNamed, public TAttLine, public TAttFill {};
###    #
###    #metaclass
###    #class MetaClass( type(TNamed), type(TAttLine), type(TAttFill) ):
###    #   pass
###    class MetaClass(type):
###        pass
###    #class MC( metaclass = TNamed):
###    #    pass
###    class CustomMeta(type):
###        def __new__(cls, name, bases, dct):
###            # Add custom logic here
###            print(f"Creating class: {name}")
###            return super().__new__(cls, name, bases, dct)
###    #cm = CustomMeta("new", (TNamed, TAttLine, TAttFill), {})
###
###   #Not to use: class PolyTest1META( TNamed,  TAttLine,  TAttFill , MetaClass) :
###   class PolyTest1META( MetaClass):
###      def __init__(self):
###         super(TNamed, self).__init__()
###         super(TAttLine, self).__init__()
###         super(TAttFill, self).__init__()

#Defining a MetaClass
ProcessLine("""
//class MetaClass: public TNamed, public TAttLine, public TAttFill{};
class MetaClass: public TNamed, public TPolyLine{};

//Not to use: class MetaClass: public TNamed, public TAttLine, public TAttFill, public TPad{};
""")
MetaClass = ROOT.MetaClass

#Note: 
#      It would have been easy if we were able to add a TPad-class in MetaClass, but 
#      unfortunately we can't. See the error below.
#      '''
#      input_line_46:3:33: warning: direct base 'TAttLine' is inaccessible due to ambiguity:
#          class MetaClass -> class TAttLine
#          class MetaClass -> class TPad -> class TVirtualPad -> class TAttLine [-Winaccessible-base]
#      class MetaClass: public TNamed, public TAttLine, public TAttFill, public TPad{};
#                                      ^~~~~~~~~~~~~~~
#      input_line_46:3:50: warning: direct base 'TAttFill' is inaccessible due to ambiguity:
#          class MetaClass -> class TAttFill
#          class MetaClass -> class TPad -> class TVirtualPad -> class TAttFill [-Winaccessible-base]
#      class MetaClass: public TNamed, public TAttLine, public TAttFill, public TPad{};
#      '''
      
            

#Using MetClass in Python.
#Not to use: class PolyTest1( TNamed,  TAttLine,  TAttFill ) :
class PolyTest1( MetaClass  ) :
   #public:
   #PolyTest1(unsigned nVertices);
   def __init__(self, nVertices):
      super().__init__()
      
      self.__fXs__ = vector["Double_t"](nVertices)
      self.__fYs__ = vector["Double_t"](nVertices)
      self.kNPointsDefault = 10000
      #self.kNPointsDefault = 10
      
      self.pad = TPad("PolyTest1-name", "PolyTest1-title", 0, 0, 1, 1)
      self.polyline_fill = TPolyLine()
      self.polyline_line = TPolyLine()

      #Not to use:
      #super(TNamed, self).__init__()
      #self.myNamed = TNamed("polygon_compression_test1", "polygon_compression_test1")
      self.SetName("polygon_compression_test1")
      self.SetTitle("polygon_compression_test1")

      self.Reset(nVertices)
      pass


   #void
   def Paint(self, notUsed : Option_t = "notUsed"):
      pass

   #void
   def Reset(self, nVertices : UInt_t ):
      pass

   #Note: 
   #     If we translate this section, enum, it would be adding bad-code
   #     to this script. Best way is to define a constant in-and-outside 
   #     __init__. 

   #private:
   ##enum {
   ##   kNPointsDefault = 10000#minimal number of points.
   ##};

   #class MyEnum(Enum):
   #   #minimal number of points.
   #   kNPointsDefault = 10000
   #   k2 = 10000
   #   k3 = 10000
   ##  ...
   

   #minimal number of points.
   kNPointsDefault = 10000
   #kNPointsDefault = 10

   __fXs__ = vector["Double_t"]()
   __fYs__ = vector["Double_t"]()


#_____________________________________________________________
# Deprecated.
def PolyTest1_PolyTest1(self, nVertices : UInt_t):
   super().__init__()

   self.__fXs__ = vector["Double_t"](nVertices)
   self.__fYs__ = vector["Double_t"](nVertices)
   self.kNPointsDefault = 10000
   #self.kNPointsDefault = 10

   #super(TNamed, self).__init__("AF", "AF")
   #self.myNamed = TNamed("polygon_compression_test1", "polygon_compression_test1")
   self.SetName("polygon_compression_test1")
   self.SetTitle("polygon_compression_test1")

   self.__fXs__ = vector["Double_t"](nVertices)
   self.__fYs__ = vector["Double_t"](nVertices)

   self.kNPointsDefault = 10000
   #self.kNPointsDefault = 10

   self.pad = TPad("PolyTest1-name", "PolyTest1-title", 0, 0, 1, 1)

   self.polyline_fill = TPolyLine()
   self.polyline_line = TPolyLine()

   self.Reset(nVertices)
      
   pass
   

#_____________________________________________________________
#void
def PolyTest1_Reset(self, nVertices : UInt_t):

   #Some canvas must already exist by this point.
   assert(gPad != nullptr ), "Reset, gPad is null"

   #We need a gRandom to exist.
   assert(gRandom != nullptr ), "Reset, gRandom is null"
   
   if nVertices < self.kNPointsDefault:
      Warning("Reset", "resetting nVertices parameter to %u" % UInt_t( self.kNPointsDefault ) )
      nVertices = self.kNPointsDefault
      
   
   #Resizing vectors.
   self.__fXs__.resize(nVertices)
   self.__fYs__.resize(nVertices)
   
   #Initializing parameters.
   xMin = 0.
   xMax = 0.
   yMin = 0.
   yMax = 0.
   #to C-types
   c_xMin = c_double(0.)
   c_xMax = c_double(0.)
   c_yMin = c_double(0.)
   c_yMax = c_double(0.)

   #Getting parameters c_from.
   gPad.GetRange(c_xMin, c_yMin, c_xMax, c_yMax)
   self.pad.GetRange(c_xMin, c_yMin, c_xMax, c_yMax)
   #to Python-types
   xMin = c_xMin.value
   xMax = c_xMax.value
   yMin = c_yMin.value
   yMax = c_yMax.value
   #Debug: print( xMin, xMax, yMin, yMax)


   #Checking parameters validity.
   assertion = (xMax - xMin > 0 and yMax - yMin > 0 )
   assert assertion, "Reset, invalid canvas' ranges"
   
   #Computing middle points.
   xCentre = xMin + 0.5 * (xMax - xMin)
   yCentre = yMin + 0.5 * (yMax - yMin)
   
   #Partitioning.
   r = TMath.Min(xMax - xMin, yMax - yMin) * 0.8 / 2
   angle = TMath.TwoPi() / (nVertices - 1)
   
   #Loop over nVertices and updating values.
   #for (unsigned i = 0; i < nVertices - 1; ++i) {
   for i in range(0, nVertices - 1 , 1):
      currR = r + gRandom.Rndm() * r * 0.01
      self.__fXs__[i] = xCentre + currR * TMath.Cos(angle * i)
      self.__fYs__[i] = yCentre + currR * TMath.Sin(angle * i)
      
   #Circular definition: the end is the start. #Avoid approximations.
   self.__fXs__[nVertices - 1] = self.__fXs__[0]
   self.__fYs__[nVertices - 1] = self.__fYs__[0]
   

#_____________________________________________________________
#void
def PolyTest1_Paint(self, notUsed: Option_t = '''notUsed'''):


   assert(gPad != nullptr ), "Paint, gPad is null"

   c_fXs = to_c( list(self.__fXs__) )
   c_fYs = to_c( list(self.__fYs__) )
   
   #Lines derived from .C version. 
   #TAttFill.Modify()
   #self.Modify()
  
   #Setting-up for PaintFillArea
   #
   #gPad.PaintFillArea( Int_t( self.__fXs__.size() ), self.__fXs__.data(), self.__fYs__.data())
   #gPad.PaintFillArea( Int_t( self.__fXs__.size() ), c_fXs, c_fYs)
   self.polyline_fill.SetPolyLine( Int_t( self.__fXs__.size() ), c_fXs, c_fYs)

   
   #Lines derived from .C version. 
   #TAttLine.Modify()
   #self.Modify()

   #Setting-up for PaintPolyLine 
   #
   #gPad.PaintPolyLine( Int_t( self.__fXs__.size() ), self.__fXs__.data(), self.__fYs__.data())
   #gPad.PaintPolyLine( Int_t( self.__fXs__.size() ), c_fXs, c_fYs)
   self.polyline_line.SetPolyLine( Int_t( self.__fXs__.size() ), c_fXs, c_fYs)
  
   


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#Loading functions into PolyTest1-class.
#PolyTest1.__init__ = PolyTest1_PolyTest1 # Deprecated.
PolyTest1.Paint    = PolyTest1_Paint
PolyTest1.Reset    = PolyTest1_Reset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



#void
def polytest1():

   global cnv
   cnv = TCanvas("cnv", "cnv")
   cnv.cd()
   
   global polygon
   polygon = PolyTest1(1000000)
   #polygon = PolyTest1(10)
   #polygon = PolyTest1(10000)

   polygon.Paint()

   polygon.polyline_line.SetLineColor(kBlue)
   polygon.polyline_line.SetLineWidth(2)
   polygon.polyline_line.Draw("")

   polygon.polyline_fill.SetFillColor(kRed)
   polygon.polyline_fill.Draw("f")

   #Attach a polygon to a canvas.
   polygon.Draw()
   
   cnv.Update()


if __name__ == "__main__":
   polytest1()
