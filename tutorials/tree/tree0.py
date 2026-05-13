## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## Simple Event class example.
##
##
## Execute it as:
## ~~~
##   IPython [1]: %run tree0.py
## ~~~
##
## Note : 
##       You'll have to copy it first to a directory where 
##       you have write access permissions!
##
##
## Additional Remarks:
## 
## Effect of ClassDef() and ClassImp() macros.-
##
##    After running this script create an instance of `Det` and `Event` classes.
##    ~~~
##      IPython [2]: d = Det()
##      IPython [3]: e = Event()
##    ~~~
##    Now you can see the effect of the ClassDef() and ClassImp() ROOT macros.
##    "Det" class doesn't have "ClassDef" and "ClassImp" implemented, 
##    whereas "Event" class includes both. If you want to see changes, uncomment
##    the lines in "Det" containing "ClassDef" and "ClassImp".
##    For instance, "e" knows who it is:
##    ~~~
##      IPython [4]: print( e.Class_Name() )
##    ~~~
##    whereas "d" does not.
##    ~~~
##      IPython [5]: print( d.Class_Name() ) # Raises error. No member name.
##    ~~~
##
##    The new methods that were added by the ClassDef()/Imp() macros can be 
##    listed with:
##    ~~~
##      IPython [6]: [ _ for _ in dir( Event ) if not _.startswith("_") ]
##      IPython [7]: [ _ for _ in dir( Det   ) if not _.startswith("_") ]
##    ~~~
##    You'll see their differences.
##
##
## \macro_code
##
## \author Heiko.Scheit@mpi-hd.mpg.de
## \translator P. P.


import ROOT
import ctypes
from array import array



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TRandom,
                   TStyle,
                   TTree,
                   TObject,
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
)

# maths
from ROOT import (
                   sin,
                   cos,
                   sqrt,
)

# types
from ROOT import (
                   Double_t,
                   Bool_t,
                   Float_t,
                   Int_t,
                   nullptr,
)
#
from ctypes import (
                     c_double,
                     c_float,
                     )


# utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args, end="")
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

# constants
from ROOT import (
                   kBlue,
                   kRed,
                   kGreen,
)

# globals
from ROOT import (
                   gStyle,
                   gPad,
                   gRandom,
                   gBenchmark,
                   gROOT,
)


# seamlessly integration with C
from ROOT.gROOT import ProcessLine
# 
# ProcessLine = gROOT.ProcessLine


# Serialization (saving the object to a file)
def save_my_class(instance, filename):
    with open(filename, 'wb') as file:  # 'wb' means write in binary mode
        pickle.dump(instance, file)  # Serialize the instance and write to the file

# Deserialization (loading the object from a file)
def load_my_class(filename):
    with open(filename, 'rb') as file:  # 'rb' means read in binary mode
        return pickle.load(file)  # Read the file and deserialize the object

#
# Equivalent functions of ClassImp and ClassDef
# in Python. We take advante of "pickle" module.
#
# Serialize object.
def ClassDef( instance, filename ):
   with open( filename, "wb" ) as file:
      pickle.dump( instance, file )
#
# Deserialize object.
def ClassImp( filename ):
   with open( filename, "rb" ) as file:
      pickle.load( file )



#
# Note :
#       We will skip this part, since TObject and other classes
#       can NOT be inherited from ROOT by now.
#      
#
# #
# # Each detector gives an energy and time signal.
# #
# # class Det  : public TObject  {
# # class Det( TObject ):
# class Det :
#    #
#    #public:
#    #
#    e = c_double() # energy # Double_t
#    t = c_double() # time   # Double_t
#    
#    def __init__(self, ):
#       #
#       # Here comes the tricky part.
#       # super( TObject, self ).__init__()
#       # TObject.__init__(self, )
#       #
#       tobj = TObject( )
#    #
#    # One intermediate layer.
#    def __getattribute__( self, name )
#       return self.tobj.__getattribute( name )
#    
#    
# 
# # ClassImp(Det)
# 
# 
# 
# #
# # 
# # class Event { 
# class Event :
#    # TObject is not required by this example.
# 
#    # Say there are two detectors (a and b) in the experiment.
#    #public:
#    a = Det()  
#    b = Det() 
#    #ClassDefOverride(Event, 1)
#    
#    def __init__( self, )
#       pass
# 
# #ClassImp(Event)


ProcessLine("""
// class Det  : public TObject  {
class Det { // each detector gives an energy and time signal
 public:
   Double_t e; // energy
   Double_t t; // time

   //  ClassDef(Det,1)
};

// ClassImp(Det)

// class Event { //TObject is not required by this example
class Event : public TObject {
 public:
   Det a; // say there are two detectors (a and b) in the experiment
   Det b;
   ClassDefOverride(Event, 1)
};

ClassImp(Event)

""")
#
Event = ROOT.Event
Det   = ROOT.Det



# void
def tree0() :
   #
   # Create a TTree.
   global tree
   tree = TTree("tree", "treelibrated tree")  # TTree
   # 
   # Note :
   #         A tree calibrated with tree specifications
   #         is a "treelibrated tree".

   #
   # Create an instance of Event.
   global e
   e = Event()  # Event
   
   #
   # Create a branch with Event as a type.
   tree.Branch("event", e)
   
   #
   # Fill some events with random numbers for 
   # for the energy values.
   #
   nevent = 10000  # Int_t
   #for (Int_t iev = 0; iev < nevent; iev++) {
   for iev in range(0, nevent, 1):
      #
      # Status Report
      if (iev % 1000 == 0)  :
         print( "Processing event " , iev , "...")
         
      #
      # The two energies follow a Gauss distribution.
      #
      ea = c_float() # Float_t
      eb = c_float() # Float_t
      gRandom.Rannor(ea, eb)  

      # Loading-up values on the event. 
      e.a.e = ea.value
      e.b.e = eb.value

      # Loading-up flying time randomly.
      # Uniform and Normal for start and end.
      e.a.t = gRandom.Rndm()                
      e.b.t = e.a.t + gRandom.Gaus(0., .1)  
      # ... 'resolution time' is sigma = 0.1 .

      #
      # Fill the tree with the current event.
      tree.Fill()  
      
   
   #
   # Start the viewer.
   # Here you can investigate the structure of your Event class.
   tree.StartViewer()
   
   # Uncomment to set a different style.
   # gROOT.SetStyle("Plain")    
   
   # Now draw some tree variables.
   global c1
   c1 = TCanvas()  # TCanvas
   c1.Divide(2, 2)

   #
   c1.cd(1)
   # Energy of det a.
   tree.Draw("a.e")                                 
   # Same but with a condition on energy b, and is scaled by 3
   tree.Draw("a.e", "3*(-.2<b.e  b.e<.2)", "same")  

   #
   c1.cd(2)
   # One energy against the other.
   tree.Draw("b.e:a.e", "", "colz")  

   #
   c1.cd(3)
   # Time of b with errorbars.
   tree.Draw("b.t", "", "e")     
   # Overlay time of the detector a.
   tree.Draw("a.t", "", "same")  

   #
   c1.cd(4)
   # Plot time of b against time of a.
   tree.Draw("b.t:a.t")  
   
   #
   c1.Update()
   c1.Draw()
   
   #
   print( "\n" )
   print( "You can now examine the structure of your tree in the TreeViewer.")
   print( "\n" )
   


if __name__ == "__main__":
   tree0()
