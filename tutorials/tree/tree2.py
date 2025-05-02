## \file
## \ingroup tutorial_tree
## \notebook
##
##
## This example illustrates how to make a Tree from variables or arrays
## in a C struct - without a dictionary, by creating the branches for
## builtin types (int, float, double) and arrays explicitly.
##
## See tree2a.py for the same example but using a class with a dictionary
## instead of a C-struct.
##
## Specifically in this example, we are mapping a C-struct to one of the Geant4
## common blocks `/gctrak/`. In a typical particle physics analysis scenario,
## these common blocks are filled by some `Geant4` script analysis within a loop. 
## Thus, in the filling process, at each step of such loop, only the `TTree.Fill` 
## method should be called.
##
## In here, our example emulates the Geant4 step routines.
## To run it, simply do:
##
## ~~~
## IPython[1]: %run tree2.py
## ~~~
## If want to explore the generated root file "tree2.root", do: 
## ~~~
## IPython[2]: new_browser = TBroser()
## ~~~
## 
## Enjoy it with glee! and learn by earn!
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.



from enum import Enum, auto
from struct import Struct
from array import array
import os
import ctypes
import numpy as np

import ROOT



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   addressof,
                   TCanvas,
                   TFile,
                   TH2,
                   TMath,
                   TRandom,
                   TTree,
                   TBrowser,
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
from ctypes import c_double
#
#float32 = np.float32
#int32 = np.int32
#bool_ = np.bool_
# Warning!... Use it with care:
#int = np.int32
#float = np.float32
# because, type(1) is still int.

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

# seamlessly C integration
from ROOT.gROOT import ProcessLine


# to avoid memory leaks 
try : 
   os.remove("tree2.root")
except FileNotFoundError:
   pass



# variables
# Number of Mechanisms or Processes.
MAXMEC = 30 # Int_t 
#MAXMEC = 10 # Int_t 
#MAXMEC = 5  # Int_t 


# Not to use:
# typedef
# #class Geant4Struct( Struct )  :
# class Geant4Struct( )  :
#    vect      = [ Float_t() for _ in range(7) ]      
#    getot     = Float_t()                            
#    gekin     = Float_t()                            
#    vout      = [ Float_t() for _ in range(7) ]      
#    nmec      = Int_t()                              
#    lmec      = [ Int_t() for _ in range(MAXMEC) ]   
#    namec     = [ Int_t() for _ in range(MAXMEC) ]   
#    nstep     = Int_t()                              
#    pid       = Int_t()                              
#    destep    = Float_t()                            
#    destel    = Float_t()                            
#    safety    = Float_t()                            
#    sleng     = Float_t()                            
#    step      = Float_t()                            
#    snext     = Float_t()                            
#    sfield    = Float_t()                            
#    tofg      = Float_t()                            
#    gekrat    = Float_t()                            
#    upwght    = Float_t()                            
# Instead:
# 

def float32( size ) : return np.zeros( size, dtype=np.float32 )
def int32( size )   : return np.zeros( size, dtype=np.int32 )

# Prototype for a future version.
#class float32:
#   def __init__( self, size ):
#      return 
#   def SetValue()
#   def GetValue()
#   def 

#class Gctrak_t(Struct):
class Geant4Struct(Struct):
   #
   # Define the format string in "_format" according to your needs.
   # For example: 
   # _format = '7f f f I 10I f I'  
   # which means: 
   #        7 floats, 3 more floats, 1 int, 10 ints, 1 float, 1 int
   #

   _format = f'7f f f 7f i {MAXMEC}i {MAXMEC}i i f f f f f f f f f f f'  

   #
   def __init__(self):
       super().__init__(self._format)
       self.vect      = float32( 7 )     # Float_t [7]             7f
       self.getot     = float32( 1 )     # Float_t [1]              f            
       self.gekin     = float32( 1 )     # Float_t [1]              f            
       self.vout      = float32( 7 )     # Float_t [7]             7f
       self.nmec      = int32( 1 )       # Int_t   [1]              i            
       self.lmec      = int32( MAXMEC )  # Int_t   [MAXMEC] {MAXMEC}i
       self.namec     = int32( MAXMEC )  # Int_t   [MAXMEC] {MAXMEC}i
       self.nstep     = int32( 1 )       # Int_t   [1]              i            
       self.pid       = float32( 1 )     # Int_t   [1]              i              
       self.destep    = float32( 1 )     # Float_t [1]              f
       self.destel    = float32( 1 )     # Float_t [1]              f
       self.safety    = float32( 1 )     # Float_t [1]              f
       self.sleng     = float32( 1 )     # Float_t [1]              f
       self.step      = float32( 1 )     # Float_t [1]              f
       self.snext     = float32( 1 )     # Float_t [1]              f
       self.sfield    = float32( 1 )     # Float_t [1]              f
       self.tofg      = float32( 1 )     # Float_t [1]              f
       self.gekrat    = float32( 1 )     # Float_t [1]              f
       self.upwght    = float32( 1 )     # Float_t [1]              f
   #
   # TODO: Automatize this process. Given a string literal with 
   #       a structure information, create the desired structure.
   #

   # So, root can handle data properly.
   #
   #def pack(self):
   #   return struct.pack(self._format,
   #                      *self.vect, self.getot, self.gekin, *self.vout,
   #                      self.nmec, *self.lmec, *self.namec, self.nstep,
   #                      self.pid, self.destep, self.destel, self.safety,
   #                      self.sleng, self.step, self.snext, self.sfield,
   #                      self.tofg, self.gekrat, self.upwght)

   #def unpack(self, buffer):
   #   unpacked_data = struct.unpack(self._format, buffer)
   #   self.vect = np.array(unpacked_data[0:7], dtype=np.float32)
   #   self.getot = np.float32(unpacked_data[7])
   #   self.gekin = np.float32(unpacked_data[8])
   #   self.vout = np.array(unpacked_data[9:16], dtype=np.float32)
   #   self.nmec = np.int32(unpacked_data[16])
   #   self.lmec = np.array(unpacked_data[17:17+MAXMEC], dtype=np.int32)
   #   self.namec = np.array(unpacked_data[17+MAXMEC:17+2*MAXMEC], dtype=np.int32)
   #   self.nstep = np.int32(unpacked_data[17+2*MAXMEC])
   #   self.pid = np.int32(unpacked_data[18+2*MAXMEC])
   #   self.destep = np.float32(unpacked_data[19+2*MAXMEC])
   #   self.destel = np.float32(unpacked_data[20+2*MAXMEC])
   #   self.safety = np.float32(unpacked_data[21+2*MAXMEC])
   #   self.sleng = np.float32(unpacked_data[22+2*MAXMEC])
   #   self.step = np.float32(unpacked_data[23+2*MAXMEC])
   #   self.snext = np.float32(unpacked_data[24+2*MAXMEC])
   #   self.sfield = np.float32(unpacked_data[25+2*MAXMEC])
   #   self.tofg = np.float32(unpacked_data[26+2*MAXMEC])
   #   self.gekrat = np.float32(unpacked_data[27+2*MAXMEC])
   #   self.upwght = np.float32(unpacked_data[28+2*MAXMEC])



# Work in progress.
#
#Gctrak_t = Geant4Struct()
# Gctrak_t = Geant4Struct


#
# Instead :
#           Using C-struct.
#
ProcessLine(f'''

   typedef struct {{
   Float_t vect[7];
   Float_t getot;
   Float_t gekin;
   Float_t vout[7];
   Int_t   nmec;
   Int_t   lmec[{ MAXMEC }];
   Int_t   namec[{ MAXMEC }];
   Int_t   nstep;
   Int_t   pid;
   Float_t destep;
   Float_t destel;
   Float_t safety;
   Float_t sleng;
   Float_t step;
   Float_t snext;
   Float_t sfield;
   Float_t tofg;
   Float_t gekrat;
   Float_t upwght;
}} Gctrak_t;

''')
Gctrak_t = ROOT.Gctrak_t




# void
def helixStep(step : Float_t, vect : list[Float_t], vout : list[Float_t]) :

   """
   This function extrapolates track in a constant field.
   And outputs a vector with new particle positions.

   step  ... smoothness in the track 
   vecto ... previous particle positons
   vout  ... particle position after 

   """

   # Index assignation for k__space_time_and_momentum_values.
   # Note: Little script in case indexes become huge.
   # TODO: Improve : 
   #          Enum class : auto() unnecesary, it could be done
   #          by assigning items inside the class, and then calling a
   #          method `.organize`.
   #          locales.update() could with a method too `update_locals_with_items`. 
   #
   # enum
   class Evect( Enum ):
      kX   = auto ( 0 ) # 1 is the default of auto(). We need 0.
      kY   = auto (   )
      kZ   = auto (   )
      kPX  = auto (   )
      kPY  = auto (   )
      kPZ  = auto (   )
      kPP  = auto (   )
   # 
   Evect_enum_dict = { item.name : item.value for item in Evect }
   #
   #locals().update( Evect_enum_dict ) # Error: doesn't crete variables.
   globals().update( Evect_enum_dict ) # TakeCare. But better than `execute()`.


   # Note: Important.
   #       Here it is the right place to talk about types.
   #       In what follows, the right way to initialize whatever physical
   #       variable is using numpy-types if you want to integrate them with 
   #       TTree class of ROOT.
   #       Python and numpy types are different, even though they are found
   #       in the same environment: Python. Why? 
   #       Answer.- Shortly, numpy is based mostly on LAPACK, Boost, BLAS. 
   #                The well renowned C++ libraries for numerical computing.
   #                All of which use, again mostly, C-types and C-structures
   #                based on 32 or 64 bits. Hence the name float32, int32, ... . 
   #                Python2 and 3 are written entirely in C. 
   #                `float` corresponds to the C-double type, and `int` does not
   #                correspond to a directly to a C-type. Its details are
   #                because Python handles memory differently, and int are 
   #                managed dynamically(run-time). Lower values correspond to int32,
   #                higher values to int64 and huge numbers to longlong...
   #                Which, by the way, are not equivalent to numpy.float32 
   #                or numpy.int32.
   #                There are currently works dealing with this issue. Something like
   #                dynamyc handling types transformation between languages.
   #                It has emerged a recently interest in computer science about this.
   #                thanks to the not-so-recent-but-still-hot-topic theme of bindings
   #                between languages, specially C++ and Python; two strong suits
   #                whatever good scientist has to know: Speed and Mutability, 
   #                respectively. 
   #                
   #       It could be fantastic if we find a way to automatize the use 
   #       of Python numbers as numpy types, seamlessly. TODO.
   #       
   #       Meanwhile, we have to use np.float32 and np.int32 throughly at
   #       initializing numbers. No need in operations.
   #       


   # 
   # Magnetic field in kilogauss units.
   field =  20  # Float_t

   # Momentum conservation.
   vout[kPP]     = vect[kPP]

   #
   h4     =  field * 2.99792e-4  # Float_t                               
   rho    =  -h4 / vect[kPP]     # Float_t                               
   tet    =  rho * step          # Float_t                               
   tsint  =  tet * tet / 6       # Float_t                               
   sintt  =  1 - tsint           # Float_t                               
   sint   =  tet * sintt         # Float_t                               
   cos1t  =  tet / 2             # Float_t                               

   #
   f1 =  step * sintt                # Float_t                               
   f2 =  step * cos1t                # Float_t                               
   f3 =  step * tsint * vect[kPZ]    # Float_t                               
   f4 =  -tet * cos1t                # Float_t                               
   f5 =  sint                        # Float_t                               
   f6 =  tet * cos1t * vect[kPZ]     # Float_t                               

   #
   vout[kX]   =   vect[kX] + (f1 * vect[kPX] - f2 * vect[kPY])    
   vout[kY]   =   vect[kY] + (f1 * vect[kPY] + f2 * vect[kPX])    
   vout[kZ]   =   vect[kZ] + (f1 * vect[kPZ] + f3)                
   vout[kPX]  =   vect[kPX] + (f4 * vect[kPX] - f5 * vect[kPY])   
   vout[kPY]  =   vect[kPY] + (f4 * vect[kPY] + f5 * vect[kPX])   
   vout[kPZ]  =   vect[kPZ] + (f4 * vect[kPZ] + f6)               

   #
   return vout

   

# void
def tree2w() :

   """
   This function creates tree2.root from a Tree file.
   
   """

   #
   # 1. 
   # Create a file. And create a Tree with few branches with
   # a subset of `gctrak`.
   #
   global f, t2
   f = TFile("tree2.root", "recreate")
   # f = TFile("tree2.root", "update") # In case you need it.
   #
   t2 = TTree("t2", "A Tree with data from a fake Geant4.")


   #
   # Note: 
   #     There is a problem when TBufferFile::WriteFastArray.
   #     The Pythonization of TTree sets up the max buffer value
   #     upto 1GB for speeding-up processes in Python.
   #
   # Increase buffer by 2GB
   #t2.IncrementTotalBuffers(2000000000)  
   #t2.IncrementTotalBuffers(0)  
   #
   # Set to 10 GB
   # t2.SetMaxVirtualSize(10 * 1024 * 1024 * 1024)  
   # #
   # # Set the maximum virtual size (in bytes)
   # t2.SetMaxTreeSize(10000000000)  # Set max tree size to 10GB
   #
   # Auto-flush every 100,000 entries
   #t2.SetAutoFlush(100000)
   #
   # Note: 
   #       In the .C version, SetMaxVirtualSize, IncrementTotalBuffers,
   #       SetMaxTreeSize are unnecesary.



   #
   global gstep
   gstep = Gctrak_t()
   #
   # Using the C-struct, 
   # create branches for each member of the structure.
   t2.Branch ( "getot"  , addressof ( gstep , "getot"  ) , "getot/F"           )
   t2.Branch ( "gekin"  , addressof ( gstep , "gekin"  ) , "gekin/F"           )
   t2.Branch ( "nmec"   , addressof ( gstep , "nmec"   ) , "nmec/I"            )
   t2.Branch ( "destep" , addressof ( gstep , "destep" ) , "destep/F"          )
   t2.Branch ( "pid"    , addressof ( gstep , "pid"    ) , "pid/I"             )
   t2.Branch ( "destep" , addressof ( gstep , "destep" ) , "destep/F"          )
   t2.Branch ( "destel" , addressof ( gstep , "destel" ) , "destel/F"          )
   t2.Branch ( "safety" , addressof ( gstep , "safety" ) , "safety/F"          )
   t2.Branch ( "sleng"  , addressof ( gstep , "sleng"  ) , "sleng/F"           )
   t2.Branch ( "step"   , addressof ( gstep , "step"   ) , "step/F"            )
   t2.Branch ( "snext"  , addressof ( gstep , "snext"  ) , "snext/F"           )
   t2.Branch ( "sfield" , addressof ( gstep , "sfield" ) , "sfield/F"          )
   t2.Branch ( "tofg"   , addressof ( gstep , "tofg"   ) , "tofg/F"            )
   t2.Branch ( "gekrat" , addressof ( gstep , "gekrat" ) , "gekrat/F"          )
   t2.Branch ( "upwght" , addressof ( gstep , "upwght" ) , "upwght/F"          )
   #
   t2.Branch ( "vect"   , gstep.vect                     , "vect[7]/F"         )
   t2.Branch ( "lmec"   , gstep.lmec                     , f"lmec[{MAXMEC}]/I" )
   # 

   # Note :
   #        For simple types inherited from C-struct, 
   #        we have to pass its complete address like:
   #        `addressof( c_struct , "its_member" )`
   #        However, for C-array types, passing its reference
   #        is enough like:
   #        ` c_struct.its_array_member `

      
   # Note :
   # Not to use: 
   # `t2.Branch( "getot", gstep.getot ) `  
   # Because:
   #             `.Branch` method with two arguments
   #              ( a.k.a. template method for inference of type)
   #              is not implemented in python
   #              like in ROOT version.
   

   #
   # 2.
   # Initialize particle physical parameters at initial point.
   #
   px = py = pz = p = charge = 0.  # Float_t
   mass        = 0.137             # Float_t
   newParticle = True              # Bool_t
   #
   # And initialize particle simulation parameters.
   gstep.step          =  0.1 
   gstep.destep        =  0.0
   gstep.nmec          =  0 
   gstep.pid           =  0 
   #
   # Initialize the vector output.
   global vout
   vout = np.zeros( 7 , dtype=np.float32 )
   #
   # for the steps
   # of the helix simulation.
   

   #
   # 3.
   # Transport particles.
   #
   #for (Int_t i = 0; i < 10000; i++) {
   for i in range(0, 10000, 1):
      #
      # Generate a new particle if necessary.
      if (newParticle)  :
         #
         px     = gRandom.Gaus(0, .02)
         py     = gRandom.Gaus(0, .02)
         pz     = gRandom.Gaus(0, .02)
         p      = TMath.Sqrt(px * px + py * py + pz * pz)
         charge = 1
         #
         if (gRandom.Rndm() < 0.5)  :
            charge = -1
         #
         #
         gstep.pid      +=  1            
         #
         gstep.vect[0]   =  0
         gstep.vect[1]   =  0
         gstep.vect[2]   =  0
         gstep.vect[3]   =  px / p
         gstep.vect[4]   =  py / p
         gstep.vect[5]   =  pz / p
         gstep.vect[6]   =  p * charge
         #
         gstep.getot     =  TMath.Sqrt(p * p + mass * mass)
         gstep.gekin     =  gstep.getot - mass
         #
         #
         newParticle     =  False
         
      
      #
      # 3.1.
      # Fill the Tree with current step parameters.
      #
      t2.Fill()
      #
      # Status report:
      if i % 1000 == 0:
         print( "Filled entry number: ", i )
      
      #
      # 3.2.
      # Transport particle in magnetic field.
      # One step at a time.
      vout = helixStep(gstep.step, gstep.vect, vout) 
      
      #
      # 3.3.
      # Apply energy loss. Due to the transport.
      #
      gstep.destep   =   gstep.step * gRandom.Gaus(0.0002, 0.00001)  
      gstep.gekin   -=   gstep.destep                                
      gstep.getot    =   gstep.gekin + mass                          
      #
      gstep.vect[0]  =   vout[0]  
      gstep.vect[1]  =   vout[1]  
      gstep.vect[2]  =   vout[2]  
      gstep.vect[3]  =   vout[3]  
      gstep.vect[4]  =   vout[4]  
      gstep.vect[5]  =   vout[5]  
      gstep.vect[6]  =   charge * TMath.Sqrt(gstep.getot * gstep.getot - mass * mass)  
      #
      #
      gstep.nmec     =   Int_t( 5 * gRandom.Rndm() ) 
      #
      #for (Int_t l = 0; l < gstep.nmec; l++)  :
      for l in range(0, gstep.nmec, 1 ) : 
         gstep.lmec[l] =  l  
      #
      if (gstep.gekin < 0.001)  :
         newParticle = True
      #
      if (TMath.Abs(gstep.vect[2]) > 30)  :
         newParticle = True
         
      
   
   #
   # 4.
   # Save the Tree header. 
   #
   t2.Write()
   #
   # The file will NOT be automatically closed
   # when going out of the function scope.
   # So, close it.
   f.Close()
   


# void
def tree2r() :

   """
   This functions reads the Tree generated by `tree2w` and fills
   the only histrogram we are interested on inside the destep branch.
   
   Important Note:
   We use `global` at creating the TFile and TTree objects
   because we want to keep these objects alive when we leave
   this function.

   """

   #
   global f, t2
   f = TFile("tree2.root")  # TFile
   t2 = f.Get("t2")  # (TTree *)


   #
   global destep
   destep = np.zeros( 1, "f" ) # np.float32() # static Float_t
   #
   global b_destep
   b_destep = t2.GetBranch("destep") # TBranch *
   b_destep.SetAddress(destep)
   
   #
   # Create one histogram. Let's say... for the entries of the branch '.destep'.
   global hdestep
   hdestep = TH1F("hdestep", "destep in Mev", 100, 1e-5, 3e-5)  # TH1F
   
   #
   # Read only the "destep" branch for all entries.
   #
   global nentries
   nentries = t2.GetEntries()  # Long64_t
   #
   #for (Long64_t i = 0  i < nentries  i++) {
   for i in range(0, nentries, 1):
      b_destep.GetEntry(i)
      hdestep.Fill(destep)
      
   
   #
   # Fill a 3-d scatter plot with the particle step coordinates.
   #
   global c1
   c1 = TCanvas("c1", "c1", 600, 800)  # TCanvas
   #
   c1.SetFillColor(42)
   c1.Divide(1, 2)

   #
   c1.cd(1)
   #
   # Style.
   hdestep.SetFillColor(45)
   #
   # Calculus.
   hdestep.Fit("gaus")
   #
   # Save.
   # hdestep.Write() # If you want to save your plot and results in the root file.
   #
   # Plot.
   hdestep.Draw("")
   c1.Update()

   #
   c1.cd(2)
   #
   # Style.
   gPad.SetFillColor(37)
   t2.SetMarkerColor(kRed)
   #
   # Plot.
   t2.Draw("vect[0]:vect[1]:vect[2]")
   c1.Update()
   

   # Note:
   #      We did not close the file, because 
   #      we wanted to keep the generated histograms
   #      for a stylized canvas.
   # So, don't do:
   # f.Close()
   # 
   # Unless you have written it before, by doing:
   # hdestep.Write()
   # If that is the case, you can close the file:
   # f.Close() # If needed.

   #
   # Allow to use the TTree after the function's end.
   t2.ResetBranchAddresses()
   #
   # Explore the root file with.
   # new_browser = TBrowser()


   
   

# void
def tree2() :
   #
   # Write.
   tree2w()
   #
   # Read.
   tree2r()
   


if __name__ == "__main__":
   tree2()
