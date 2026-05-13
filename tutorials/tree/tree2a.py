## \file
## \ingroup tutorial_tree
## \notebook
##
##
## This example is similar to tree2.py, but uses a class instead of a C-struct.
##
## Here, we are "mapping" a class to one of the Geant4
## common blocks /gctrak/. In particle physics simulations, this block will be filled
## by Geant4 in a loop and only the `TTree.Fill` method should be called at.
##
## To run the example, do:
##
## ~~~
## IPython [0]: tree2a.py
## ~~~
##
## Note: Since Input/Output (I/O) is involved, 
##       ACLiC has to be invoked to create 
##       a dictionary for the class Gctrak.
##
##
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import numpy as np

import ctypes
from array import array

import ROOT



# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       )

# classes
from ROOT import (
                   TBrowser,
                   TCanvas,
                   TFile,
                   TH2,
                   TMath,
                   TROOT,
                   TRandom,
                   TTree,
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
                   TObject,
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
from ctypes import (
                     c_double,
                     c_float,
                     c_int,
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



# seamleasly integration
from ROOT.gROOT import ProcessLine


# Note :
#        Please, update this script when the issue of 
#        "Inheriting a ROOT class in a Python class" 
#        is solved. This:
#        `from ROOT import TObject` 
#        `class C( TObject ): pass `
#        shouldn't generate error.
#
#        Ref.- : https://github.com/root-project/root/issues/16520
#
#        The update should be done by removing the intermediate
#        layer that puts the `.__getattr__` method.
#        By removing it, the performance will increased in terms
#        of one operation.
#        It is not dramatic in this particular case, where we
#        need TObject only for the Tree and its Branches 
#        generation.

#
#        Also, the tricky way we inherit here TObject in a Python 
#        class has limitations. One was already mention, the 
#        intermediate layer, and the other is initialization 
#        process. The new child class, inherited from TObject,
#        has a different kind of initialization, and a different
#        managament of setting-up its members.
#        For more sophisticated/complex ROOT classes, 
#        a warning must be done before using this tricky 
#        way of inheritance.
#


# variables
MAXMEC = 30

ProcessLine(f'''
//const Int_t MAXMEC = {MAXMEC};

class Gctrak : public TObject {{
 public:
   Float_t vect[7];
   Float_t getot;
   Float_t gekin;
   Float_t vout[7]; //! not persistent
   Int_t   nmec;
   Int_t * lmec;  //[nmec]
   Int_t * namec; //[nmec]
   Int_t   nstep; //! not persistent
   Int_t   pid;
   Float_t destep;
   Float_t destel; //! not persistent
   Float_t safety; //! not persistent
   Float_t sleng;  //! not persistent
   Float_t step;   //! not persistent
   Float_t snext;  //! not persistent
   Float_t sfield; //! not persistent
   Float_t tofg;   //! not persistent
   Float_t gekrat; //! not persistent
   Float_t upwght; //! not persistent

   Gctrak() {{
      lmec  = nullptr;
      namec = nullptr;
   }}

   //ClassDefOverride(Gctrak, 1)

   // The following methods are useful
   // for cppyy.LowLeveView.
   //
   //Int_t * namec;   //[nmec]
   //Int_t * lmec;    //[nmec]
   //Float_t vout[7]; //
   //Float_t vect[7];

   //  void Set_namec( Int_t * namec_input){{
   //     this->namec = namec_input;
   //  }}

   //  void Set_lmec( Int_t * lmec_input){{
   //     this->lmec = lmec_input;
   //  }}

   //  void Set_vout( const Float_t vout_input [7] ){{
   //     for (int i = 0; i < 7; ++i) {{
   //      vout[i] = vout_input[i];
   //     }}
   //  }}

   //  void Set_vect( const Float_t vect_input [7] ){{
   //     for (int i = 0; i < 7; ++i) {{
   //      vect[i] = vect_input[i];
   //     }}
   //  }}

   
}};

''')
#
Gctrak = ROOT.Gctrak



# void
def helixStep(step : Float_t, vect : np.array, vout : np.array) :

   #
   # Extrapolate the track in a constant magnetic field.
   #
   # Magnetic field in kilogauss.
   field = 20  # Float_t

   #
   #enum Evect { kX, kY, kZ, kPX, kPY, kPZ, kPP ]
   #
   kX   =  0
   kY   =  1
   kZ   =  2
   kPX  =  3
   kPY  =  4
   kPZ  =  5
   kPP  =  6
  


   #
   vout[kPP]     = vect[kPP]

   #
   # Constants.               
   h4     = field * 2.99792e-4    # Float_t
   rho    = -h4 / vect[kPP]       # Float_t
   tet    = rho * step            # Float_t
   tsint  = tet * tet / 6         # Float_t
   sintt  = 1 - tsint             # Float_t
   sint   = tet * sintt           # Float_t
   cos1t  = tet / 2               # Float_t

   #
   f1 = step * sintt              # Float_t
   f2 = step * cos1t              # Float_t
   f3 = step * tsint * vect[kPZ]  # Float_t
   f4 = -tet * cos1t              # Float_t
   f5 = sint                      # Float_t
   f6 = tet * cos1t * vect[kPZ]   # Float_t

   #
   vout[kX]      = vect[kX]  + ( f1 * vect[kPX] - f2 * vect[kPY] )
   vout[kY]      = vect[kY]  + ( f1 * vect[kPY] + f2 * vect[kPX] )
   vout[kZ]      = vect[kZ]  + ( f1 * vect[kPZ] + f3             )
   vout[kPX]     = vect[kPX] + ( f4 * vect[kPX] - f5 * vect[kPY] )
   vout[kPY]     = vect[kPY] + ( f4 * vect[kPY] + f5 * vect[kPX] )
   vout[kPZ]     = vect[kPZ] + ( f4 * vect[kPZ] + f6             )


   return vout
   

# void
def tree2aw() :

   #
   # Create a tree2.root file with a Tree.
   
   # Create the file, the Tree and a few branches with
   # a subset of Geant4-block-gctrak.
   #
   global f, t2, gstep
   f = TFile("tree2.root", "recreate")
   # 
   t2 = TTree("t2", "a Tree with data from a fake Geant3")
   #
   try:
      globals()["gstep"]
      if gstep : del gstep
   except:
      pass
   gstep = Gctrak()  # Gctrak
   # 
   t2.Branch("track", gstep, 8000, 1)
   

   #
   # Initialize particle parameters at starting point.
   #
   px = py = pz = p = charge = 0. # Float_t 
   #
   mass = 0.137       # Float_t
   #
   # vout = [ Float_t() for _ in range(7) ]
   #vout = np.array( [ Float_t() ] * 7 , "d" )
   vout = np.zeros( 7 , "d" )
   #
   newParticle = True # Bool_t
   #
   # Not to use:
   #gstep.lmec         = np.zeros( MAXMEC , "d" ) # Int_t[MAXMEC]  # new
   #gstep.namec        = np.zeros( MAXMEC , "d" ) # Int_t[MAXMEC]  # new
   #
   # Instead:
   #gstep.Set_lmec ( np.zeros( MAXMEC , "i" ) ) # Int_t[MAXMEC]  # new
   #gstep.Set_namec( np.zeros( MAXMEC , "i" ) ) # Int_t[MAXMEC]  # new
   # 
   # Event better:
   gstep_lmec = [ 0 ] * MAXMEC 
   gstep_namec = [ 0 ] * MAXMEC
   #
   gstep.lmec     = ( c_int * MAXMEC )( * gstep_lmec )   # Int_t[MAXMEC]  # new
   gstep.namec    = ( c_int * MAXMEC )( * gstep_namec )   # Int_t[MAXMEC]  # new
   #
   #gstep.Set_lmec(  ( c_int * MAXMEC )()  )
   #gstep.Set_namec(  ( c_int * MAXMEC )()  )


   #  
   gstep.step         = 0.1
   gstep.destep       = 0
   gstep.nmec         = 0
   #gstep.nmec     = Int_t( 5 * gRandom.Rndm() )
   gstep.pid          = 0
   
   #
   # Transport the particle.
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
         gstep.pid     += 1
         #
         # Error :
         #        gstep.vect[ i ] = float() 
         #        Doesnt' work properly. 
         #        Error on fill the Tree not on assigment.
         # Instead :
         # - - - 
         gstep_vect = [ float() ] * 7
         #
         gstep_vect[0]  = 0
         gstep_vect[1]  = 0
         gstep_vect[2]  = 0
         gstep_vect[3]  = px / p
         gstep_vect[4]  = py / p
         gstep_vect[5]  = pz / p
         gstep_vect[6]  = p * charge
         #
         gstep.vect = ( c_float * 7 )( * gstep_vect )
         #
         # - - - 
         #
         gstep.getot    = TMath.Sqrt(p * p + mass * mass)
         gstep.gekin    = gstep.getot - mass

         #
         newParticle     = False
         
      
      #
      #        Fill the Tree with the current step parameters.
      #
      t2.Fill()
      #
      gROOT.Remove( gstep ) # IMPORTANT for PyROOT. Otherwise will raise a memory issue.
      # Error:
      #        There is no error now.
      #        TODO: Fix the types at handling data from 
      #        python to C. Improve and backwards.
      #
      

      #
      # Transport the particle in the magnetic field
      # by one step. For the next step.
      #
      #vout = helixStep(gstep.step, gstep.vect, vout) # Warning! No gstep.vect
      vout = helixStep(gstep.step, gstep_vect, vout)
      # Note:
      #       gstep_vect was defined previously, 
      #       It is impossible to handle float * c-type
      #       without raising memory issues in ROOT.
      #
      

      #
      # Apply energy loss. For the next step.
      #
      gstep.destep   = gstep.step * gRandom.Gaus(0.0002, 0.00001)
      gstep.gekin   -= gstep.destep
      gstep.getot    = gstep.gekin + mass
      #
      # Not to use:
      #gstep.vect[0]  = vout[0]
      #gstep.vect[1]  = vout[1]
      #gstep.vect[2]  = vout[2]
      #gstep.vect[3]  = vout[3]
      #gstep.vect[4]  = vout[4]
      #gstep.vect[5]  = vout[5]
      #gstep.vect[6]  = charge * TMath.Sqrt(gstep.getot * gstep.getot - mass * mass)
      #
      # Instead:
      # - - -
      gstep_vect     = [ 0. ] * 7
      #
      gstep_vect[0]  = vout[0] # Sotimes here
      gstep_vect[1]  = vout[1] # comes 
      gstep_vect[2]  = vout[2] # more complex 
      gstep_vect[3]  = vout[3] # calculations 
      gstep_vect[4]  = vout[4] # than just a simple
      gstep_vect[5]  = vout[5] # gstep_vect = [ vout[i] for i in range(7) ]
      gstep_vect[6]  = charge * TMath.Sqrt(gstep.getot * gstep.getot - mass * mass)
      #
      gstep.vect =  ( c_float * 7  ) ( * gstep_vect )
      # - - -
      # Note:  
      #        No matter how you calculate or assign the list-python values,
      #        what matters is how we load it into C++.
      #        In other words, this line should persist:
      #        `gstep.vect =  ( c_float * 7  ) ( * gstep_vect )`



      #
      gstep.nmec     = Int_t( 5 * gRandom.Rndm() )
      #
      #for (Int_t l = 0; l < gstep->nmec; l++)  :
      for l in range( 0, gstep.nmec, 1 ): 
         gstep.lmec = c_int(  l )
         gstep.namec = c_int( l + 100 )
         pass
      #
      # Not to use: 
      #             gstep.lmec = c_int(  l )
      #             gstep.namec = c_int( l + 100 )
      #
      # Instead: Unnecessary now, because the gstep.lmec accepts c_int
      #          And the TTree.Fill fixes the size of gstep.nmec.
      #          Keep in mind for next edition.
      # - - - 
      #gstep_lmec  = [ l        for l in range( 0, gstep.nmec, 1 ) ]
      #gstep_namec = [ l + 100  for l in range( 0, gstep.nmec, 1 ) ]
      ##
      #gstep.lmec = ( c_int * gstep.nmec )( * gstep_lmec ) 
      #gstep.namec = ( c_int * gstep.nmec )( * gstep_namec ) 
      # - - - 
         

      #
      if (gstep.gekin < 0.001)  :
         newParticle = True

      #   
      if (TMath.Abs(gstep_vect[2]) > 30)  : # Warning! Use gstep_vect no gstep.vect
         newParticle = True
      
   
   # Save the Tree header.
   t2.Write()

   # The file will NOT be automatically closed
   # when going out of the function scope.
   # So, let's close it. Shall we?
   #
   # Error:
   f.Close()
   


# void
def tree2ar() :

   #
   # Read the Tree generated by "tree2w" function 
   # and fill one histogram.
   

   # 
   global f, t2
   f  = TFile("tree2.root") # TFile
   t2 = f.Get("t2")         # (TTree *)
   #
   # Note: 
   #      We use "global" to create the TFile and TTree objects
   #      because we want to keep these objects alive when we leave
   #      this function.
   

   #
   # We are only interested in the "destep" branch.
   #
   global b_destep
   b_destep = t2.GetBranch("destep") # TBranch *


   #
   global gstep
   try:
      globals()["gstep"]
      if gstep : del gstep
   except:
      pass
   gstep    = Gctrak() # nullptr # Gctrak * # No need to create a new one
   #gROOT.Remove( gstep )
   #
   t2.SetBranchAddress("track", gstep)
   
   #
   # Create one histogram.
   #
   global hdestep
   hdestep = TH1F("hdestep", "destep in Mev", 100, 1e-5, 3e-5)  # TH1F

   
   #
   # Read only the "destep" branch for all entries.
   #
   nentries = t2.GetEntries(); # Long64_t
   #
   #for (Long64_t i = 0; i < nentries; i++) {
   for i in range(0, nentries, 1):
      #
      b_destep.GetEntry( i )
      #
      hdestep.Fill( gstep.destep )
      
   
   #
   # We fill a 3-dimension scatter plot with
   # coordinates of the particles' steps.
   #
   # Note :
   #        We do not close the file because
   #        we want to keep the generated histograms.
   #
   global c1
   c1 = TCanvas("c1", "c1", 600, 800)  # TCanvas
   #
   c1.SetFillColor(42)
   c1.Divide(1, 2)
   c1.cd(1)
   #
   hdestep.SetFillColor(45)
   hdestep.Fit("gaus")
   #
   c1.cd(2)
   gPad.SetFillColor(37)
   t2.SetMarkerColor(kRed)
   #
   t2.Draw("vect[0]:vect[1]:vect[2]")

   #
   c1.Update()
   c1.Draw()


   #
   if (gROOT.IsBatch())  :
      return
      
   
   # Old fashioned way.
   # Invoke the x3d viewer.
   #gPad.GetViewer3D("x3d")
   # 
   # Still, even though it is old, it has a good exciting view.
   # Nevertheless, "gl" has some more computing features.

   # New.
   #
   gPad.GetViewer3D("gl")
   


# void
def tree2a() :
   
   # Write Tree.
   tree2aw()

   # Read Tree.
   tree2ar()
   


if __name__ == "__main__":
   tree2a()
