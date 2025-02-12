## \file
## \ingroup tutorial_tree
## \notebook
##
## Write and read STL vectors in a tree.
##
## \macro_image
## \macro_code
##
## \author The ROOT Team
## \translator P. P.


import ROOT
import ctypes
from array import array


# standard library
from ROOT import std
from ROOT.std import (
                       make_shared,
                       unique_ptr,
                       vector,
                       )

# classes
from ROOT import (
                   TCanvas,
                   TH1F,
                   TGraph,
                   TLatex,
                   TBranch,
                   TBenchmark,
                   TCanvas,
                   TFile,
                   TFrame,
                   TRandom,
                   TSystem,
                   TTree,
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

#globals
from ROOT import (
                  gSystem,
                  gStyle,
                  gPad,
                  gRandom,
                  gBenchmark,
                  gROOT,

                  )





# void
def write() :
   
   #
   global f
   f = TFile.Open("hvector.root", "RECREATE") # TFile *
   
   if ( not f ) :
      return
     
   #
   # Create one histogram of 
   # a particle momentum in x axis $p_x$. 
   global hpx
   hpx = TH1F("hpx", "This is the px distribution", 100, -4, 4)  # TH1F
   hpx.SetFillColor(48)

   #
   # Collection of momentums.
   global vpx, vpy, vpz, vrand 
   vpx    =  std.vector[ "Float_t" ]( )
   vpy    =  std.vector[ "Float_t" ]( )
   vpz    =  std.vector[ "Float_t" ]( )
   vrand  =  std.vector[ "Float_t" ]( )


   #
   # Create a TTree.
   global t
   t = TTree("tvec", "Tree with vectors")  # TTree
   #
   # Set up the Tree.
   t.Branch("vpx"   , vpx   )
   t.Branch("vpy"   , vpy   )
   t.Branch("vpz"   , vpz   )
   t.Branch("vrand" , vrand )
   
   #
   # Create a new canvas.
   global c1
   c1 = TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500)  # TCanvas
   
   # 
   gRandom.SetSeed(   )
   #
   # Status report variable. Frame rate.
   #kUPDATE = 1000 # Int_t
   kUPDATE = 100 # Int_t
   #
   #for (Int_t i = 0; i < 25000; i++) {
   #for i in range(0, 25000, 1): # error with Rannor. error  with Gaus.
   for i in range(0, 2500, 1): # error with Rannor.   ok with Gaus.
   # for i in range(0, 150, 1):  # ok with Rannor.    ok with Gaus.
   # for i in range(0, 1500, 1): # ok with Rannor.    ok with Gaus.
      #
      # Random number particles.
      npx = Int_t( gRandom.Rndm(1) * 15 )  # Int_t
      # 
      vpx.clear()
      vpy.clear()
      vpz.clear()
      vrand.clear()
      
      # for (Int_t j = 0; j < npx; ++j) {
      for j in range(0, npx, 1):
         #
         # Not to use :
         #               `.Rannor` generates memory leaks
         #               for intensive loops 
         #               of order ~10000.
         # Generate px, py, pz.
         #px = c_float()
         #py = c_float()
         #gRandom.Rannor(px, py)
         #px = px.value
         #py = py.value

         #
         # Generate px, py, pz.
         px = gRandom.Gaus( 0, 1 )
         py = gRandom.Gaus( 0, 1 )
         #
         pz = px * px + py * py
         # Fill histogram and std vectors.
         hpx.Fill( px )
         # 
         # Inherited from .C ...
         vpx.emplace_back( px )
         vpy.emplace_back( py )
         vpz.emplace_back( pz )
         # ... and the Pythonic way.
         # vpx  +=  [ px ]
         # vpy  +=  [ py ]
         # vpz  +=  [ pz ]
         #
         # Python takes (1) operation : []
         #              (2) operation : +=
         # std.vector takes (1) operation : .emplace_back( )
         # Your choice.
         

         #
         # Additional random number.
         random = gRandom.Rndm(1)  # Float_t
         vrand.emplace_back( random )
         # vrand  += [ random ]

         
      #
      # Frame rendering. Animation.  
      if (i and (i % kUPDATE) == 0)  :
         #
         if (i == kUPDATE)  :
            hpx.Draw()
         #   
         c1.Modified()
         c1.Update()
         if (gSystem.ProcessEvents())  :
            break
            
      #       
      t.Fill()
   #   
   f.Write()
   f.Close()
   
   #
   gROOT.Remove( f )
   f.Clear() 
   f.Delete()
   del f

   #
   gROOT.Remove( c1 )
   c1.Clear()
   del c1
   

# void
def read() :
   
   global f
   f = TFile.Open("hvector.root", "READ") # TFile *
   
   if ( not f ) :
      return
      
   
   global t
   t = TTree()
   f.GetObject["TTree"]("tvec", t)
   # Or:
   # t = f.Get("tvec")
   #
   # Note:
   #       `.GetObject` is a template method.
   #       In its .C version, the argument of 
   #       this template is infered automatically.
   #       Here in .py, the inference is not done.
   #       We have to declare the template argument
   #       manually:
   #       my_file = TFile( ... )
   #       my_tree = TTree()
   #       my_file.GetObject["TTree"]( "the_tree", my_tree )
   #     
   #       Alternatively, if the type isn't a complex structure
   #       or even mixing-ups of std::vector<TH1, TF1>
   #       You could jsutg simply use:
   #       my_tree = my_file.Get("the_tree")
   #       
   
   #
   global vpx
   vpx = std.vector[ "float" ]( ) # nullptr # *
   
   #
   # Create a new canvas.
   global c1
   c1 = TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500)  # TCanvas
   
   # Frame rate.
   #kUPDATE = 1000 # Int_t
   kUPDATE = 100 # Int_t

   
   #
   global bvpx
   bvpx = TBranch() # nullptr # TBranch *
   t.SetBranchAddress("vpx", vpx, bvpx)
   
   
   #
   # Create one histograms
   global h
   h = TH1F("h", "This is the px distribution", 100, -4, 4)  # TH1F
   h.SetFillColor(48)
   
   #
   #for (Int_t i = 0; i < 25000; i++) {
   #for i in range(0, 25000, 1): # error
   for i in range(0, 2500, 1): # ok
   #for i in range(0, 50, 1): # ok
   #for i in range(0, 1, 1): # ok
   #for i in range(0, 2, 1): # ok
   #for i in range(0, 10, 1): # error
      #
      vpx.clear()
      tentry = t.LoadTree( i )  # Long64_t
      bvpx.GetEntry( tentry )
      # 
      #for (UInt_t j = 0; j < vpx->size(); ++j)  :
      for j in range( 0, vpx.size(), 1 ):
         # 
         h.Fill( vpx.at( j ) )
         
      #
      # Frame Rendering. Animation. 
      if (i and (i % kUPDATE) == 0)  :
         #
         if (i == kUPDATE)  :
            h.Draw()
            c1.Update()
            
         #
         c1.Modified()
         c1.Update()
         # But NOT draw.
         #
         if (gSystem.ProcessEvents())  :
            break
            
         
      
   
   #
   # Since we passed the address of a local variable, 
   # we need to remove it.
   # 
   t.ResetBranchAddresses()
   

# void
def hvector() :
   #
   gBenchmark.Start("hvector")
   
   #
   write()
   read()
   
   #
   gBenchmark.Show("hvector")
   


if __name__ == "__main__":
   hvector()
   # TODO:
   # Error at exit.
   # Bug likely caused by a .C extension.
