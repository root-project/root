## \file
## \ingroup tutorial_spectrum
## \notebook
##
## Examples showing how to use TSpectrum2Painter (the SPEC option)
##
## \macro_image
## \macro_code
##
## \author: Olivier Couet, Miroslav Morhac
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
                   TRandom3,
                   TH2F,
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

# ctypes
from ctypes import (
                     c_double,
                     c_float,
                     )

#utils
def to_c( ls ):
   return ( c_double * len( ls ) )( * ls )

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



# void
def spectrumpainter() :

   
   global h2
   h2 = TH2F( "h2", "h2", 40, -8, 8, 40, -9, 9 )

   # #
   # c_px = c_float() # Float_t
   # c_py = c_float() # Float_t
   # #
   # #for (Int_t i = 0; i < 50000; i++) {
   # # Error:
   # #for i in range(0, 50000, 1):
   # # Ok:
   # for i in range(0, 9000, 1):
   #    # from Gaussian normal distribution: mean = 0 , std dev = 1
   #    gRandom.Rannor(c_px, c_py)
   #    #
   #    px = c_px.value 
   #    py = c_py.value 
   #    #
   #    h2.Fill(px, py)
   #    h2.Fill(px + 4, py - 4, 0.5)
   #    h2.Fill(px + 4, py + 4, 0.25)
   #    h2.Fill(px - 3, py + 4, 0.125)
   # Note: 
   #      Due to JIT problems: iterations around O(10^4) generates memory leaks
   #                           but no memory leaks under iterations below O( 10^3);
   #      We will use another randon generator instead ... 
   #      Could be "TRandom3". And it goes like ...
    
   global random_gen_3
   random_gen_3 = TRandom3() 
   
   for i in range(0, 50000, 1):
      
      px = random_gen_3.Gaus( 0, 1 )
      py = random_gen_3.Gaus( 0, 1 )
      
      h2.Fill ( px     , py             )
      h2.Fill ( px + 4 , py - 4 , 0.5   )
      h2.Fill ( px + 4 , py + 4 , 0.25  )
      h2.Fill ( px - 3 , py + 4 , 0.125 )

   # Note:
   #       Since the purpose of this tutorial is to show how to use "the painter of spectrums"
   #       doesn't matter which generator we use. However, in numerical studies, consider
   #       carefully which one do you choose, some are better for performance, some are better
   #       for randomness : TRandom3 and Rannor have their slight differences.
      
   
   
   global c1
   c1 = TCanvas("c1", "Illustration of 2D graphics", 10, 10, 800, 700)
   c1.Divide(2, 2)
   

   c1.cd(1)
   h2.Draw("SPEC dm(2,10) zs(1)")

   c1.cd(2)
   h2.Draw("SPEC dm(1,10) pa(2,1,1) ci(1,4,8) a(15,45,0)")

   c1.cd(3)
   h2.Draw("SPEC dm(2,10) pa(1,1,1) ci(1,1,1) a(15,45,0) s(1,1)")

   c1.cd(4)
   h2.Draw("SPEC dm(1,10) pa(2,1,1) ci(1,4,8) a(15,45,0) cm(1,15,4,4,1) cg(1,2)")
   


if __name__ == "__main__":
   spectrumpainter()
