## \file
## \ingroup tutorial_math
## \notebook
##
## kdTreeBinning tutorial: it bins data in cells of equal content using a kd-tree.
##
## Using the TKDTree-wrapper-class as a data-binning-structure,
## we plot a 2D-data using the TH2Poly class of ROOT.
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Bartolomeu Rabacal
## \translator P. P.


import ROOT
import ctypes

#cmath = ROOT.cmath # Not implemented.
std = ROOT.std

TKDTreeBinning = ROOT.TKDTreeBinning 
TH1F = ROOT.TH1F
TH2D = ROOT.TH2D 
TH2Poly = ROOT.TH2Poly 
TStyle = ROOT.TStyle 
TGraph2D = ROOT.TGraph2D 
TRandom3 = ROOT.TRandom3 
TCanvas = ROOT.TCanvas 
iostream = ROOT.iostream 

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
UInt_t = ROOT.UInt_t
c_double = ctypes.c_double

#math
sqrt = std.sqrt

#utils
def to_c(ls):
   return (c_double * len(ls) )( * ls )

#C++ integration
ProcessLine = ROOT.gInterpreter.ProcessLine


#Loading TKDTreeBinning
ProcessLine("""
   #include "TKDTree.h"

   TKDTreeBinning * Load_TKDTreeBinning (UInt_t dataSize, UInt_t dataDim, const std::vector< double > &data, UInt_t nBins=100, bool adjustBinEdges=false)
   {
      return new TKDTreeBinning(dataSize, dataDim, data, nBins, adjustBinEdges);
   }

""")




# void
def kdTreeBinning() :
   
   # ----------------------------------------------------------
   #  Create random sample with regular binning plotting
   
   
   DATASZ = 10000
   DATADIM = 2
   NBINS = 50
   
   smp = [ Double_t() for _ in range( DATASZ * DATADIM ) ]
   # Works without converting to C-type. But you cannot.
   # Read below, at defining kdBins.
   # However, an error arises if you run the script more than twice.       
   # So, let it be.
   smp = to_c(smp) 
  
   
   mu = [0,2]
   sig = [2,3]
   r = TRandom3()
   r.SetSeed(1)

   #for (UInt_t i = 0; i < DATADIM; ++i)
   #    for (UInt_t j = 0; j < DATASZ; ++j)
   for i in range(0, DATADIM, 1):
      for j in range(0, DATASZ, 1): 
         smp[DATASZ * i + j] = r.Gaus(mu[i], sig[i])

   
   h1bins = UInt_t( sqrt( NBINS ) )
   
   global h1
   h1 = TH2D("h1BinTest", "Regular binning", h1bins, -5., 5., h1bins, -5., 5.)
   #for (UInt_t j = 0; j < DATASZ; ++j)
   for j in range(0, DATASZ, 1):
      h1.Fill(smp[j], smp[DATASZ + j])
   
   
   # ------------------------------------------------------------------------------------
   # Create a KDTreeBinning object with the TH2Poly class's method for plotting.
   global kdBins 
   #DOING
   #BP:
   kdBins = TKDTreeBinning(DATASZ, DATADIM, smp, NBINS)
   print("Not run this script twice.")
   print("TKDTreeBinning doesn't have deallocation implemented in pyroot.")
   #if you un-comment the lines below, an error arises. 
   """
   Fatal Python error: none_dealloc: deallocating None: bug likely caused by a refcount error in a C extension
   Python runtime state: initialized
   """
   print("""
   #del kdBins;
   #kdBins = ROOT.Load_TKDTreeBinning(DATASZ, DATADIM, smp, NBINS);
   """)
   
   
   nbins = kdBins.GetNBins()
   dim = kdBins.GetDim()
   
   binsMinEdges = kdBins.GetBinsMinEdges()
   binsMaxEdges = kdBins.GetBinsMaxEdges()
   
   global h2pol
   h2pol = TH2Poly( "h2PolyBinTest"
                  , "KDTree binning"
                  , kdBins.GetDataMin(0)
                  , kdBins.GetDataMax(0)
                  , kdBins.GetDataMin(1)
                  , kdBins.GetDataMax(1) 
                    )
   
   #for (UInt_t i = 0; i < nbins; ++i) {
   for i in range(0, nbins, 1): 
      edgeDim = i * dim
      h2pol.AddBin( binsMinEdges[edgeDim]
                  , binsMinEdges[edgeDim + 1]
                  , binsMaxEdges[edgeDim]
                  , binsMaxEdges[edgeDim + 1]
                    )
      
   
   #for (i = 1; i <= kdBins.GetNBins(); ++i)
   for i in range(1, kdBins.GetNBins()+1, 1): 
      h2pol.SetBinContent(i, kdBins.GetBinDensity(i - 1))
   
   print("Bin with minimum density: " , kdBins.GetBinMinDensity())
   print("Bin with maximum density: " , kdBins.GetBinMaxDensity())
   
   global c1
   c1 = TCanvas("glc1", "TH2Poly from a kdTree",0,0,600,800)
   c1.Divide(1,3)
   c1.cd(1)
   h1.Draw("lego")
   
   c1.cd(2)
   h2pol.Draw("COLZ L")
   c1.Update()
   
   #-------------------------------------------------
   # Draw another equivalent plot by showing the data points.
   
   
   z = std.vector["Double_t"](DATASZ, 0.)
   #for (i = 0; i < DATASZ; ++i)
   for i in range(0, DATASZ, 1):
      z[i] = h2pol.GetBinContent(h2pol.FindBin(smp[i], smp[DATASZ + i])) # Double_t
   
   #BP: 
   #Not to use: g = TGraph2D(DATASZ, smp, smp[DATASZ], z[0]) 
   #Note:
   #      TGraph2D has the next constructor.
   """
   	TGraph2D (Int_t n, Double_t *x, Double_t *y, Double_t *z)
 	Graph2D constructor with three vectors of doubles as input. 
   Ref:
        https://root.cern/doc/master/TGraph2D_8cxx_source.html#l00301
   """
   #DOING: Fixing...
   c_smp_1 = to_c( smp[ : DATASZ ] )
   c_smp_2 = to_c( smp[ DATASZ : ] )
   c_z   = to_c(z)
   #Remember: 
   #          smp contains DATASZ * DATADIM  elements.
   #          So, the fist DATASZ-elements belongs to an array. 
   #          and the rest to another array. 
   #          In this case, c_smp_1 and c_smp_2 are the arrays.
   # 
   #          If you have an experiment with more than two dimensions.
   #          which is often the case, use the next convention:
   #          c_smp_1 = ...
   #          c_smp_2 = ...
   #          ... = ... 
   #          c_smp_n = ...
   #          or use nest a list in another list [ [...], [...], [...] ].

   global g
   g = TGraph2D(DATASZ, c_smp_1, c_smp_2, c_z) 
   # To avoid potential memory leak.
   g.SetNameTitle("TGraph2D", "g: Graph2D") 
   smp = list(c_smp_1) + list(c_smp_2) # In case of coming-back.

   g.SetMarkerStyle(20)
   
   c1.cd(3)
   g.Draw("pcol")
   c1.Update()
   
   # ---------------------------------------------------------
   # Make a new TH2Poly where bins are ordered by the density.
   
   global h2polrebin 
   h2polrebin = TH2Poly( "h2PolyBinTest"
                       , "KDTree binning"
                       , kdBins.GetDataMin(0)
                       , kdBins.GetDataMax(0)
                       , kdBins.GetDataMin(1)
                       , kdBins.GetDataMax(1)
                         )
   h2polrebin.SetFloat()
   
   #---------------------------------
   # Sort the bins by their density.
   
   kdBins.SortBinsByDensity()
   
   #for (UInt_t i = 0; i < kdBins->GetNBins(); ++i) {
   for i in range(0, kdBins.GetNBins(), 1):
      binMinEdges = kdBins.GetBinMinEdges(i)
      binMaxEdges = kdBins.GetBinMaxEdges(i)
      h2polrebin.AddBin(binMinEdges[0], binMinEdges[1], binMaxEdges[0], binMaxEdges[1])
      
   
   #for (UInt_t i = 1; i <= kdBins->GetNBins(); ++i) {
   for i in range(1, kdBins.GetNBins() + 1, 1):
      h2polrebin.SetBinContent(i, kdBins.GetBinDensity(i - 1))
      
   
   print(f"Bin with minimum density: " , kdBins.GetBinMinDensity())
   print(f"Bin with maximum density: " , kdBins.GetBinMaxDensity())
   
   # Now, make a vector with bin number vs. position.
   #for (i = 0; i < DATASZ; ++i)
   for i in range(0, DATASZ, 1):
      z[i] = h2polrebin.FindBin(smp[i], smp[DATASZ + i]) # Double_t 
   
   #In case of redifing our data.
   c_smp_1 = to_c( smp[ : DATASZ ] )
   c_smp_2 = to_c( smp[ DATASZ : ] )
   c_z   = to_c(z)
   
   #Not to use: g2 = TGraph2D(DATASZ, smp, smp[DATASZ], z[0])
   global g2
   g2 = TGraph2D(DATASZ, c_smp_1, c_smp_2, c_z) 
   # To avoid potential memory leak.
   g2.SetNameTitle("TGraph2D", "g2: Graph2D") 
   
   smp = list(c_smp_1) + list(c_smp_2) # In case of coming-back.

   g2.SetMarkerStyle(20)
   
   
   # Plot the new TH2Poly (ordered one) and TGraph2D.
   # The new TH2Poly has to be same as the old one, and the TGraph2D
   # should be similar to the previous one.
   # It is now made by using a 'z' value as the bin number.
   global c4
   c4 = TCanvas("glc4", "TH2Poly from a kdTree (Ordered)",50,50,800,800)
   
   c4.Divide(2,2)
   c4.cd(1)
   h2polrebin.Draw("COLZ L");  # draw as scatter plot
   
   c4.cd(2)
   g2.Draw("pcol")
   
   c4.Update()
   
   # Make also a 1D-binned histogram.
   
   global kdX, kdY
   #Not to use:
   #kdX = TKDTreeBinning(DATASZ, 1, smp[0], 20)
   #kdY = TKDTreeBinning(DATASZ, 1, smp[DATASZ], 40)
   kdX = TKDTreeBinning(DATASZ, 1, c_smp_1, 20)
   kdY = TKDTreeBinning(DATASZ, 1, c_smp_2, 40)
   
   
   kdX.SortOneDimBinEdges()
   kdY.SortOneDimBinEdges()
   
   global hX 
   hX = TH1F("hX", "X projection", kdX.GetNBins(), kdX.GetOneDimBinEdges())
   #for(int i=0; i<kdX->GetNBins(); ++i) {
   for i in range(0, kdX.GetNBins(), 1):
      hX.SetBinContent(i+1, kdX.GetBinDensity(i))
      
   
   global hY
   hY = TH1F("hY", "Y Projection", kdY.GetNBins(), kdY.GetOneDimBinEdges())
   #for(int i=0; i<kdY->GetNBins(); ++i) {
   for i in range(0, kdY.GetNBins(), 1):
      hY.SetBinContent(i+1, kdY.GetBinDensity(i))
      
   
   
   c4.cd(3)
   hX.Draw()
   c4.cd(4)
   hY.Draw()
     
   #To avoid potential memory leak if you run this script again.
   #ROOT.gROOT.DeleteAll() 
   #Special objects don't have deallocation implemented. 
   #ROOT.gROOT.Remove(h1)
   #ROOT.gROOT.Remove(kdBins)
   #kdBins.Clear()
   #del kdBins
   


if __name__ == "__main__":
   kdTreeBinning()
