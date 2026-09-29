# \file
# \ingroup tutorial_math
# \notebook
# Example of first few Legendre Polynomials
#
# Draws a graph.
#
# \macro_image
# \macro_code
#
# \author Lorenzo Moneta


import ROOT
TMath = 		 ROOT.TMath
TF1 = 		 ROOT.TF1
TCanvas = 		 ROOT.TCanvas
TLegend = 		 ROOT.TLegend
TLegendEntry = 		 ROOT.TLegendEntry
Math = 		 ROOT.Math
TSystem = 		 ROOT.TSystem

#globals
gROOT = ROOT.gROOT
gSystem = ROOT.gSystem


def Legendre():

   gSystem.Load("libMathMore.so")
   
   global Canvas
   Canvas =  TCanvas("DistCanvas", "Legendre polynomials example", 10, 10, 750, 600)
   Canvas.SetGrid()
   
   global leg
   leg =  TLegend(0.5, 0.7, 0.4, 0.89)

   #Drawin the set of Legendre functions.
   #Not to use: 
   # L = [TF1() ] * 5
   global L
   L =  [ TF1() for i in range(5) ] 
   #for ( nu = 0 nu <= 4 nu++)
   for nu in range(0, 4+1, 1):
      # Creating the Legendre-Polynomial.
      L[nu] = TF1("L_0", "ROOT::Math::legendre([0], x)", -1, 1)
      # Setting-up functions. 
      L[nu].SetParameters(nu, 0.0)
      L[nu].SetLineStyle(1)
      L[nu].SetLineWidth(2)
      L[nu].SetLineColor(nu+1)
      
   # Setting-up Canvas.
   L[0].SetMaximum(1)
   L[0].SetMinimum(-1)
   L[0].SetTitle("Legendre polynomials")
   
   #Adding Legends for the Legendre-Functions into 'leg'
   leg.AddEntry(L[0].DrawCopy(), " L_0(x)", "l")
   leg.AddEntry(L[1].DrawCopy("same"), " L_1(x)", "l")
   leg.AddEntry(L[2].DrawCopy("same"), " L_2(x)", "l")
   leg.AddEntry(L[3].DrawCopy("same"), " L_3(x)", "l")
   leg.AddEntry(L[4].DrawCopy("same"), " L_4(x)", "l")
   leg.Draw()
   
   #Drawing the legend into the Canvas.   
   Canvas.cd()
   
if __name__ == "__main__":
   Legendre()
