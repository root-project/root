## \file
## \ingroup tutorial_graphics
## \notebook
##
## This scrpit plots the Amplitude of a Hydrogen Atom.
##
## Also, it visualizes the Amplitude of a Hydrogen Atom in the n = 2, l = 0, m = 0 state;
## and demonstrates how TH2F can be used in Quantum Mechanics.
##
## The formula for Hydrogen in this energy state is \f$ \psi_{200} = \frac{1}{4\sqrt{2\pi}a_0 ^{\frac{3}{2}}}(2-\frac{\sqrt{x^2+y^2}}{a_0})e^{-\frac{\sqrt{x^2+y^2}}{2a_0}} \f$
##
## \macro_image
## \macro_code
##
## \author Advait Dhingra
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TH2F = ROOT.TH2F
TMath = ROOT.TMath
TLatex = ROOT.TLatex

#types
Double_t = ROOT.Double_t

#constants
kCividis = ROOT.kCividis

#maths
sqrt = ROOT.sqrt

#globals
gStyle = ROOT.gStyle



# double
def WaveFunction(x : Double_t, y : Double_t) :

   r = sqrt(x*x + y*y)

   # Wavefunction formula for psi 2,0,0
   #w = (1/pow((4*sqrt(2*TMath.Pi()) 1), 1.5)) * (2 - (r / 1)*pow(TMath.E(), (-1  r)/2)) 
   w = (1/pow((4*sqrt(2*TMath.Pi())* 1), 1.5)) * (2 - (r / 1)*pow(TMath.E(), (-1 * r)/2))

   # Amplitude
   return w*w; 
   
   

# void
def schroedinger_hydrogen() :
   
   global h2D 
   h2D = TH2F("Hydrogen Atom",
   "Hydrogen in n = 2, l = 0, m = 0 state; Position in x direction; Position in y direction",
   200, -10, 10, 200, -10, 10)
   
   #for (float i = -10; i < 10; i += 0.01) {
   #   for (float j = -10; j < 10; j += 0.01) {
   i = -10
   while( i < 10 ):
      i += 0.01
      j = -10
      while( j < 10 ):
         j += 0.01

         h2D.Fill(i, j, WaveFunction(i, j))
         
      
   
   gStyle.SetPalette(kCividis)
   gStyle.SetOptStat(0)
   
   global c1
   c1 = TCanvas("c1", "Schroedinger's Hydrogen Atom", 750, 1500)
   c1.Divide(1, 2)
   
   global c1_1
   c1_1 = c1.cd(1)
   c1_1.SetRightMargin(0.14)

   h2D.GetXaxis().SetLabelSize(0.03)
   h2D.GetYaxis().SetLabelSize(0.03)
   h2D.GetZaxis().SetLabelSize(0.03)
   h2D.SetContour(50)
   h2D.Draw("colz")
   
   global l
   l = TLatex(-10, -12.43, "The Electron is more likely to be found in the yellow areas and less likely to be found in the blue areas.")
   l.SetTextFont(42)
   l.SetTextSize(0.02)
   l.Draw()
   
   global c1_2
   c1_2 = c1.cd(2)
   c1_2.SetTheta(42.)
   
   global h2Dc
   h2Dc = h2D.Clone() # TH2D
   h2Dc.SetTitle("3D view of probability amplitude;;")
   h2Dc.Draw("surf2")
   


if __name__ == "__main__":
   schroedinger_hydrogen()
