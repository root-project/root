## \file
## \ingroup tutorial_graphics
##
## This script illustrates how to animate a picture using a TTimer-object.
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TStyle = ROOT.TStyle 
TCanvas = ROOT.TCanvas 
TF2 = ROOT.TF2 
TMath = ROOT.TMath
TTimer = ROOT.TTimer 

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad

#variables
pi = Double_t()
f2 = TF2()
t = Int_t(0)
phi = Double_t(30)

# void
def anim() :

   gStyle.SetCanvasPreferGL(True)
   gStyle.SetFrameFillColor(42)

   global c1
   c1 = TCanvas("c1")
   c1.SetFillColor(17)
   
   global pi
   pi = TMath.Pi()

   global f2
   f2 = TF2("f2","sin(2*x)*sin(2*y)*[0]",0,pi,0,pi)
   f2.SetParameter(0,1)
   f2.SetNpx(15)
   f2.SetNpy(15)
   f2.SetMaximum(1)
   f2.SetMinimum(-1)
   f2.Draw("glsurf1")

   global timer
   timer = TTimer(20)
   #Not to use:
   #timer.SetCommand("Animate()")
   #Instead:
   command = """
   TPython::Eval( \"Animate()\" ) ;
   """ 
   timer.SetCommand(command)
   timer.TurnOn()
   
# void
def Animate() :

   #just in case the canvas has been deleted
   if (not ROOT.gROOT.GetListOfCanvases().FindObject("c1")) : return

   global t, phi
   t += 0.05*pi
   phi += 2

   f2.SetParameter(0, TMath.Cos(t))
   gPad.SetPhi(phi)

   gPad.Modified()
   gPad.Update()
   


if __name__ == "__main__":
   anim()
