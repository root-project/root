## \file
## \ingroup tutorial_geom
## This macro generates
## with 2 views of the NA49 detector using the old obsolete geometry package.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT


TPad = ROOT.TPad
TPaveLabel = ROOT.TPaveLabel
TFile = ROOT.TFile
TCanvas = ROOT.TCanvas

gBenchmark = ROOT.gBenchmark

# void
def na49view() :
   global c1
   c1 = TCanvas("c1","The NA49 canvas",200,10,700,780)
   
   global gBenchmark
   gBenchmark.Start("na49view")
   
   global All, tof
   All = TPad("all","A Global view of NA49",0.02,0.02,0.48,0.82,28)
   tof = TPad("tof","One Time Of Flight element",0.52,0.02,0.98,0.82,28)
   All.Draw()
   tof.Draw()
   global na49title
   na49title = TPaveLabel(0.04,0.86,0.96,0.98,"Two views of the NA49 detector")
   na49title.SetFillColor(32)
   na49title.Draw()
   #
   global nageom
   nageom = TFile("na49.root")
   if (not nageom or nageom.IsZombie()): 
      print("na49.root was NOT found")
      print("Please, execute na49.C or na49.py first, to generate its geometry.")
      print("Then, execute na49geomfile.C or na49geomfile.py to save its geometry in a rootfile.")
      return

   # Note the difference between n49 object, and na49 geometry.
   global n49, na49
   n49 = ROOT.gROOT.FindObject("na49") #(TGeometry) 
   na49 = ROOT.na49
   n49.SetBomb(1.2)
   n49.cd();     #Set current geometry
   All.cd();     #Set current pad
   na49.Draw()
   c1.Update()
   tof.cd()
   global TOFR1
   TOFR1 = n49.GetNode("TOFR1")
   TOFR1.Draw()
   c1.Update()
   
   gBenchmark.Show("na49view")
   
   # Note: 
   # To have a better and dynamic view of any of these pads,
   # you can click-in the middle button of your mouse.
   # Then select "View with x3d" option in the VIEW menu of the Canvas.
   # Once in x3d, you are in wireframe mode by default.
   # You can switch to any of these other options:
   #   - Hidden Line mode by typing E
   #   - Solid mode by typing R
   #   - Wireframe mode by typing W
   #   - Stereo mode by clicking S (and you will need special glasses, though)
   #   - To leave x3d type Q
   

   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   myvars = [x for x in dir() if not x.startswith("__")]
   #myvars = [x for x in vars(self) ] 
   for var in myvars: 
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         #exec(f"ROOT.gROOT.Remove(self.{var})")
         print("deleting", var)
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!

if __name__ == "__main__":
   na49view()
