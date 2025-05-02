## \file
## \ingroup tutorial_geom
## Draw a "representative" TXTRU shape
##
## \macro_image
## \macro_code
##
## \author Robert Hatcher (rhatcher@fnal.gov) 2000.09.06
## \translator P. P.


import ROOT
import ctypes

TMath = ROOT.TMath
TCanvas = ROOT.TCanvas
TGeometry = ROOT.TGeometry
TXTRU = ROOT.TXTRU
TNode = ROOT.TNode

Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double

gPad = ROOT.gPad


# void
def xtruDraw() :
   global canvas
   canvas = TCanvas("xtru","Example XTRU object",200,10,640,640)
   
   # Create a new geometry
   global geometry
   geometry = TGeometry("geometry","geometry")
   geometry.cd()
   
   atxtru = TXTRU("atxtru","atxtru","void",5,2)
   
   # outline and z segment specifications
   
   x = [
      -177.292,   -308.432,   -308.432,   -305.435,   -292.456,    -280.01
      ,    -241.91,    -241.91,   -177.292,   -177.292,    177.292,    177.292
      ,     241.91,     241.91,     280.06,    297.942,    305.435,    308.432
      ,    308.432,    177.292,    177.292,   -177.292
   ] 
   y = [
           154.711,    23.5712,     1.1938,     1.1938,     8.6868,     8.6868
      ,    -3.7592,   -90.0938,   -154.711,   -190.602,   -190.602,   -154.711
      ,   -90.0938,    -3.7592,     8.6868,     8.6868,     1.1938,     1.1938
      ,    23.5712,    154.711,    190.602,    190.602
   ]   
   z = [0.00,      500.0]
   scale = [1.00,       1.00]
   x0 = [0,          0]
   y0 = [0,          0]
   
   i = Int_t() 
   
   nxy = len(x)
   for  i in range(0, nxy):  
      atxtru.DefineVertex(i,x[i],y[i])
      
   
   nz = len(z) 
   for  i in range(0, nz):  
      atxtru.DefineSection(i,z[i],scale[i],x0[i],y0[i])
      
   
   # Define a TNode where this example resides in the TGeometry
   # Draw the TGeometry
   
   global anode
   anode = TNode("anode","anode",atxtru)
   anode.SetLineColor(1)
   
   geometry.Draw("3xd")
   
   # Tweak the pad scales so as not to distort the shape
   
   global gPad
   thisPad = gPad
   if thisPad:
      view = thisPad.GetView()
      if (not view): return

      Min = Max = center = [Double_t()]*3
      c_Min = (c_double *3)(*Min)
      c_Max = (c_double *3)(*Max)
      c_center = (c_double *3)(*center)
      view.GetRange(c_Min,c_Max)
      Min = list(c_Min)
      Max = list(c_Max)
      center = list(c_center)

      # Find the boxed center
      for  i in range(0, 3):  center[i] = 0.5*(Max[i]+Min[i])
      MaxSide = 0
      # Find the largest side
      for  i in range(0, 3):  MaxSide = TMath.Max(MaxSide,Max[i]-center[i])
      # Adjust scales:
      for  i in range(0, 3):  
         Max[i] = center[i] + MaxSide
         Min[i] = center[i] - MaxSide
         
      c_Min = (c_double *3)(*Min)
      c_Max = (c_double *3)(*Max)
      c_center = (c_double *3)(*center)
      view.SetRange(c_Min,c_Max)

      ireply = Int_t()
      thisPad.Modified()
      thisPad.Update()
      
   
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
         print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!
   


if __name__ == "__main__":
   xtruDraw()
