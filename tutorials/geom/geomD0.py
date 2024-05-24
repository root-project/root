## \file
## \ingroup tutorial_geom
## Script drawing a detector geometry (here D0).
##
## by default the geometry is drawn using the GL viewer
## Using the TBrowser, you can select other components
## if the file containing the geometry is not found in the local
## directory, it is automatically read from the ROOT web site.
##   - run with `%run geomD0.py`    top level detectors are transparent
##   - or       `%run geomD0.py 1`  top level detectors are visible
##
## \image html geom_geomD0.png width=800px
## \macro_code
##
## \authors Bertrand Bellenot, Rene Brun
## \translator P. P.


import ROOT

TGeoManager = ROOT.TGeoManager
TGeoVolume = ROOT.TGeoVolume
TCanvas = ROOT.TCanvas

Int_t = ROOT.Int_t

gGeoManager = ROOT.gGeoManager

# void
def geomD0(allVisible=0) :

   TGeoManager.Import("http://root.cern.ch/files/d0.root")

   global gGeoManager
   gGeoManager.DefaultColors()
   #gGeoManager.SetMaxVisNodes(40000)
   gGeoManager.SetMaxVisNodes(4000)
   #gGeoManager.SetVisLevel(4)
   if not allVisible:
      RecursiveInvisible(gGeoManager.GetVolume("D0-"))
      RecursiveInvisible(gGeoManager.GetVolume("D0+"))
      RecursiveInvisible(gGeoManager.GetVolume("D0WZ"))
      RecursiveInvisible(gGeoManager.GetVolume("D0WL"))
      RecursiveTransparency(gGeoManager.GetVolume("MUON"), 90)
      
   global myVol 
   myVol = gGeoManager.GetVolume("D0")

   #myVol.Draw("ogl")
   myVol.Draw("x3d")
   
   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   gb = [ i for i in globals()]
   myvars = [x for x in dir() if not x.startswith("__")]
   myvars += [x for x in gb if not x.startswith("__")]
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

# void
def RecursiveInvisible(vol) :
   vol.InvisibleAll()
   nd = vol.GetNdaughters()
   for i in range(nd):
      RecursiveInvisible(vol.GetNode(i).GetVolume())
      
   

# void
def RecursiveTransparency(vol, transp) :
   vol.SetTransparency(transp)
   nd = vol.GetNdaughters()
   for i in range(nd):
      RecursiveTransparency(vol.GetNode(i).GetVolume(), transp)
      
   


if __name__ == "__main__":
   geomD0()
