## \file
## \ingroup tutorial_geom
## Script drawing a detector geometry (here ATLAS).
##
## by default the geometry is drawn using the GL viewer
## Using the TBrowser, you can select other components
## if the file containing the geometry is not found in the local
## directory, it is automatically read from the ROOT web site.
##
## \image html geom_geomAtlas.png width=800px
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT


TGeoManager = ROOT.TGeoManager
TBrowser = ROOT.TBrowser

gGeoManager = ROOT.gGeoManager

# void
def geomAtlas() :
   ROOT.gROOT.GetListOfBrowsers().Delete()

   global gGeoManager
   TGeoManager.Import("http://root.cern.ch/files/atlas.root")
   #gGeoManager.DefaultColors()
   gGeoManager.SetMaxVisNodes(5000)
   #gGeoManager.SetMaxVisNodes(50000)
   #gGeoManager.SetVisLevel(4)
   global myVol
   myVol = gGeoManager.GetVolume("ATLS")

   global b
   b = TBrowser()
   b.Add(gGeoManager)
   b.Add(myVol)
   b.BrowseObject(myVol)
   # If you want to see myVol outside TBrowser use: "ogl" or "x3d" options.
   #myVol.Draw("ogl")
   #myVol.Draw("x3d")
   myVol.Draw("")
   b.Show()

   
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


if __name__ == "__main__":
   geomAtlas()
