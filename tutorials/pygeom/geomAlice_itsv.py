## \file
## \ingroup tutorial_geom
## Script drawing a detector geometry (here ITSV from Alice).
##
## By default the geometry is drawn using the GL viewer
## Using the TBrowser, you can select other components
## if the file containing the geometry is not found in the local
## directory, it is automatically read from the ROOT web site.
##
## \image html geom_geomAlice_itsv.png width=800px
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

TBrowser = ROOT.TBrowser
TGeoManager = ROOT.TGeoManager
gGeoManager = ROOT.gGeoManager


# void
def geomAlice_itsv() :
   ROOT.gROOT.GetListOfBrowsers().Delete()
   global gGeoManager
   TGeoManager.Import("http://root.cern.ch/files/alice2.root")
   gGeoManager.DefaultColors()
   global Vol_ITSV
   Vol_ITSV = gGeoManager.GetVolume("ITSV")
   
   global b
   b = TBrowser()
   b.Add(gGeoManager)
   b.Add(Vol_ITSV) 
   b.BrowseObject(Vol_ITSV)
   # If you want to see Vol_ITSV outside TBroswer use: "ogl" or "x3d" options.
   #Vol_ITSV.Draw("ogl") 
   #Vol_ITSV.Draw("x3d") 
   Vol_ITSV.Draw("") 
   b.Show()

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
   geomAlice_itsv()
