## \file
## \ingroup tutorial_geom
## Script drawing a detector geometry (here ALICE).
##
## by default the geometry is drawn using the GL viewer
## Using the TBrowser, you can select other components
## if the file containing the geometry is not found in the local
## directory, it is automatically read from the ROOT web site.
##
## \image html geom_geomAlice.png width=800px
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

TGeoManager = ROOT.TGeoManager
TBrowser = ROOT.TBrowser
TCanvas = ROOT.TCanvas

Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine
gGeoManager = ROOT.gGeoManager

# void
def geomAlice() :

   ROOT.gROOT.GetListOfBrowsers().Delete()

   TGeoManager.Import("http://root.cern.ch/files/alice2.root")
   global gGeoManager
   gGeoManager.DefaultColors()
   #gGeoManager.SetVisLevel(4)
   gGeoManager.GetVolume("HALL").InvisibleAll()
   gGeoManager.GetVolume("ZDCC").InvisibleAll()
   gGeoManager.GetVolume("ZDCA").InvisibleAll()
   gGeoManager.GetVolume("L3MO").InvisibleAll()
   gGeoManager.GetVolume("YOUT1").InvisibleAll()
   gGeoManager.GetVolume("YOUT2").InvisibleAll()
   gGeoManager.GetVolume("YSAA").InvisibleAll()
   gGeoManager.GetVolume("RB24").InvisibleAll()
   gGeoManager.GetVolume("RB26Pipe").InvisibleAll()
   gGeoManager.GetVolume("DDIP").InvisibleAll()
   gGeoManager.GetVolume("DCM0").InvisibleAll()
   #gGeoManager.GetVolume("PPRD").InvisibleAll()
   gGeoManager.GetVolume("BRS1").InvisibleAll()
   gGeoManager.GetVolume("BRS4").InvisibleAll()
   #gGeoManager.GetVolume("Dipole").InvisibleAll()
   gGeoManager.GetVolume("ZN1").InvisibleAll()
   gGeoManager.GetVolume("Q13T").InvisibleAll()
   gGeoManager.GetVolume("ZP1").InvisibleAll()
   gGeoManager.GetVolume("QTD1").InvisibleAll()
   gGeoManager.GetVolume("QTD2").InvisibleAll()
   gGeoManager.GetVolume("QBS7").InvisibleAll()
   gGeoManager.GetVolume("QA07").InvisibleAll()
   gGeoManager.GetVolume("MD1V").InvisibleAll()
   gGeoManager.GetVolume("QTD3").InvisibleAll()
   gGeoManager.GetVolume("QTD4").InvisibleAll()
   gGeoManager.GetVolume("QTD5").InvisibleAll()
   gGeoManager.GetVolume("QBS3").InvisibleAll()
   gGeoManager.GetVolume("QBS4").InvisibleAll()
   gGeoManager.GetVolume("QBS5").InvisibleAll()
   gGeoManager.GetVolume("QBS6").InvisibleAll()

   #gGeoManager.GetVolume("ALIC").Draw("ogl")
   #gGeoManager.GetVolume("ALIC").Draw("x3d")
   # We are not going to display on a TCanvas, instead
   # instead we are going to display on TBrowser.
   volume = gGeoManager.GetVolume("ALIC")

   global myb
   myb = TBrowser()
   
   myb.Add(gGeoManager)
   myb.Add(volume)
   myb.BrowseObject(volume)
   #volume.Draw("ogl")
   #volume.Draw("x3d")
   volume.Draw("")
   myb.Show()
   
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
   geomAlice()
