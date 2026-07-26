## \file
## \ingroup tutorial_webgui
##  Web-based geometry viewer for CMS geometry
##
## \macro_code
##
## \author Sergey Linev
## \translator P. P.


import ROOT

RGeomViewer = ROOT.RGeomViewer
TGeoManager = ROOT.TGeoManager 
TGeoVolume = ROOT.TGeoVolume 
TFile = ROOT.TFile 

gGeoManager = ROOT.gGeoManager
std = ROOT.std



# void
def web_cms(split = False) :
   TFile.SetCacheFileDir(".")
   
   TGeoManager.Import("https://root.cern/files/cms.root")
   
   global gGeoManager 
   gGeoManager.DefaultColors()
   gGeoManager.SetVisLevel(4)
   gGeoManager.GetVolume("TRAK").InvisibleAll()
   gGeoManager.GetVolume("HVP2").SetTransparency(20)
   gGeoManager.GetVolume("HVEQ").SetTransparency(20)
   gGeoManager.GetVolume("YE4").SetTransparency(10)
   gGeoManager.GetVolume("YE3").SetTransparency(20)
   gGeoManager.GetVolume("RB2").SetTransparency(99)
   gGeoManager.GetVolume("RB3").SetTransparency(99)
   gGeoManager.GetVolume("COCF").SetTransparency(99)
   gGeoManager.GetVolume("HEC1").SetLineColor(7)
   gGeoManager.GetVolume("EAP1").SetLineColor(7)
   gGeoManager.GetVolume("EAP2").SetLineColor(7)
   gGeoManager.GetVolume("EAP3").SetLineColor(7)
   gGeoManager.GetVolume("EAP4").SetLineColor(7)
   gGeoManager.GetVolume("HTC1").SetLineColor(2)
   
   viewer = RGeomViewer(gGeoManager)
   
   # select volume to draw
   viewer.SelectVolume("CMSE")
   
   # specify JSROOT draw options - here clipping on X,Y,Z axes
   viewer.SetDrawOptions("clipxyz")
   
   # set default limits for number of visible nodes and faces
   # when viewer created, initial values exported from TGeoManager
   viewer.SetLimits()
   
   viewer.SetShowHierarchy(not split)
   
   # start web browser
   viewer.Show()
   
   # destroy viewer only when connection to client is closed
   viewer.ClearOnClose(viewer)
   
   if split:
      # create separate widget with geometry hierarchy only
      hier = std.make_shared<ROOT.RGeomHierarchy>(viewer.Description())
      
      # start web browser with hierarchy
      hier.Show()
      
      # destroy widget only when connection to client is closed
      hier.ClearOnClose(hier)
      
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
   web_cms()
   print("\nCheck your browser to explore the CMS detector")
