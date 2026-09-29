## \file
## \ingroup tutorial_geom
## Macro allowing to vizualize tessellations from Wavefront's .obj format.
##
## \image html geom_visualizeWavefrontObj.png width=500px
## \macro_code
##
## \author Andrei Gheata
## \translator P. P.


import ROOT

TROOT = ROOT.TROOT 
TColor = ROOT.TColor 
TDatime = ROOT.TDatime 
TRandom3 = ROOT.TRandom3 
TGeoManager = ROOT.TGeoManager 
TGeoTessellated = ROOT.TGeoTessellated 
TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium
TGeoVolume = ROOT.TGeoVolume

TString = ROOT.TString

gRandom = ROOT.gRandom
gGeoManager = ROOT.gGeoManager

#______________________________________________________________________________
# int
def randomColor() :

   global gRandom
   gRandom = TRandom3()
   dt = TDatime() 
   gRandom.SetSeed(dt.GetTime())
   ci = TColor.GetFreeColorIndex()
   color = TColor(ci, gRandom.Rndm(), gRandom.Rndm(), gRandom.Rndm())
   return ci
   

#______________________________________________________________________________
# void
def visualizeWavefrontObj(dot_obj_file="", check = False) :
   # Input a file in .obj format (https:#en.wikipedia.org/wiki/Wavefront_.obj_file)
   # The file should have a single object inside, only vertex and faces information is used
   
   name = TString(dot_obj_file)
   sfile = TString(dot_obj_file)
   if sfile.IsNull():
      sfile = ROOT.gROOT.GetTutorialsDir()
      sfile += "/geom/teddy.obj"
      
   name.ReplaceAll(".obj", "")
   ROOT.gROOT.GetListOfCanvases().Delete()
   global gGeoManager
   if (gGeoManager):
      del gGeoManager
   TGeoManager(str(name), "Imported from .obj file")
   gGeoManager = ROOT.gGeoManager

   #Not to use: mat = TGeoMaterial("Al", 26.98, 13, 2.7)
   mat = TGeoMaterial("Al") 
   mat.SetA(26.98) 
   mat.SetZ(13)
   mat.SetDensity(2.7)
   med = TGeoMedium("MED", 1, mat)
   top = gGeoManager.MakeBox("TOP", med, 10, 10, 10)
   gGeoManager.SetTopVolume(top)
   
   sfile = TString(sfile)
   tsl = TGeoTessellated.ImportFromObjFormat(sfile.Data(), check)
   if (not tsl): return
   tsl.ResizeCenter(5.)
   
   vol = TGeoVolume(str(name), tsl, med)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol, 1)
   gGeoManager.CloseGeometry()
   if (not ROOT.gROOT.IsBatch()): 
      #top.Draw("ogl")
      top.Draw("x3d")
   
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
         print("deleting", var, "from ROOT.gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!


if __name__ == "__main__":
   visualizeWavefrontObj()
