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

TVirtualGeoConverter = ROOT.TVirtualGeoConverter 
TView = ROOT.TView 
TString = ROOT.TString

#Not implemented
#printf = ROOT.printf


gRandom = ROOT.gRandom
gGeoManager = ROOT.gGeoManager
gPad = ROOT.gPad


#______________________________________________________________________________
# int
def randomColor() :
   gRandom = TRandom3()
   dt = TDatime()
   gRandom.SetSeed(dt.GetTime())
   ci = TColor.GetFreeColorIndex()
   color = TColor(ci, gRandom.Rndm(), gRandom.Rndm(), gRandom.Rndm())
   return ci
   

#______________________________________________________________________________
# void
def tessellatedNav(dot_obj_file = "", check = False) :
  
   # Input a file in .obj format (https:#en.wikipedia.org/wiki/Wavefront_.obj_file)
   # The file should have a single object inside, only vertex and faces information is used
   name = TString(dot_obj_file)
   sfile = TString(dot_obj_file)
   if TString(sfile).IsNull():
      sfile = ROOT.gROOT.GetTutorialsDir()
      sfile += "/geom/teddy.obj"
      
   name.ReplaceAll(".obj", "")
   ROOT.gROOT.GetListOfCanvases().Delete()
   global gGeoManager
   if (gGeoManager):
      del gGeoManager
   geom = TGeoManager(str(name), "Imported from .obj file")
   mat = TGeoMaterial("Al", 26.98, 13, 2.7)
   med = TGeoMedium("MED", 1, mat)
   top = geom.MakeBox("TOP", med, 10, 10, 10)
   geom.SetTopVolume(top)
   
   sfile = TString(sfile)
   tsl = TGeoTessellated.ImportFromObjFormat(sfile.Data(), check)
   if (not tsl):
      return
   tsl.ResizeCenter(5.)
   
   vol = TGeoVolume(str(name), tsl, med)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol, 1)
   geom.CloseGeometry()
   
   # Convert to VecGeom tessellated solid
   converter = TVirtualGeoConverter.Instance(geom)
   if not converter:
      print("Raytracing a tessellated shape without VecGeom support will just draw a box\n")
      
   else:
      converter.ConvertGeometry()
      
   
   if (ROOT.gROOT.IsBatch()):
      return

   # Set the view
   #top.Draw()
   top.Draw("x3d")
   global gPad
   view = gPad.GetView()
   if (not view):
      print("not view")
      return
   view.Top()
   
   # Raytracing will call VecGeom navigation
   top.Raytrace()

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
   tessellatedNav()
