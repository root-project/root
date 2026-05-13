## \file
## \ingroup tutorial_geom
## Combinatorial Solid Geometry example.
##
## \macro_code
##
## \author Andrei Gheata
## \translator P. P.


import ROOT

TControlBar = ROOT.TControlBar

TGeoManager = ROOT.TGeoManager
TGeoMaterial = ROOT.TGeoMaterial
TGeoVolume = ROOT.TGeoVolume
TGeoMedium = ROOT.TGeoMedium
TGeoTranslation = ROOT.TGeoTranslation 
TGeoCombiTrans = ROOT.TGeoCombiTrans  
TGeoRotation = ROOT.TGeoRotation
TGeoPgon = ROOT.TGeoPgon
TGeoSphere = ROOT.TGeoSphere
TGeoCompositeShape = ROOT.TGeoCompositeShape
TGeoBBox = ROOT.TGeoBBox
TGeoTorus = ROOT.TGeoTorus

TCanvas = ROOT.TCanvas
TPaveText = ROOT.TPaveText

gGeoManager = ROOT.gGeoManager
gPad = ROOT.gPad

#FLAGS
raytracing = True 

def DelROOTObjs():
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   #myvars = [x for x in dir() if not x.startswith("__")]
   myvars = [x for x in globals() if not x.startswith("_")]
   #myvars = [x for x in vars(self) ] 
   for var in myvars: 
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!

#________________________________________________________
# TCanvas
def create_canvas(title = str(), divide = True) :
   global c, gPad
   c = ROOT.gROOT.GetListOfCanvases().FindObject("csg_canvas")
   if c:
      c.Clear()
      c.Update()
      c.SetTitle(title)
      
   else:
      c = TCanvas("csg_canvas", title, 700,1000)
      
   
   if divide:
      c.Divide(1,2,0,0)
      c.cd(2)
      gPad.SetPad(0,0,1,0.4)
      c.cd(1)
      gPad.SetPad(0,0.4,1,1)
      
   
   DelROOTObjs()
   return c
   


#______________________________________________________________________________
# void
def MakePicture() :
   global gGeoManager, raytracing, gPad

   is_raytracing = gGeoManager.GetTopVolume().IsRaytracing()
   if (is_raytracing != raytracing):
      gGeoManager.GetTopVolume().SetVisRaytrace(raytracing)
      gPad.Modified()
      gPad.Update()
   print("")
      
   

   DelROOTObjs()

#______________________________________________________________________________
# void
def s_union() :
   global c, gGeoManager
   global xtru, mat, med, top
   global pgon, sph, tr, cs, vol
   global pt, text
   c = create_canvas("Union boolean operation")
   
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   
   xtru = TGeoManager("xtru", "poza12")
   mat = TGeoMaterial("Al")
   mat.SetA(26.98)
   mat.SetZ(13)
   mat.SetDensity(2.7)
   med = TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   
   # define shape components with names
   pgon = TGeoPgon("pg",0.,360.,6,2)
   pgon.DefineSection(0,0,0,20)
   pgon.DefineSection(1, 30,0,20)
   
   sph = TGeoSphere("sph", 40., 45., 5., 175., 0., 340.)
   # define named geometrical transformations with names
   tr = TGeoTranslation(0., 0., 45.)
   tr.SetName("tr")
   # register all used transformations
   tr.RegisterYourself()
   # create the composite shape based on a Boolean expression
   cs = TGeoCompositeShape("mir", "sph:tr + pg")
   
   vol = TGeoVolume("COMP1",cs)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(100)
   top.Draw()
   MakePicture()
   
   c.cd(2)
   pt = TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoCompositeShape - composite shape class")
   text.SetTextColor(2)
   pt.AddText("----- It's an example of boolean union operation : A + B")
   pt.AddText("----- A == part of sphere (5-175, 0-340), B == pgon")
   pt.AddText(" ")
   pt.SetAllWith("-----","color",4)
   pt.SetAllWith("-----","font",72)
   pt.SetAllWith("-----","size",0.04)
   pt.SetTextAlign(12)
   pt.SetTextSize(.044)
   pt.Draw()
   c.cd(1)
   print("")
   

   DelROOTObjs()

#______________________________________________________________________________
# void
def s_intersection() :
   global c, gGeoManager
   global xtru, mat, med, top
   global box, sph, tr, cs, vol
   global pt, text
   c = create_canvas("Intersection boolean operation")
   
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   
   xtru = TGeoManager("xtru", "poza12")
   mat = TGeoMaterial("Al", 26.98,13,2.7)
   med = TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   
   # define shape components with names
   box = TGeoBBox("bx", 40., 40., 40.)
   sph = TGeoSphere("sph", 40., 45.)
   # define named geometrical transformations with names
   tr = TGeoTranslation(0., 0., 45.)
   tr.SetName("tr")
   # register all used transformations
   tr.RegisterYourself()
   # create the composite shape based on a Boolean expression
   cs = TGeoCompositeShape("mir", "sph:tr * bx")
   
   vol = TGeoVolume("COMP2",cs)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(100)
   top.Draw()
   MakePicture()
   
   c.cd(2)
   
   pt = TPaveText(0.01,0.01,0.99,0.99)
   
   pt.SetLineColor(1)
   
   text = pt.AddText("TGeoCompositeShape - composite shape class")
   
   text.SetTextColor(2)
   pt.AddText("----- Here is an example of boolean intersection operation : A * B")
   pt.AddText("----- A == sphere (with inner radius non-zero), B == box")
   pt.AddText(" ")
   pt.SetAllWith("-----","color",4)
   pt.SetAllWith("-----","font",72)
   pt.SetAllWith("-----","size",0.04)
   pt.SetTextAlign(12)
   pt.SetTextSize(0.044)
   pt.Draw()
   c.cd(1)
   print("")
   

   DelROOTObjs()

#______________________________________________________________________________
# void
def s_difference() :
   global c, gGeoManager
   global xtru, mat, med, top
   global tor, sph, cs, vol 
   global pt, text

   c = create_canvas("Difference boolean operation")
   
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   
   xtru = TGeoManager("xtru", "poza12")
   mat = TGeoMaterial("Al", 26.98,13,2.7)
   med = TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   
   # define shape components with names
   tor = TGeoTorus("tor", 45., 15., 20., 45., 145.)
   sph = TGeoSphere("sph", 20., 45., 0., 180., 0., 270.)
   # create the composite shape based on a Boolean expression
   cs = TGeoCompositeShape("mir", "sph - tor")
   
   vol = TGeoVolume("COMP3",cs)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(60)
   top.Draw()
   MakePicture()
   
   c.cd(2)
   
   pt = TPaveText(.01, .01, .99, .99)
   
   pt.SetLineColor(1)
   
   text = pt.AddText("TGeoCompositeShape - composite shape class")
   
   text.SetTextColor(2)
   
   pt.AddText("----- It's an example of boolean difference: A - B")
   pt.AddText("----- A == part of sphere (0-180, 0-270), B == partial torus (45-145)")
   pt.AddText(" ")
   pt.SetAllWith("-----","color",4)
   pt.SetAllWith("-----","font",72)
   pt.SetAllWith("-----","size",0.04)
   pt.SetTextAlign(12)
   pt.SetTextSize(0.044)
   pt.Draw()
   c.cd(1)
   print("")
   

   DelROOTObjs()

#______________________________________________________________________________
# void
def s_complex() :
   global c, gGeoManager, xtru, mat, med, top
   global box, box1, sph, sph1
   global tr, tr1, tr2, tr3
   global cs, vol, text, pt


   c = create_canvas("A * B - C")
   
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   
   xtru = TGeoManager("xtru", "poza12")
   mat = TGeoMaterial("Al", 26.98,13,2.7)
   med = TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   
   # define shape components with names
   box = TGeoBBox("box", 20., 20., 20.)
   box1 = TGeoBBox("box1", 5., 5., 5.)
   sph = TGeoSphere("sph", 5., 25.)
   sph1 = TGeoSphere("sph1", 1., 15.)
   # create the composite shape based on a Boolean expression
   tr = TGeoTranslation(0., 30., 0.)
   tr1 = TGeoTranslation(0., 40., 0.)
   tr2 = TGeoTranslation(0., 30., 0.)
   tr3 = TGeoTranslation(0., 30., 0.)
   tr.SetName("tr")
   tr1.SetName("tr1")
   tr2.SetName("tr2")
   tr3.SetName("tr3")
   # register all used transformations
   tr.RegisterYourself()
   tr1.RegisterYourself()
   tr2.RegisterYourself()
   tr3.RegisterYourself()
   
   cs = TGeoCompositeShape("mir", "(sph * box) + (sph1:tr - box1:tr1)")
   
   vol = TGeoVolume("COMP4",cs)
   # vol.SetLineColor(randomColor())
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   top.Draw()
   MakePicture()
   
   c.cd(2)
   pt = TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoCompositeShape - composite shape class")
   text.SetTextColor(2)
   pt.AddText("----- (sphere * box) + (sphere - box) ")
   
   pt.AddText(" ")
   pt.SetAllWith("-----","color",4)
   pt.SetAllWith("-----","font",72)
   pt.SetAllWith("-----","size",0.04)
   pt.SetTextAlign(12)
   pt.SetTextSize(0.044)
   pt.Draw()
   c.cd(1)
   print("")
   
   

   DelROOTObjs()

#______________________________________________________________________________
# void
def raytrace() :
   global raytracing 
   raytracing = not raytracing
   
   if gGeoManager and gPad:
      top = gGeoManager.GetTopVolume()
      drawn = gPad.GetListOfPrimitives().FindObject(top)
      if (drawn):
         top.SetVisRaytrace(raytracing)
      
      print("raytrace {:d}\n".format(raytracing))
      gPad.Modified()
      gPad.Update()
   print("")
      
   

   DelROOTObjs()

#______________________________________________________________________________
# void
def Help() :
   global c, welcome, hdemo
   c = create_canvas("Help to run demos", False)
   
   welcome = TPaveText(.1,.8,.9,.97)
   welcome.AddText("Welcome to the new geometry package")
   welcome.SetTextFont(32)
   welcome.SetTextColor(4)
   welcome.SetFillColor(24)
   welcome.Draw()
   
   hdemo = TPaveText(.05,.05,.95,.7)
   hdemo.SetTextAlign(12)
   hdemo.SetTextFont(52)
   hdemo.AddText("- Demo for building TGeo composite shapes")
   hdemo.AddText(" ")
   hdemo.AddText(" .... s_union() : Union boolean operation")
   hdemo.AddText(" .... s_difference() : Difference boolean operation")
   hdemo.AddText(" .... s_intersection() : Intersection boolean operation")
   hdemo.AddText(" .... s_complex() : Combination of (A * B) + (C - D)")
   hdemo.AddText(" ")
   hdemo.SetAllWith("....","color",2)
   hdemo.SetAllWith("....","font",72)
   hdemo.SetAllWith("....","size",0.03)
   
   hdemo.Draw()
   print("")
   

   DelROOTObjs()

#______________________________________________________________________________
def Py(function_name):
   return f'TPython::Exec("{function_name}()")'

# void
#def csgdemo () :
class csgdemo :
   ROOT.gSystem.Load("libGeom")
   bar = TControlBar("vertical", "TGeo composite shapes",20,20)
   
   bar.AddButton("How to run  ",Py("Help"),"Instructions ")
   bar.AddButton("Union ", Py("s_union"), "A + B ")
   bar.AddButton("Intersection ", Py("s_intersection"), "A * B ")
   bar.AddButton("Difference ", Py("s_difference"), "A - B ")
   bar.AddButton("Complex composite", Py("s_complex"), "(A * B) + (C - D)")
   bar.AddButton("RAY-TRACE ON/OFF",Py("raytrace"),"Toggle ray-tracing mode")
   bar.Show()
   
   ROOT.gROOT.Remove(bar)
   
if __name__ == "__main__":
   csgdemo()
