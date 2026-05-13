# \file
# \ingroup tutorial_geom
# GUI to draw the geometry shapes.
#
# \macro_code
#
# \author Andrei Gheata
# \translator P. P.


import ROOT
import ctypes
import sys
from typing import cast

TMath = 		 ROOT.TMath
TControlBar = 		 ROOT.TControlBar
TRandom3 = 		 ROOT.TRandom3
TROOT = 		 ROOT.TROOT
TSystem = 		 ROOT.TSystem
TVirtualPad = 		 ROOT.TVirtualPad
TCanvas = 		 ROOT.TCanvas
TVirtualGeoPainter = 		 ROOT.TVirtualGeoPainter
TGeoManager = 		 ROOT.TGeoManager
TGeoNode = 		 ROOT.TGeoNode
TView = 		 ROOT.TView
TColor = 		 ROOT.TView
TPaveText = 		 ROOT.TPaveText
TGeoBBox = 		 ROOT.TGeoBBox
TGeoPara = 		 ROOT.TGeoPara
TGeoTube = 		 ROOT.TGeoTube
TGeoCone = 		 ROOT.TGeoCone
TGeoEltu = 		 ROOT.TGeoEltu
TGeoSphere = 		 ROOT.TGeoSphere
TGeoTorus = 		 ROOT.TGeoTorus
TGeoTrd1 = 		 ROOT.TGeoTrd1
TGeoTrd2 = 		 ROOT.TGeoTrd2
TGeoParaboloid = 		 ROOT.TGeoParaboloid
TGeoHype = 		 ROOT.TGeoHype
TGeoPcon = 		 ROOT.TGeoPcon
TGeoPgon = 		 ROOT.TGeoPgon
TGeoArb8 = 		 ROOT.TGeoArb8
TGeoXtru = 		 ROOT.TGeoXtru
TGeoCompositeShape = 		 ROOT.TGeoCompositeShape
TGeoTessellated = 		 ROOT.TGeoTessellated
TGeoPhysicalNode = 		 ROOT.TGeoPhysicalNode

TProcessEventTimer = ROOT.TProcessEventTimer

kTRUE = ROOT.kTRUE
kFALSE = ROOT.kFALSE
char = ROOT.char
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
strcmp = ROOT.strcmp
Vertex_t = ROOT.Tessellated.Vertex_t

TObject = ROOT.TObject
nullptr = ROOT.nullptr
gRandom = ROOT.gRandom

Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine
TString = ROOT.TString
vector = ROOT.std.vector
std = ROOT.std
c_double = ctypes.c_double
c_int = ctypes.c_int

TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium
TGeoPatternFinder = ROOT.TGeoPatternFinder
TGeoVolume = ROOT.TGeoVolume
TGeoTranslation = ROOT.TGeoTranslation

TProcesssEventTimer = ROOT.TProcessEventTimer 


global gGeoManager, gPad
gPad = ROOT.gPad
gGeoManager = ROOT.gGeoManager

#FLAGS
comments = kTRUE
raytracing = kFALSE
grotate = kFALSE
axis = kTRUE
RndmColors = kFALSE 

# most useful global variables
c = mat = med = top = Slice = pt = text = vol  = None
vol = slicex = slicey = None
pgon = None
myMatrixObj = _alignment = None
c_help = None



# printf is a C function for formatted strings.
# redefining it at here.
def printf(string, *args):
   print(string.format(args))


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

#Since TGeoMaterial isn´t a fully Pythonized class.
#We will define a function for its constructor.
# The constructor TGeoMaterial(name, A, Z, rho) crashes:
"""
#   TGeoMaterial::TGeoMaterial(const char* name, double a, double z, double rho, double radlen = 0, double intlen = 0) =>
    SegmentationViolation: segfault in C++; program state was reset
"""

def makeTGeoMaterial(name, a, z, rho):
   mat = TGeoMaterial(name)
   mat.SetA(a)
   mat.SetZ(z)
   mat.SetDensity(rho)
   mat.Print()
   return mat

   DelROOTObjs()


#___________________________________
def MakePicture():
   global gPad, gGeoManager, view, is_raytracing   

   if gPad and gGeoManager: 
      view = gPad.GetView()
      if view:
            #view.RotateView(248,66)
         if (axis):
            view.ShowAxis()
         
      is_raytracing = gGeoManager.GetTopVolume().IsRaytracing()
      if (is_raytracing != raytracing):
         gGeoManager.GetTopVolume().SetVisRaytrace(raytracing)
         gPad.Modified()
         gPad.Update()
      
   else :
      print("No global Pad found! Define a Canvas First")
      print("Or No global GeoManager found! Define your Geometry First")
      print(" You can Make a Picture if you have nothing to show")

   
   DelROOTObjs()


#___________________________________
def AddMemberInfo(pave : TPaveText, datamember:char, value : Double_t, comment : char):
   global line, text
   line = TString(datamember)
   while (line.Length() < 10):
      line.Append(" ")
   line.Append(("= {:5.2f} => {:s}".format(value, comment)))
   text = pave.AddText(line.Data())
   #text.SetTextColor(4)
   text.SetTextAlign(12)
   

   DelROOTObjs()


#___________________________________
def AddMemberInfo(pave = TPaveText(), datamember = char(), value = Int_t(), comment = char()):
   global line, text
   line = TString(datamember)
   while (line.Length() < 10) :
      line.Append(" ")
   line.Append(("= {:5.2f} => {:s}".format(value, comment)))
   text = pave.AddText(line.Data())
   #text.SetTextColor(4)
   text.SetTextAlign(12)
   

   DelROOTObjs()


#___________________________________
def AddFinderInfo( pave = TPaveText(), pf = TObject(), iaxis = Int_t()):
   global finder, volume, sh, text
   #No: finder = TGeoPatternFinder(pf) #dynamic_cast
   #No: finder = ROOT.BindObject(pf, TGeoPatternFinder ) #
   # We use a pythonic dynamic caster. Which is more-or-less equivalent.
   # See documentation of typying-module. Specifically, see cast-method for more information.
   finder = cast(pf, TGeoPatternFinder)
   if not pave or not pf or not iaxis: 
      return
   volume = finder.GetVolume()
   sh = volume.GetShape()
   text = pave.AddText(("Division of {:s} on axis {:d} ({:s})".format(volume.GetName(), iaxis, sh.GetAxisName(iaxis))))
   text.SetTextColor(3)
   text.SetTextAlign(12)
   AddMemberInfo(pave, "fNdiv", finder.GetNdiv(), "number of divisions")
   AddMemberInfo(pave, "fStart", finder.GetStart(), "start divisioning position")
   AddMemberInfo(pave, "fStep", finder.GetStep(), "division step")
   

   DelROOTObjs()


#___________________________________
def AddExecInfo(pave=TPaveText() , name = nullptr, axisinfo = nullptr):
   global text 
   #pave:TPaveText is a mutable object. No need for global pave.
   if name and axisinfo:
      text = pave.AddText(("Execute: {:s}(iaxis, ndiv, start, step) to divide this.".format(name)) )
      text.SetTextColor(4)
      pave.AddText(("----- IAXIS can be {:s}".format(axisinfo)))
      pave.AddText("----- NDIV must be a positive integer")
      pave.AddText("----- START must be a valid axis offset within shape range on divided axis")
      pave.AddText("----- STEP is the division step. START+NDIVSTEP must be in range also")
      pave.AddText("----- If START and STEP are omitted, all range of the axis will be divided")
      pave.SetAllWith("-----","color",2)
      pave.SetAllWith("-----","font",72)
      pave.SetAllWith("-----","size",0.04)
   elif name:
      text = pave.AddText(("Execute: {:s}()".format(name)))
      text.SetTextColor(4)
      
   
   pave.AddText(" ")
   pave.SetTextSize(0.044)
   pave.SetTextAlign(12)
   


   DelROOTObjs()


#___________________________________
def SavePicture(name : char, objcanvas : TCanvas, objvol : TObject, iaxis : Int_t, step : Double_t):

   global c, vol, fname
   if type(objcanvas) == TObject and type(objvol) == TObject :
      print("Define canvas and volume first!")
      return
 
   c = objcanvas
   vol = objvol
   if not c or not vol: 
      return
   c.cd()
   fname = TString()
   if (iaxis == 0):
      fname.Form("t_{:s}.gif".format(name))
   elif (step == 0):
      fname.Form("t_{:s}div{:s}.gif".format(name, vol.GetShape().GetAxisName(iaxis)))
   else:
      fname.Form("t_{:s}divstep{:s}.gif".format(name, vol.GetShape().GetAxisName(iaxis)))
   
   c.Print(fname.Data())
   c.SaveAs(fname.Data())
   

   DelROOTObjs()


#___________________________________
def randomColor():
   global RndmColors, gRandom
   if RndmColors:   
      color = 7. * gRandom.Rndm()
      return (1+Int_t(color))
   else :
      return 1 # Default color in TGeoVolume
   DelROOTObjs()


#___________________________________
def SwitchRandomColors():
   global RndmColors
   RndmColors = not RndmColors
   toggle = "ON" if RndmColors else "OFF"
   print(f"Random Colors {toggle}")
   DelROOTObjs()

   DelROOTObjs()


#___________________________________
def SwitchComments():
   global comments 
   comments = not comments   
   toggle = "ON" if comments else "OFF"
   print(f"Comments {toggle}")

   DelROOTObjs()


#___________________________________
def raytrace():
   global raytracing, gGeoManager, gPad, top, drawn 
   raytracing = not raytracing
   if gGeoManager and gPad:
      top = gGeoManager.GetTopVolume()
      drawn = gPad.GetListOfPrimitives().FindObject(top)
      if drawn:
         top.SetVisRaytrace(raytracing)
      gPad.Modified()
      gPad.Update()
   else:
      print("at raytrace()", "Define your Geometry and Canvas First")
      return      

   

   DelROOTObjs()


#___________________________________
def Help():
   global c_help, welcome, hdemo 
   c_help = ROOT.gROOT.GetListOfCanvases().FindObject("geom_help")
   if c_help:
      c_help.Clear()
      c_help.Update()
      c_help.cd()
   else:
      c_help =  TCanvas("geom_help","Help to run demos",200,10,700,600)
      
   
   welcome =  TPaveText(.1,.8,.9,.97)
   welcome.AddText("Welcome to the new geometry package")
   welcome.SetTextFont(32)
   welcome.SetTextColor(4)
   welcome.SetFillColor(24)
   welcome.Draw()
   
   hdemo =  TPaveText(.05,.05,.95,.7)
   hdemo.SetTextAlign(12)
   hdemo.SetTextFont(52)
   hdemo.AddText("- Demo for building TGeo basic shapes and simple geometry. Shape parameters are")
   hdemo.AddText("  displayed in the right pad")
   hdemo.AddText("- Click left mouse button to execute one demo")
   hdemo.AddText("- While pointing the mouse to the pad containing the geometry, do:")
   hdemo.AddText("- .... click-and-move to rotate")
   hdemo.AddText("- .... press j/k to zoom/unzoom")
   hdemo.AddText("- .... press l/h/u/i to move the view center around")
   hdemo.AddText("- Click Ray-trace ON/OFF to toggle ray-tracing")
   hdemo.AddText("- Use <View with x3d> from the <View> menu to get an x3d view")
   hdemo.AddText("- .... same methods to rotate/zoom/move the view")
   hdemo.AddText("- Execute box(1,8) to divide a box in 8 equal slices along X")
   hdemo.AddText("- Most shapes can be divided on X,Y,Z,Rxy or Phi :")
   hdemo.AddText("- .... root[0] <shape>(IAXIS, NDIV, START, STEP);")
   hdemo.AddText("  .... IAXIS = 1,2,3 meaning (X,Y,Z) or (Rxy, Phi, Z)")
   hdemo.AddText("  .... NDIV  = number of slices")
   hdemo.AddText("  .... START = start slicing position")
   hdemo.AddText("  .... STEP  = division step")
   hdemo.AddText("- Click Comments ON/OFF to toggle comments")
   hdemo.AddText("- Click Ideal/Align geometry to see how alignment works")
   hdemo.AddText(" ")
   hdemo.SetAllWith("....","color",2)
   hdemo.SetAllWith("....","font",72)
   hdemo.SetAllWith("....","size",0.03)
   
   hdemo.Draw()
   
   c_help.Update()
   

   DelROOTObjs()


#___________________________________
def autorotate():

   global grotate, gPad, view, gGeoManager, painter, longit, dphi, irep, timer, grotate
   grotate = not grotate
   if not grotate:
      ROOT.gROOT.SetInterrupt(kTRUE)
      return
      
   if not gPad: 
      return
   view = gPad.GetView()
   if not view:
      return
   if not gGeoManager:
      return
   painter = gGeoManager.GetGeomPainter()
   if not painter:
      return
   longit = view.GetLongitude()
   #lat = view.GetLatitude()
   #psi = view.GetPsi()
   dphi = 1.
   irep = c_int() 
   timer =  TProcessEventTimer(5)
   ROOT.gROOT.SetInterrupt(kFALSE)
   while (grotate):
      if timer.ProcessEvents():
         break
      if ROOT.gROOT.IsInterrupted():
         break
      longit += dphi
      longit -= 360.
      if not gPad:
         grotate = kFALSE
         return
         
      view = gPad.GetView()
      if not view:
         grotate = kFALSE
         return
         
      view.SetView(longit,view.GetLatitude(),view.GetPsi(),irep)
      gPad.Modified()
      gPad.Update()

   del timer
   

   DelROOTObjs()


#___________________________________
def axes():
   global axis, view, gPad
   axis = not axis
   view = gPad.GetView() if gPad else nullptr
   if(view):
      view.ShowAxis()
   toggle = "ON" if (axis) else "OFF" 
   if not view: 
      print(f"Axes are {toggle}. But run your geometry first.")
   else:
      print(f"Axes are {toggle}. ")
   

   DelROOTObjs()


#___________________________________
def create_canvas(title = char()):
   global c, comments
   c = ROOT.gROOT.GetListOfCanvases().FindObject("geom_draw")
   if c:
      c.Clear()
      c.Update()
      c.SetTitle(title)
   else:
      #c =  TCanvas("geom_draw", title, 700, 1000)
      c =  TCanvas("geom_draw", title, 500, 700)
      
   if comments:
      c.Divide(1,2,0,0)
      c.GetPad(2).SetPad(0,0,1,0.4)
      c.GetPad(1).SetPad(0,0.4,1,1)
      c.cd(1)
      
   return c
   

   DelROOTObjs()


#___________________________________
def box(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, vol, _box, text 
   global vol 
   c = create_canvas("A simple box")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("box", "A simple box")
   TGeoManager("box", "A simple box")
 
   #Note: Define your material this way:
   #global mat
   mat =  TGeoMaterial("Al") #, 26.98,13,2.7)
   mat.SetA(26.98)
   mat.SetZ(13.)
   mat.SetDensity(2.7)
   #Not to use: mat = TGeoMaterial("Al", 26.98,13,2.7) 
   #BP mat =  TGeoMaterial("Al", 26.98,13,2.7)
   global med, top
   med =  TGeoMedium("MED", 1, mat)
   top = gGeoManager.MakeBox("TOP", med , 100, 100, 100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeBox("BOX",med, 20,30,40)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis > 0) and (iaxis < 4):
      global Slice
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice:
         return
      Slice.SetLineColor(randomColor())
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()

   if not comments:
      return

   c.cd(2)
   global pt
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   global _box
   _box = vol.GetShape()
   global text
   text = pt.AddText("TGeoBBox - box class")
   
   text.SetTextColor(2)
   AddMemberInfo(pt, "fDX", _box.GetDX(), "half length in X")
   AddMemberInfo(pt, "fDY", _box.GetDY(), "half length in Y")
   AddMemberInfo(pt, "fDZ", _box.GetDZ(), "half length in Z")
   AddMemberInfo(pt, "fOrigin[0]", (_box.GetOrigin())[0], "box origin on X")
   AddMemberInfo(pt, "fOrigin[1]", (_box.GetOrigin())[1], "box origin on Y")
   AddMemberInfo(pt, "fOrigin[2]", (_box.GetOrigin())[2], "box origin on Z")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "box", "1, 2 or 3 (X, Y, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.Update()
   c.Draw()
   #SavePicture("box",c,vol,iaxis,step)
   c.cd(1)
   
   # Print Status of function:
   print("box() function works fine!.")
   


   DelROOTObjs()


#___________________________________
def para(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _para

   c = create_canvas("A parallelepiped")
   
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")

   #new TGeoManager("para", "A parallelepiped")
   TGeoManager("para", "A parallelepiped")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakePara("PARA",med, 20,30,40,30,15,30)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis > 0) and (iaxis < 4):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice:
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 1-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments:
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   #para = (TGeoPara ) vol.GetShape()
   para = vol.GetShape()
   text = pt.AddText("TGeoPara - parallelepiped class")
    
   text.SetTextColor(2)
   AddMemberInfo(pt, "fX", para.GetX(), "half length in X")
   AddMemberInfo(pt, "fY", para.GetY(), "half length in Y")
   AddMemberInfo(pt, "fZ", para.GetZ(), "half length in Z")
   AddMemberInfo(pt, "fAlpha", para.GetAlpha(), "angle about Y of the Z bases")
   AddMemberInfo(pt, "fTheta", para.GetTheta(), "inclination of para axis about Z")
   AddMemberInfo(pt, "fPhi", para.GetPhi(), "phi angle of para axis")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "para", "1, 2 or 3 (X, Y, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("para",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def tube(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _tube
   
   c = create_canvas("A tube")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("tube", "poza2")
   TGeoManager("tube", "poza2")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeTube("TUBE",med, 20,30,40)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis > 0) and (iaxis < 4):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 1-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   tube = (vol.GetShape())
   #TText *text = pt->AddText("TGeoTube - tube class");
   text = pt.AddText("TGeoTube - tube class")
   # text = ROOT.text 
    
   text.SetTextColor(2)
   AddMemberInfo(pt,"fRmin",tube.GetRmin(),"minimum radius")
   AddMemberInfo(pt,"fRmax",tube.GetRmax(),"maximum radius")
   AddMemberInfo(pt,"fDZ",  tube.GetDZ(),  "half length in Z")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "tube", "1, 2 or 3 (Rxy, Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("tube",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def tubeseg(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _tubeseg
   
   c = create_canvas("A tube segment")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("tubeseg", "poza3")
   TGeoManager("tubeseg", "poza3")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeTubs("TUBESEG",med, 20,30,40,-30,270)
   vol.SetLineColor(randomColor())
   if (iaxis > 0) and (iaxis < 4):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 1-3.\n", iaxis)
      return
      
   
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   tubeseg = (vol.GetShape())
   text = pt.AddText("TGeoTubeSeg - tube segment class")
   #TTert *text = pt->AddText("TGeoTubeSeg - tube segment class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fRmin",tubeseg.GetRmin(),"minimum radius")
   AddMemberInfo(pt,"fRmax",tubeseg.GetRmax(),"maximum radius")
   AddMemberInfo(pt,"fDZ",  tubeseg.GetDZ(),  "half length in Z")
   AddMemberInfo(pt,"fPhi1",tubeseg.GetPhi1(),"first phi limit")
   AddMemberInfo(pt,"fPhi2",tubeseg.GetPhi2(),"second phi limit")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "tubeseg", "1, 2 or 3 (Rxy, Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("tubeseg",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def ctub(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _ctub

   c = create_canvas("A cut tube segment")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("ctub", "poza3")
   TGeoManager("ctub", "poza3")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   theta = 160. * TMath.Pi()/180.
   phi = 30. * TMath.Pi()/180.
   #TODO Double_t nlow[3]
   nlow =[Double_t()]*3 
   nlow[0] = TMath.Sin(theta)*TMath.Cos(phi)
   nlow[1] = TMath.Sin(theta)*TMath.Sin(phi)
   nlow[2] = TMath.Cos(theta)
   theta = 20. * TMath.Pi()/180.
   phi = 60. * TMath.Pi()/180.
   #TODO Double_t nhi[3]
   nhi = [Double_t]*3 
   nhi[0] = TMath.Sin(theta)*TMath.Cos(phi)
   nhi[1] = TMath.Sin(theta)*TMath.Sin(phi)
   nhi[2] = TMath.Cos(theta)
   vol = gGeoManager.MakeCtub("CTUB",med, 20,30,40,-30,250, nlow[0], nlow[1], nlow[2], nhi[0],nhi[1],nhi[2])
   vol.SetLineColor(randomColor())
   if (iaxis == 1 or iaxis == 2):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 1-2.\n", iaxis)
      return
      
   
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   tubeseg = (vol.GetShape())
   text = pt.AddText("TGeoTubeSeg - tube segment class")
   #TText *text = pt->AddText("TGeoTubeSeg - tube segment class");
   # text = ROOT.text 
    
   text.SetTextColor(2)
   AddMemberInfo(pt,"fRmin",tubeseg.GetRmin(),"minimum radius")
   AddMemberInfo(pt,"fRmax",tubeseg.GetRmax(),"maximum radius")
   AddMemberInfo(pt,"fDZ",  tubeseg.GetDZ(),  "half length in Z")
   AddMemberInfo(pt,"fPhi1",tubeseg.GetPhi1(),"first phi limit")
   AddMemberInfo(pt,"fPhi2",tubeseg.GetPhi2(),"second phi limit")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "ctub", "1 or 2")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("tubeseg",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def cone(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _cone
   
   c = create_canvas("A cone")

   global gGeoManager 
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("cone", "poza4")
   TGeoManager("cone", "poza4")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeCone("CONE",med, 40,10,20,35,45)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   if (iaxis == 2 or iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 2-3.\n", iaxis)
      return
      
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   global gvol
   gvol = vol
   cone = (vol.GetShape())
   text = pt.AddText("TGeoCone - cone class")
   #TText *text = pt->AddText("TGeoCone - cone class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fDZ",  cone.GetDZ(),    "half length in Z")
   AddMemberInfo(pt,"fRmin1",cone.GetRmin1(),"inner radius at -dz")
   AddMemberInfo(pt,"fRmax1",cone.GetRmax1(),"outer radius at -dz")
   AddMemberInfo(pt,"fRmin2",cone.GetRmin2(),"inner radius at +dz")
   AddMemberInfo(pt,"fRmax2",cone.GetRmax2(),"outer radius at +dz")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "cone", "2 or 3 (Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("cone",c,vol,iaxis,step)


   DelROOTObjs()


#___________________________________
def coneseg(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _coneseg
   
   c = create_canvas("A cone segment")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("coneseg", "poza5")
   TGeoManager("coneseg", "poza5")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeCons("CONESEG",med, 40,30,40,10,20,-30,250)
   vol.SetLineColor(randomColor())
   #   vol->SetLineWidth(2);
   top.AddNode(vol,1)
   if (iaxis >= 2 and iaxis <= 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 2-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   coneseg = (vol.GetShape())
   text = pt.AddText("TGeoConeSeg - coneseg class")
   #TText *text = pt->AddText("TGeoConeSeg - coneseg class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt, "fDZ", coneseg.GetDZ(), "half length in Z")
   AddMemberInfo(pt, "fRmin1", coneseg.GetRmin1(), "inner radius at -dz")
   AddMemberInfo(pt, "fRmax1", coneseg.GetRmax1(), "outer radius at -dz")
   AddMemberInfo(pt, "fRmin2", coneseg.GetRmin1(), "inner radius at +dz")
   AddMemberInfo(pt, "fRmax2", coneseg.GetRmax1(), "outer radius at +dz")
   AddMemberInfo(pt, "fPhi1", coneseg.GetPhi1(), "first phi limit")
   AddMemberInfo(pt, "fPhi2", coneseg.GetPhi2(), "second phi limit")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "coneseg", "2 or 3 (Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("coneseg",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def eltu(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _eltu
   
   c = create_canvas("An Elliptical tube")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("eltu", "poza6")
   TGeoManager("eltu", "poza6")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeEltu("ELTU",med, 30,10,40)
   vol.SetLineColor(randomColor())
   #vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if(iaxis >= 2 and iaxis <= 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 2-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   eltu = (vol.GetShape())
   text = pt.AddText("TGeoEltu - eltu class")
   #TText *text = pt->AddText("TGeoEltu - eltu class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fA",eltu.GetA(), "semi-axis along x")
   AddMemberInfo(pt,"fB",eltu.GetB(), "semi-axis along y")
   AddMemberInfo(pt,"fDZ", eltu.GetDZ(),  "half length in Z")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "eltu", "2 or 3 (Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("eltu",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def sphere():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _sphere
   
   c = create_canvas("A spherical sector")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("sphere", "poza7")
   TGeoManager("sphere", "poza7")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeSphere("SPHERE",med, 30,40,60,120,30,240)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   #TGeoSphere *sphere = (TGeoSphere*)(vol->GetShape());
   sphere = (vol.GetShape())
   text = pt.AddText("TGeoSphere- sphere class")
   #TText *text = pt->AddText("TGeoSphere- sphere class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt, "fRmin", sphere.GetRmin(), "inner radius")
   AddMemberInfo(pt, "fRmax", sphere.GetRmax(), "outer radius")
   AddMemberInfo(pt, "fTheta1", sphere.GetTheta1(), "lower theta limit")
   AddMemberInfo(pt, "fTheta2", sphere.GetTheta2(), "higher theta limit")
   AddMemberInfo(pt, "fPhi1", sphere.GetPhi1(), "lower phi limit")
   AddMemberInfo(pt, "fPhi2", sphere.GetPhi2(), "higher phi limit")
   AddExecInfo(pt, "sphere")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("sphere",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def torus():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _torus
   
   c = create_canvas("A toroidal segment")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("torus", "poza2")
   TGeoManager("torus", "poza2")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeTorus("TORUS",med, 40,20,25,0,270)
   vol.SetLineColor(randomColor())
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   tor = (vol.GetShape())
   text = pt.AddText("TGeoTorus - torus class")
   #TText *text = pt->AddText("TGeoTorus - torus class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt, "fR", tor.GetR(), "radius of the ring")
   AddMemberInfo(pt, "fRmin", tor.GetRmin(), "minimum radius")
   AddMemberInfo(pt, "fRmax", tor.GetRmax(), "maximum radius")
   AddMemberInfo(pt, "fPhi1", tor.GetPhi1(), "starting phi angle")
   AddMemberInfo(pt, "fDphi", tor.GetDphi(), "phi range")
   AddExecInfo(pt, "torus")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)


   DelROOTObjs()


#___________________________________
def trd1(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _trd1
   
   c = create_canvas("A trapezoid with dX varying")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("trd1", "poza8")
   TGeoManager("trd1", "poza8")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeTrd1("Trd1",med, 10,20,30,40)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis == 2 or iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice:
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed range is 2-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   #trd1 = (TGeoTrd1 ) vol.GetShape()
   trd1 = vol.GetShape()
   text = pt.AddText("TGeoTrd1 - Trd1 class")
   #TText *text = pt->AddText("TGeoTrd1 - Trd1 class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fDx1",trd1.GetDx1(),"half length in X at lower Z surface(-dz)")
   AddMemberInfo(pt,"fDx2",trd1.GetDx2(),"half length in X at higher Z surface(+dz)")
   AddMemberInfo(pt,"fDy",trd1.GetDy(),"half length in Y")
   AddMemberInfo(pt,"fDz",trd1.GetDz(),"half length in Z")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "trd1", "2 or 3 (Y, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("trd1",c,vol,iaxis,step)


   DelROOTObjs()


#___________________________________
def parab():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _parab
   
   c = create_canvas("A paraboloid segment")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("parab", "paraboloid")
   TGeoManager("parab", "paraboloid")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeParaboloid("PARAB",med,0, 40, 50)
   #par = vol.GetShape()
   par = vol.GetShape()
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoParaboloid - Paraboloid class")
   #TText *text = pt->AddText("TGeoParaboloid - Paraboloid class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fRlo",par.GetRlo(),"radius at Z=-dz")
   AddMemberInfo(pt,"fRhi",par.GetRhi(),"radius at Z=+dz")
   AddMemberInfo(pt,"fDz",par.GetDz(),"half-length on Z axis")
   pt.AddText("----- A paraboloid is described by the equation:")
   pt.AddText("-----    z = a*r*r + b;   where: r = x*x + y*y")
   pt.AddText("----- Create with:    TGeoParaboloid *parab = new TGeoParaboloid(rlo, rhi, dz);")
   pt.AddText("-----    dz:  half-length in Z (range from -dz to +dz")
   pt.AddText("-----    rlo: radius at z=-dz given by: -dz = a*rlo*rlo + b")
   pt.AddText("-----    rhi: radius at z=+dz given by:  dz = a*rhi*rhi + b")
   pt.AddText("-----      rlo != rhi; both >= 0")
   AddExecInfo(pt, "parab")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)


   DelROOTObjs()


#___________________________________
def hype():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _hype
   
   c = create_canvas("A hyperboloid")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("hype", "hyperboloid")
   TGeoManager("hype", "hyperboloid")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeHype("HYPE",med,10, 45 ,20,45,40)
   #hype = vol.GetShape()
   hype = vol.GetShape()
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoHype - Hyperboloid class")
   #TText *text = pt->AddText("TGeoHype - Hyperboloid class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt, "fRmin", hype.GetRmin(), "minimum inner radius")
   AddMemberInfo(pt, "fStIn", hype.GetStIn(), "inner surface stereo angle [deg]")
   AddMemberInfo(pt, "fRmax", hype.GetRmax(), "minimum outer radius")
   AddMemberInfo(pt, "fStOut",hype.GetStOut(),"outer surface stereo angle [deg]")
   AddMemberInfo(pt, "fDz",   hype.GetDz(),   "half-length on Z axis")
   pt.AddText("----- A hyperboloid is described by the equation:")
   pt.AddText("-----    r^2 - (tan(stereo)*z)^2 = rmin^2;   where: r = x*x + y*y")
   pt.AddText("----- Create with:    TGeoHype *hype = new TGeoHype(rin, stin, rout, stout, dz);")
   pt.AddText("-----      rin < rout; rout > 0")
   pt.AddText("-----      rin = 0; stin > 0 => inner surface conical")
   pt.AddText("-----      stin/stout = 0 => corresponding surface cylindrical")
   AddExecInfo(pt, "hype")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)


   DelROOTObjs()


#___________________________________
def pcon(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _pcon
   
   c = create_canvas("A polycone")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("pcon", "poza10")
   TGeoManager("pcon", "poza10")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakePcon("PCON",med, -30.0,300,4)
   pcon = (vol.GetShape())
   pcon.DefineSection(0,0,15,20)
   pcon.DefineSection(1,20,15,20)
   pcon.DefineSection(2,20,15,25)
   pcon.DefineSection(3,50,15,20)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis == 2 or iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   if iaxis: # TODO:in C++ version is if. Shouldn't it be elif?
      printf("Wrong division axis {:d}. Allowed range is 2-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoPcon - pcon class")
   #TText *text = pt->AddText("TGeoPcon - pcon class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fPhi1",pcon.GetPhi1(),"lower phi limit")
   AddMemberInfo(pt,"fDphi",pcon.GetDphi(),"phi range")
   AddMemberInfo(pt,"fNz",pcon.GetNz(),"number of z planes")

   for j in range(pcon.GetNz()) :
      line = TString(\
            "fZ[{}]={:5.2f}  fRmin[{}]={:5.2f}  fRmax[{}]={:5.2f}".format(
            j, pcon.GetZ()[j], j, pcon.GetRmin()[j], j, pcon.GetRmax()[j] )
      )
      text = pt.AddText(line.Data())
      text.SetTextColor(4)
      text.SetTextAlign(12)
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "pcon", "2 or 3 (Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("pcon",c,vol,iaxis,step)


   DelROOTObjs()


#___________________________________
def pgon(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _pgon
   
   c = create_canvas("A polygone")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("pgon", "poza11")
   TGeoManager("pgon", "poza11")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,150,150,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakePgon("PGON",med, -45.0,270.0,4,4)
   pgon = (vol.GetShape())
   pgon.DefineSection(0,-70,45,50)
   pgon.DefineSection(1,0,35,40)
   pgon.DefineSection(2,0,30,35)
   pgon.DefineSection(3,70,90,100)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis == 2 or iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: return
      Slice.SetLineColor(randomColor())
   if iaxis: # TODO: shouldn´t it be elif?
      printf("Wrong division axis {:d}. Allowed range is 2-3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoPgon - pgon class")
   #TText *text = pt->AddText("TGeoPgon - pgon class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt, "fPhi1",  pgon.GetPhi1(),  "lower phi limit")
   AddMemberInfo(pt, "fDphi",  pgon.GetDphi(),  "phi range")
   AddMemberInfo(pt, "fNedges",pgon.GetNedges(),"number of edges")
   AddMemberInfo(pt, "fNz",    pgon.GetNz(),    "number of z planes")
   for j in range(pgon.GetNz() ):
      line = TString(
          "fZ[{}]={:5.2f}  fRmin[{}]={:5.2f}  fRmax[{}]={:5.2f}".format( 
          j,pgon.GetZ()[j],j,pgon.GetRmin()[j],j,pgon.GetRmax()[j])
      )
      text = pt.AddText(line.Data())
      text.SetTextColor(4)
      text.SetTextAlign(12)
      
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "pgon", "2 or 3 (Phi, Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("pgon",c,vol,iaxis,step);
   
   
   DelROOTObjs()


#___________________________________
def arb8():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _arb8
   global vert, vol, arb
   
   c = create_canvas("An arbitrary polyhedron")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("arb8", "poza12")
   _arb8 = TGeoManager("arb8", "poza12")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   arb =  TGeoArb8(20)
   arb.SetVertex(0,-30,-25)
   arb.SetVertex(1,-25,25)
   arb.SetVertex(2,5,25)
   arb.SetVertex(3,25,-25)
   arb.SetVertex(4,-28,-23)
   arb.SetVertex(5,-23,27)
   arb.SetVertex(6,-23,27)
   arb.SetVertex(7,13,-27)
   vol =  TGeoVolume("ARB8",arb,med)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoArb8 - arb8 class")
   #TText *text = pt->AddText("TGeoArb8 - arb8 class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fDz",arb.GetDz(),"Z half length")
   vert = arb.GetVertices()
   text = pt.AddText("Vertices on lower Z plane:")
   text.SetTextColor(3)
   for i in range(0, 4):
      text = pt.AddText( (" fXY[{:d}] = ({:5.2f}, {:5.2f})".format(
      i, vert[2*i], vert[2*i+1]))
      )
      text.SetTextSize(0.043)
      text.SetTextColor(4)
      
   text = pt.AddText("Vertices on higher Z plane:")
   text.SetTextColor(3)
   for i in range(4, 8): 
      text = pt.AddText( (" fXY[{:d}] = ({:5.2f}, {:5.2f})".format( 
      i, vert[2*i], vert[2*i+1]))
      )
      text.SetTextSize(0.043)
      text.SetTextColor(4)
      
   
   AddExecInfo(pt, "arb8")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("arb8",c,vol,iaxis,step)


   DelROOTObjs()


#___________________________________
def trd2(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _trd2
   global vol, tr2d
   
   c = create_canvas("A trapezoid with dX and dY varying with Z")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("trd2", "poza9")
   TGeoManager("trd2", "poza9")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeTrd2("Trd2",med, 10,20,30,10,40)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed is only 3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   _trd2 = (vol.GetShape())
   text = pt.AddText("TGeoTrd2 - Trd2 class")
   #TText *text = pt->AddText("TGeoTrd2 - Trd2 class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fDx1",_trd2.GetDx1(),"half length in X at lower Z surface(-dz)")
   AddMemberInfo(pt,"fDx2",_trd2.GetDx2(),"half length in X at higher Z surface(+dz)")
   AddMemberInfo(pt,"fDy1",_trd2.GetDy1(),"half length in Y at lower Z surface(-dz)")
   AddMemberInfo(pt,"fDy2",_trd2.GetDy2(),"half length in Y at higher Z surface(-dz)")
   AddMemberInfo(pt,"fDz",_trd2.GetDz(),"half length in Z")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "_trd2", "only 3 (Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("_trd2",c,vol,iaxis,step);


   DelROOTObjs()


#___________________________________
def trap(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _trap
   global vol 
   
   c = create_canvas("A more general trapezoid")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("trap", "poza10")
   TGeoManager("trap", "poza10")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeTrap("Trap",med, 30,15,30,20,10,15,0,20,10,15,0)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed is only 3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   _trap = (vol.GetShape())
   text = pt.AddText("TGeoTrap - Trapezoid class")
   #TText *text = pt->AddText("TGeoTrap - Trapezoid class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fDz",_trap.GetDz(),"half length in Z")
   AddMemberInfo(pt,"fTheta",_trap.GetTheta(),"theta angle of trapezoid axis")
   AddMemberInfo(pt,"fPhi",_trap.GetPhi(),"phi angle of trapezoid axis")
   AddMemberInfo(pt,"fH1",_trap.GetH1(),"half length in y at -fDz")
   AddMemberInfo(pt,"fAlpha1",_trap.GetAlpha1(),"angle between centers of x edges and y axis at -fDz")
   AddMemberInfo(pt,"fBl1",_trap.GetBl1(),"half length in x at -dZ and y=-fH1")
   AddMemberInfo(pt,"fTl1",_trap.GetTl1(),"half length in x at -dZ and y=+fH1")
   AddMemberInfo(pt,"fH2",_trap.GetH2(),"half length in y at +fDz")
   AddMemberInfo(pt,"fBl2",_trap.GetBl2(),"half length in x at +dZ and y=-fH1")
   AddMemberInfo(pt,"fTl2",_trap.GetTl2(),"half length in x at +dZ and y=+fH1")
   AddMemberInfo(pt,"fAlpha2",_trap.GetAlpha2(),"angle between centers of x edges and y axis at +fDz")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "trap", "only 3 (Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("_trap",c,vol,iaxis,step)


   DelROOTObjs()


#___________________________________
def gtra(iaxis = Int_t(0), ndiv = Int_t(8), start = Double_t(0), step = Double_t(0)):
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _gtra
   global vol, Slice
   
   c = create_canvas("A twisted trapezoid")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("gtra", "poza11")
   TGeoManager("gtra", "poza11")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeGtra("Gtra",med, 30,15,30,30,20,10,15,0,20,10,15,0)
   print("vol", vol)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   if (iaxis == 3):
      Slice = vol.Divide("SLICE",iaxis,ndiv,start,step)
      if not Slice: 
         return
      Slice.SetLineColor(randomColor())
   elif iaxis:
      printf("Wrong division axis {:d}. Allowed is only 3.\n", iaxis)
      return
      
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   _gtrap = (vol.GetShape())
   text = pt.AddText("TGeoGtra - Twisted trapezoid class")
   #TText *text = pt->AddText("TGeoGtra - Twisted trapezoid class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fDz",_gtrap.GetDz(),"half length in Z")
   AddMemberInfo(pt,"fTheta",_gtrap.GetTheta(),"theta angle of trapezoid axis")
   AddMemberInfo(pt,"fPhi",_gtrap.GetPhi(),"phi angle of trapezoid axis")
   AddMemberInfo(pt,"fTwist",_gtrap.GetTwistAngle(), "twist angle")
   AddMemberInfo(pt,"fH1",_gtrap.GetH1(),"half length in y at -fDz")
   AddMemberInfo(pt,"fAlpha1",_gtrap.GetAlpha1(),"angle between centers of x edges and y axis at -fDz")
   AddMemberInfo(pt,"fBl1",_gtrap.GetBl1(),"half length in x at -dZ and y=-fH1")
   AddMemberInfo(pt,"fTl1",_gtrap.GetTl1(),"half length in x at -dZ and y=+fH1")
   AddMemberInfo(pt,"fH2",_gtrap.GetH2(),"half length in y at +fDz")
   AddMemberInfo(pt,"fBl2",_gtrap.GetBl2(),"half length in x at +dZ and y=-fH1")
   AddMemberInfo(pt,"fTl2",_gtrap.GetTl2(),"half length in x at +dZ and y=+fH1")
   AddMemberInfo(pt,"fAlpha2",_gtrap.GetAlpha2(),"angle between centers of x edges and y axis at +fDz")
   AddFinderInfo(pt, vol.GetFinder(), iaxis)
   AddExecInfo(pt, "gtra", "only 3 (Z)")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   #   SavePicture("gtra",c,vol,iaxis,step)


   DelROOTObjs()


#___________________________________
def xtru():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _xtru
   global vol, xtru, x, y, c_x, c_y
   
   c = create_canvas("A twisted trapezoid")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("xtru", "poza12")
   TGeoManager("xtru", "poza12")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   gGeoManager.SetTopVolume(top)
   vol = gGeoManager.MakeXtru("XTRU",med,4)
   _xtru = vol.GetShape()
   x = [-30,-30,30,30,15,15,-15,-15]
   y = [-30,30,30,-30,-30,15,15,-30]
   c_x = (c_double*8)(*x)
   c_y = (c_double*8)(*y)
   _xtru.DefinePolygon(8,c_x,c_y)
   _xtru.DefineSection(0,-40, -20., 10., 1.5)
   _xtru.DefineSection(1, 10, 0., 0., 0.5)
   _xtru.DefineSection(2, 10, 0., 0., 0.7)
   _xtru.DefineSection(3, 40, 10., 20., 0.9)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(80)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoXtru - Polygonal extrusion class")
   #TText *text = pt->AddText("TGeoXtru - Polygonal extrusion class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fNvert",_xtru.GetNvert(),"number of polygone vertices")
   AddMemberInfo(pt,"fNz",_xtru.GetNz(),"number of Z sections")
   pt.AddText("----- Any Z section is an arbitrary polygone")
   pt.AddText("----- The shape can have an arbitrary number of Z sections, as for pcon/pgon")
   pt.AddText("----- Create with:    _xtru = TGeoXtru(nz) # nz is number of Z sections.")
   pt.AddText("----- Define the blueprint polygon :")
   pt.AddText("-----                 x = [-30,-30,30,30,15,15,-15,-15]")
   pt.AddText("-----                 y = [-30,30,30,-30,-30,15,15,-30]")
   pt.AddText("-----                 _tru.DefinePolygon(8,x,y)")
   pt.AddText("----- Define translations/scales of the blueprint for Z sections :")
   pt.AddText("-----                 _xtru.DefineSection(i, Zsection, x0, y0, scale);")
   pt.AddText("----- Sections have to be defined in increasing Z order")
   pt.AddText("----- 2 sections can be defined at same Z (not for first/last sections)")
   AddExecInfo(pt, "xtru")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)


   DelROOTObjs()


#___________________________________
def tessellated():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _tessellated
   global tsl, sqrt5, vert, vol
   
   # Create a [triacontahedron solid](https://en.wikipedia.org/wiki/Rhombic_triacontahedron)
   
   
   c = create_canvas("A tessellated shape")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #new TGeoManager("tessellated", "tessellated")
   TGeoManager("tessellated", "tessellated")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,10,10,10)
   gGeoManager.SetTopVolume(top)
   tsl =  TGeoTessellated("triaconthaedron", 30)
   sqrt5 = TMath.Sqrt(5.)
   # Tranlation of C++ line : std.vector<Tessellated.Vertex_t> vert
   #                   into :
   vert = vector(Vertex_t)()
   vert.reserve(120)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), -1)
   vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5))
   vert.emplace_back(-1, 1, -1)
  
   vert.emplace_back(1, 1, -1)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), -1)
   vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5))
   vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5))
  
   vert.emplace_back(1, 1, -1)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), -1)
   vert.emplace_back(0.5 * (-1 + sqrt5),  0.5 * (1 + sqrt5), 0)
   vert.emplace_back(0.5 * (1 + sqrt5), 1, 0)
  
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5), 0)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), -1)
   vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5), 0)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), 1)
  
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5), 0)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), -1)
   vert.emplace_back(-1, 1, -1)
   vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0)
  
   vert.emplace_back(1, 1, -1)
   vert.emplace_back(0.5 * (1 + sqrt5), 1, 0)
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (1 - sqrt5))
   vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5))
  
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (1 - sqrt5))
   vert.emplace_back(0.5 * (1 + sqrt5), -1, 0)
   vert.emplace_back(1, -1, -1)
   vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5))
  
   vert.emplace_back(1, -1, -1)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1)
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5))
   vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5))
  
   vert.emplace_back(1, 0, 0.5 * (-1 - sqrt5))
   vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5))
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5))
  
   vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5), 0)
   vert.emplace_back(0.5 * (1 + sqrt5), 1, 0)
   vert.emplace_back(1, 1, 1)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), 1)
  
   vert.emplace_back(0.5 * (1 + sqrt5), 1, 0)
   vert.emplace_back(1, 1, 1)
   vert.emplace_back(1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (-1 + sqrt5))
  
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (1 - sqrt5))
   vert.emplace_back(0.5 * (1 + sqrt5), 1, 0)
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (-1 + sqrt5))
   vert.emplace_back(0.5 * (1 + sqrt5), -1, 0)
  
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5), 0)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), 1)
   vert.emplace_back(-1, 1, 1)
   vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0)
  
   vert.emplace_back(0, 0.5 * (1 + sqrt5), 1)
   vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(-1, 1, 1)
  
   vert.emplace_back(1, 1, 1)
   vert.emplace_back(0, 0.5 * (1 + sqrt5), 1)
   vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5))
   vert.emplace_back(1, 0, 0.5 * (1 + sqrt5))
  
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(0, 0.5 * (-1 + sqrt5), 0.5 * (1 + sqrt5))
   vert.emplace_back(1, 0, 0.5 * (1 + sqrt5))
  
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5))
   vert.emplace_back(1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(1, -1, 1)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1)
  
   vert.emplace_back(0.5 * (1 + sqrt5), 0, 0.5 * (-1 + sqrt5))
   vert.emplace_back(0.5 * (1 + sqrt5), -1, 0)
   vert.emplace_back(1, -1, 1)
   vert.emplace_back(1, 0, 0.5 * (1 + sqrt5))
  
   vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(-1, 1, 1)
   vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0)
   vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (-1 + sqrt5))
  
   vert.emplace_back(-1, -1, 1)
   vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (-1 + sqrt5))
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0)
  
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (1 + sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (1 + sqrt5))
   vert.emplace_back(-1, -1, 1)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1)
  
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0)
   vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (1 - sqrt5))
   vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0)
   vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (-1 + sqrt5))
  
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0)
   vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (1 - sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5))
   vert.emplace_back(-1, -1, -1)
  
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1)
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5), 0)
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0)
   vert.emplace_back(-1, -1, -1)
  
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5), 0)
   vert.emplace_back(0.5 * (-1 - sqrt5), -1, 0)
   vert.emplace_back(-1, -1, 1)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1)
  
   vert.emplace_back(-1, 1, -1)
   vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5))
   vert.emplace_back(0.5 * (-1 - sqrt5), 0, 0.5 * (1 - sqrt5))
   vert.emplace_back(0.5 * (-1 - sqrt5), 1, 0)
  
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1)
   vert.emplace_back(0, 0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5))
   vert.emplace_back(-1, 0, 0.5 * (-1 - sqrt5))
   vert.emplace_back(-1, -1, -1)
  
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1)
   vert.emplace_back(0.5 * (1 - sqrt5), 0.5 * (-1 - sqrt5), 0)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1)
   vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5), 0)
  
   vert.emplace_back(1, -1, -1)
   vert.emplace_back(0.5 * (1 + sqrt5), -1, 0)
   vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5), 0)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), -1)
  
   vert.emplace_back(0.5 * (1 + sqrt5), -1, 0)
   vert.emplace_back(1, -1, 1)
   vert.emplace_back(0, 0.5 * (-1 - sqrt5), 1)
   vert.emplace_back(0.5 * (-1 + sqrt5), 0.5 * (-1 - sqrt5), 0)
  
   
   tsl.AddFacet(vert[0], vert[1], vert[2], vert[3])
   tsl.AddFacet(vert[4], vert[7], vert[6], vert[5])
   tsl.AddFacet(vert[8], vert[9], vert[10], vert[11])
   tsl.AddFacet(vert[12], vert[15], vert[14], vert[13])
   tsl.AddFacet(vert[16], vert[17], vert[18], vert[19])
   tsl.AddFacet(vert[20], vert[21], vert[22], vert[23])
   tsl.AddFacet(vert[24], vert[25], vert[26], vert[27])
   tsl.AddFacet(vert[28], vert[29], vert[30], vert[31])
   tsl.AddFacet(vert[32], vert[35], vert[34], vert[33])
   tsl.AddFacet(vert[36], vert[39], vert[38], vert[37])
   tsl.AddFacet(vert[40], vert[41], vert[42], vert[43])
   tsl.AddFacet(vert[44], vert[45], vert[46], vert[47])
   tsl.AddFacet(vert[48], vert[51], vert[50], vert[49])
   tsl.AddFacet(vert[52], vert[55], vert[54], vert[53])
   tsl.AddFacet(vert[56], vert[57], vert[58], vert[59])
   tsl.AddFacet(vert[60], vert[63], vert[62], vert[61])
   tsl.AddFacet(vert[64], vert[67], vert[66], vert[65])
   tsl.AddFacet(vert[68], vert[71], vert[70], vert[69])
   tsl.AddFacet(vert[72], vert[73], vert[74], vert[75])
   tsl.AddFacet(vert[76], vert[77], vert[78], vert[79])
   tsl.AddFacet(vert[80], vert[81], vert[82], vert[83])
   tsl.AddFacet(vert[84], vert[87], vert[86], vert[85])
   tsl.AddFacet(vert[88], vert[89], vert[90], vert[91])
   tsl.AddFacet(vert[92], vert[93], vert[94], vert[95])
   tsl.AddFacet(vert[96], vert[99], vert[98], vert[97])
   tsl.AddFacet(vert[100], vert[101], vert[102], vert[103])
   tsl.AddFacet(vert[104], vert[107], vert[106], vert[105])
   tsl.AddFacet(vert[108], vert[111], vert[110], vert[109])
   tsl.AddFacet(vert[112], vert[113], vert[114], vert[115])
   tsl.AddFacet(vert[116], vert[117], vert[118], vert[119])
   
   vol =  TGeoVolume("TRIACONTHAEDRON", tsl, med)
   vol.SetLineColor(randomColor())
   vol.SetLineWidth(2)
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoTessellated - Tessellated shape class")
   #TText *text = pt->AddText("TGeoTessellated - Tessellated shape class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   AddMemberInfo(pt,"fNfacets",tsl.GetNfacets(),"number of facets")
   AddMemberInfo(pt,"fNvertices",tsl.GetNvertices(),"number of vertices")
   pt.AddText("----- A tessellated shape is defined by the number of facets")
   pt.AddText("-----    facets can be added using AddFacet")
   pt.AddText("----- Create with:    TGeoTessellated *tsl = new TGeoTessellated(nfacets);")
   AddExecInfo(pt, "tessellated")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)


   DelROOTObjs()


#___________________________________
def composite():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _composite
   global tr, pgon, sph, cs
   global xtru
   
   c = create_canvas("A Boolean shape composition")
   
   global gGeoManager
   if gGeoManager: 
      print("deleting previous Geometry ...")
      #Important!!!
      #Not to Use: del gGeoManager
      #Not to Use: GeoManager.Delete()
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager") 

   #new TGeoManager("xtru", "poza12")
   xtru = TGeoManager("xtru", "poza12")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,100)
   #return
   #BP return 
   
   gGeoManager.SetTopVolume(top)
   # define shape components with names
   pgon =  TGeoPgon("pg",0.,360.,6,2)
   pgon.DefineSection(0,0,0,20)
   pgon.DefineSection(1, 30,0,20)

   #new TGeoSphere("sph", 40., 45.)
   sph = TGeoSphere("sph", 40., 45.)
   
   # define named geometrical transformations with names
   tr =  TGeoTranslation("tr", 0., 0., 45.)
   # register all used transformations; otherwise TGeoCompositeShape will not find it.
   tr.RegisterYourself()
   
   # create the composite shape based on a Boolean expression
   cs =  TGeoCompositeShape("composite", "((sph:tr)*pg)")
   
   vol =  TGeoVolume("COMP",cs)
   vol.SetLineColor(randomColor())
   top.AddNode(vol,1)
   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(100)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("TGeoCompositeShape - composite shape class")
   #TText *text = pt->AddText("TGeoCompositeShape - composite shape class");
   # text = ROOT.text 
   
   text.SetTextColor(2)
   pt.AddText("----- Define the shape components and don't forget to name them")
   pt.AddText("----- Define geometrical transformations that apply to shape components")
   pt.AddText("----- Name all transformations and register them")
   pt.AddText("----- Define the composite shape based on a Boolean expression")
   pt.AddText("                TGeoCompositeShape(\"someName\", \"expression\")")
   pt.AddText("----- Expression is made of <shapeName:transfName> components related by Boolean operators")
   pt.AddText("----- Boolean operators can be: (+) union, (-) subtraction and () intersection")
   pt.AddText("----- Use parenthesis in the expression to force precedence")
   AddExecInfo(pt, "composite")
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)



class myMatrix(ROOT.TGeoMatrix):
   def __init__(self):
      pass


#___________________________________
def ideal():
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _ideal
   global vol, slicex, slicey
   global pgon
   global myMatrixObj, _alignment
   
   # This is an ideal geometry. In real life, some geometry pieces are moved/rotated
   # with respect to their ideal positions. This is called alignment. Alignment
   # operations can be handled by TGeo starting from a CLOSED geometry (applied a posteriori)
   # Alignment is handled by PHYSICAL NODES, representing an unique object in geometry.
   #
   # Creating physical nodes:
   # 1. node = gGeoManager.MakePhysicalNode(path)
   #   - creates a physical node represented by path
   #   - path can be : "TOP_1/A_2/B_3"
   #   - B_3 is the 'final node' e.g. the logical node represented by this physical node
   #   - type(node) is TGeoPhysicalNode
   # 2. node = gGeoManager.MakePhysicalNode()
   #   - creates a physical node representing the current modeller state
   #   - type(node) is TGeoPhysicalNode
   
   # Setting visualisation options for `node`:
   # 1.   node.SetVisibility(flag) # set node visible(*) or invisible
   #    - default flag = True
   # 2.   node.SetIsVolAtt(flag)   # set line attributes to match the ones of the volumes in the branch
   #    - default flag = True
   #    - when called with False- the attributes defined for the physical node will be taken
   #       node.SetLineColor(color)
   #       node.SetLineWidth(width)
   #       node.SetLineStyle(style)
   # 3.   node.SetVisibleFull(flag) # not only last node in the branch is visible (default)
   #
   # Activating/deactivating physical nodes drawing - not needed in case of alignment
   
   # Aligning physical nodes
   #==========================
   #   node.Align(newmat = TGeoMatrix(), newshape = TGeoShape(), check = Bool_t(kFALSE))
   #   where ...
   #   newmat : new matrix to replace final node LOCAL matrix
   #   newshape : new shape to replace final node shape
   #   check : optional check if the new aligned node is overlapping
   # gGeoManager.SetDrawExtraPaths(flag = Bool_t())
   
   c = create_canvas("Ideal geometry")
   
   global gGeoManager
   if gGeoManager: 
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")

   #new TGeoManager("alignment", "Ideal geometry")
   _alignment = TGeoManager("alignment", "Ideal geometry")
   mat =  makeTGeoMaterial("Al", 26.98,13,2.7)
   med =  TGeoMedium("MED",1,mat)
   top = gGeoManager.MakeBox("TOP",med,100,100,10)
   gGeoManager.SetTopVolume(top)
   slicex = top.Divide("SX",1,10,-100,10)
   slicey = slicex.Divide("SY",2,10,-100,10)
   vol = gGeoManager.MakePgon("CELL",med,0.,360.,6,2)
   global gvol
   gvol = vol
   global gslicey
   gslicey = slicey
   pgon = (vol.GetShape())
   pgon.DefineSection(0,-5,0.,2.)
   pgon.DefineSection(1,5,0.,2.)
   vol.SetLineColor(randomColor())
   #Bug: AddNode method receives three arguments, the lost one is a TGeoMatrix, an abstract class.
   #Not to use: slicey.AddNode(vol,1)
   #Instead, we create an instance of the abstract-class TGeoMatrix :
   myMatrixObj = myMatrix() # TGeoMatrix 
   slicey.AddNode(vol, 1, myMatrixObj, "")

   gGeoManager.CloseGeometry()
   gGeoManager.SetNsegments(10)
   
   top.Draw()
   
   MakePicture()
   if not comments: 
      return
   c.cd(2)
   global pt
   pt =  TPaveText(0.01,0.01,0.99,0.99)
   pt.SetLineColor(1)
   text = pt.AddText("Ideal / Aligned geometry")
   text.SetTextColor(2)
   pt.AddText("-- Create physical nodes for the objects you want to align")
   pt.AddText("-- You must start from a valid CLOSED geometry")
   pt.AddText("    TGeoPhysicalNode *node = gGeoManager->MakePhysicalNode(const char *path)")
   pt.AddText("    + creates a physical node represented by path, e.g. TOP_1/A_2/B_3")
   pt.AddText("    node->Align(TGeoMatrix *newmat, TGeoShape *newshape, Bool_t check=kFALSE)")
   pt.AddText("    + newmat = new matrix to replace final node LOCAL matrix")
   pt.AddText("    + newshape = new shape to replace final node shape")
   pt.AddText("    + check = optional check if the new aligned node is overlapping")
   pt.AddText(" ")
   pt.SetAllWith("--","color",4)
   pt.SetAllWith("--","font",72)
   pt.SetAllWith("--","size",0.04)
   pt.SetAllWith("+","color",2)
   pt.SetAllWith("+","font",72)
   pt.SetAllWith("+","size",0.04)
   pt.SetTextAlign(12)
   pt.SetTextSize(0.044)
   pt.Draw()
   c.Update()
   c.Draw()

   c.cd(1)
   print("ideal() function executed fine")
   
   
   DelROOTObjs()


#___________________________________
def align() :
   global c, gGeoManager, mat, med, top, Slice, pt, text, vol, _align
   global node, tr, name, List, gPad 
   
   if not gGeoManager: 
      print("Nothing aligned. Run ideal geometry before")
      return
   if strcmp(gGeoManager.GetName(),"alignment"):
      print("Click: <Ideal geometry> first\n")
      return
      
   List = gGeoManager.GetListOfPhysicalNodes()
   zero_translation = TGeoTranslation(0, 0, 0)

   for i in range(1, 11):
      for j in range(1, 11):
         node = nullptr
         name = ("TOP_1/SX_{:d}/SY_{:d}/CELL_1".format(i,j) )
         if List:
            #print("List located")
            node = List.At(10*(i-1)+j-1)
         if not node:
            node = gGeoManager.MakePhysicalNode(name)
 
         # Aligment of nodes `if node.IsAligned(): ` will be implemented differently.

         tr = TGeoTranslation() # Initialize translation
         
         if not node.IsAligned():  # was´t set to any alingment before.
            if (i,j) == (1,1) : print("your geometry is being un-aligned")
            # Un-aligning node, then
            tr.SetTranslation(2.*gRandom.Rndm(), 2.*gRandom.Rndm(), 0.) # x-rand, y-rand, z-0
         else:
            tr_i_j = node.GetNode().GetMatrix()
            
            if tr_i_j == zero_translation:
               if (i,j) == (1,1) : print("your geometry is being un-aligned again. ")
               #print(f"Node {name} is aligned ")
               # Note: node.IsAligned() functions in a different context. 
               #       if True, that means that node has been already aligned by its "Align" method.
               #       node.Align(TGeoTranslation(0,0,0)) returns True.
               #       IsAligned-method doesn't compare with TGeoTranslation(0,0,0)--a non translation operation--.
               #       Be careful too use IsAlign. 
               #       Here we are checking if node is aligned with respect to TGeoTranslation(0,0,0).
               
               # Un-aligning node, then
               tr.SetTranslation(2.*gRandom.Rndm(), 2.*gRandom.Rndm(), 0.) # x-rand, y-rand, z-0
            elif tr_i_j != zero_translation:
               if (i,j) == (1,1) : print("your geometry is being restored ")
               # Modifying:
               # tr =  TGeoTranslation(2.*gRandom.Rndm(), 2.*gRandom.Rndm(),0.)
               # Set tr, to Un-align after. # Modify, It is a bit redundant.
               # However, if you call the ideal() function again, it'll re-align 
               # its configuration "apparently"; but it won't. ideal() always 
               # creates a "new" ideal-configuration.
               # Into:
               # Re-align ideally
               # tr =  TGeoTranslation(0, 0, 0)
               tr.SetTranslation(0, 0, 0)
            
         node.Align(tr) # Or Un-align, depending on its previous set-up.
         
      
   if gPad:
      gPad.Modified()
      gPad.Update()
   print("align() was being executed")

      
   
def Py(function_name):
   Py_x = 'TPython::Exec("{}");'
   Py_name = Py_x.format(function_name)
   return Py_name

#______________________________________________________________________________
#def geodemo ():
class geodemo :
      #Help() 
      #pass
   # IP[1] %run geodemo.py
   # IP[2] box()           # draw a TGeoBBox with description
   #
   # The box can be divided on one axis.
   #
   # IP[3] box(iaxis, ndiv, start, step)
   #
   # where: iaxis = 1,2 or 3, meaning (X,Y,Z) or (Rxy, phi, Z) depending on shape type
   #        ndiv  = number of slices
   #        start = starting position (must be in shape range)
   #        step  = division step
   # If step=0, all range of a given axis will be divided
   #
   # The same procedure can be performed for visualizing other kinds of shapes.
   # When drawing one shape after another, the old geometry/canvas will be deleted.
   # 
   
   #TControlBar Doesn´t function properly.
   bar =  TControlBar("vertical", "TGeo shapes",10,10)
   #ProcessLine("""
   #TControlBar *bar = new TControlBar("vertical", "TGeo shapes",10,10);
   #""")
   #bar = ROOT.bar

   bar.AddButton("How to run  ",  Py('Help()'),  "Instructions for running this macro")
   bar.AddButton("Arb8        ",  Py("arb8()"),  "An arbitrary polyhedron defined by vertices (max 8) sitting on 2 parallel planes")
   bar.AddButton("Box         ",  Py("box()"),  "A box shape.")
   bar.AddButton("Composite   ",  Py("composite()"),  "A composite shape")
   bar.AddButton("Cone        ",  Py("cone()"),  "A conical tube")
   bar.AddButton("Cone segment",  Py("coneseg()"),  "A conical segment")
   bar.AddButton("Cut tube    ",  Py("ctub()"),  "A cut tube segment")
   bar.AddButton("Elliptical tube",  Py("eltu()"),  "An elliptical tube")
   bar.AddButton("Extruded poly",  Py("xtru()"),  "A general polygone extrusion")
   bar.AddButton("Hyperboloid  ",  Py("hype()"),  "A hyperboloid")
   bar.AddButton("Paraboloid  ",  Py("parab()"),  "A paraboloid")
   bar.AddButton("Polycone    ",  Py("pcon()"),  "A polycone shape")
   bar.AddButton("Polygone    ",  Py("pgon()"),  "A polygone")
   bar.AddButton("Parallelepiped",  Py("para()"),  "A parallelepiped shape")
   bar.AddButton("Sphere      ",  Py("sphere()"),  "A spherical sector")
   bar.AddButton("Trd1        ",  Py("trd1()"),  "A trapezoid with dX varying with Z")
   bar.AddButton("Trd2        ",  Py("trd2()"),  "A trapezoid with both dX and dY varying with Z")
   bar.AddButton("Trapezoid   ",  Py("trap()"),  "A general trapezoid")
   bar.AddButton("Torus       ",  Py("torus()"),  "A toroidal segment")
   bar.AddButton("Tube        ",  Py("tube()"),  "A tube with inner and outer radius")
   bar.AddButton("Tube segment",  Py("tubeseg()"),  "A tube segment")
   bar.AddButton("Twisted trap",  Py("gtra()"),  "A twisted trapezoid")
   bar.AddButton("Tessellated ",  Py("tessellated()"),  "A tessellated shape")
   bar.AddButton("Aligned (ideal)",  Py("ideal()"),  "An ideal (aligned) geometry")
   bar.AddButton("Un-aligned / re-aligned",  Py("align()"),  "Some alignment operations")
   bar.AddButton("RAY-TRACE ON/OFF",  Py("raytrace()"),  "Toggle ray-tracing mode")
   bar.AddButton("RANDOM COLORS ON/OFF",  Py("SwitchRandomColors()"),  "Geometrical Figures with Random Colors ?")
   bar.AddButton("COMMENTS  ON/OFF",  Py("SwitchComments()"),  "Toggle explanations pad ON/OFF")
   bar.AddButton("AXES ON/OFF",  Py("axes()"),  "Toggle axes ON/OFF")
   bar.AddButton("AUTOROTATE ON/OFF",  Py("autorotate()"),  "Toggle autorotation ON/OFF")
   bar.Show()
   ROOT.gROOT.SaveContext()
   global gRandom
   gRandom =  TRandom3()
   global gbar
   gbar = bar
   
   ROOT.gROOT.Remove(bar)




"""
GeoDemo.AddExecInfo = AddExecInfo
GeoDemo.AddFinderInfo = AddFinderInfo
GeoDemo.AddMemberInfo = AddMemberInfo
GeoDemo.MakePicture = MakePicture
GeoDemo.SavePicture = SavePicture
GeoDemo.align = align
GeoDemo.arb8 = arb8
GeoDemo.autorotate = autorotate
GeoDemo.axes = axes
GeoDemo.box = box
GeoDemo.composite = composite
GeoDemo.cone = cone
GeoDemo.coneseg = coneseg
GeoDemo.create_canvas = create_canvas
GeoDemo.ctub = ctub
GeoDemo.eltu = eltu
GeoDemo.geodemo = geodemo
GeoDemo.gtra = gtra
GeoDemo.hype = hype
GeoDemo.ideal = ideal
GeoDemo.para = para
GeoDemo.parab = parab
GeoDemo.pcon = pcon
GeoDemo.pgon = pgon
GeoDemo.randomColor = randomColor
GeoDemo.raytrace = raytrace
GeoDemo.sphere = sphere
GeoDemo.tessellated = tessellated
GeoDemo.torus = torus
GeoDemo.trap = trap
GeoDemo.trd1 = trd1
GeoDemo.trd2 = trd2
GeoDemo.tube = tube
GeoDemo.tubeseg = tubeseg
GeoDemo.xtru = xtru
"""

   
if __name__ == "__main__":

   myGeoDemo = geodemo() # Pops-up a Window manager.
   
   # inside geodemo functions:
   """
   AddExecInfo()
   AddFinderInfo()
   AddMemberInfo()
   MakePicture()
   SavePicture()
   align()
   arb8()
   autorotate()
   axes()
   box()
   composite()
   cone()
   coneseg()
   create_canvas()
   ctub()
   eltu()
   geodemo()
   gtra()
   hype()
   ideal()
   para()
   :arab()
   pcon()
   pgon()
   randomColor()
   raytrace()
   sphere()
   tessellated()
   torus()
   trap()
   trd1()
   trd2()
   tube()
   tubeseg()
   xtru()
   """
   """
   AddExecInfo() 
   # Checking... [ good ]
   AddFinderInfo()
   # Checking... [ good ]
   AddMemberInfo()
   # Checking... [ good ]
   MakePicture()
   # Checking... [ good ]
   SavePicture()
   # Checking... [ good ]
   align()
   # Checking... [ good ]
   arb8()
   # Checking... [ good ]
   autorotate()
   # Checking... [ good ]
   axes()
   # Checking... [ good ]
   # box()
   # Checking... [ good but only once. Check memory leek]
   composite()
   cone()
   coneseg()
   create_canvas()
   ctub()
   eltu()
   geodemo()
   gtra()
   hype()
   ideal()
   para()
   parab()
   pcon()
   pgon()
   randomColor()
   raytrace()
   sphere()
   tessellated()
   torus()
   trap()
   trd1()
   #BP
   trd2()
   #sys.exit()
   tube()
   tubeseg()
   xtru()
   print("End of Checking", "All functions are good")
   """

  
