## \file
## \ingroup tutorial_geom
## Draw a sample of TXTRU shapes some convex, concave (and possibly malformed)
##
## Change the next flags to test alternative specifications:
##   makecw = False # Make Counter-Clockwise order of x-y points.
##   reversez = False # Reverse z-points order. 
##   domalformed = False # Do Malformed Polygons.
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
TGeoManager = ROOT.TGeoManager
TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium
TGeoTranslation = ROOT.TGeoTranslation
TGeoRotation = ROOT.TGeoRotation
TGeoCombiTrans = ROOT.TGeoCombiTrans 

TBRIK = ROOT.TBRIK
TNode = ROOT.TNode
TXTRU = ROOT.TXTRU
TCONE = ROOT.TCONE

kRed = ROOT.kRed
kYellow = ROOT.kYellow
kBlue = ROOT.kBlue

Double_t = ROOT.Double_t
Float_t = ROOT.Float_t 
Int_t = ROOT.Int_t 
c_double = ctypes.c_double

gStyle = ROOT.gStyle
gGeoIdentity = ROOT.gGeoIdentity
gPad = ROOT.gPad

# Not completely implemented in pyroot. Improve its Pythonization.
# sprintf = ROOT.sprintf 

def sprintfPy(buffer, FormatString, *args):
   # Note: In Python a string is not a mutable object.
   buffer = FormatString.format(*args)
   return buffer

Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine
gGeoManager = ROOT.gGeoManager

#def xtruSamples() :
class xtruSamples:
 def __init__(self) :
   print("running xtruSamples class")
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager

   # One normally specifies the x-y points in counter-clockwise order;
   # flip this to TRUE to test that it doesn't matter.
   makecw = False
   
   # One normally specifies the z points in increasing z order;
   # flip this to TRUE to test that it doesn't matter.
   reversez = False
   
   # One shouldn't be creating malformed polygons
   # but to test what happens when one does here's a flag.
   # The effect will be only apparent in solid rendering mode
   domalformed = False
   #  domalformed = kTRUE;
   global c1 
   c1 = TCanvas("c1","sample TXTRU Shapes",200,10,640,640)
   
   # Create a new geometry
   global geom
   geom = TGeometry("sample","sample")
   ProcessLine(
   """
   TGeometry* geom = new TGeometry("sample","sample");
   """)
   #global geom
   #geom = ROOT.geom

   geom.cd()

   
   # Define the complexity of the drawing
   zseg = 6 # either 2 or 6
   extra_figures = 1 # make extra z "arrow" visible
   
   unit = 1
   
   global world, worldnode
   # Create a large BRIK to embed things into
   bigdim = 12.5*unit
   world = TBRIK("world","world","void",bigdim,bigdim,bigdim)
   
   # Create the main node, make it invisible
   worldnode = TNode("worldnode","world node",world)
   worldnode.SetVisibility(0)
   worldnode.cd()
   
   # Canonical shape ... gets further modified by scale factors
   # to create convex (and malformed) versions
   x = [  -0.50, -1.20, 1.20, 0.50, 0.50, 1.20, -1.20, -0.50]
   y = [  -0.75, -2.00, -2.00, -0.75, 0.75, 2.00, 2.00, 0.75]
   z = [  -0.50, -1.50, -1.50, 1.50, 1.50, 0.50]
   s = [  0.50, 1.00, 1.50, 1.50, 1.00, 0.50]
   # In C++ we use sizeof() sometimes to measure the size of an array.
   # In Python we can achive that by using len() instead.
   # C++: Int_t   nxy = sizeof(x)/sizeof(Float_t);
   #nxy = Int_t(len(x)/len(Float_t))
   nxy = len(x)
   convexscale = [  7.0, -1.0, 1.5]
   
   icolor = [  1, 2, 3, 2, 2, 2, 4, 2, 6]
   
   global mytxtru, txtrunode, mytxtru_list, txtrunode_list
   global zbrik, zbriknode, txtr, txtrunode, zcone, zconenode 
   # xycase and zcase:  0=convex, 1=malformed, 2=concave
   # this will either create a 2x2 matrix of shapes
   # or a 3x3 array (if displaying malformed versions)
   mytxtru_list=[]
   txtrunode_list=[]

   for zcase in range( 0, 3):
      if (zcase == 1 and not domalformed): continue
      for xycase in range( 0, 3):
         if (xycase == 1 and not domalformed): continue
         #print("zcase", zcase, "xycase", xycase) 
         name = " "*9
         name = sprintfPy(name,"txtru{:1d}{:1d}{:1d}".format(xycase,zcase,zseg))
         #print("name", name)
         mytxtru = TXTRU(name,name,"void",8,2)
         mytxtru_list.append(mytxtru)
         
         xsign = -1 if makecw else 1 
         zsign = -1 if (reversez) else 1
         
         # set the vertex points
         for i in range(0, nxy):
            xtmp = x[i] * xsign
            ytmp = y[i]
            if (i==0 or i==3 or i==4 or i==7): xtmp *= convexscale[xycase]
            if (xycase==2): xtmp *=2
            mytxtru.DefineVertex(i,xtmp,ytmp)
            
         # set the z segment positions and scales
         j = 0
         #Not to use: for (i, j) in [(i,i) in range(0,zseg)]:
         #Not to use: for (i, j) in zip(range(0,zseg), range(0,zseg)): 
         # It is important the outside counter 'j'. 
         for i in range(0, zseg):
            #print(" i j", i, j)
            ztmp = z[i] * zsign
            if (i==0 or i==5): ztmp *= convexscale[zcase]
            if (zcase==2): ztmp *= 2.5
            if (zseg>2 and zcase!=2 and (i==1 or i==4)): continue
            mytxtru.DefineSection(j,ztmp,s[i])
            (j:=j+1)
         
         #New TNode
         txtrunode = TNode(name,name,mytxtru)
         txtrunode.SetLineColor(icolor[3*zcase+xycase])
         pos_scale = 10 if (domalformed) else 6
         xpos = (xycase-1) * pos_scale * unit
         ypos = (zcase-1) * pos_scale * unit
         txtrunode.SetPosition(xpos,ypos,0.)
         txtrunode_list.append(txtrunode)
         
   #It will draw all the geometries until now on the same canvas.
   #geom.Draw()
   #sys.exit()
      
   
   global zcone, zconenode, zbrik, zbriknode 
   # Some extra shapes to show the direction of "z"

   zhalf = 0.5 * bigdim
   rmax = 0.03 * bigdim
   zcone = TCONE("zcone","zcone","void",zhalf,0.,rmax,0.,0.)
   zcone.SetVisibility(extra_figures)
   zconenode = TNode("zconenode","zconenode",zcone)
   zconenode.SetLineColor(3)
   
   dzstub = 2 * rmax
   zbrik = TBRIK("zbrik","zbrik","void",rmax,rmax,dzstub)
   zbrik.SetVisibility(extra_figures)
   zbriknode = TNode("zbriknode","zbriknode",zbrik)
   zbriknode.SetPosition(0.,0.,zhalf+dzstub)
   zbriknode.SetLineColor(3)
   
   #geom.ls()
    
   # It will draw all geometries until now(xtru's and a cone and a brik) in the same canvas.
   # use extra_figures to set-up this.
   geom.Draw()
  
   # Tweak the pad so that it displays the entire geometry undistorted
   # Little Note: gPad was defined at the top of this script.
   global gPad 
   gPad = ROOT.gPad
   #print("gPad", gPad)

   #input("continue")
   thisPad = gPad
   if thisPad:
      view = thisPad.GetView()
      if (not view): 
         print("no view found")
         return
      #Not to Use: Min = Max = center = [Double_t()]*3
      Min = [c_double() for i in range(3)]
      Max = [c_double() for i in range(3)]
      center = [c_double() for i in range(3)]
      c_Min = (c_double*3)(*Min)
      c_Max = (c_double*3)(*Max)
      c_center = (c_double*3)(*center)
      
      view.GetRange(c_Min, c_Max)
      Min = list(c_Min)
      Max = list(c_Max)
      center = list(c_center)
      
      # Find the boxed center
      for i in range(0, 3):center[i] = 0.5*(Max[i]+Min[i])
      maxSide = 0
      # Find the largest side
      for i in range(0, 3):maxSide = TMath.Max(maxSide,Max[i]-center[i])
      # C++ statement: file: \\ Adjust scales:
      for i in range(0, 3):
         Max[i] = center[i] + maxSide
         Min[i] = center[i] - maxSide
      
      c_Min = (c_double*3)(*Min)
      c_Max = (c_double*3)(*Max)
      c_center = (c_double*3)(*center)
      view.SetRange(c_Min, c_Max)

      thisPad.Modified()
      thisPad.Update()
      thisPad.Draw()
      geom.Draw()
      
   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   myvars = [x for x in dir() if not x.startswith("__")]
   myvars += [x for x in globals() if not x.startswith("_")]
   #myvars = [x for x in vars(self) ] 
   for var in myvars: 
      #print(var)
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         exec(f"ROOT.gROOT.Remove(self.{var})")
         print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes too much memory. Try without exec.
         #TODO: replace much by toomuch. English language error.
      except :
         pass 
   # Now, it works!!!
   


if __name__ == "__main__":
   xtruSamples()
