## \file
## \ingroup tutorial_geom
## Definition of a simple geometry (the 4 ROOT characters)
##
## \macro_image
## \macro_code
##
## \author Andrei Gheata
## \translator P. P.


import ROOT
TCanvas = ROOT.TCanvas
TGeoManager = ROOT.TGeoManager
TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium
TGeoTranslation = ROOT.TGeoTranslation
TGeoRotation = ROOT.TGeoRotation
TGeoCombiTrans = ROOT.TGeoCombiTrans 

kRed = ROOT.kRed
kYellow = ROOT.kYellow
kBlue = ROOT.kBlue

gStyle = ROOT.gStyle
gGeoIdentity = ROOT.gGeoIdentity
gGeoManager = ROOT.gGeoManager

Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine


#def rootgeom(self.vis = True):
class rootgeom:
 def __init__(self, vis = True):
   global gGeoManager
   if gGeoManager:
      ROOT.gROOT.Remove(ROOT.gGeoManager)
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager

   global gStyle
   gStyle = ROOT.gStyle
   gStyle.SetCanvasPreferGL(True)


   self.geom = TGeoManager("simple1", "Simple geometry")
   #ProcessLine(
   #"""
   #TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");
   #""")
   #self.geom = ROOT.geom
   
   #--- define some materials
   self.matVacuum = TGeoMaterial("Vacuum", 0,0,0)
   self.matAl = TGeoMaterial("Al", 26.98,13,2.7)
   #   #--- define some media
   self.Vacuum = TGeoMedium("Vacuum",1, self.matVacuum)
   self.Al = TGeoMedium("Root Material",2, self.matAl)
   
   #--- define the transformations
   self.tr1 = TGeoTranslation(20., 0, 0.)
   self.tr2 = TGeoTranslation(10., 0., 0.)
   self.tr3 = TGeoTranslation(10., 20., 0.)
   self.tr4 = TGeoTranslation(5., 10., 0.)
   self.tr5 = TGeoTranslation(20., 0., 0.)
   self.tr6 = TGeoTranslation(-5., 0., 0.)
   self.tr7 = TGeoTranslation(7.5, 7.5, 0.)

   self.rot1 = TGeoRotation("rot1", 90., 0., 90., 270., 0., 0.)
   self.combi1 = TGeoCombiTrans(7.5, -7.5, 0., self.rot1)

   self.tr8 = TGeoTranslation(7.5, -5., 0.)
   self.tr9 = TGeoTranslation(7.5, 20., 0.)
   self.tr10 = TGeoTranslation(85., 0., 0.)
   self.tr11 = TGeoTranslation(35., 0., 0.)
   self.tr12 = TGeoTranslation(-15., 0., 0.)
   self.tr13 = TGeoTranslation(-65., 0., 0.)
   self.tr14 = TGeoTranslation(0,0,-100)

   self.combi2 = TGeoCombiTrans(0,0,100,
      TGeoRotation("rot2",90,180,90,90,180,0))
   self.combi3 = TGeoCombiTrans(100,0,0,
      TGeoRotation("rot3",90,270,0,0,90,180))
   self.combi4 = TGeoCombiTrans(-100,0,0,
      TGeoRotation("rot4",90,90,0,0,90,0))
   self.combi5 = TGeoCombiTrans(0,100,0,
      TGeoRotation("rot5",0,0,90,180,90,270))
   self.combi6 = TGeoCombiTrans(0,-100,0,
      TGeoRotation("rot6",180,0,90,180,90,90))
   
   #--- make the top container volume
   worldx = 110.
   worldy = 50.
   worldz = 5.
   self.top = self.geom.MakeBox("TOP", self.Vacuum, 270., 270., 120.)
   self.geom.SetTopVolume(self.top)
   self.replica = self.geom.MakeBox("REPLICA", self.Vacuum,120,120,120)
   self.replica.SetVisibility(False)
   self.rootbox = self.geom.MakeBox("ROOT", self.Vacuum, 110., 50., 5.)
   self.rootbox.SetVisibility(False)
   
   #--- make letter 'R'
   self.R = self.geom.MakeBox("R", self.Vacuum, 25., 25., 5.)
   self.R.SetVisibility(False)
   self.bar1 = self.geom.MakeBox("bar1", self.Al, 5., 25, 5.)
   self.bar1.SetLineColor(kRed)
   self.R.AddNode( self.bar1, 1,  self.tr1 )
   self.bar2 = self.geom.MakeBox("bar2", self.Al, 5., 5., 5.)
   self.bar2.SetLineColor(kRed)
   self.R.AddNode( self.bar2, 1,  self.tr2 )
   self.R.AddNode( self.bar2, 2,  self.tr3 )
   self.tub1 = self.geom.MakeTubs("tub1", self.Al, 5., 15., 5., 90., 270.)
   self.tub1.SetLineColor(kRed)
   self.R.AddNode( self.tub1, 1,  self.tr4 )
   self.bar3 = self.geom.MakeArb8("bar3", self.Al, 5.)
   self.bar3.SetLineColor(kRed)
   self.arb =  self.bar3.GetShape()
   self.arb.SetVertex(0, 15., -5.)
   self.arb.SetVertex(1, 0., -25.)
   self.arb.SetVertex(2, -10., -25.)
   self.arb.SetVertex(3, 5., -5.)
   self.arb.SetVertex(4, 15., -5.)
   self.arb.SetVertex(5, 0., -25.)
   self.arb.SetVertex(6, -10., -25.)
   self.arb.SetVertex(7, 5., -5.)
   global gGeoIdentity
   self.R.AddNode(self.bar3, 1, gGeoIdentity)
   
   #--- make letter 'O'
   self.O = self.geom.MakeBox("O",  self.Vacuum, 25., 25., 5.)
   self.O.SetVisibility(False)
   self.bar4 = self.geom.MakeBox("bar4", self.Al, 5., 7.5, 5.)
   self.bar4.SetLineColor(kYellow)
   self.O.AddNode(self.bar4, 1, self.tr5)
   self.O.AddNode(self.bar4, 2, self.tr6)
   self.tub2 = self.geom.MakeTubs("tub1", self.Al, 7.5, 17.5, 5., 0., 180.)
   self.tub2.SetLineColor(kYellow)
   self.O.AddNode(self.tub2, 1, self.tr7)
   self.O.AddNode(self.tub2, 2, self.combi1)
   
   #--- make letter 'T'
   self.T = self.geom.MakeBox("T", self.Vacuum, 25., 25., 5.)
   self.T.SetVisibility(False)
   self.bar5 = self.geom.MakeBox("bar5", self.Al, 5., 20., 5.)
   self.bar5.SetLineColor(kBlue)
   self.T.AddNode(self.bar5, 1, self.tr8)
   self.bar6 = self.geom.MakeBox("bar6", self.Al, 17.5, 5., 5.)
   self.bar6.SetLineColor(kBlue)
   self.T.AddNode(self.bar6, 1, self.tr9)
   
   
   
   self.rootbox.AddNode(self.R, 1, self.tr10)
   self.rootbox.AddNode(self.O, 1, self.tr11)
   self.rootbox.AddNode(self.O, 2, self.tr12)
   self.rootbox.AddNode(self.T, 1, self.tr13)
   
   self.replica.AddNode(self.rootbox, 1, self.tr14)
   self.replica.AddNode(self.rootbox, 2, self.combi2)
   self.replica.AddNode(self.rootbox, 3, self.combi3)
   self.replica.AddNode(self.rootbox, 4, self.combi4)
   self.replica.AddNode(self.rootbox, 5, self.combi5)
   self.replica.AddNode(self.rootbox, 6, self.combi6)
   
   self.top.AddNode(self.replica, 1, TGeoTranslation(-150, -150, 0))
   self.top.AddNode(self.replica, 2, TGeoTranslation(150, -150, 0))
   self.top.AddNode(self.replica, 3, TGeoTranslation(150, 150, 0))
   self.top.AddNode(self.replica, 4, TGeoTranslation(-150, 150, 0))
   
   #--- close the geometry
   self.geom.CloseGeometry()
   
   #--- draw the ROOT box.
   # by default the picture will appear in the standard ROOT TPad.
   #if you have activated the following line in system.rootrc,
   #it will appear in the GL viewer
   ##Viewer3D.DefaultDrawOption:   ogl
   
   self.geom.SetVisLevel(4)
   if (vis):
      #self.c1 = TCanvas("c1")
      # "ogle" and "x3d" work fine.
      #top.Draw("x3d")
      #top.Draw("ogl")
      #top.Draw("ogle")
      self.top.Draw("c1")
      #c1.Update()
      #c1.Draw()
   

   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   print("Deleting objs from gROOT")
   myvars = [x for x in dir(self) if not x.startswith("__")]
   for var in myvars: 
      try:
         #exec(f"ROOT.gROOT.Remove({var})")
         exec(f"ROOT.gROOT.Remove(self.{var})")
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   #ROOT.gROOT.Remove(self.geom)
   ROOT.gROOT.Remove(gStyle)
   ROOT.gROOT.Remove(gGeoManager)
   ROOT.gROOT.Remove(gGeoIdentity)
   ROOT.gROOT.GetListOfCanvases().Clear()
   ROOT.gROOT.GetListOfCanvases().Delete()
   # Now, it works!!!



if __name__ == "__main__":
   myROOTGeometry = rootgeom()
