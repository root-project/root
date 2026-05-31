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

Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine


def rootgeom(vis = True):
   # global gStyle
   # gStyle = ROOT.gStyle
   # gStyle.SetCanvasPreferGL(True)

   #geom = TGeoManager("simple1", "Simple geometry")
   ProcessLine(
   """
   TGeoManager *geom = new TGeoManager("simple1", "Simple geometry");
   """)
   global geom
   geom = ROOT.geom
   
   #--- define some materials
   matVacuum = TGeoMaterial("Vacuum", 0,0,0)
   matAl = TGeoMaterial("Al", 26.98,13,2.7)
   #   #--- define some media
   Vacuum = TGeoMedium("Vacuum",1, matVacuum)
   Al = TGeoMedium("Root Material",2, matAl)
   
   #--- define the transformations
   tr1 = TGeoTranslation(20., 0, 0.)
   tr2 = TGeoTranslation(10., 0., 0.)
   tr3 = TGeoTranslation(10., 20., 0.)
   tr4 = TGeoTranslation(5., 10., 0.)
   tr5 = TGeoTranslation(20., 0., 0.)
   tr6 = TGeoTranslation(-5., 0., 0.)
   tr7 = TGeoTranslation(7.5, 7.5, 0.)

   rot1 = TGeoRotation("rot1", 90., 0., 90., 270., 0., 0.)
   combi1 = TGeoCombiTrans(7.5, -7.5, 0., rot1)

   tr8 = TGeoTranslation(7.5, -5., 0.)
   tr9 = TGeoTranslation(7.5, 20., 0.)
   tr10 = TGeoTranslation(85., 0., 0.)
   tr11 = TGeoTranslation(35., 0., 0.)
   tr12 = TGeoTranslation(-15., 0., 0.)
   tr13 = TGeoTranslation(-65., 0., 0.)
   tr14 = TGeoTranslation(0,0,-100)

   combi2 = TGeoCombiTrans(0,0,100,
      TGeoRotation("rot2",90,180,90,90,180,0))
   combi3 = TGeoCombiTrans(100,0,0,
      TGeoRotation("rot3",90,270,0,0,90,180))
   combi4 = TGeoCombiTrans(-100,0,0,
      TGeoRotation("rot4",90,90,0,0,90,0))
   combi5 = TGeoCombiTrans(0,100,0,
      TGeoRotation("rot5",0,0,90,180,90,270))
   combi6 = TGeoCombiTrans(0,-100,0,
      TGeoRotation("rot6",180,0,90,180,90,90))
   
   #--- make the top container volume
   worldx = 110.
   worldy = 50.
   worldz = 5.
   top = geom.MakeBox("TOP", Vacuum, 270., 270., 120.)
   geom.SetTopVolume(top)
   replica = geom.MakeBox("REPLICA", Vacuum,120,120,120)
   replica.SetVisibility(False)
   rootbox = geom.MakeBox("ROOT", Vacuum, 110., 50., 5.)
   rootbox.SetVisibility(False)
   
   #--- make letter 'R'
   R = geom.MakeBox("R", Vacuum, 25., 25., 5.)
   R.SetVisibility(False)
   bar1 = geom.MakeBox("bar1", Al, 5., 25, 5.)
   bar1.SetLineColor(kRed)
   R.AddNode(bar1, 1, tr1)
   bar2 = geom.MakeBox("bar2", Al, 5., 5., 5.)
   bar2.SetLineColor(kRed)
   R.AddNode(bar2, 1, tr2)
   R.AddNode(bar2, 2, tr3)
   tub1 = geom.MakeTubs("tub1", Al, 5., 15., 5., 90., 270.)
   tub1.SetLineColor(kRed)
   R.AddNode(tub1, 1, tr4)
   bar3 = geom.MakeArb8("bar3", Al, 5.)
   bar3.SetLineColor(kRed)
   arb = bar3.GetShape()
   arb.SetVertex(0, 15., -5.)
   arb.SetVertex(1, 0., -25.)
   arb.SetVertex(2, -10., -25.)
   arb.SetVertex(3, 5., -5.)
   arb.SetVertex(4, 15., -5.)
   arb.SetVertex(5, 0., -25.)
   arb.SetVertex(6, -10., -25.)
   arb.SetVertex(7, 5., -5.)
   R.AddNode(bar3, 1, gGeoIdentity)
   
   #--- make letter 'O'
   O = geom.MakeBox("O", Vacuum, 25., 25., 5.)
   O.SetVisibility(False)
   bar4 = geom.MakeBox("bar4", Al, 5., 7.5, 5.)
   bar4.SetLineColor(kYellow)
   O.AddNode(bar4, 1, tr5)
   O.AddNode(bar4, 2, tr6)
   tub2 = geom.MakeTubs("tub1", Al, 7.5, 17.5, 5., 0., 180.)
   tub2.SetLineColor(kYellow)
   O.AddNode(tub2, 1, tr7)
   O.AddNode(tub2, 2, combi1)
   
   #--- make letter 'T'
   T = geom.MakeBox("T", Vacuum, 25., 25., 5.)
   T.SetVisibility(False)
   bar5 = geom.MakeBox("bar5", Al, 5., 20., 5.)
   bar5.SetLineColor(kBlue)
   T.AddNode(bar5, 1, tr8)
   bar6 = geom.MakeBox("bar6", Al, 17.5, 5., 5.)
   bar6.SetLineColor(kBlue)
   T.AddNode(bar6, 1, tr9)
   
   
   
   rootbox.AddNode(R, 1, tr10)
   rootbox.AddNode(O, 1, tr11)
   rootbox.AddNode(O, 2, tr12)
   rootbox.AddNode(T, 1, tr13)
   
   replica.AddNode(rootbox, 1, tr14)
   replica.AddNode(rootbox, 2, combi2)
   replica.AddNode(rootbox, 3, combi3)
   replica.AddNode(rootbox, 4, combi4)
   replica.AddNode(rootbox, 5, combi5)
   replica.AddNode(rootbox, 6, combi6)
   
   top.AddNode(replica, 1, TGeoTranslation(-150, -150, 0))
   top.AddNode(replica, 2, TGeoTranslation(150, -150, 0))
   top.AddNode(replica, 3, TGeoTranslation(150, 150, 0))
   top.AddNode(replica, 4, TGeoTranslation(-150, 150, 0))
   
   #--- close the geometry
   geom.CloseGeometry()
   
   #--- draw the ROOT box.
   # by default the picture will appear in the standard ROOT TPad.
   #if you have activated the following line in system.rootrc,
   #it will appear in the GL viewer
   ##Viewer3D.DefaultDrawOption:   ogl
   
   geom.SetVisLevel(4)
   if (vis):
      # "ogle" and "3xd" work fine.
      top.Draw("ogle")
      #top.Draw("3xd")
   



if __name__ == "__main__":
   rootgeom()
