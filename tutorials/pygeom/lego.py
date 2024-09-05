## \file
## \ingroup tutorial_geom
## Drawing a figure, made of lego block, using ROOT geometry class.
## It is mandatory to have OpenGL installed alongside with ROOT.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: `%run lego.py` in ipython3 interpreter, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to all of ROOT team
##
## \image html geom_lego.png width=800px
## \macro_code
##
## \author Soon Gi Kwon(1116won@hanmail.net), Dept. of Physics, Univ. of Seoul
## \translator P. P.


import ROOT

TSystem = ROOT.TSystem
TGeoManager = ROOT.TGeoManager

TGeoCombiTrans = 		 ROOT.TGeoCombiTrans
TGeoRotation = 		 ROOT.TGeoRotation
TGeoMaterial = 		 ROOT.TGeoMaterial  
TGeoMedium = 		 ROOT.TGeoMedium  
TGeoVolume = 		 ROOT.TGeoVolume 
TGeoTranslation = 		 ROOT.TGeoTranslation 
TGeoCombiTrans = 		 ROOT.TGeoCombiTrans


Declare = 		 ROOT.gInterpreter.Declare 
ProcessLine = 		 ROOT.gInterpreter.ProcessLine 
gGeoManager = ROOT.gGeoManager



def sprintfPy(buffer, StringFormat, *args):
   buffer = StringFormat.format(*args)
   return buffer


#def lego():
class lego:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager

   # geom =  TGeoManager("geom","My first 3D geometry")
   ProcessLine('''
   TGeoManager *geom = new TGeoManager("geom","My first 3D geometry");
   ''')
   global geom
   geom = ROOT.geom

   
   
   
   vacuum = TGeoMaterial("vacuum",0,0,0) # TGeoMaterial
   Fe = TGeoMaterial("Fe",55.845,26,7.87) # TGeoMaterial
   
   
   
   Air = TGeoMedium("Vacuum",0,vacuum) # TGeoMedium
   Iron = TGeoMedium("Iron",1,Fe) # TGeoMedium
   
   
   # create volume
   top = geom.MakeBox("top",Air,100,100,100) # TGeoVolume
   geom.SetTopVolume(top)
   geom.SetTopVisible(False)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom->SetTopVisible(1);
   
   #----------------------------------------------------------------------
   
   ha1 = geom.MakeSphere("ha1",Iron,0,10,80,90,0,360) # TGeoVolume
   ha1.SetLineColor(41)
   top.AddNodeOverlap(ha1,1,TGeoCombiTrans(0,0,4,TGeoRotation("ha1",0,0,0)))
   
   ha2 = geom.MakeSphere("ha2",Iron,0,7,90,180,0,360) # TGeoVolume
   ha2.SetLineColor(41)
   top.AddNodeOverlap(ha2,1,TGeoCombiTrans(0,0,4,TGeoRotation("ha2",0,180,0)))
   
   ha3 = geom.MakeSphere("ha3",Iron,0,7.3,80,90,0,360) # TGeoVolume
   ha3.SetLineColor(2)
   top.AddNodeOverlap(ha3,1,TGeoCombiTrans(0,0,4.8,TGeoRotation("ha3",0,0,0)))
   
   
   h1 = geom.MakeTubs("h1",Iron,0,6,4.5,0,0) # TGeoVolume
   h1.SetLineColor(5)
   top.AddNodeOverlap(h1,1,TGeoCombiTrans(0,0,0,TGeoRotation("h1",0,0,0)))
   
   h2 = geom.MakeSphere("h2",Iron,0,7.5,0,52.5,0,360) # TGeoVolume
   h2.SetLineColor(5)
   top.AddNodeOverlap(h2,1,TGeoCombiTrans(0,0,0,TGeoRotation("h2",0,0,0)))
   
   h3 = geom.MakeSphere("h3",Iron,0,7.5,0,52.5,0,360) # TGeoVolume
   h3.SetLineColor(5)
   top.AddNodeOverlap(h3,1,TGeoCombiTrans(0,0,0,TGeoRotation("h3",180,180,0)))
   
   h4 = geom.MakeTubs("h4",Iron,2.5,3.5,1.5,0,0) # TGeoVolume
   h4.SetLineColor(5)
   top.AddNodeOverlap(h4,1,TGeoCombiTrans(0,0,7.5,TGeoRotation("h4",0,0,0)))
   
   
   
   t1_1 = geom.MakeTubs("t1_1",Iron,0,0.8,1,0,360) # TGeoVolume
   t1_1.SetLineColor(12)
   top.AddNodeOverlap(t1_1,1,TGeoCombiTrans(-5,2,1.5,TGeoRotation("t1_1",-90,90,0)))
   
   t2_1 = geom.MakeTubs("t2_1",Iron,0,0.8,1,0,360) # TGeoVolume
   t2_1.SetLineColor(12)
   top.AddNodeOverlap(t2_1,1,TGeoCombiTrans(-5,-2,1.5,TGeoRotation("t2_1",-90,90,0)))
   
   fb1 = geom.MakeTubs("fb1",Iron,2,2.3,1,100,260) # TGeoVolume
   fb1.SetLineColor(12)
   top.AddNodeOverlap(fb1,1,TGeoCombiTrans(-5,0,-1,TGeoRotation("fb1",90,90,90)))
   
   
   
   m1 = geom.MakeBox("m1",Iron,7,8,4) # TGeoVolume
   m1.SetLineColor(2)
   top.AddNodeOverlap(m1,1,TGeoCombiTrans(0,0,-17,TGeoRotation("m1",90,90,0)))
   
   m2 = geom.MakeTubs("m2",Iron,0,1,7,90,180) # TGeoVolume
   m2.SetLineColor(2)
   top.AddNodeOverlap(m2,1,TGeoCombiTrans(-3,0,-9,TGeoRotation("m2",0,90,0)))
   
   m3 = geom.MakeTubs("m3",Iron,0,1,7,0,90) # TGeoVolume
   m3.SetLineColor(2)
   top.AddNodeOverlap(m3,1,TGeoCombiTrans(3,0,-9,TGeoRotation("m3",0,90,0)))
   
   m4 = geom.MakeBox("m4",Iron,3,7,0.5) # TGeoVolume
   m4.SetLineColor(2)
   top.AddNodeOverlap(m4,1,TGeoCombiTrans(0,0,-8.5,TGeoRotation("m4",90,0,90)))
   
   m5 = geom.MakeTubs("m5",Iron,0,1.5,1.2,0,0) # TGeoVolume
   m5.SetLineColor(5)
   top.AddNodeOverlap(m5,1,TGeoCombiTrans(0,0,-7.8,TGeoRotation("m5",0,0,0)))
   
   m6 = geom.MakeTrd2("m6",Iron,4,4,0,2,8) # TGeoVolume
   m6.SetLineColor(2)
   top.AddNodeOverlap(m6,1,TGeoCombiTrans(0,-7,-17,TGeoRotation("m6",0,180,0)))
   
   m7 = geom.MakeTrd2("m7",Iron,4,4,0,2,8) # TGeoVolume
   m7.SetLineColor(2)
   top.AddNodeOverlap(m7,1,TGeoCombiTrans(0,7,-17,TGeoRotation("m7",0,180,0)))
   
   
   md1 = geom.MakeBox("md1",Iron,4,8.5,0.7) # TGeoVolume
   md1.SetLineColor(37)
   top.AddNodeOverlap(md1,1,TGeoCombiTrans(0,0,-25.5,TGeoRotation("md1",0,0,0)))
   
   md2 = geom.MakeBox("md2",Iron,3,0.4,2) # TGeoVolume
   md2.SetLineColor(37)
   top.AddNodeOverlap(md2,1,TGeoCombiTrans(0,0,-28,TGeoRotation("md2",0,0,0)))
   
   d1 = geom.MakeTrd2("d1",Iron,3,4,4,4,7) # TGeoVolume
   d1.SetLineColor(37)
   top.AddNodeOverlap(d1,1,TGeoCombiTrans(-4.8,4.5,-35,TGeoRotation("d1",90,45,-90)))
   
   d2 = geom.MakeTrd2("d2",Iron,3,4,4,4,7) # TGeoVolume
   d2.SetLineColor(37)
   top.AddNodeOverlap(d2,1,TGeoCombiTrans(0,-4.5,-37,TGeoRotation("d2",0,0,0)))
   
   d3 = geom.MakeTubs("d3",Iron,0,4,3.98,0,180) # TGeoVolume
   d3.SetLineColor(37)
   top.AddNodeOverlap(d3,1,TGeoCombiTrans(0,4.5,-30.2,TGeoRotation("d3",0,90,-45)))
   
   d4 = geom.MakeTubs("d4",Iron,0,4,3.98,0,180) # TGeoVolume
   d4.SetLineColor(37)
   top.AddNodeOverlap(d4,1,TGeoCombiTrans(0,-4.5,-30,TGeoRotation("d4",0,90,0)))
   
   d5 = geom.MakeBox("d5",Iron,4,4,1) # TGeoVolume
   d5.SetLineColor(37)
   top.AddNodeOverlap(d5,1,TGeoCombiTrans(-10.2,4.5,-39,TGeoRotation("d5",90,45,-90)))
   
   d6 = geom.MakeBox("d6",Iron,4,4,1) # TGeoVolume
   d6.SetLineColor(37)
   top.AddNodeOverlap(d6,1,TGeoCombiTrans(-1,-4.5,-43.4,TGeoRotation("d6",0,0,0)))
   
   
   
   a1 = geom.MakeTubs("a1",Iron,0,1.5,4,0,0) # TGeoVolume
   a1.SetLineColor(1)
   top.AddNodeOverlap(a1,1,TGeoCombiTrans(0,10,-15.1,TGeoRotation("a1",0,20,45)))
   
   a2 = geom.MakeSphere("a2",Iron,0,1.48,0,180,0,200) # TGeoVolume
   a2.SetLineColor(1)
   top.AddNodeOverlap(a2,1,TGeoCombiTrans(0,8.6,-11.5,TGeoRotation("a2",120,80,20)))
   
   a3 = geom.MakeTubs("a3",Iron,0,1.5,2.2,0,0) # TGeoVolume
   a3.SetLineColor(1)
   top.AddNodeOverlap(a3,1,TGeoCombiTrans(0,11.3,-20.6,TGeoRotation("a3",300,0,40)))
   
   a4 = geom.MakeTubs("a4",Iron,0,1,1,0,0) # TGeoVolume
   a4.SetLineColor(5)
   top.AddNodeOverlap(a4,1,TGeoCombiTrans(0,11.3,-23.8,TGeoRotation("a4",75,0,30)))
   
   a5 = geom.MakeTubs("a5",Iron,1.5,2.5,2,0,270) # TGeoVolume
   a5.SetLineColor(5)
   top.AddNodeOverlap(a5,1,TGeoCombiTrans(0,11.3,-26.5,TGeoRotation("a5",-90,90,00)))
   
   
   
   
   a1_1 = geom.MakeTubs("a1_1",Iron,0,1.5,4,0,0) # TGeoVolume
   a1_1.SetLineColor(1)
   top.AddNodeOverlap(a1_1,1,TGeoCombiTrans(0,-10,-15.1,TGeoRotation("a1_1",0,-20,-45)))
   
   a2_1 = geom.MakeSphere("a2_1",Iron,0,1.48,0,180,0,200) # TGeoVolume
   a2_1.SetLineColor(1)
   top.AddNodeOverlap(a2_1,1,TGeoCombiTrans(0,-8.6,-11.5,TGeoRotation("a2_1",120,80,-20)))
   
   a3_1 = geom.MakeTubs("a3_1",Iron,0,1.5,2.2,0,0) # TGeoVolume
   a3_1.SetLineColor(1)
   top.AddNodeOverlap(a3_1,1,TGeoCombiTrans(0,-11.3,-20.6,TGeoRotation("a3_1",-300,0,-40)))
   
   a4_1 = geom.MakeTubs("a4_1",Iron,0,1,1,0,0) # TGeoVolume
   a4_1.SetLineColor(5)
   top.AddNodeOverlap(a4_1,1,TGeoCombiTrans(0,-11.3,-23.8,TGeoRotation("a4_1",-75,0,-30)))
   
   a5=geom.MakeTubs("a5_1",Iron,1.5,2.5,2,0,270)
   a5.SetLineColor(5)
   top.AddNodeOverlap(a5,1,TGeoCombiTrans(0,-11.3,-26.5,TGeoRotation("a5",90,90,00)))
   
   
   #**********************************NO,2******************
   
   
   ha_1 = geom.MakeSphere("ha_1",Iron,0,10,80,90,0,360) # TGeoVolume
   ha_1.SetLineColor(6)
   top.AddNodeOverlap(ha_1,1,TGeoCombiTrans(0,36,4,TGeoRotation("ha_1",0,0,0)))
   
   ha_2 = geom.MakeTubs("ha_2",Iron,0,6,5,0,0) # TGeoVolume
   ha_2.SetLineColor(6)
   top.AddNodeOverlap(ha_2,1,TGeoCombiTrans(0,36,10,TGeoRotation("ha_2",0,180,0)))
   
   ha_3 = geom.MakeTubs("ha_3",Iron,0,1,12,0,0) # TGeoVolume
   ha_3.SetLineColor(28)
   top.AddNodeOverlap(ha_3,1,TGeoCombiTrans(0,36,8,TGeoRotation("ha_3",0,90,0)))
   
   ha_4 = geom.MakeTubs("ha_4",Iron,0,1,3,0,0) # TGeoVolume
   ha_4.SetLineColor(28)
   top.AddNodeOverlap(ha_4,1,TGeoCombiTrans(0,22,10,TGeoRotation("ha_4",0,0,0)))
   
   ha_5 = geom.MakeTubs("ha_5",Iron,0,1,3,0,0) # TGeoVolume
   ha_5.SetLineColor(28)
   top.AddNodeOverlap(ha_5,1,TGeoCombiTrans(0,46,10,TGeoRotation("ha_5",0,0,0)))
   
   ha_6 = geom.MakeTubs("ha_6",Iron,0,1,3,0,0) # TGeoVolume
   ha_6.SetLineColor(28)
   top.AddNodeOverlap(ha_6,1,TGeoCombiTrans(0,24,10,TGeoRotation("ha_6",0,0,0)))
   
   ha_7 = geom.MakeTubs("ha_7",Iron,0,1,3,0,0) # TGeoVolume
   ha_7.SetLineColor(28)
   top.AddNodeOverlap(ha_7,1,TGeoCombiTrans(0,48,10,TGeoRotation("ha_7",0,0,0)))
   
   ha_8 = geom.MakeBox("ha_8",Iron,2,0.5,2) # TGeoVolume
   ha_8.SetLineColor(19)
   top.AddNodeOverlap(ha_8,1,TGeoCombiTrans(-4.2,36,9,TGeoRotation("ha_8",0,45,0)))
   
   
   ha_9 = geom.MakeBox("ha_9",Iron,2,0.5,2) # TGeoVolume
   ha_9.SetLineColor(19)
   top.AddNodeOverlap(ha_9,1,TGeoCombiTrans(-4.2,36,9,TGeoRotation("ha_9",0,135,0)))
   
   
   
   h_1 = geom.MakeTubs("h_1",Iron,0,6,4.5,0,0) # TGeoVolume
   h_1.SetLineColor(5)
   top.AddNodeOverlap(h_1,1,TGeoCombiTrans(0,36,0,TGeoRotation("h_1",0,0,0)))
   
   h_2 = geom.MakeSphere("h_2",Iron,0,7.5,0,52.5,0,360) # TGeoVolume
   h_2.SetLineColor(5)
   top.AddNodeOverlap(h_2,1,TGeoCombiTrans(0,36,0,TGeoRotation("h_2",0,0,0)))
   
   h_3 = geom.MakeSphere("h_3",Iron,0,7.5,0,52.5,0,360) # TGeoVolume
   h_3.SetLineColor(5)
   top.AddNodeOverlap(h_3,1,TGeoCombiTrans(0,36,0,TGeoRotation("h_3",180,180,0)))
   
   h_4 = geom.MakeTubs("h_4",Iron,2.5,3.5,1.5,0,0) # TGeoVolume
   h_4.SetLineColor(5)
   top.AddNodeOverlap(h_4,1,TGeoCombiTrans(0,36,7.5,TGeoRotation("h_4",0,0,0)))
   
   
   fa1 = geom.MakeTubs("fa1",Iron,0,0.5,1,0,360) # TGeoVolume
   fa1.SetLineColor(12)
   top.AddNodeOverlap(fa1,1,TGeoCombiTrans(-5,38,1.5,TGeoRotation("fa1",-90,90,0)))
   
   fa2 = geom.MakeTubs("fa2",Iron,0,0.5,1,0,360) # TGeoVolume
   fa2.SetLineColor(12)
   top.AddNodeOverlap(fa2,1,TGeoCombiTrans(-5,34,1.5,TGeoRotation("fa2",-90,90,0)))
   
   fa1_1 = geom.MakeTubs("fa1_1",Iron,1,1.2,1,0,360) # TGeoVolume
   fa1_1.SetLineColor(12)
   top.AddNodeOverlap(fa1_1,1,TGeoCombiTrans(-5,38,1.5,TGeoRotation("fa1_1",-90,90,0)))
   
   fa2_1 = geom.MakeTubs("fa2_1",Iron,1,1.2,1,0,360) # TGeoVolume
   fa2_1.SetLineColor(12)
   top.AddNodeOverlap(fa2_1,1,TGeoCombiTrans(-5,34,1.5,TGeoRotation("fa2_1",-90,90,0)))
   
   fa3 = geom.MakeTubs("fa3",Iron,2,2.3,1,90,270) # TGeoVolume
   fa3.SetLineColor(12)
   top.AddNodeOverlap(fa3,1,TGeoCombiTrans(-5,36,-1,TGeoRotation("fa3",90,90,90)))
   
   
   
   m_1 = geom.MakeBox("m_1",Iron,7,8,4) # TGeoVolume
   m_1.SetLineColor(25)
   top.AddNodeOverlap(m_1,1,TGeoCombiTrans(0,36,-17,TGeoRotation("m_1",90,90,0)))
   
   m_2 = geom.MakeTubs("m_2",Iron,0,1,7,90,180) # TGeoVolume
   m_2.SetLineColor(25)
   top.AddNodeOverlap(m_2,1,TGeoCombiTrans(-3,36,-9,TGeoRotation("m_2",0,90,0)))
   
   m_3 = geom.MakeTubs("m_3",Iron,0,1,7,0,90) # TGeoVolume
   m_3.SetLineColor(25)
   top.AddNodeOverlap(m_3,1,TGeoCombiTrans(3,36,-9,TGeoRotation("m_3",0,90,0)))
   
   m_4 = geom.MakeBox("m_4",Iron,3,7,0.5) # TGeoVolume
   m_4.SetLineColor(25)
   top.AddNodeOverlap(m_4,1,TGeoCombiTrans(0,36,-8.5,TGeoRotation("m_4",90,0,90)))
   
   m_5 = geom.MakeTubs("m_5",Iron,0,1.5,1.2,0,0) # TGeoVolume
   m_5.SetLineColor(5)
   top.AddNodeOverlap(m_5,1,TGeoCombiTrans(0,36,-7.8,TGeoRotation("m_5",0,0,0)))
   
   m_6 = geom.MakeTrd2("m_6",Iron,4,4,0,2,8) # TGeoVolume
   m_6.SetLineColor(25)
   top.AddNodeOverlap(m_6,1,TGeoCombiTrans(0,29,-17,TGeoRotation("m_6",0,180,0)))
   
   m_7 = geom.MakeTrd2("m_7",Iron,4,4,0,2,8) # TGeoVolume
   m_7.SetLineColor(25)
   top.AddNodeOverlap(m_7,1,TGeoCombiTrans(0,43,-17,TGeoRotation("m_7",0,180,0)))
   
   
   md_1 = geom.MakeBox("md_1",Iron,4,8.5,0.7) # TGeoVolume
   md_1.SetLineColor(48)
   top.AddNodeOverlap(md_1,1,TGeoCombiTrans(0,36,-25.5,TGeoRotation("md_1",0,0,0)))
   
   md_2 = geom.MakeBox("md_2",Iron,3,0.4,2) # TGeoVolume
   md_2.SetLineColor(48)
   top.AddNodeOverlap(md_2,1,TGeoCombiTrans(0,36,-28,TGeoRotation("md_2",0,0,0)))
   
   d_1 = geom.MakeTrd2("d_1",Iron,3,4,4,4,7) # TGeoVolume
   d_1.SetLineColor(48)
   top.AddNodeOverlap(d_1,1,TGeoCombiTrans(0,40.5,-37.2,TGeoRotation("d_1",0,0,0)))
   
   d_2 = geom.MakeTrd2("d_2",Iron,3,4,4,4,7) # TGeoVolume
   d_2.SetLineColor(48)
   top.AddNodeOverlap(d_2,1,TGeoCombiTrans(0,31.5,-37.2,TGeoRotation("d_2",0,0,0)))
   
   d_3 = geom.MakeTubs("d_3",Iron,0,4,3.98,0,180) # TGeoVolume
   d_3.SetLineColor(48)
   top.AddNodeOverlap(d_3,1,TGeoCombiTrans(0,40.5,-30.2,TGeoRotation("d_3",0,90,0)))
   
   d_4 = geom.MakeTubs("d_4",Iron,0,4,3.98,0,180) # TGeoVolume
   d_4.SetLineColor(48)
   top.AddNodeOverlap(d_4,1,TGeoCombiTrans(0,31.5,-30.2,TGeoRotation("d_4",0,90,0)))
   
   d_5 = geom.MakeBox("d_5",Iron,4,4,1) # TGeoVolume
   d_5.SetLineColor(48)
   top.AddNodeOverlap(d_5,1,TGeoCombiTrans(-1,40.5,-43.7,TGeoRotation("d_5",0,0,0)))
   
   d_6 = geom.MakeBox("d_6",Iron,4,4,1) # TGeoVolume
   d_6.SetLineColor(48)
   top.AddNodeOverlap(d_6,1,TGeoCombiTrans(-1,31.5,-43.7,TGeoRotation("d_6",0,0,0)))
   
   
   a_1 = geom.MakeTubs("a_1",Iron,0,1.5,4,0,0) # TGeoVolume
   a_1.SetLineColor(45)
   top.AddNodeOverlap(a_1,1,TGeoCombiTrans(0,46,-15.1,TGeoRotation("a_1",0,20,45)))
   
   a_2 = geom.MakeSphere("a_2",Iron,0,1.48,0,180,0,200) # TGeoVolume
   a_2.SetLineColor(45)
   top.AddNodeOverlap(a_2,1,TGeoCombiTrans(0,44.6,-11.5,TGeoRotation("a_2",120,80,20)))
   
   a_3 = geom.MakeTubs("a_3",Iron,0,1.5,2.2,0,0) # TGeoVolume
   a_3.SetLineColor(45)
   top.AddNodeOverlap(a_3,1,TGeoCombiTrans(0,47.3,-20.6,TGeoRotation("a_3",300,0,40)))
   
   a_4 = geom.MakeTubs("a_4",Iron,0,1,1,0,0) # TGeoVolume
   a_4.SetLineColor(12)
   top.AddNodeOverlap(a_4,1,TGeoCombiTrans(0,47.3,-23.8,TGeoRotation("a_4",75,0,30)))
   
   a_5 = geom.MakeTubs("a_5",Iron,1.5,2.5,2,0,270) # TGeoVolume
   a_5.SetLineColor(12)
   top.AddNodeOverlap(a_5,1,TGeoCombiTrans(0,47.3,-26.5,TGeoRotation("a_5",-90,90,0)))
   
   
   
   
   Aa1 = geom.MakeTubs("Aa1",Iron,0,1.5,4,0,0) # TGeoVolume
   Aa1.SetLineColor(45)
   top.AddNodeOverlap(Aa1,1,TGeoCombiTrans(0,26,-15.1,TGeoRotation("Aa1",0,-20,-45)))
   
   Aa2 = geom.MakeSphere("Aa2",Iron,0,1.48,0,180,0,200) # TGeoVolume
   Aa2.SetLineColor(45)
   top.AddNodeOverlap(Aa2,1,TGeoCombiTrans(0,27.4,-11.5,TGeoRotation("Aa2",120,80,-20)))
   
   Aa3 = geom.MakeTubs("Aa3",Iron,0,1.5,2.2,0,0) # TGeoVolume
   Aa3.SetLineColor(45)
   top.AddNodeOverlap(Aa3,1,TGeoCombiTrans(0,24.7,-20.6,TGeoRotation("Aa3",-300,0,-40)))
   
   Aa4 = geom.MakeTubs("Aa4",Iron,0,1,1,0,0) # TGeoVolume
   Aa4.SetLineColor(12)
   top.AddNodeOverlap(Aa4,1,TGeoCombiTrans(0,24.7,-23.8,TGeoRotation("Aa4",-75,0,-30)))
   
   Aa5 = geom.MakeTubs("Aa5",Iron,1.5,2.5,2,0,270) # TGeoVolume
   Aa5.SetLineColor(12)
   top.AddNodeOverlap(Aa5,1,TGeoCombiTrans(0,24.7,-26.5,TGeoRotation("Aa5",90,90,00)))
   
   
   
   bag1 = geom.MakeBox("bag1",Iron,10,4,6) # TGeoVolume
   bag1.SetLineColor(19)
   top.AddNodeOverlap(bag1,1,TGeoCombiTrans(0,48,-36,TGeoRotation("bag1",0,0,0)))
   
   bag2 = geom.MakeTubs("bag2",Iron,3,4,1,180,360) # TGeoVolume
   bag2.SetLineColor(19)
   top.AddNodeOverlap(bag2,1,TGeoCombiTrans(0,48,-30,TGeoRotation("bag2",0,270,0)))
   
   
   well = geom.MakeBox("well",Iron,5,10,3) # TGeoVolume
   well.SetLineColor(18)
   top.AddNodeOverlap(well,1,TGeoCombiTrans(-26.5,-17,-42,TGeoRotation("well",0,0,0)))
   
   
   K5 = geom.MakeTubs("K5",Iron,0,3,3,0,0) # TGeoVolume
   K5.SetLineColor(18)
   top.AddNodeOverlap(K5,1,TGeoCombiTrans(-27,-12.5,-39,TGeoRotation("K5",0,0,0)))
   
   K4 = geom.MakeTubs("K4",Iron,0,3,3,0,0) # TGeoVolume
   K4.SetLineColor(18)
   top.AddNodeOverlap(K4,1,TGeoCombiTrans(-27,-21.5,-39,TGeoRotation("K4",0,0,0)))
   
   
   
   #==============Board=========
   nB = " "*100
   Z = Y = 0
   bo1 = TGeoVolume()
   
   while(Y<6):
      while(Z<10):
         nB = sprintfPy(nB, "B{:d}_Y{:d}",Z,Y)
         bo1=geom.MakeTubs(nB,Iron,0,3,3,0,0)
         bo1.SetLineColor(8)
         top.AddNodeOverlap(bo1,1,TGeoCombiTrans(-27+(Y*9),-21.5+(Z*9),-45,TGeoRotation("bo1",0,0,0)))
         (Z:=Z+1)
         
      (Y:=Y+1)
      Z=0
      
   
   
   bo2 = geom.MakeBox("bo2",Iron,27,45,3) # TGeoVolume
   bo2.SetLineColor(8)
   top.AddNodeOverlap(bo2,1,TGeoCombiTrans(-4.5,18,-48,TGeoRotation("bo2",0,0,0)))
   
   
   top.SetVisibility(False)
   geom.CloseGeometry()
   
   #top.Draw("ogl")
   top.Draw("x3d")
   #input("\n\nPress Enter to quit lego() function")

   
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
   lego()
