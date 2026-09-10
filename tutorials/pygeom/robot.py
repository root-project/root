## \file
## \ingroup tutorial_geom
## Drawing a famous Korean robot, TaekwonV, using ROOT geometry class.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: `%run robot.py` in ipython3 interpreter, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to all of ROOT team
##
## \image html geom_robot.width = png
## \macro_code
##
## \author Jin Hui Hwang, Dept. of Physics, Univ. of Seoul
## \translator P. P.

import ROOT

TGeoManager =  		ROOT.TGeoManager
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

sin = ROOT.sin
cos = ROOT.cos
#Cos = ROOT.TMath.Cos
#Sin = ROOT.TMath.Sin



def sprintfPy(buffer, StringFormat, *args):
   buffer = StringFormat.format(*args)
   return buffer

#def robot():
class robot:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager
   
   #Robot =  TGeoManager("Robot","This is Taegwon V")
   ProcessLine('''
   TGeoManager *Robot = new TGeoManager("Robot","This is Taegwon V");
   ''')
   Robot = ROOT.Robot
   
   vacuum = TGeoMaterial("vacuum",0,0,0)
   Fe = TGeoMaterial("Fe",55.845,26,7.87)
   
   Air = TGeoMedium("Vacuum",0,vacuum)
   Iron = TGeoMedium("Iron",1,Fe)
   
   # create volume
   
   top = Robot.MakeBox("top",Air,1000,1000,1000)
   Robot.SetTopVolume(top)
   Robot.SetTopVisible(False)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom->SetTopVisible(1);
   
   
   
   # head
   Band = Robot.MakeEltu("Band",Iron,20,20,2.5)
   Band.SetLineColor(12)
   Band.SetFillColor(12)
   Band_b = Robot.MakeSphere("Band_b",Iron,0,2,0,180,180,360)
   Band_b.SetLineColor(2)
   Band_b.SetFillColor(2)
   Head = Robot.MakeSphere("Head",Iron,0,19,0,180,180,360)
   Head.SetLineColor(17)
   Head.SetFillColor(17)
   Horn = Robot.MakeSphere("Horn",Iron,0,10,60,180,240,300)
   
   # drawing head
   top.AddNodeOverlap(Band,1, TGeoTranslation(0,0,90))
   Phi = 3.14
   N = 10
   
   # (i = int; i<=N; i++)
   #for i in range(i<=N):
   for i in range(N+1):
      top.AddNodeOverlap(Band_b,1, TGeoCombiTrans(sin(2*Phi/N*i)*19,-cos(2*Phi/N*i)*19,90,
       TGeoRotation("R1",-90+(360/N*i),-90,90)))
      
   top.AddNodeOverlap(Head,1, TGeoCombiTrans(0,0,87.5, TGeoRotation("R2",0,-90,0)))
   
   name = " "*50
   pcs = 30
   # (i = int; i<pcs; i++)
   #for i in range(i<pcs):
   for i in range(1, pcs):
      name = sprintfPy(name,"Horn{:d}",i)
      Horn=Robot.MakeSphere(name,Iron,
         10- 10/pcs*i, 10, 180-(120/pcs)*i, 180-((120/pcs)*(i-1)), 240, 300)
      Horn.SetLineColor(2)
      Horn.SetFillColor(2)
      top.AddNodeOverlap(Horn,1, TGeoCombiTrans(0,8,102, TGeoRotation("R2",0,140,0)))
      top.AddNodeOverlap(Horn,1, TGeoCombiTrans(0,-8,102, TGeoRotation("R2",180,140,0)))
      
   
   # face
   Migan = Robot.MakeGtra("Migan",Iron,3,0,0,0,3,2,11,0,3,3,11,0)
   Migan.SetLineColor(17)
   Migan.SetFillColor(17)
   Ko = Robot.MakeGtra("Ko",Iron,7,0,0,0,3,1,5,0,3,2,5,0)
   Ko.SetLineColor(17)
   Ko.SetFillColor(17)
   Ko_m = Robot.MakeBox("Ko_m",Iron,2,8,4)
   Ko_m.SetLineColor(17)
   Ko_m.SetFillColor(17)
   Bol_1 = Robot.MakeBox("Bol_1",Iron,7,5.5,7)
   Bol_1.SetLineColor(17)
   Bol_1.SetFillColor(17)
   Bol_2 = Robot.MakeGtra("Bol_2",Iron,1,0,0,0,7,0,9,0,7,0,9,0)
   Bol_2.SetLineColor(17)
   Bol_2.SetFillColor(17)
   Noon = Robot.MakeBox("Noon",Iron,1,10,5)
   Noon.SetLineColor(12)
   Noon.SetFillColor(12)
   Tuck = Robot.MakeBox("Tuck",Iron,2,10,5.5)
   Tuck.SetLineColor(2)
   Tuck.SetFillColor(2)
   Tuck_1 = Robot.MakeBox("Tuck_1",Iron,2,9,1)
   Tuck_1.SetLineColor(2)
   Tuck_1.SetFillColor(2)
   Tuck_2 = Robot.MakeBox("Tuck_2",Iron,3,1,14)
   Tuck_2.SetLineColor(2)
   Tuck_2.SetFillColor(2)
   Tuck_j = Robot.MakeSphere("Tuck_j",Iron,0,3.5,0,180,0,360)
   Tuck_j.SetLineColor(5)
   Tuck_j.SetFillColor(5)
   Ear = Robot.MakeCons("Ear",Iron,1,0,3,0,3,0,360)
   Ear.SetLineColor(5)
   Ear.SetFillColor(5)
   Ear_2 = Robot.MakeCone("Ear_2",Iron,5,0,0,0,3)
   Ear_2.SetLineColor(5)
   Ear_2.SetFillColor(5)
   
   # drawing face
   top.AddNodeOverlap(Migan,1, TGeoCombiTrans(-15,0,88, TGeoRotation("R2",-90,40,0)))
   top.AddNodeOverlap(Ko,1, TGeoCombiTrans(-15,0,76.5, TGeoRotation("R2",-90,-20,0)))
   top.AddNodeOverlap(Ko_m,1, TGeoTranslation(-9,0,68))
   top.AddNodeOverlap(Bol_1,1, TGeoCombiTrans(-7,2,76, TGeoRotation("R2",-30,-10,0)))
   top.AddNodeOverlap(Bol_1,1, TGeoCombiTrans(-7,-2,76, TGeoRotation("R2",30,10,0)))
   top.AddNodeOverlap(Bol_2,1, TGeoCombiTrans(-6.5,-10.5,76, TGeoRotation("R2",-15,-90,-30)))
   top.AddNodeOverlap(Bol_2,1, TGeoCombiTrans(-4,-12.5,82.5, TGeoRotation("R2",-20,-90,-95)))
   top.AddNodeOverlap(Bol_2,1, TGeoCombiTrans(-7.5,10.5,76, TGeoRotation("R2",20,-90,-30)))
   top.AddNodeOverlap(Bol_2,1, TGeoCombiTrans(-4,12.5,82.5, TGeoRotation("R2",20,-90,-95)))
   top.AddNodeOverlap(Noon,1, TGeoCombiTrans(-5,-7,86, TGeoRotation("R2",60,0,0)))
   top.AddNodeOverlap(Noon,1, TGeoCombiTrans(-5,7,86, TGeoRotation("R2",-60,0,0)))
   top.AddNodeOverlap(Tuck,1, TGeoTranslation(-12,0,62.5))
   # (i = int; i<10; i++)
   for i in range(10):
      top.AddNodeOverlap(Tuck_1,1, TGeoCombiTrans(-4.2,11,61+i, TGeoRotation("R2",90,-20,20)))
      top.AddNodeOverlap(Tuck_1,1, TGeoCombiTrans(-4.2,-11,61+i, TGeoRotation("R2",90,-20,-20)))
      
   top.AddNodeOverlap(Tuck_2,1, TGeoTranslation(2,-15.1,76))
   top.AddNodeOverlap(Tuck_2,1, TGeoTranslation(2,15.1,76))
   top.AddNodeOverlap(Tuck_j,1, TGeoTranslation(-13,0,62.5))
   top.AddNodeOverlap(Ear,1, TGeoCombiTrans(2,-16.5,80, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(Ear,1, TGeoCombiTrans(2,16.5,80, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Ear_2,1, TGeoCombiTrans(2,-20,80, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(Ear_2,1, TGeoCombiTrans(2,20,80, TGeoRotation("R2",0,90,0)))
   
   
   # (i = int; i<28; i+=1)
   for i in range(1, 28):
      a = i*2
      Hear = Robot.MakeCons("Hear",Iron,3,13+a,16+a,13+a,16+a,-60-a,60+a)
      if i<27:
         Hear.SetLineColor(12)
         Hear.SetFillColor(12)
         
      else:
         Hear.SetLineColor(2)
         Hear.SetFillColor(2)
         
      top.AddNodeOverlap(Hear,1, TGeoTranslation(0,0,89-i))
      
   # (i = int; i<28; i+=1)
   for i in range(1, 28):
      a = i*2
      Hear = Robot.MakeCons("Hear",Iron,3,13+a,16+a,13+a,16+a,-70-a,-60-a)
      Hear.SetLineColor(2)
      Hear.SetFillColor(2)
      top.AddNodeOverlap(Hear,1, TGeoTranslation(0,0,89-i))
      
   # (i = int; i<28; i+=1)
   for i in range(1, 28):
      a = i*2
      Hear = Robot.MakeCons("Hear",Iron,3,13+a,16+a,13+a,16+a,60+a,70+a)
      Hear.SetLineColor(2)
      Hear.SetFillColor(2)
      top.AddNodeOverlap(Hear,1, TGeoTranslation(0,0,89-i))
      
   
   # neck
   Mock = Robot.MakeTrd2("Mock",Iron,1,1,7,6.5,20)
   Mock.SetLineColor(17)
   Mock.SetFillColor(17)
   Mock_1 = Robot.MakeTrd2("Mock_1",Iron,1,1,6,5,20)
   Mock_1.SetLineColor(17)
   Mock_1.SetFillColor(17)
   Mock_s = Robot.MakeTrd2("Mock_s",Iron,1,1,5,4.5,20)
   Mock_s.SetLineColor(17)
   Mock_s.SetFillColor(17)
   
   # drawing neck
   top.AddNodeOverlap(Mock,1, TGeoCombiTrans(-5,4.7,50, TGeoRotation("R2",-30,0,-10)))
   top.AddNodeOverlap(Mock,1, TGeoCombiTrans(-5,-4.7,50, TGeoRotation("R2",30,0,10)))
   top.AddNodeOverlap(Mock_1,1, TGeoCombiTrans(11,-4,50, TGeoRotation("R2",130,-8,10)))
   top.AddNodeOverlap(Mock_1,1, TGeoCombiTrans(11,4,50, TGeoRotation("R2",-130,8,-10)))
   top.AddNodeOverlap(Mock_s,1, TGeoCombiTrans(2.5,9,50, TGeoRotation("R2",90,0,0)))
   top.AddNodeOverlap(Mock_s,1, TGeoCombiTrans(2.5,-9,50, TGeoRotation("R2",90,0,0)))
   
   
   # chest
   Gasem = Robot.MakeBox("Gasem",Iron,16,50,20)
   Gasem.SetLineColor(12)
   Gasem.SetFillColor(12)
   Gasem_b1 = Robot.MakeSphere("Gasem_b1",Iron,0,15,0,180,0,360)
   Gasem_b1.SetLineColor(12)
   Gasem_b1.SetFillColor(12)
   Gasem_b2 = Robot.MakeSphere("Gasem_b2",Iron,0,13,0,180,0,360)
   Gasem_b2.SetLineColor(12)
   Gasem_b2.SetFillColor(12)
   Gasem_1 = Robot.MakeEltu("Gasem_1",Iron,13,13,20)
   Gasem_1.SetLineColor(12)
   Gasem_1.SetFillColor(12)
   Gasem_2 = Robot.MakeEltu("Gasem_2",Iron,13,13,19)
   Gasem_2.SetLineColor(12)
   Gasem_2.SetFillColor(12)
   Gasem_3 = Robot.MakeCone("Gasem_3",Iron,19,0,13,0,15)
   Gasem_3.SetLineColor(12)
   Gasem_3.SetFillColor(12)
   Gasem_4 = Robot.MakeEltu("Gasem_4",Iron,15,15,16)
   Gasem_4.SetLineColor(12)
   Gasem_4.SetFillColor(12)
   Gasem_5 = Robot.MakeEltu("Gasem_5",Iron,13,13,16)
   Gasem_5.SetLineColor(12)
   Gasem_5.SetFillColor(12)
   Gasem_m1 = Robot.MakeBox("Gasem_m1",Iron,19,19,5)
   Gasem_m1.SetLineColor(12)
   Gasem_m1.SetFillColor(12)
   Gasem_m2 = Robot.MakeTrd2("Gasem_m2",Iron,13,15,2,2,19)
   Gasem_m2.SetLineColor(12)
   Gasem_m2.SetFillColor(12)
   V = Robot.MakeTrd2("V",Iron,2,2,22,30,4)
   V.SetLineColor(2)
   V.SetFillColor(2)
   V_m = Robot.MakeBox("V_m",Iron,2,7,1)
   V_m.SetLineColor(2)
   V_m.SetFillColor(2)
   
   # drawing chest
   top.AddNodeOverlap(Gasem,1, TGeoTranslation(4,0,19))
   top.AddNodeOverlap(Gasem_b1,1, TGeoTranslation(-12,50,35))
   top.AddNodeOverlap(Gasem_b1,1, TGeoTranslation(-12,-50,35))
   top.AddNodeOverlap(Gasem_b1,1, TGeoTranslation(20,50,35))
   top.AddNodeOverlap(Gasem_b1,1, TGeoTranslation(20,-50,35))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(-12,50,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(-12,-50,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(20,50,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(20,-50,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(20,10,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(20,-10,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(-12,10,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(-12,-10,-5))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(20,10,35))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(20,-10,35))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(-12,10,35))
   top.AddNodeOverlap(Gasem_b2,1, TGeoTranslation(-12,-10,35))
   top.AddNodeOverlap(Gasem_1,1, TGeoCombiTrans(20,31,-5, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_1,1, TGeoCombiTrans(20,-31,-5, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_1,1, TGeoCombiTrans(-12,31,-5, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_1,1, TGeoCombiTrans(-12,-31,-5, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_2,1, TGeoCombiTrans(20,10,13, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_2,1, TGeoCombiTrans(20,-10,13, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_2,1, TGeoCombiTrans(-12,10,13, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_2,1, TGeoCombiTrans(-12,-10,13, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(-12,50,16, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(-12,-50,16, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(20,50,16, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(20,-50,16, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(-12,31,35, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(-12,-31,35, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(20,31,35, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(Gasem_3,1, TGeoCombiTrans(20,-31,35, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_4,1, TGeoCombiTrans(4,-50,35, TGeoRotation("R2",90,90,0)))
   top.AddNodeOverlap(Gasem_4,1, TGeoCombiTrans(4,50,35, TGeoRotation("R2",90,90,0)))
   top.AddNodeOverlap(Gasem_5,1, TGeoCombiTrans(4,-50,-5, TGeoRotation("R2",90,90,0)))
   top.AddNodeOverlap(Gasem_5,1, TGeoCombiTrans(4,50,-5, TGeoRotation("R2",90,90,0)))
   top.AddNodeOverlap(Gasem_m1,1, TGeoCombiTrans(-22,30,16, TGeoRotation("R2",90,88,0)))
   top.AddNodeOverlap(Gasem_m1,1, TGeoCombiTrans(-22,-30,16, TGeoRotation("R2",90,88,0)))
   top.AddNodeOverlap(Gasem_m1,1, TGeoCombiTrans(29,30,16, TGeoRotation("R2",90,92,0)))
   top.AddNodeOverlap(Gasem_m1,1, TGeoCombiTrans(29,-30,16, TGeoRotation("R2",90,92,0)))
   top.AddNodeOverlap(Gasem_m2,1, TGeoCombiTrans(2,-62,16, TGeoRotation("R2",0,3,0)))
   top.AddNodeOverlap(Gasem_m2,1, TGeoCombiTrans(2,62,16, TGeoRotation("R2",0,-3,0)))
   top.AddNodeOverlap(Gasem_m2,1, TGeoCombiTrans(2,-30,47.5, TGeoRotation("R2",0,87,0)))
   top.AddNodeOverlap(Gasem_m2,1, TGeoCombiTrans(2,30,47.5, TGeoRotation("R2",0,-87,0)))
   top.AddNodeOverlap(Gasem_m2,1, TGeoCombiTrans(2,-30,-16, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gasem_m2,1, TGeoCombiTrans(2,30,-16, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(V,1, TGeoCombiTrans(-30,18.3,16, TGeoRotation("R2",0,-135,0)))
   top.AddNodeOverlap(V,1, TGeoCombiTrans(-30,-18.3,16, TGeoRotation("R2",0,135,0)))
   top.AddNodeOverlap(V_m,1, TGeoTranslation(-30,-37,35))
   top.AddNodeOverlap(V_m,1, TGeoTranslation(-30,37,35))
   
   # abdomen
   Bea = Robot.MakeEltu("Bea",Iron,20,37,25)
   Bea.SetLineColor(17)
   Bea.SetFillColor(17)
   Bea_d = Robot.MakeEltu("Bea_d",Iron,21,36,5)
   Bea_d.SetLineColor(12)
   Bea_d.SetFillColor(12)
   Beakop = Robot.MakeEltu("Beakop",Iron,15,25,5)
   Beakop.SetLineColor(10)
   Beakop.SetFillColor(10)
   
   # drawing abdomen
   top.AddNodeOverlap(Bea,1, TGeoTranslation(3,0,-30))
   top.AddNodeOverlap(Bea_d,1, TGeoTranslation(3,0,-10))
   top.AddNodeOverlap(Beakop,1, TGeoCombiTrans(-12.1,0,-50,  TGeoRotation("R2",90,90,0)))
   
   # Gungdi
   Gungdi = Robot.MakeEltu("Gungdi",Iron,25,50,18)
   Gungdi.SetLineColor(12)
   Gungdi.SetFillColor(12)
   Gungdi_d = Robot.MakeEltu("Gungdi_d",Iron,5,5,5)
   Gungdi_d.SetLineColor(2)
   Gungdi_d.SetFillColor(2)
   
   # drawing Gungdi
   top.AddNodeOverlap(Gungdi,1, TGeoTranslation(3,0,-70))
   # (i = int; i<30; i++)
   for i in range(30):
      Gungdi_j = Robot.MakeEltu("Gungdi_j",Iron,24-0.2*i,49-0.5*i,1)
      Gungdi_j.SetLineColor(12)
      Gungdi_j.SetFillColor(12)
      top.AddNodeOverlap(Gungdi_j,1, TGeoTranslation(3,0,-51+0.5*i))
      
   # (i = int; i<40; i++)
   for i in range(40):
      if i<16:
         Gungdi_h = Robot.MakeEltu("Gungdi_h",Iron,24-0.1*i,49-0.3*i,1)
         Gungdi_h.SetLineColor(12)
         Gungdi_h.SetFillColor(12)
         top.AddNodeOverlap(Gungdi_h,1, TGeoTranslation(3,0,-88-0.5*i))
         
      else:
         Gungdi_h = Robot.MakeEltu("Gungdi_h",Iron,27-0.3*i,52-0.5*i,1)
         Gungdi_h.SetLineColor(12)
         Gungdi_h.SetFillColor(12)
         top.AddNodeOverlap(Gungdi_h,1, TGeoTranslation(3,0,-89-0.5*i))
         
      
   top.AddNodeOverlap(Gungdi_d,1, TGeoCombiTrans(3,-45,-62, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gungdi_d,1, TGeoCombiTrans(3,-45,-78, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gungdi_d,1, TGeoCombiTrans(3,45,-62, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Gungdi_d,1, TGeoCombiTrans(3,45,-78, TGeoRotation("R2",0,90,0)))
   
   # feet
   Jang = Robot.MakeEltu("Jang",Iron,18,18,50)
   Jang.SetLineColor(17)
   Jang.SetFillColor(17)
   Jong = Robot.MakeEltu("Jong",Iron,22,22,50)
   Jong.SetLineColor(12)
   Jong.SetFillColor(12)
   Bal = Robot.MakeSphere("Bal",Iron,0,22,0,180,180,360)
   Bal.SetLineColor(12)
   Bal.SetFillColor(12)
   
   # drawing Dary
   top.AddNodeOverlap(Jang,1, TGeoCombiTrans(3,-25,-120, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Jang,1, TGeoCombiTrans(3,25,-120, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Jong,1, TGeoCombiTrans(3,-25,-220, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Jong,1, TGeoCombiTrans(3,25,-220, TGeoRotation("R2",0,0,0)))
   # (i = int; i<30; i++)
   for i in range(30):
      Mu = Robot.MakeCons("Mu",Iron,1,0,22.1,0,22.1,120+2*i,-120-2*i)
      Mu.SetLineColor(4)
      Mu.SetFillColor(4)
      top.AddNodeOverlap(Mu,1, TGeoTranslation(3,-25,-171-i))
      top.AddNodeOverlap(Mu,1, TGeoTranslation(3,25,-171-i))
      
      
   top.AddNodeOverlap(Bal,1, TGeoCombiTrans(-10,-25,-270, TGeoRotation("R2",270,-90,0)))
   top.AddNodeOverlap(Bal,1, TGeoCombiTrans(-10,25,-270, TGeoRotation("R2",270,-90,0)))
   
   # arms
   S = Robot.MakeSphere("S",Iron,0,25,0,180,180,360)
   S.SetLineColor(17)
   S.SetFillColor(17)
   S_1 = Robot.MakeSphere("S_1",Iron,0,15,0,180,0,360)
   S_1.SetLineColor(17)
   S_1.SetFillColor(17)
   Pal = Robot.MakeEltu("Pal",Iron,15,15,30)
   Pal.SetLineColor(17)
   Pal.SetFillColor(17)
   Fal = Robot.MakeEltu("Fal",Iron,17,17,30)
   Fal.SetLineColor(4)
   Fal.SetFillColor(4)
   Bbul = Robot.MakeCone("Bbul",Iron,8,0,0,0,5)
   Bbul.SetLineColor(17)
   Bbul.SetFillColor(17)
   
   # drawing arms
   top.AddNodeOverlap(S,1, TGeoCombiTrans(3,73,30, TGeoRotation("R2",0,-30,0)))
   top.AddNodeOverlap(S,1, TGeoCombiTrans(3,-73,30, TGeoRotation("R2",0,210,0)))
   top.AddNodeOverlap(S_1,1, TGeoCombiTrans(3,-73,27, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(S_1,1, TGeoCombiTrans(3,73,27, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Pal,1, TGeoCombiTrans(3,-73,-5, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Pal,1, TGeoCombiTrans(3,73,-5, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Fal,1, TGeoCombiTrans(3,-73,-60, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Fal,1, TGeoCombiTrans(3,73,-60, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Bbul,1, TGeoCombiTrans(3,-97,-72, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(Bbul,1, TGeoCombiTrans(3,-97,-48, TGeoRotation("R2",0,-90,0)))
   top.AddNodeOverlap(Bbul,1, TGeoCombiTrans(3,97,-72, TGeoRotation("R2",0,90,0)))
   top.AddNodeOverlap(Bbul,1, TGeoCombiTrans(3,97,-48, TGeoRotation("R2",0,90,0)))
   
   # hands
   Son_d = Robot.MakeBox("Son_d",Iron,15,15,7)
   Son_d.SetLineColor(4)
   Son_d.SetFillColor(4)
   Son_g = Robot.MakeBox("Son_g",Iron,4,10,4)
   Son_g.SetLineColor(4)
   Son_g.SetFillColor(4)
   Son_g1 = Robot.MakeBox("Son_g1",Iron,6,6,6)
   Son_g1.SetLineColor(4)
   Son_g1.SetFillColor(4)
   Son_g2 = Robot.MakeBox("Son_g2",Iron,8,3,3)
   Son_g2.SetLineColor(4)
   Son_g2.SetFillColor(4)
   Last_b = Robot.MakeCone("Last_b",Iron,10,0,0,0,4)
   Last_b.SetLineColor(17)
   Last_b.SetFillColor(17)
   Last = Robot.MakeSphere("Last",Iron,0,3,0,180,0,360)
   Last.SetLineColor(2)
   Last.SetFillColor(2)
   
   #drawing hands
   top.AddNodeOverlap(Son_d,1, TGeoCombiTrans(3,-80,-105, TGeoRotation("R2",0,90,0)))
   # (i = int; i<4; i++)
   for i in range(4):
      top.AddNodeOverlap(Son_g,1, TGeoCombiTrans(-6+6*i,-72,-118, TGeoRotation("R2",0,-10,0)))
      
   # (i = int; i<4; i++)
   for i in range(4):
      top.AddNodeOverlap(Son_g,1, TGeoCombiTrans(-6+6*i,-67,-113, TGeoRotation("R2",0,110,0)))
      
   top.AddNodeOverlap(Son_g1,1, TGeoCombiTrans(-5,-70,-98, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Son_g2,1, TGeoCombiTrans(-5,-65,-102, TGeoRotation("R2",0,60,0)))
   top.AddNodeOverlap(Son_d,1, TGeoCombiTrans(3,80,-105, TGeoRotation("R2",0,90,0)))
   # (i = int; i<4; i++)
   for i in range(4):
      top.AddNodeOverlap(Son_g,1, TGeoCombiTrans(-6+6*i,72,-118, TGeoRotation("R2",0,10,0)))
      
   # (i = int; i<4; i++)
   for i in range(4):
      top.AddNodeOverlap(Son_g,1, TGeoCombiTrans(-6+6*i,67,-113, TGeoRotation("R2",0,70,0)))
      
   top.AddNodeOverlap(Son_g1,1, TGeoCombiTrans(-5,70,-98, TGeoRotation("R2",0,0,0)))
   top.AddNodeOverlap(Son_g2,1, TGeoCombiTrans(-5,65,-102, TGeoRotation("R2",0,60,0)))
   top.AddNodeOverlap(Last_b,1, TGeoCombiTrans(3,-88,-103, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last_b,1, TGeoCombiTrans(12,-88,-103, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last_b,1, TGeoCombiTrans(-7,-88,-103, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last_b,1, TGeoCombiTrans(3,88,-103, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last_b,1, TGeoCombiTrans(12,88,-103, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last_b,1, TGeoCombiTrans(-7,88,-103, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last,1, TGeoCombiTrans(3,-88,-112, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last,1, TGeoCombiTrans(12,-88,-112, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last,1, TGeoCombiTrans(-7,-88,-112, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last,1, TGeoCombiTrans(3,88,-112, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last,1, TGeoCombiTrans(12,88,-112, TGeoRotation("R2",0,180,0)))
   top.AddNodeOverlap(Last,1, TGeoCombiTrans(-7,88,-112, TGeoRotation("R2",0,180,0)))
   
   # (i = int; i<20; i+=1)
   for i in range(1, 20):
      if i<7:
         Effect = Robot.MakeCons("Effect",Iron,3,20/sin(i),21/sin(i),20/sin(i),21/sin(i),0,70)
         Effect.SetLineColor(9)
         Effect.SetFillColor(9)
         top.AddNodeOverlap(Effect,1, TGeoTranslation(3,0,-280))
         
      if 6<i and i<10:
         Effect = Robot.MakeCons("Effect",Iron,5,20/sin(i),21/sin(i),20/sin(i),21/sin(i),50,120)
         Effect.SetLineColor(38)
         Effect.SetFillColor(38)
         top.AddNodeOverlap(Effect,1, TGeoTranslation(3,0,-280))
         
      if 9<i and i<20:
         Effect = Robot.MakeCons("Effect",Iron,4,20/sin(i),21/sin(i),20/sin(i),21/sin(i),200,330)
         Effect.SetLineColor(33)
         Effect.SetFillColor(33)
         top.AddNodeOverlap(Effect,1, TGeoTranslation(3,0,-280))
         
      
   
   
   #close geometry
   top.SetVisibility(False)
   Robot.CloseGeometry()
   
   # in GL viewer
   #top.Draw("ogl")
   top.Draw("x3d")
   #TODO: solve name conflicts with sprintfPy. Declare a name inside every loop which makes an object.
   # for ...
   #    name = sprintfPy(name, "PatternName{}_{}", i, j) # Abstract line to be added.
   #    Object = TGeoManager_Object.MakeObject( name, ...)
   #
   #input("Press enter to quit Robot")
   

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
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!



if __name__ == "__main__":
   robot()
