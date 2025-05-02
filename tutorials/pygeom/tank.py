## \file
## \ingroup tutorial_geom
## Drawing a fine tank, using ROOT geometry class.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: `.x tank.C` in ROOT terminal, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to all of ROOT team
##
## \image html geom_tank.png width=800px
## \macro_code
##
## \author Dong Gyu Lee (ravirus@hanmail.net), Dept. of Physics, Univ. of Seoul
## \translator P. P.


import ROOT

#Not to use:from ROOT import TGeoManager
TGeoManager = ROOT.TGeoManager
TGeoTranslation = ROOT.TGeoTranslation
TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium 
TGeoCombiTrans = ROOT.TGeoCombiTrans 
TGeoRotation = ROOT.TGeoRotation


#TMath = ROOT.TMath
#Sin = ROOT.TMath.Sin
#Cos = ROOT.TMath.Cos
sin = ROOT.sin
cos = ROOT.cos

Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine

gGeoManager = ROOT.gGeoManager

def sprintfPy(buffer, FormatString, *args):
   buffer = FormatString.format(*args)
   return buffer

#def tank():
class tank:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager

   #Not to use: geom =  TGeoManager("geom","My 3D Project")
   Declare('''
   TGeoManager *geom = new TGeoManager("geom","My 3D Project");
   ''')
   geom = ROOT.geom 
   
   
   #------------------Creat materials-----------------------------
   vacuum =  TGeoMaterial("vacuum",0,0,0)
   Fe =  TGeoMaterial("Fe",55.84,26.7,7.87)
   Cu =  TGeoMaterial("Cu",63.549,29,8.92)
   
   #------------------Creat media----------------------------------
   Air =  TGeoMedium("Air",0,vacuum)
   Iron =  TGeoMedium("Iron",1,Fe)
   
   #------------------Create TOP volume----------------------------
   top = geom.MakeBox("top",Air,100,100,100)
   geom.SetTopVolume(top)
   geom.SetTopVisible(False)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom->SetTopVisible(1);
   
   
   #-----------------Create Object volume--------------------------
   
   
   #Now, we start real shape
   
   #UpperBody
   pl =geom.MakeBox("pl",Iron,210,93,20)
   pl.SetLineColor(42)
   pl1 =geom.MakeBox("pl1",Iron,217,50,5)
   pl1.SetLineColor(42)
   pl2 =geom.MakeTrd2("pl2",Iron,219,150,50,40,10)
   pl2.SetLineColor(42)
   plu =geom.MakeTrd2("plu",Iron,210,70,100,100,5)
   plu.SetLineColor(42)
   top.AddNodeOverlap(plu,1,TGeoTranslation(0,0,-105))
   sp =geom.MakeTubs("sp",Iron,30,40,50,10,60)#Small Plate front
   sp.SetLineColor(42)
   
   #Top which will have the gun
   tp =geom.MakeSphere("tp",Iron,0,100,67,90,0,360)#tp is Top with gun
   tp.SetLineColor(12)
   tp1 =geom.MakeSphere("tp1",Iron,90,190,0,29,0,360)#tp1 is Top with roof
   tp1.SetLineColor(12)
   mgg =geom.MakeTubs("mgg",Iron,0,25,30,42,136)#Main Gun Guard
   mgg.SetLineColor(12)
   mgg1 =geom.MakeTrd2("mgg1",Iron,30.5,45,19,30,35)
   mgg1.SetLineColor(12)
   
   top.AddNodeOverlap(mgg1,1,TGeoCombiTrans(-57,0,-63,TGeoRotation("mgg",90,90,0)))
   top.AddNodeOverlap(mgg,1,TGeoCombiTrans(-75,0,-63,TGeoRotation("mgg",0,90,90)))
   
   #Small Top infront Top
   stp =geom.MakeSphere("stp",Iron,0,30,67,90,0,360)#Top for driver
   stp.SetLineColor(12)
   stp1 =geom.MakeSphere("stp1",Iron,115,120,0,12,0,360)#Top with roof
   stp1.SetLineColor(12)
   stpo1 =geom.MakeBox("stpo1",Iron,3,1,5)
   stpo1.SetLineColor(42)#Small T P Option 1
   
   top.AddNodeOverlap(stpo1,1,TGeoTranslation(-93,-32,-95))
   top.AddNodeOverlap(stpo1,1,TGeoTranslation(-93,-38,-95))
   top.AddNodeOverlap(stp,1,TGeoTranslation(-120,-35,-108))
   top.AddNodeOverlap(stp1,1,TGeoCombiTrans(-185,-35,-168,TGeoRotation("stp1",90,40,0)))
   
   #The Main Gun1 with AddNodeOverlap
   mg1 =geom.MakeCone("mg1",Iron,160,4,5,4,7)
   mg1.SetLineColor(12)
   top.AddNodeOverlap(mg1,1,TGeoCombiTrans(-220,0,-53,TGeoRotation("bs",90,94,0)))
   mg1o1 =geom.MakeCone("mg1o1",Iron,40,4.1,8,4.1,8)
   mg1o1.SetLineColor(12)#
   top.AddNodeOverlap(mg1o1,1,TGeoCombiTrans(-220,0,-53,TGeoRotation("bs",90,94,0)))
   
   
   #Underbody
   underbody =geom.MakeTrd2("underbody",Iron,160,210,93,93,30)
   underbody.SetLineColor(28)
   bs =geom.MakeTubs("bs",Iron,0,20,93,10,270)
   bs.SetLineColor(42)
   bsp =geom.MakeTubs("bsp",Iron,0,20,30,10,270)
   bsp.SetLineColor(42)
   
   Tip =geom.MakeCone("Tip",Iron,21,0,24,0,24) #Tip is wheel
   Tip.SetLineColor(12)
   Tip1 =geom.MakeCone("Tip1",Iron,10,23,30,25,30)
   Tip1.SetLineColor(14)
   Tip2 =geom.MakeCone("Tip2",Iron,30,0,7,0,7)
   Tip2.SetLineColor(42)
   
   wheel =geom.MakeCone("wheel",Iron,30,0,7,0,7)
   wheel.SetLineColor(42)
   wheel1 =geom.MakeCone("wheel1",Iron,21,0,16,0,16) #innner wheel
   wheel1.SetLineColor(14)
   wheel2 =geom.MakeCone("wheel2",Iron,10,15,22,15,22) #outter wheel
   wheel2.SetLineColor(12)
   
   Tip0 =geom.MakeCone("Tip0",Iron,30,0,7,0,7)
   Tip0.SetLineColor(12)
   Tip01 =geom.MakeCone("Tip01",Iron,10,7,10.5,7,10.5)
   Tip0.SetLineColor(14)
   
   #cycle of chain with AddNodeOverlap
   #char name[50]
   name = " "*50
   #WH = TGeoVolume() #piece of chain
   #whp = TGeoVolume() 
   #who = TGeoVolume() 
   ProcessLine('''
   TGeoVolume *WH;//piece of chain
   TGeoVolume *whp;
   TGeoVolume *who;
   ''') 
   WH = ROOT.WH
   whp = ROOT.whp
   who = ROOT.who
   
   #consist upper chain
   for i in range(26):
      name = sprintfPy(name,"wh{:d}",i)
      WH = geom.MakeBox(name,Iron,5.5,22,2)
      whp = geom.MakeBox(name,Iron,5,2.1,4)
      who = geom.MakeBox(name,Iron,2,6,1)
      WH.SetLineColor(12)
      whp.SetLineColor(14)
      who.SetLineColor(42)
      top.AddNodeOverlap(WH,1,TGeoTranslation(-195+(15*i),-120,-125))
      top.AddNodeOverlap(WH,1,TGeoTranslation(-195+(15*i),120,-125))
      
      top.AddNodeOverlap(whp,1,TGeoTranslation(-195+(15*i),-120,-127))
      top.AddNodeOverlap(whp,1,TGeoTranslation(-195+(15*i),120,-127))
      
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-195+(15*i),-127,-123, TGeoRotation("who",-15,0,0)))
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-195+(15*i),-113,-123, TGeoRotation("who",15,0,0)))
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-195+(15*i),127,-123, TGeoRotation("who",15,0,0)))
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-195+(15*i),113,-123, TGeoRotation("who",-15,0,0)))
      
      
   #chain connector
   WHl = geom.MakeBox(name,Iron,187.5,5,1)
   WHl.SetLineColor(12)
   top.AddNodeOverlap(WHl,1,TGeoTranslation(-7.5,-129,-125))
   top.AddNodeOverlap(WHl,1,TGeoTranslation(-7.5,-111,-125))
   top.AddNodeOverlap(WHl,1,TGeoTranslation(-7.5,111,-125))
   top.AddNodeOverlap(WHl,1,TGeoTranslation(-7.5,129,-125))
   
   #just one side
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(34*(3.14/180))),-120,-150+(25*cos(34*(3.14/180))), TGeoRotation("who",90,34,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(68*(3.14/180))),-120,-150+(25*cos(68*(3.14/180))), TGeoRotation("who",90,68,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(102*(3.14/180))),-120,-150+(25*cos(102*(3.14/180))), TGeoRotation("who",90,102,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180))),-120,-150+(25*cos(136*(3.14/180))), TGeoRotation("who",90,136,-90)))
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-12,-120,-150+(25*cos(136*(3.14/180)))-10, TGeoRotation("who",90,140,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-24,-120,-150+(25*cos(136*(3.14/180)))-20, TGeoRotation("who",90,142,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-37,-120,-150+(25*cos(136*(3.14/180)))-30, TGeoRotation("who",90,145,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-50,-120,-150+(25*cos(136*(3.14/180)))-40, TGeoRotation("who",90,149,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(34*(3.14/180))),-120,-150+(22.8*cos(34*(3.14/180))), TGeoRotation("whp",90,34,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(68*(3.14/180))),-120,-150+(22.8*cos(68*(3.14/180))), TGeoRotation("whp",90,68,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(102*(3.14/180))),-120,-150+(22.8*cos(102*(3.14/180))), TGeoRotation("whp",90,102,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180))),-120,-150+(22.8*cos(136*(3.14/180))), TGeoRotation("whp",90,136,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-12,-120,-150+(22.8*cos(136*(3.14/180)))-10, TGeoRotation("whp",90,140,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-24,-120,-150+(22.8*cos(136*(3.14/180)))-20, TGeoRotation("whp",90,142,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-37,-120,-150+(22.8*cos(136*(3.14/180)))-30, TGeoRotation("whp",90,145,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-50,-120,-150+(22.8*cos(136*(3.14/180)))-40, TGeoRotation("whp",90,149,-90)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(34*(3.14/180))),-127,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",97.5,34,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(68*(3.14/180))),-127,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",97.5,68,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(102*(3.14/180))),-127,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",97.5,102,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180))),-127,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",97.5,136,-97.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-12,-127,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",97.5,140,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-24,-127,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",97.5,142,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-37,-127,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",97.5,145,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-50,-127,-150+(27*cos(136*(3.14/180)))-40, TGeoRotation("who",97.5,149,-97.5)))
   #--------------------------
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(34*(3.14/180))),-113,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",82.5,34,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(68*(3.14/180))),-113,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",82.5,68,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(102*(3.14/180))),-113,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",82.5,102,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180))),-113,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",82.5,136,-82.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-12,-113,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",82.5,140,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-24,-113,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",82.5,142,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-37,-113,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",82.5,145,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-50,-113,-150+(27*cos(136*(3.14/180)))-40, TGeoRotation("who",82.5,149,-82.5)))
   
   
   chc0 =geom.MakeTubs("chc0",Iron,24.5,26.5,5,-34,0)#Small Plate front
   chc0.SetLineColor(12)
   chc1 =geom.MakeTubs("chc1",Iron,24.5,26.5,5,-68,-34)#Small Plate front
   chc1.SetLineColor(12)
   chc2 =geom.MakeTubs("chc2",Iron,24.5,26.5,5,-102,-68)#Small Plate front
   chc2.SetLineColor(12)
   chc3 =geom.MakeTubs("chc3",Iron,24.5,26.5,5,-136,-102)#Small Plate front
   chc3.SetLineColor(12)
   
   top.AddNodeOverlap(chc0,1,TGeoCombiTrans(180,-129,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1,1,TGeoCombiTrans(180,-129,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2,1,TGeoCombiTrans(180,-129,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3,1,TGeoCombiTrans(180,-129,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chc0,1,TGeoCombiTrans(180,-111,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1,1,TGeoCombiTrans(180,-111,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2,1,TGeoCombiTrans(180,-111,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3,1,TGeoCombiTrans(180,-111,-150,TGeoRotation("chc3",0,90,90)))
   
   chcl =geom.MakeBox("chcl",Iron,5,5,1)
   chcl.SetLineColor(12)
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-6,-111,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-18,-111,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-30,-111,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,145,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-43,-111,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,149,-90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-6,-129,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-18,-129,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-30,-129,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,145,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-43,-129,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,149,-90)))
   
   chc4 =geom.MakeTubs("chc4",Iron,31.5,34.5,5,-175,-145)#Small Plate front
   chc4.SetLineColor(12)
   top.AddNodeOverlap(chc4,1,TGeoCombiTrans(130,-111,-180,TGeoRotation("chc3",0,90,90)))
   top.AddNodeOverlap(chc4,1,TGeoCombiTrans(130,-129,-180,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(34*(3.14/180))),-120,-150+(25*cos(34*(3.14/180))), TGeoRotation("who",90,-34,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(68*(3.14/180))),-120,-150+(25*cos(68*(3.14/180))), TGeoRotation("who",90,-68,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(102*(3.14/180))),-120,-150+(25*cos(102*(3.14/180))), TGeoRotation("who",90,-102,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180))),-120,-150+(25*cos(136*(3.14/180))), TGeoRotation("who",90,-136,-90)))
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+12,-120,-150+(25*cos(136*(3.14/180)))-10, TGeoRotation("who",90,-140,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+24,-120,-150+(25*cos(136*(3.14/180)))-20, TGeoRotation("who",90,-142,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+35,-120,-150+(25*cos(136*(3.14/180)))-30, TGeoRotation("who",90,-139,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+48,-120,-150+(25*cos(136*(3.14/180)))-41, TGeoRotation("who",90,-153,-90)))
   
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(34*(3.14/180))),-120,-150+(22.8*cos(34*(3.14/180))), TGeoRotation("whp",90,-34,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(68*(3.14/180))),-120,-150+(22.8*cos(68*(3.14/180))), TGeoRotation("whp",90,-68,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(102*(3.14/180))),-120,-150+(22.8*cos(102*(3.14/180))), TGeoRotation("whp",90,-102,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180))),-120,-150+(22.8*cos(136*(3.14/180))), TGeoRotation("whp",90,-136,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+12,-120,-150+(22.8*cos(136*(3.14/180)))-10, TGeoRotation("whp",90,-140,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+24,-120,-150+(22.8*cos(136*(3.14/180)))-20, TGeoRotation("whp",90,-142,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+35,-120,-150+(22.8*cos(136*(3.14/180)))-30, TGeoRotation("whp",90,-139,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+48,-120,-150+(22.8*cos(136*(3.14/180)))-41, TGeoRotation("whp",90,-153,-90)))
   
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(34*(3.14/180))),-127,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",97.5,-34,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(68*(3.14/180))),-127,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",97.5,-68,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(102*(3.14/180))),-127,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",97.5,-102,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180))),-127,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",97.5,-136,-97.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+12,-127,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",97.5,-140,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+24,-127,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",97.5,-142,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+35,-127,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",97.5,-139,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+48,-127,-150+(27*cos(136*(3.14/180)))-41, TGeoRotation("who",97.5,-153,-97.5)))
   #-------------------------
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(34*(3.14/180))),-113,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",82.5,-34,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(68*(3.14/180))),-113,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",82.5,-68,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(102*(3.14/180))),-113,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",82.5,-102,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180))),-113,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",82.5,-136,-82.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+12,-113,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",82.5,-140,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+24,-113,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",82.5,-142,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+35,-113,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",82.5,-139,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+48,-113,-150+(27*cos(136*(3.14/180)))-41, TGeoRotation("who",82.5,-153,-82.5)))
   
   
   chc0i =geom.MakeTubs("chc0i",Iron,24.5,26.5,5,0,34)#Small Plate front
   chc0i.SetLineColor(12)
   chc1i =geom.MakeTubs("chc1i",Iron,24.5,26.5,5,34,68)#Small Plate front
   chc1i.SetLineColor(12)
   chc2i =geom.MakeTubs("chc2i",Iron,24.5,26.5,5,68,102)#Small Plate front
   chc2i.SetLineColor(12)
   chc3i =geom.MakeTubs("chc3i",Iron,24.5,26.5,5,102,136)#Small Plate front
   chc3i.SetLineColor(12)
   
   top.AddNodeOverlap(chc0i,1,TGeoCombiTrans(-195,-129,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1i,1,TGeoCombiTrans(-195,-129,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2i,1,TGeoCombiTrans(-195,-129,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3i,1,TGeoCombiTrans(-195,-129,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chc0i,1,TGeoCombiTrans(-195,-111,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1i,1,TGeoCombiTrans(-195,-111,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2i,1,TGeoCombiTrans(-195,-111,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3i,1,TGeoCombiTrans(-195,-111,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+ 6,-129,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,-140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+18,-129,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,-142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+29,-129,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,-139,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+41,-129,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,-138,-90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+ 6,-111,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,-140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+18,-111,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,-142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+29,-111,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,-139,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+41,-111,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,-138,-90)))
   
   chc4i =geom.MakeTubs("chc4i",Iron,31.5,33,5,145,175)#Small Plate front
   chc4i.SetLineColor(12)
   top.AddNodeOverlap(chc4i,1,TGeoCombiTrans(-150,-111,-180,TGeoRotation("chc3",0,90,90)))
   top.AddNodeOverlap(chc4i,1,TGeoCombiTrans(-150,-129,-180,TGeoRotation("chc3",0,90,90)))
   
   
   #just other side
   
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(34*(3.14/180))),120,-150+(25*cos(34*(3.14/180))), TGeoRotation("who",90,34,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(68*(3.14/180))),120,-150+(25*cos(68*(3.14/180))), TGeoRotation("who",90,68,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(102*(3.14/180))),120,-150+(25*cos(102*(3.14/180))), TGeoRotation("who",90,102,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180))),120,-150+(25*cos(136*(3.14/180))), TGeoRotation("who",90,136,-90)))
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-12,120,-150+(25*cos(136*(3.14/180)))-10, TGeoRotation("who",90,140,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-24,120,-150+(25*cos(136*(3.14/180)))-20, TGeoRotation("who",90,142,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-37,120,-150+(25*cos(136*(3.14/180)))-30, TGeoRotation("who",90,145,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-50,120,-150+(25*cos(136*(3.14/180)))-40, TGeoRotation("who",90,149,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(34*(3.14/180))),120,-150+(22.8*cos(34*(3.14/180))), TGeoRotation("whp",90,34,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(68*(3.14/180))),120,-150+(22.8*cos(68*(3.14/180))), TGeoRotation("whp",90,68,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(102*(3.14/180))),120,-150+(22.8*cos(102*(3.14/180))), TGeoRotation("whp",90,102,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180))),120,-150+(22.8*cos(136*(3.14/180))), TGeoRotation("whp",90,136,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-12,120,-150+(22.8*cos(136*(3.14/180)))-10, TGeoRotation("whp",90,140,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-24,120,-150+(22.8*cos(136*(3.14/180)))-20, TGeoRotation("whp",90,142,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-37,120,-150+(22.8*cos(136*(3.14/180)))-30, TGeoRotation("whp",90,145,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(180+(22.8*sin(136*(3.14/180)))-50,120,-150+(22.8*cos(136*(3.14/180)))-40, TGeoRotation("whp",90,149,-90)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(34*(3.14/180))),113,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",97.5,34,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(68*(3.14/180))),113,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",97.5,68,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(102*(3.14/180))),113,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",97.5,102,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180))),113,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",97.5,136,-97.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-12,113,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",97.5,140,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-24,113,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",97.5,142,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-37,113,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",97.5,145,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-50,113,-150+(27*cos(136*(3.14/180)))-40, TGeoRotation("who",97.5,149,-97.5)))
   #--------------------------
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(34*(3.14/180))),127,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",82.5,34,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(68*(3.14/180))),127,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",82.5,68,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(102*(3.14/180))),127,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",82.5,102,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180))),127,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",82.5,136,-82.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-12,127,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",82.5,140,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-24,127,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",82.5,142,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-37,127,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",82.5,145,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(180+(27*sin(136*(3.14/180)))-50,127,-150+(27*cos(136*(3.14/180)))-40, TGeoRotation("who",82.5,149,-82.5)))
   
   
   top.AddNodeOverlap(chc0,1,TGeoCombiTrans(180,129,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1,1,TGeoCombiTrans(180,129,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2,1,TGeoCombiTrans(180,129,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3,1,TGeoCombiTrans(180,129,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chc0,1,TGeoCombiTrans(180,111,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1,1,TGeoCombiTrans(180,111,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2,1,TGeoCombiTrans(180,111,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3,1,TGeoCombiTrans(180,111,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-6,111,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-18,111,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-30,111,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,145,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-43,111,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,149,-90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-6,129,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-18,129,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-30,129,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,145,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(180+(25*sin(136*(3.14/180)))-43,129,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,149,-90)))
   
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(34*(3.14/180))),120,-150+(25*cos(34*(3.14/180))), TGeoRotation("who",90,-34,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(68*(3.14/180))),120,-150+(25*cos(68*(3.14/180))), TGeoRotation("who",90,-68,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(102*(3.14/180))),120,-150+(25*cos(102*(3.14/180))), TGeoRotation("who",90,-102,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180))),120,-150+(25*cos(136*(3.14/180))), TGeoRotation("who",90,-136,-90)))
   
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+12,120,-150+(25*cos(136*(3.14/180)))-10, TGeoRotation("who",90,-140,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+24,120,-150+(25*cos(136*(3.14/180)))-20, TGeoRotation("who",90,-142,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+35,120,-150+(25*cos(136*(3.14/180)))-30, TGeoRotation("who",90,-139,-90)))
   top.AddNodeOverlap(WH,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+48,120,-150+(25*cos(136*(3.14/180)))-41, TGeoRotation("who",90,-153,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(34*(3.14/180))),120,-150+(22.8*cos(34*(3.14/180))), TGeoRotation("whp",90,-34,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(68*(3.14/180))),120,-150+(22.8*cos(68*(3.14/180))), TGeoRotation("whp",90,-68,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(102*(3.14/180))),120,-150+(22.8*cos(102*(3.14/180))), TGeoRotation("whp",90,-102,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180))),120,-150+(22.8*cos(136*(3.14/180))), TGeoRotation("whp",90,-136,-90)))
   
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+12,120,-150+(22.8*cos(136*(3.14/180)))-10, TGeoRotation("whp",90,-140,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+24,120,-150+(22.8*cos(136*(3.14/180)))-20, TGeoRotation("whp",90,-142,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+35,120,-150+(22.8*cos(136*(3.14/180)))-30, TGeoRotation("whp",90,-139,-90)))
   top.AddNodeOverlap(whp,1,TGeoCombiTrans(-195-(22.8*sin(136*(3.14/180)))+48,120,-150+(22.8*cos(136*(3.14/180)))-41, TGeoRotation("whp",90,-153,-90)))
   
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(34*(3.14/180))),113,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",97.5,-34,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(68*(3.14/180))),113,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",97.5,-68,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(102*(3.14/180))),113,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",97.5,-102,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180))),113,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",97.5,-136,-97.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+12,113,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",97.5,-140,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+24,113,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",97.5,-142,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+35,113,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",97.5,-139,-97.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+48,113,-150+(27*cos(136*(3.14/180)))-41, TGeoRotation("who",97.5,-153,-97.5)))
   #-------------------------
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(34*(3.14/180))),127,-150+(27*cos(34*(3.14/180))), TGeoRotation("who",82.5,-34,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(68*(3.14/180))),127,-150+(27*cos(68*(3.14/180))), TGeoRotation("who",82.5,-68,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(102*(3.14/180))),127,-150+(27*cos(102*(3.14/180))), TGeoRotation("who",82.5,-102,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180))),127,-150+(27*cos(136*(3.14/180))), TGeoRotation("who",82.5,-136,-82.5)))
   
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+12,127,-150+(27*cos(136*(3.14/180)))-10, TGeoRotation("who",82.5,-140,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+24,127,-150+(27*cos(136*(3.14/180)))-20, TGeoRotation("who",82.5,-142,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+35,127,-150+(27*cos(136*(3.14/180)))-30, TGeoRotation("who",82.5,-139,-82.5)))
   top.AddNodeOverlap(who,1,TGeoCombiTrans(-195-(27*sin(136*(3.14/180)))+48,127,-150+(27*cos(136*(3.14/180)))-41, TGeoRotation("who",82.5,-153,-82.5)))
   
   
   top.AddNodeOverlap(chc0i,1,TGeoCombiTrans(-195,129,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1i,1,TGeoCombiTrans(-195,129,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2i,1,TGeoCombiTrans(-195,129,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3i,1,TGeoCombiTrans(-195,129,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chc0i,1,TGeoCombiTrans(-195,111,-150,TGeoRotation("chc0",0,90,90)))
   top.AddNodeOverlap(chc1i,1,TGeoCombiTrans(-195,111,-150,TGeoRotation("chc1",0,90,90)))
   top.AddNodeOverlap(chc2i,1,TGeoCombiTrans(-195,111,-150,TGeoRotation("chc2",0,90,90)))
   top.AddNodeOverlap(chc3i,1,TGeoCombiTrans(-195,111,-150,TGeoRotation("chc3",0,90,90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+ 6,129,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,-140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+18,129,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,-142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+29,129,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,-139,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+41,129,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,-138,-90)))
   
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+ 6,111,-150+(25*cos(136*(3.14/180)))-5, TGeoRotation("chcl",90,-140,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+18,111,-150+(25*cos(136*(3.14/180)))-15, TGeoRotation("chcl",90,-142,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+29,111,-150+(25*cos(136*(3.14/180)))-25, TGeoRotation("chcl",90,-139,-90)))
   top.AddNodeOverlap(chcl,1,TGeoCombiTrans(-195-(25*sin(136*(3.14/180)))+41,111,-150+(25*cos(136*(3.14/180)))-35, TGeoRotation("chcl",90,-138,-90)))
   
   #consist under chain
   for i in range(20):
      name = sprintfPy(name,"wh{:d}",i)
      top.AddNodeOverlap(WH,1,TGeoTranslation(-150+(15*i),-120,-212))
      top.AddNodeOverlap(WH,1,TGeoTranslation(-150+(15*i),120,-212))
      
      top.AddNodeOverlap(whp,1,TGeoTranslation(-150+(15*i),-120,-210))
      top.AddNodeOverlap(whp,1,TGeoTranslation(-150+(15*i),120,-210))
      
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-150+(15*i),-127,-214, TGeoRotation("who",15,0,0)))
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-150+(15*i),-113,-214, TGeoRotation("who",-15,0,0)))
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-150+(15*i),127,-214, TGeoRotation("who",-15,0,0)))
      top.AddNodeOverlap(who,1,TGeoCombiTrans(-150+(15*i),113,-214, TGeoRotation("who",15,0,0)))
      
   WHlu = geom.MakeBox(name,Iron,140,5,1)#chain connetor in under
   WHlu.SetLineColor(12)
   top.AddNodeOverlap(WHlu,1,TGeoTranslation(-7.5,-129,-212))
   top.AddNodeOverlap(WHlu,1,TGeoTranslation(-7.5,-111,-212))
   top.AddNodeOverlap(WHlu,1,TGeoTranslation(-7.5,129,-212))
   top.AddNodeOverlap(WHlu,1,TGeoTranslation(-7.5,111,-212))
   
   
   
   
   #Now, we put real shape
   
   top.AddNodeOverlap(underbody,1,TGeoTranslation(0,0,-160))
   top.AddNodeOverlap(pl,1,TGeoTranslation(0,0,-130))
   top.AddNodeOverlap(tp,1,TGeoTranslation(30,0,-83))
   top.AddNodeOverlap(tp1,1,TGeoTranslation(30,0,-208))
   top.AddNodeOverlap(pl2,1,TGeoTranslation(0,-120,-100))
   top.AddNodeOverlap(pl2,1,TGeoTranslation(0,120,-100))
   top.AddNodeOverlap(pl1,1,TGeoTranslation(0,-120,-115))
   top.AddNodeOverlap(pl1,1,TGeoTranslation(0,120,-115))
   top.AddNodeOverlap(bs,1,TGeoCombiTrans(180,0,-150,TGeoRotation("bs",180,90,90)))
   top.AddNodeOverlap(bsp,1,TGeoCombiTrans(-195,61.5,-150,TGeoRotation("bsp",0,90,90)))
   top.AddNodeOverlap(bsp,1,TGeoCombiTrans(-195,-61.5,-150,TGeoRotation("bsp",0,90,90)))
   
   
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-115,-132.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-45,-132.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(35,-132.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(95,-132.5,-140,TGeoRotation("Tip01",0,90,90)))
   
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-115,-107.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-45,-107.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(35,-107.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(95,-107.5,-140,TGeoRotation("Tip01",0,90,90)))
   
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(-115,-110.5,-140,TGeoRotation("Tip0",0,90,90)))
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(-45,-110.5,-140,TGeoRotation("Tip0",0,90,90)))
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(35,-110.5,-140,TGeoRotation("Tip0",0,90,90)))
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(95,-110.5,-140,TGeoRotation("Tip0",0,90,90)))
   
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(-150,-120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(-80,-120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(-10,-120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(60,-120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(130,-120,-180,TGeoRotation("Tip",0,90,90)))
   
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-150,-107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-150,-132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-80,-107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-80,-132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-10,-107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-10,-132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(60,-107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(60,-132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(130,-107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(130,-132.5,-180,TGeoRotation("Tip",0,90,90)))
   
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(-150,-112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(-80,-112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(-10,-112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(60,-112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(130,-112.5,-180,TGeoRotation("Tip",0,90,90)))
   
   top.AddNodeOverlap(wheel1,1,TGeoCombiTrans(180,-120,-150,TGeoRotation("wheel1",0,90,90)))
   top.AddNodeOverlap(wheel1,1,TGeoCombiTrans(-195,-120,-150,TGeoRotation("wheel1",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(180,-107.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(180,-132.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(-195,-107.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(-195,-132.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel,1,TGeoCombiTrans(180,-112.5,-150,TGeoRotation("wheel",0,90,90)))
   top.AddNodeOverlap(wheel,1,TGeoCombiTrans(-195,-112.5,-150,TGeoRotation("wheel2",0,90,90)))
   
   top.AddNodeOverlap(sp,1,TGeoCombiTrans(-209,-120,-149,TGeoRotation("sp",0,90,90)))#spnot
   top.AddNodeOverlap(sp,1,TGeoCombiTrans(209,-120,-149,TGeoRotation("sp1",180,90,90)))#spnot
   
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-115,132.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-45,132.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(35,132.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(95,132.5,-140,TGeoRotation("Tip01",0,90,90)))
   
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-115,107.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(-45,107.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(35,107.5,-140,TGeoRotation("Tip01",0,90,90)))
   top.AddNodeOverlap(Tip01,1,TGeoCombiTrans(95,107.5,-140,TGeoRotation("Tip01",0,90,90)))
   
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(-115,110.5,-140,TGeoRotation("Tip0",0,90,90)))
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(-45,110.5,-140,TGeoRotation("Tip0",0,90,90)))
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(35,110.5,-140,TGeoRotation("Tip0",0,90,90)))
   top.AddNodeOverlap(Tip0,1,TGeoCombiTrans(95,110.5,-140,TGeoRotation("Tip0",0,90,90)))
   
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(-150,120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(-80,120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(-10,120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(60,120,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip,1,TGeoCombiTrans(130,120,-180,TGeoRotation("Tip",0,90,90)))
   
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-150,107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-150,132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-80,107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-80,132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-10,107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(-10,132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(60,107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(60,132.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(130,107.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip1,1,TGeoCombiTrans(130,132.5,-180,TGeoRotation("Tip",0,90,90)))
   
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(-150,112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(-80,112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(-10,112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(60,112.5,-180,TGeoRotation("Tip",0,90,90)))
   top.AddNodeOverlap(Tip2,1,TGeoCombiTrans(130,112.5,-180,TGeoRotation("Tip",0,90,90)))
   
   top.AddNodeOverlap(wheel,1,TGeoCombiTrans(-195,112.5,-150,TGeoRotation("wheel1",0,90,90)))
   top.AddNodeOverlap(wheel,1,TGeoCombiTrans(180,112.5,-150,TGeoRotation("wheel",0,90,90)))
   top.AddNodeOverlap(wheel1,1,TGeoCombiTrans(180,120,-150,TGeoRotation("wheel1",0,90,90)))
   top.AddNodeOverlap(wheel1,1,TGeoCombiTrans(-195,120,-150,TGeoRotation("wheel1",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(180,107.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(180,132.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(-195,107.5,-150,TGeoRotation("wheel2",0,90,90)))
   top.AddNodeOverlap(wheel2,1,TGeoCombiTrans(-195,132.5,-150,TGeoRotation("wheel2",0,90,90)))
   
   top.AddNodeOverlap(sp,1,TGeoCombiTrans(-209,120,-149,TGeoRotation("sp",0,90,90)))#spnot
   top.AddNodeOverlap(sp,1,TGeoCombiTrans(209,120,-149,TGeoRotation("sp1",180,90,90)))#spnot
   top.SetVisibility(False)
   geom.CloseGeometry()
   
   
   #------------------draw on GL viewer-------------------------------
   # Note: Open GL has problems with pyroot package when compiled with root v-06-30-06. 
   # Note: ROOT Team is working hard on this stuff for next version realeases. Please be Patient!
   # Not to use: top.Draw("ogl")
   #top.Draw("ogl")
   #Use x3d instead.
   
   #------------------draw-on with X3D-------------------------------
   print("Drawing-on with X3D")
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
         print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!
   
if __name__ == "__main__":
   tank()
