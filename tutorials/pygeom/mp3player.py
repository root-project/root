## \file
## \ingroup tutorial_geom
## Drawing a mp3 type music player, using ROOT geometry class and OpenGL or 3XD.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: %run mp3player.py in the ipython3 interpreter, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to all of ROOT team
##
## \macro_image
## \macro_code
##
## \author Eun Young Kim, Dept. of Physics, Univ. of Seoul
## \translator P. P. 


import ROOT

TCanvas = 		 ROOT.TCanvas
TPaveText = 		 ROOT.TPaveText
TImage = 		 ROOT.TImage
TLine = 		 ROOT.TLine
TLatex = 		 ROOT.TLatex
TButton = 		 ROOT.TButton
TPad = ROOT.TPad

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

kFALSE = ROOT.kFALSE
kTRUE = ROOT.kTRUE

#def mp3player():
class mp3player:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager
   
   #geom = TGeoManager("geom","My first 3D geometry")
   ProcessLine('''
   TGeoManager *geom = new TGeoManager("geom","My first 3D geometry");
   ''')
   global geom
   geom = ROOT.geom
   
   #materials
   vacuum = TGeoMaterial("vacuum",0,0,0)
   Fe = TGeoMaterial("Fe",55.845,26,7.87)
   
   #create media
   
   Iron = TGeoMedium("Iron",1,Fe)
   Air = TGeoMedium("Vacuum",0,vacuum)
   
   
   #create volume
   
   top = geom.MakeBox("top",Air,800,800,800) # TGeoVolume
   geom.SetTopVolume(top)
   geom.SetTopVisible(kFALSE)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom.SetTopVisible(1)
   # geom.SetTopVisible(True)

   
   
   
   b1 = geom.MakeBox("b1",Iron,100,200,600) # TGeoVolume
   b1.SetLineColor(2)
   
   
   b2 = geom.MakeTubs("b2",Iron,0,50,200,0,90) # TGeoVolume
   b2.SetLineColor(10)
   
   
   b3 = geom.MakeTubs("b3",Iron,0,50,200,90,180) # TGeoVolume
   b3.SetLineColor(10)
   
   
   b4 = geom.MakeTubs("b4",Iron,0,50,200,180,270) # TGeoVolume
   b4.SetLineColor(10)
   
   b5 = geom.MakeTubs("b5",Iron,0,50,200,270,360) # TGeoVolume
   b5.SetLineColor(10)
   
   
   b6 = geom.MakeTubs("b6",Iron,0,50,600,0,90) # TGeoVolume
   b6.SetLineColor(10)
   
   b7 = geom.MakeTubs("b7",Iron,0,50,600,90,180) # TGeoVolume
   b7.SetLineColor(10)
   
   b8 = geom.MakeTubs("b8",Iron,0,50,600,180,270) # TGeoVolume
   b8.SetLineColor(10)
   
   b9 = geom.MakeTubs("b9",Iron,0,50,600,270,360) # TGeoVolume
   b9.SetLineColor(10)
   
   
   
   b10 = geom.MakeTubs("b10",Iron,0,50,100,0,90) # TGeoVolume
   b10.SetLineColor(10)
   
   b11 = geom.MakeTubs("b11",Iron,0,50,100,90,180) # TGeoVolume
   b11.SetLineColor(10)
   
   b12 = geom.MakeTubs("b12",Iron,0,50,100,180,270) # TGeoVolume
   b12.SetLineColor(10)
   
   b13 = geom.MakeTubs("b13",Iron,0,50,100,270,360) # TGeoVolume
   b13.SetLineColor(10)
   
   
   b14 = geom.MakeBox("b14",Iron,100,50,450) # TGeoVolume
   b14.SetLineColor(10)
   b15 = geom.MakeBox("b15",Iron,50,200,600) # TGeoVolume
   b15.SetLineColor(10)
   
   
   
   b16 = geom.MakeSphere("b16",Iron,0,50,0,90,0,90) # TGeoVolume
   b16.SetLineColor(10)
   
   b17 = geom.MakeSphere("b17",Iron,0,50,0,90,270,360) # TGeoVolume
   b17.SetLineColor(10)
   
   b18 = geom.MakeSphere("b18",Iron,0,50,0,90,180,270) # TGeoVolume
   b18.SetLineColor(10)
   
   b19 = geom.MakeSphere("b19",Iron,0,50,0,90,90,180) # TGeoVolume
   b19.SetLineColor(10)
   
   
   b20 = geom.MakeTube("b20",Iron,50,150,150) # TGeoVolume
   b20.SetLineColor(17)
   
   
   
   b21 = geom.MakeSphere("b21",Iron,0,50,90,180,0,90) # TGeoVolume
   b21.SetLineColor(10)
   
   b22 = geom.MakeSphere("b22",Iron,0,50,90,180,270,360) # TGeoVolume
   b22.SetLineColor(10)
   
   b23 = geom.MakeSphere("b23",Iron,0,50,90,180,180,270) # TGeoVolume
   b23.SetLineColor(10)
   
   b24 = geom.MakeSphere("b24",Iron,0,50,90,180,90,180) # TGeoVolume
   b24.SetLineColor(10)
   
   
   
   b25 = geom.MakeTube("b25",Iron,51,54,150) # TGeoVolume
   b25.SetLineColor(17)
   b26 = geom.MakeTube("b26",Iron,56,59,150) # TGeoVolume
   b26.SetLineColor(17)
   b27 = geom.MakeTube("b27",Iron,61,64,150) # TGeoVolume
   b27.SetLineColor(17)
   b28 = geom.MakeTube("b28",Iron,66,69,150) # TGeoVolume
   b28.SetLineColor(17)
   b29 = geom.MakeTube("b29",Iron,71,74,150) # TGeoVolume
   b29.SetLineColor(17)
   b30 = geom.MakeTube("b30",Iron,76,79,150) # TGeoVolume
   b30.SetLineColor(17)
   b31 = geom.MakeTube("b31",Iron,81,84,150) # TGeoVolume
   b31.SetLineColor(17)
   b32 = geom.MakeTube("b32",Iron,86,89,150) # TGeoVolume
   b32.SetLineColor(17)
   b33 = geom.MakeTube("b33",Iron,91,94,150) # TGeoVolume
   b33.SetLineColor(17)
   b34 = geom.MakeTube("b34",Iron,96,99,150) # TGeoVolume
   b34.SetLineColor(17)
   b35 = geom.MakeTube("b35",Iron,101,104,150) # TGeoVolume
   b35.SetLineColor(17)
   b36 = geom.MakeTube("b36",Iron,106,109,150) # TGeoVolume
   b36.SetLineColor(17)
   b37 = geom.MakeTube("b37",Iron,111,114,150) # TGeoVolume
   b37.SetLineColor(17)
   b38 = geom.MakeTube("b38",Iron,116,119,150) # TGeoVolume
   b38.SetLineColor(17)
   b39 = geom.MakeTube("b39",Iron,121,124,150) # TGeoVolume
   b39.SetLineColor(17)
   b40 = geom.MakeTube("b40",Iron,126,129,150) # TGeoVolume
   b40.SetLineColor(17)
   b41 = geom.MakeTube("b41",Iron,131,134,150) # TGeoVolume
   b41.SetLineColor(17)
   b42 = geom.MakeTube("b42",Iron,136,139,150) # TGeoVolume
   b42.SetLineColor(17)
   b43 = geom.MakeTube("b43",Iron,141,144,150) # TGeoVolume
   b43.SetLineColor(17)
   b44 = geom.MakeTube("b44",Iron,146,149,150) # TGeoVolume
   b44.SetLineColor(17)
   
   
   b45 = geom.MakeTube("b45",Iron,0,25,150) # TGeoVolume
   b45.SetLineColor(10)
   
   b46 = geom.MakeTube("b46",Iron,25,30,150) # TGeoVolume
   b46.SetLineColor(17)
   
   b47 = geom.MakeBox("b47",Iron,140,194,504) # TGeoVolume
   b47.SetLineColor(32)
   
   b48 = geom.MakeBox("b48",Iron,150,176,236) # TGeoVolume
   b48.SetLineColor(37)
   
   
   b49 = geom.MakeBox("b49",Iron,150,2,236) # TGeoVolume
   b49.SetLineColor(20)
   top.AddNodeOverlap(b49,49, TGeoTranslation(-2,179,-150))
   
   b50 = geom.MakeBox("b50",Iron,150,2,236) # TGeoVolume
   b50.SetLineColor(20)
   top.AddNodeOverlap(b50,50, TGeoTranslation(-2,-179,-150))
   
   b51 = geom.MakeBox("b51",Iron,150,176,2) # TGeoVolume
   b51.SetLineColor(20)
   top.AddNodeOverlap(b51,51, TGeoTranslation(-2,0,89))
   
   b52 = geom.MakeBox("b52",Iron,150,176,2) # TGeoVolume
   b52.SetLineColor(20)
   top.AddNodeOverlap(b52,52, TGeoTranslation(-2,0,-389))
   
   
   b53 = geom.MakeBox("b53",Iron,150,200,90) # TGeoVolume
   b53.SetLineColor(10)
   top.AddNodeOverlap(b53,53, TGeoTranslation(0,0,-510))
   
   
   
   
   
   b54 = geom.MakeBox("b54",Iron,15,254,600) # TGeoVolume
   b54.SetLineColor(37)
   top.AddNodeOverlap(b54,54, TGeoTranslation(25,0,0))
   
   r1 =  TGeoRotation("r1",90,90,0)
   
   b55 = geom.MakeTubs("b55",Iron,0,54,15,270,360) # TGeoVolume
   b55.SetLineColor(37)
   top.AddNodeOverlap(b55,55, TGeoCombiTrans(25,200,-600,r1))
   
   
   b56 = geom.MakeTubs("b56",Iron,0,54,15,180,270) # TGeoVolume
   b56.SetLineColor(37)
   top.AddNodeOverlap(b56,56, TGeoCombiTrans(25,-200,-600,r1))
   
   
   b57 = geom.MakeTubs("b57",Iron,0,54,15,0,90) # TGeoVolume
   b57.SetLineColor(37)
   top.AddNodeOverlap(b57,57, TGeoCombiTrans(25,200,600,r1))
   
   b58 = geom.MakeTubs("b58",Iron,0,54,15,90,180) # TGeoVolume
   b58.SetLineColor(37)
   top.AddNodeOverlap(b58,58, TGeoCombiTrans(25,-200,600,r1))
   
   #b59 = geom.MakePgon("b59",Iron,100,100,100,100) # TGeoVolume 
   #b59.SetLineColor(37)
   #top.AddNodeOverlap(b59,59, TGeoCombiTrans(200,200,100,r1))
   
   
   
   #IAudid
   
   
   r2 =  TGeoRotation("r2",90,90,30)
   
   b61 = geom.MakeBox("b61",Iron,5,19,150) # TGeoVolume
   b61.SetLineColor(38)
   top.AddNodeOverlap(b61,61, TGeoCombiTrans(-4,-87,-495,r2))
   
   b62 = geom.MakeBox("b62",Iron,5,19,150) # TGeoVolume
   b62.SetLineColor(38)
   top.AddNodeOverlap(b62,62, TGeoCombiTrans(-4,-65,-495,r2))
   #u
   b63 = geom.MakeBox("b63",Iron,5,15,150) # TGeoVolume
   b63.SetLineColor(38)
   top.AddNodeOverlap(b63,63, TGeoCombiTrans(-4,-40,-497,r1))
   
   b64 = geom.MakeBox("b64",Iron,5,15,150) # TGeoVolume
   b64.SetLineColor(38)
   top.AddNodeOverlap(b64,64, TGeoCombiTrans(-4,-10,-497,r1))
   
   b65 = geom.MakeTubs("b65",Iron,7,17,150,0,180) # TGeoVolume
   b65.SetLineColor(38)
   top.AddNodeOverlap(b65,65, TGeoCombiTrans(-4,-25,-490,r1))
   
   
   #D
   
   b66 = geom.MakeBox("b66",Iron,5,19,150) # TGeoVolume
   b66.SetLineColor(38)
   top.AddNodeOverlap(b66,66, TGeoCombiTrans(-4,10,-495,r1))
   
   
   b67 = geom.MakeTubs("b67",Iron,10,20,150,230,480) # TGeoVolume
   b67.SetLineColor(38)
   top.AddNodeOverlap(b67,67, TGeoCombiTrans(-4,23,-495,r1))
   
   #I
   
   b68 = geom.MakeBox("b68",Iron,5,20,150) # TGeoVolume
   b68.SetLineColor(38)
   top.AddNodeOverlap(b68,68, TGeoCombiTrans(-4,53,-495,r1))
   
   #O
   
   b69 = geom.MakeTubs("b69",Iron,10,22,150,0,360) # TGeoVolume
   b69.SetLineColor(38)
   top.AddNodeOverlap(b69,69, TGeoCombiTrans(-4,85,-495,r1))
   
   
   # I
   b60 = geom.MakeTube("b60",Iron,0,10,150) # TGeoVolume
   b60.SetLineColor(38)
   top.AddNodeOverlap(b60,60, TGeoCombiTrans(-4,-120,-550,r1))
   
   
   b70 = geom.MakeBox("b70",Iron,2,19,150) # TGeoVolume
   b70.SetLineColor(38)
   top.AddNodeOverlap(b70,70, TGeoCombiTrans(-4,-114,-495,r1))
   
   b71 = geom.MakeBox("b71",Iron,2,19,150) # TGeoVolume
   b71.SetLineColor(38)
   top.AddNodeOverlap(b71,71, TGeoCombiTrans(-4,-126,-495,r1))
   
   
   b72 = geom.MakeBox("b72",Iron,8,2,150) # TGeoVolume
   b72.SetLineColor(38)
   top.AddNodeOverlap(b72,72, TGeoCombiTrans(-4,-120,-515,r1))
   
   
   b73 = geom.MakeBox("b73",Iron,8,2,150) # TGeoVolume
   b73.SetLineColor(38)
   top.AddNodeOverlap(b73,73, TGeoCombiTrans(-4,-120,-475,r1))
   
   
   # button
   
   #r0 = nullptr; #  TGeoRotation("r0",0,0,0)
   r0 = TGeoRotation("r0",0,0,0)
   
   b74 = geom.MakeBox("b74",Iron,35,250,70) # TGeoVolume
   b74.SetLineColor(38)
   top.AddNodeOverlap(b74,74, TGeoCombiTrans(-25,10,-60,r0))
   
   b75 = geom.MakeBox("b75",Iron,35,250,35) # TGeoVolume
   b75.SetLineColor(38)
   top.AddNodeOverlap(b75,75, TGeoCombiTrans(-25,10,-175,r0))
   
   
   b76 = geom.MakeBox("b76",Iron,35,250,35) # TGeoVolume
   b76.SetLineColor(38)
   top.AddNodeOverlap(b76,76, TGeoCombiTrans(-25,10,55,r0))
   
   
   r3 =  TGeoRotation("r3",0,90,0)
   b77 = geom.MakeTubs("b77",Iron,0,70,250,180,270) # TGeoVolume
   b77.SetLineColor(38)
   top.AddNodeOverlap(b77,77, TGeoCombiTrans(10,10,-210,r3))
   
   
   b78 = geom.MakeTubs("b78",Iron,0,70,250,90,180) # TGeoVolume
   b78.SetLineColor(38)
   top.AddNodeOverlap(b78,78, TGeoCombiTrans(10,10,90,r3))
   
   
   
   #Hold
   
   b79 = geom.MakeBox("b79",Iron,40,250,150) # TGeoVolume
   b79.SetLineColor(10)
   top.AddNodeOverlap(b79,79, TGeoCombiTrans(60,0,450,r0))
   
   b80 = geom.MakeTubs("b80",Iron,50,100,250,180,270) # TGeoVolume
   b80.SetLineColor(10)
   top.AddNodeOverlap(b80,80, TGeoCombiTrans(10,0,350,r3))
   
   b81 = geom.MakeTubs("b81",Iron,50,100,250,90,180) # TGeoVolume
   b81.SetLineColor(10)
   top.AddNodeOverlap(b81,81, TGeoCombiTrans(10,0,400,r3))
   
   b82 = geom.MakeBox("b82",Iron,30,250,150) # TGeoVolume
   b82.SetLineColor(10)
   top.AddNodeOverlap(b82,82, TGeoCombiTrans(-70,0,450,r0))
   
   b83 = geom.MakeBox("b83",Iron,30,250,60) # TGeoVolume
   b83.SetLineColor(10)
   top.AddNodeOverlap(b83,83, TGeoCombiTrans(-20,0,540,r0))
   
   b85 = geom.MakeTubs("b85",Iron,0,40,240,180,270) # TGeoVolume
   b85.SetLineColor(38)
   top.AddNodeOverlap(b85,85, TGeoCombiTrans(10,10,370,r3))
   
   b84 = geom.MakeTubs("b84",Iron,0,40,240,90,180) # TGeoVolume
   b84.SetLineColor(38)
   top.AddNodeOverlap(b84,84, TGeoCombiTrans(10,10,400,r3))
   
   b86 = geom.MakeBox("b86",Iron,20,240,20) # TGeoVolume
   b86.SetLineColor(38)
   top.AddNodeOverlap(b86,86, TGeoCombiTrans(-10,10,380,r0))
   
   
   b87 = geom.MakeBox("b87",Iron,20,250,10) # TGeoVolume
   b87.SetLineColor(35)
   top.AddNodeOverlap(b87,87, TGeoCombiTrans(-10,20,385,r0))
   
   
   b88 = geom.MakeBox("b88",Iron,100,220,600) # TGeoVolume
   b88.SetLineColor(10)
   top.AddNodeOverlap(b88,88, TGeoCombiTrans(0,-30,0,r0))
   
   
   b89 = geom.MakeTube("b89",Iron,25,95,650) # TGeoVolume
   b89.SetLineColor(10)
   top.AddNodeOverlap(b89,89, TGeoCombiTrans(0,-60,0,r0))
   
   b90 = geom.MakeTube("b90",Iron,25,95,650) # TGeoVolume
   b90.SetLineColor(10)
   top.AddNodeOverlap(b90,90, TGeoCombiTrans(0,60,0,r0))
   
   
   b91 = geom.MakeBox("b91",Iron,40,200,650) # TGeoVolume
   b91.SetLineColor(10)
   top.AddNodeOverlap(b91,91, TGeoCombiTrans(70,0,0,r0))
   
   b92 = geom.MakeBox("b92",Iron,100,50,650) # TGeoVolume
   b92.SetLineColor(10)
   top.AddNodeOverlap(b92,92, TGeoCombiTrans(0,150,0,r0))
   
   b93 = geom.MakeBox("b93",Iron,100,50,650) # TGeoVolume
   b93.SetLineColor(10)
   top.AddNodeOverlap(b93,93, TGeoCombiTrans(0,-150,0,r0))
   
   
   b94 = geom.MakeBox("b94",Iron,40,200,650) # TGeoVolume
   b94.SetLineColor(10)
   top.AddNodeOverlap(b94,94, TGeoCombiTrans(-70,0,0,r0))
   
   
   b95 = geom.MakeTube("b95",Iron,25,35,650) # TGeoVolume
   b95.SetLineColor(1)
   top.AddNodeOverlap(b95,95, TGeoCombiTrans(0,-60,-10,r0))
   
   b96 = geom.MakeTube("b96",Iron,25,35,650) # TGeoVolume
   b96.SetLineColor(1)
   top.AddNodeOverlap(b96,96, TGeoCombiTrans(0,60,-10,r0))
   #usb
   
   b97 = geom.MakeBox("b97",Iron,70,70,600) # TGeoVolume
   b97.SetLineColor(17)
   top.AddNodeOverlap(b97,97, TGeoCombiTrans(0,0,57,r0))
   
   
   b98 = geom.MakeTubs("b98",Iron,0,50,600,0,90) # TGeoVolume
   b98.SetLineColor(17)
   top.AddNodeOverlap(b98,98, TGeoCombiTrans(20,60,57,r0))
   
   b99 = geom.MakeTubs("b99",Iron,0,50,600,180,270) # TGeoVolume
   b99.SetLineColor(17)
   top.AddNodeOverlap(b99,99, TGeoCombiTrans(-20,-60,57,r0))
   
   
   b100 = geom.MakeTubs("b100",Iron,0,50,600,90,180) # TGeoVolume
   b100.SetLineColor(17)
   top.AddNodeOverlap(b100,100, TGeoCombiTrans(-20,60,57,r0))
   
   
   b101 = geom.MakeTubs("b101",Iron,0,50,600,270,360) # TGeoVolume
   b101.SetLineColor(17)
   top.AddNodeOverlap(b101,101, TGeoCombiTrans(20,-60,57,r0))
   
   b102 = geom.MakeBox("b102",Iron,20,110,600) # TGeoVolume
   b102.SetLineColor(17)
   top.AddNodeOverlap(b102,102, TGeoCombiTrans(0,0,57,r0))
   
   
   b103 = geom.MakeBox("b103",Iron,15,200,600) # TGeoVolume
   b103.SetLineColor(37)
   top.AddNodeOverlap(b103,103, TGeoCombiTrans(25,0,57,r0))
   #AddNode
   top.AddNodeOverlap(b1,1, TGeoTranslation(0,0,0))
   top.AddNodeOverlap(b2,2, TGeoCombiTrans(100,0,600,r3))
   top.AddNodeOverlap(b3,3, TGeoCombiTrans(-100,0,600,r3))
   top.AddNodeOverlap(b4,4, TGeoCombiTrans(-100,0,-600,r3))
   top.AddNodeOverlap(b5,5, TGeoCombiTrans(100,0,-600,r3))
   top.AddNodeOverlap(b6,6, TGeoCombiTrans(100,200,0,r0))
   top.AddNodeOverlap(b7,7, TGeoCombiTrans(-100,200,0,r0))
   top.AddNodeOverlap(b8,8, TGeoCombiTrans(-100,-200,0,r0))
   top.AddNodeOverlap(b9,9, TGeoCombiTrans(100,-200,0,r0))
   
   top.AddNodeOverlap(b10,10, TGeoCombiTrans(0,200,600,r1))
   top.AddNodeOverlap(b11,11, TGeoCombiTrans(0,-200,600,r1))
   top.AddNodeOverlap(b12,12, TGeoCombiTrans(0,-200,-600, r1))
   top.AddNodeOverlap(b13,13, TGeoCombiTrans(0,200,-600,r1))
   top.AddNodeOverlap(b14,14, TGeoTranslation(0,200,-150))
   top.AddNodeOverlap(b15,15, TGeoTranslation(100,0,0))
   
   top.AddNodeOverlap(b16,16, TGeoCombiTrans(100,200,600,r0))
   top.AddNodeOverlap(b17,17, TGeoCombiTrans(100,-200,600,r0))
   top.AddNodeOverlap(b18,18, TGeoCombiTrans(-100,-200,600,r0))
   top.AddNodeOverlap(b19,19, TGeoCombiTrans(-100,200,600,r0))
   top.AddNodeOverlap(b20,20, TGeoCombiTrans(-3,0,350,r1))
   top.AddNodeOverlap(b21,21, TGeoCombiTrans(100,200,-600,r0))
   top.AddNodeOverlap(b22,22, TGeoCombiTrans(100,-200,-600,r0))
   top.AddNodeOverlap(b23,23, TGeoCombiTrans(-100,-200,-600,r0))
   top.AddNodeOverlap(b24,24, TGeoCombiTrans(-100,200,-600,r0))
   
   
   
   top.AddNodeOverlap(b25,25, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b26,26, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b27,27, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b28,28, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b29,29, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b30,30, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b31,31, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b32,32, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b33,33, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b34,34, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b35,35, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b36,36, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b37,37, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b38,38, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b39,39, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b40,40, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b41,41, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b42,42, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b43,43, TGeoCombiTrans(-9,0,350,r1))
   top.AddNodeOverlap(b44,44, TGeoCombiTrans(-9,0,350,r1))
   
   
   top.AddNodeOverlap(b45,45, TGeoCombiTrans(-20,0,350,r1))
   top.AddNodeOverlap(b46,46, TGeoCombiTrans(-25,0,350,r1))
   
   top.AddNodeOverlap(b47,47, TGeoTranslation(5,0,85))
   top.AddNodeOverlap(b48,48, TGeoTranslation(-2,0,-150))
   geom.CloseGeometry()
   
   global can 
   can = TCanvas("can","My virtual laboratory",800,800) # TCanvas
   
   
   #Mp3
   global pad1
   pad1 =  TPad("pad1","Pad1",0,0.5,0.5,1)
   pad1.SetFillColor(1)
   pad1.Draw()
   pad1.cd()
   top.Draw()
   
   #Sound
   can.cd()
   global pad2
   pad2 = TPad("pad2","Pad2",0.5,0.5,1,1) # TPad
   pad2.SetFillColor(10)
   pad2.Draw()
   pad2.cd()
   
   global pt
   pt =  TPaveText(0.4,0.90,0.6,0.95,"br")
   pt.SetFillColor(30)
   pt.AddText(0.5,0.5,"Musics")
   pt.Draw()
   
   global Tex
   Tex = TLatex() 
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.81,"Mariah Carey - Shake it off")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.71,"Alicia keys :)- If I ain't got you")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.61,"Michael Jackson - Billie Jean")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.51,"Christina Milian - Am to Pm")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.41,"Zapp&Roger - Slow and Easy")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.31,"Black Eyes Peas - Let's get retarded")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(31)
   Tex.DrawLatex(0.3,0.21,"Bosson - One in a Millin")
   
   Tex.SetTextSize(0.04)
   Tex.SetTextColor(15)
   Tex.DrawLatex(0.2,0.11,"Click Buttonnot not  You Can Listen to Musics")


   global but1, but2, but3, but4, but5, but6, but7, but8, but9
   location = "/usr/share/sounds/gnome/default/alerts"
   but1 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg \")', 0.2,0.8,0.25,0.85) # TButton
   but1.Draw()
   but1.SetFillColor(29)
   but2 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg\")', 0.2,0.7,0.25,.75) # TButton
   but2.Draw()
   but2.SetFillColor(29)
   but3 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg\")', 0.2,0.6,0.25,0.65) # TButton
   but3.Draw()
   but3.SetFillColor(29)
   but4 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg\")', 0.2,0.5,0.25,0.55) # TButton
   but4.Draw()
   but4.SetFillColor(29)
   but5 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg\")', 0.2,0.4,0.25,0.45) # TButton
   but5.Draw()
   but5.SetFillColor(29)
   but6 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg\")', 0.2,0.3,0.25,0.35) # TButton
   but6.Draw()
   but6.SetFillColor(29)
   but7 = TButton("", f'gSystem->Exec(\"xdg-open {location}/click.ogg\")', 0.2,0.2,0.25,0.25) # TButton
   but7.Draw()
   but7.SetFillColor(29)
   
   #introduction
   can.cd()


   global pad3
   pad3 = TPad("pad3","Pad3",0,0,1,0.5) # TPad
   pad3.SetFillColor(10)
   pad3.Draw()
   pad3.cd()
   
   #image = TImage.Open("mp3.jpg")
   #image.Draw()
   
   global pad4
   pad4 = TPad("pad4","Pad4",0.6,0.1,0.9,0.9) # TPad
   pad4.SetFillColor(1)
   pad4.Draw()
   pad4.cd()
   
   global L
   L = TLine()
   
   Tex.SetTextSize(0.08)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.85,"IAudio U3 Mp3 Player")
   
   L.SetLineColor(10)
   L.SetLineWidth(3)
   L.DrawLine(0.05, 0.83,0.90, 0.83)
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.75,"+ Color LCD")
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.65,"+ 60mW High Generating Power")
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.55,"+ GUI Theme Skin")
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.45,"+ Noble White&Black")
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.35,"+ Text Viewer+Image Viewer")
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.25,"+ 20 Hours Playing")
   
   Tex.SetTextSize(0.06)
   Tex.SetTextColor(10)
   Tex.DrawLatex(0.06,0.15,"+ The Best Quality of Sound")
   
   
   pad1.cd()
   pad1.Update()
   #input("Press Enter to close mp3player()")


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
   mp3player()
