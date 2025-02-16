## \file
## \ingroup tutorial_geom
## Drawing the Cheongwadae building which is 
## the Presidential Residence of the Republic of Korea, 
## using the ROOT geometry class and OpenGL or X3D.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: `%run cheongwadae.py` in ipython3 interpreter, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to all of ROOT team
##
## \image html geom_cheongwadae.png width=800px
## \macro_code
##
## \author Hee Jun Shin (s-heejun@hanmail.net), Dept. of Physics, Univ. of Seoul
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

#def cheongwadae():
class cheongwadae:
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
   
   #material
   vacuum =  TGeoMaterial("vacuum",0,0,0)
   Fe =  TGeoMaterial("Fe",55.845,26,7.87)
   
   #creat media
   Air =  TGeoMedium("Vacuum",0,vacuum)
   Iron =  TGeoMedium("Iron",1,Fe)
   
   #creat volume
   top = geom.MakeBox("top",Air,300,300,300)
   geom.SetTopVolume(top)
   geom.SetTopVisible(False)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom->SetTopVisible(1);
   
   nBlocks = " "*100
   N = 0
   f=0
   di = [0, 30]
   mBlock = TGeoVolume() 
   
   # for(int k=0; k<7; k++)
   for k in range(7):
      # for(int i=0; i<20; i++)
      for i in range(20):
         nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
         mBlock = geom.MakeBox(nBlocks, Iron, 0.6,1.8,63)
         mBlock.SetLineColor(20)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-10.6-(2.6*i),-17.8+(6*k),0))
         
         nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
         mBlock = geom.MakeBox(nBlocks, Iron, 0.7,1.8,58)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-11.9-(2.6*i),-17.8+(6*k),0))
         
      nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
      mBlock = geom.MakeBox(nBlocks, Iron, 26,1.2,63)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-36,-14.8+(6*k),0))
      
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 26,2,63)
   mBlock.SetLineColor(20)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-36,-21.6,0))
   
   # for(int k=0; k<7; k++)
   for k in range(7):
      # for(int i=0; i<20; i++)
      for i in range(20):
         nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
         mBlock = geom.MakeBox(nBlocks, Iron, 0.6,1.8,63)
         mBlock.SetLineColor(20)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-10.6-(2.6*i),-17.8+(6*k),0))
         nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
         mBlock = geom.MakeBox(nBlocks, Iron, 0.7,1.8,58)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-11.9-(2.6*i),-17.8+(6*k),0))
         
         
      nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
      mBlock = geom.MakeBox(nBlocks, Iron, 26,1.2,63)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-36,-14.8+(6*k),0))
      
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,22,58)
   mBlock.SetLineColor(2)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,0))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 3.5,8,0.1)
   mBlock.SetLineColor(13)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(4,-14,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 3.5,8,0.1)
   mBlock.SetLineColor(13)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-4,-14,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,20,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,17,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,14,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,11,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,8,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,5,60))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.2,0.1)
   mBlock.SetLineColor(1)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,2,60))
   
   # for(int k=0; k<7; k++)
   for k in range(7):
      # for(int i=0; i<20; i++)
      for i in range(20):
         nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
         mBlock = geom.MakeBox(nBlocks, Iron, 0.6,1.8,63)
         mBlock.SetLineColor(20)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(10.6+(2.6*i),-17.8+(6*k),0))
         nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
         mBlock = geom.MakeBox(nBlocks, Iron, 0.7,1.8,58)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(11.9+(2.6*i),-17.8+(6*k),0))
         
         
      nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
      mBlock = geom.MakeBox(nBlocks, Iron, 26,1.2,63)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(36,-14.8+(6*k),0))
      
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 26,2,63)
   mBlock.SetLineColor(20)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(36,-21.6,0))
   
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 82,2,82)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,24,0))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 85,0.5,85)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,26,0))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 88,2,88)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-24,0))
   
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 30, 0, 180, 0, 180)
   mBlock.SetLineColor(32)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,24,0))
   
   nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 0.1,30,0.1)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,40,0))
   
   nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
   mBlock = geom.MakeTubs(nBlocks,Iron, 0,30,4,360,360)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,27,0,  TGeoRotation("r1",0,90,0)))
   
   # for(int i=0; i<8; i++)
   for i in range(8):
      nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
      mBlock = geom.MakeBox(nBlocks,Iron, 2,22,2)
      mBlock.SetLineColor(18)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-70+(20*i),0,80))
      
   
   # for(int i=0; i<8; i++)
   for i in range(8):
      nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
      mBlock = geom.MakeBox(nBlocks,Iron, 2,22,2)
      mBlock.SetLineColor(18)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-70+(20*i),0,-80))
      
   
   # for(int i=0; i<7; i++)
   for i in range(7):
      nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
      mBlock = geom.MakeBox(nBlocks,Iron, 2,22,2)
      mBlock.SetLineColor(18)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-70,0,-80+(23*i)))
      
   
   # for(int i=0; i<7; i++)
   for i in range(7):
      nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
      mBlock = geom.MakeBox(nBlocks,Iron, 2,22,2)
      mBlock.SetLineColor(18)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(70,0,-80+(23*i)))
      
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 100,0.5,160)
   mBlock.SetLineColor(41)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-26,40))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 10,0.01,160)
   mBlock.SetLineColor(19)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-25,40))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-22,170))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-25,170))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-22,150))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-25,150))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-22,130))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-25,130))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-22,110))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(15,-25,110))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-22,170))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-25,170))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-22,150))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-25,150))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-22,130))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-25,130))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-22,110))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeSphere(nBlocks, Iron, 0, 5, 0, 180, 0, 180)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-25,110))
   
   nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 0.1,10,0.1)
   mBlock.SetLineColor(12)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(20,-15,110))
   
   nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 5,3,0.1)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(25,-8,110))
   
   nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 0.1,10,0.1)
   mBlock.SetLineColor(12)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-20,-15,110))
   
   nBlocks = sprintfPy(nBlocks, "ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 5,3,0.1)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-15,-8,110))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 7,1.5,5)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-24,88))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 7,1,5)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-24,92))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 7,0.5,5)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-24,96))
   
   nBlocks = sprintfPy(nBlocks, "f{:d}_bg{:d}",f,(N:=N+1))
   mBlock = geom.MakeBox(nBlocks, Iron, 7,0.1,5)
   mBlock.SetLineColor(18)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,-24,100))
   
   geom.CloseGeometry()
   top.SetVisibility(False)
   
   #top.Draw("ogl")
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
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!



if __name__ == "__main__":
   cheongwadae()
