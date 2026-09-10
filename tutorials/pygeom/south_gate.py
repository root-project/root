## \file
## \ingroup tutorial_geom
## Drawing a famous Korean gate, the South gate, called Namdeamoon in Korean, using ROOT geometry class.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: `%run south_gate.py` inside ipython3 interpreter, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to everyone inside the ROOT team
##
## \image html geom_south_gate.png width=800px
## \macro_code
##
## \author Lan Hee Yang(yangd5d5@hotmail.com), Dept. of Physics, Univ. of Seoul
## \translator P. P.

import ROOT

#from ROOT import TGeoManager
TGeoManager = ROOT.TGeoManager
TGeoMaterial = ROOT.TGeoMaterial
TGeoVolume = ROOT.TGeoVolume
TGeoMedium = ROOT.TGeoMedium
TGeoTranslation = ROOT.TGeoTranslation 
TGeoCombiTrans = ROOT.TGeoCombiTrans  
TGeoRotation = ROOT.TGeoRotation

ProcessLine = ROOT.gInterpreter.ProcessLine

gGeoManager = ROOT.gGeoManager

def sprintfPy(buffer, FormatString, *args):
   buffer = FormatString.format(*args)
   return buffer

#def south_gate():
class south_gate:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager
   
   #TGeoManager geom= TGeoManager("geom","My first 3D geometry")
   ProcessLine('''
   TGeoManager *geom= new TGeoManager("geom","My first 3D geometry");
   ''')
   geom = ROOT.geom

   
   vacuum =  TGeoMaterial("vacuum",0,0,0) #a,z,rho
   Fe =  TGeoMaterial("Fe",55.845,26,7.87)
   
   #Create media
   
   Air =  TGeoMedium("Vacuum",0,vacuum)
   Iron =  TGeoMedium("Iron",1,Fe)
   
   #Create volume
   
   top = geom.MakeBox("top",Air,1000,1000,1000)
   geom.SetTopVolume(top)
   geom.SetTopVisible(False)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom.SetTopVisible(1)
   

   #base
   
   nBlocks = ""*100
   i=1
   N = 0
   f=0
   di = [int()]*2
   di[0] = 0
   di[1] = 30
   mBlock = TGeoVolume() 
   
   while f<11:
      while i<14:
         if i==6 and f<8:
            i = i+3
            
         
         nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

         mBlock = geom.MakeBox(nBlocks, Iron, 29,149,9)
         mBlock.SetLineColor(20)
         if f<8:
            if i<=5 and f<8:
               top.AddNodeOverlap(mBlock,1, TGeoTranslation(-120-((i-1)*60)-di[f%2],5,5+(20*f)))
            elif i>5 and f<8:
               top.AddNodeOverlap(mBlock,1, TGeoTranslation(120+((i-9)*60)  +di[f%2],5,5+(20*f)))
               
         else:
            top.AddNodeOverlap(mBlock,1, TGeoTranslation(-420+(i*60)-di[f%2],5,5+(20*f)))
            
         (i:=i+1)
         if i>=14 and f>=8 and f%2 == 1:
            nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

            mBlock = geom.MakeBox(nBlocks, Iron, 29,149,9)
            mBlock.SetLineColor(20)
            top.AddNodeOverlap(mBlock,1, TGeoTranslation(-420+(i*60)-di[f%2],5,5+(20*f)))
            (i:=i+1)
            
         if (f%2 ==0):
            nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

            mBlock = geom.MakeBox(nBlocks, Iron, 14.5,149,9)
            mBlock.SetLineColor(20)
            top.AddNodeOverlap(mBlock,1, TGeoTranslation(-405,5,5+(20*f)))
            nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

            mBlock = geom.MakeBox(nBlocks, Iron, 14.5,149,9)
            mBlock.SetLineColor(20)
            top.AddNodeOverlap(mBlock,1, TGeoTranslation(405,5,5+(20*f)))
         elif f<5:
            nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

            mBlock = geom.MakeBox(nBlocks, Iron, 14.5,149,9)
            mBlock.SetLineColor(20)
            top.AddNodeOverlap(mBlock,1, TGeoTranslation(-105,5,5+(20*f)))
            nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

            mBlock = geom.MakeBox(nBlocks, Iron, 14.5,149,9)
            mBlock.SetLineColor(20)
            top.AddNodeOverlap(mBlock,1, TGeoTranslation(105,5,5+(20*f)))
            
            
         
      nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",8,(N:=N+1))

      mBlock = geom.MakeBox(nBlocks, Iron, 40,149,9)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-80,5,145))
      nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",8,(N:=N+1))

      mBlock = geom.MakeBox(nBlocks, Iron, 40,149,9)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(80,5,145))
      
      nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",7,(N:=N+1))

      mBlock = geom.MakeBox(nBlocks, Iron, 15,149,9)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-75,5,125))
      nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",7,(N:=N+1))

      mBlock = geom.MakeBox(nBlocks, Iron, 15,149,9)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(75,5,125))
      
      nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",6,(N:=N+1))

      mBlock = geom.MakeBox(nBlocks, Iron, 24,149,9)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(-95,5,105))
      nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",6,(N:=N+1))

      mBlock = geom.MakeBox(nBlocks, Iron, 24,149,9)
      mBlock.SetLineColor(20)
      top.AddNodeOverlap(mBlock,1, TGeoTranslation(95,5,105))
      
      
      
      i=1
      (f:=f+1)
      
      
   
   
   
   
   #wall
   
   f=0
   while f<5:
      i=0
      while i<65:
         nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

         mBlock = geom.MakeBox(nBlocks, Iron, 5.8,3,3.8)
         mBlock.SetLineColor(25)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-384+(i*12),137,218+(f*8)))
         (i:=i+1)
         
         
      (f:=f+1)
      
   
   
   
   f=0
   while f<5:
      i=0
      while i<65:
         nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

         mBlock = geom.MakeBox(nBlocks, Iron, 5.8,3,3.8)
         mBlock.SetLineColor(25)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-384+(i*12),-137,218+(f*8)))
         (i:=i+1)
         
         
      (f:=f+1)
      
   
   
   
   
   
   f=0
   while f<7:
      i=0
      while i<22:
         nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

         mBlock = geom.MakeBox(nBlocks, Iron, 3,5.8,3.8)
         mBlock.SetLineColor(25)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(-384,-126+(i*12),218+(f*8)))
         (i:=i+1)
         
         
      (f:=f+1)
      
   
   
   
   f=0
   while f<7:
      i=0
      while i<22:
         nBlocks = sprintfPy(nBlocks,"f{:d}_bg{:d}",f,(N:=N+1))

         mBlock = geom.MakeBox(nBlocks, Iron, 3,5.8,3.8)
         mBlock.SetLineColor(25)
         top.AddNodeOverlap(mBlock,1, TGeoTranslation(384,-126+(i*12),218+(f*8)))
         (i:=i+1)
         
         
      (f:=f+1)
      
   
   
   # arch
   
   k=0
   i=0
   
   while i<5:
      while(k<10):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 70,89,14, (i*36)+0.5, (i+1)*36-0.5)
         mBlock.SetLineColor(20)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,-130+(k*30),70,  TGeoRotation("r1",0,90,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 9,149,17)
   mBlock.SetLineColor(20)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(80,5,14))
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 9,149,18)
   mBlock.SetLineColor(20)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(80,5,51))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 9,149,17)
   mBlock.SetLineColor(20)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-80,5,14))
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 9,149,18)
   mBlock.SetLineColor(20)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(-80,5,51))
   
   
   
   
   
   #wall's kiwa
   
   k=0
   i=0
   
   while i<5:
      while(k<52):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 1,3,7, 0, 180)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-382+(k*15),137,255,  TGeoRotation("r1",90,90,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<52):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 2.5,3,7, 0, 180)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-382+(k*15),-137,255,  TGeoRotation("r1",90,90,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<20):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 2.5,3,6, 0, 180)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-382,-123+(k*13),271,  TGeoRotation("r1",0,90,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<20):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 2.5,3,7, 0, 180)
         mBlock.SetLineColor(12)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(382,-123+(k*13),271,  TGeoRotation("r1",0,90,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   ##########################################################################################################/
   
   
   # 1 floor
   
   
   k=0
   i=0
   
   while i<5:
      while(k<7):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 0,5,56, 0, 360)
         mBlock.SetLineColor(50)
         
         if k<=2:
            
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),80,260,  TGeoRotation("r1",0,0,0)))
         if k>=4:
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),80,260,  TGeoRotation("r1",0,0,0)))
            
         
         
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<7):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 0,5,56, 0, 360)
         mBlock.SetLineColor(50)
         
         if k<=2:
            
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),-80,260,  TGeoRotation("r1",0,0,0)))
         if k>=4:
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),-80,260,  TGeoRotation("r1",0,0,0)))
            
         
         
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   # ||=====||======||=====||=====||=====||=====||
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 298,78,8)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,300))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 298,78,5)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,320))
   
   
   
   #1
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,8)
         mBlock.SetLineColor(8)
         
         # Block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),70,300,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,8)
         mBlock.SetLineColor(8)
         # no indented block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),-70,300,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,8)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-290,0,300,  TGeoRotation("r1",90,0,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,8)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(290,0,300,  TGeoRotation("r1",90,0,0)))
   
   
   
   
   
   
   #2
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,5)
         mBlock.SetLineColor(8)
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),70,320,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,5)
         mBlock.SetLineColor(8)
          
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),-70,320,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,5)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-290,0,320,  TGeoRotation("r1",90,0,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,5)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(290,0,320,  TGeoRotation("r1",90,0,0)))
   
   
   
   
   
   
   
   
   
   
   
   #___||____||_____||____||____||____||____||
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),78,345,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),-78,345,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,-78+(k*33),345,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(300,-78+(k*33),345,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   #        ||#  ||#  ||#  ||#
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),90,342,  TGeoRotation("r1",0,-45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),-90,342,  TGeoRotation("r1",0,45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-318,-78+(k*33),345,  TGeoRotation("r1",-90,45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(318,-78+(k*33),345,  TGeoRotation("r1",90,45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   #   ## || / / / / / / / || / / / / / / / / || / / / / / / / / / / /
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 330,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,-107,362,  TGeoRotation("r1",0,-45,0)))
   
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 330,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,107,362,  TGeoRotation("r1",0,45,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 110,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(330,0,362,  TGeoRotation("r1",90,-45,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 110,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-330,0,362,  TGeoRotation("r1",90,45,0)))
   
   
   
   
   ############### add box
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,2)
         mBlock.SetLineColor(8)
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-270+(k*100),-108,362,  TGeoRotation("r1",0,-45,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,2)
         mBlock.SetLineColor(8)
          
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-270+(k*100),108,362,  TGeoRotation("r1",0,45,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,2)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(331,0,362,  TGeoRotation("r1",90,-45,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,2)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-331,0,362,  TGeoRotation("r1",90,45,0)))
   
   
   
   
   
   ######################################################################################################
   
   # 2nd floor
   
   
   k=0
   i=0
   
   while i<5:
      while(k<7):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 0,5,30, 0, 360)
         mBlock.SetLineColor(50)
         
         if k<=2:
            
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),80,465,  TGeoRotation("r1",0,0,0)))
         if k>=4:
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),80,465,  TGeoRotation("r1",0,0,0)))
            
         
         
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<7):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 0,5,30, 0, 360)
         mBlock.SetLineColor(50)
         
         if k<=2:
            
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),-80,465,  TGeoRotation("r1",0,0,0)))
         if k>=4:
            top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*100),-80,465,  TGeoRotation("r1",0,0,0)))
            
         
         
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   # ||=====||======||=====||=====||=====||=====||
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 302,80,8)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,480))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 302,80,5)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,500))
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 305,80,2.5)
   mBlock.SetLineColor(50)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,465))
   
   
   ###############add box
   
   
   
   
   
   
   #1
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,8)
         mBlock.SetLineColor(8)
         
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),71,480,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,8)
         mBlock.SetLineColor(8)
          
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),-71,480,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,8)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-293,0,480,  TGeoRotation("r1",90,0,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,8)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(293,0,480,  TGeoRotation("r1",90,0,0)))
   
   
   
   
   
   
   #2
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,5)
         mBlock.SetLineColor(8)
         
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),71,500,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,5)
         mBlock.SetLineColor(8)
       
         # no indentatino block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-250+(k*100),-71,500,  TGeoRotation("r1",0,0,0)))
            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,5)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-293,0,500,  TGeoRotation("r1",90,0,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,5)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(293,0,500,  TGeoRotation("r1",90,0,0)))
   
   
   
   
   
   
   
   
   
   
   
   #  1 ___||____||_____||____||____||____||____||
   
   
   k=0
   i=0
   
   while i<5:
      while(k<25):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 1.5,5,15)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*25),78,450,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<25):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 1.5,5,15)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*25),-78,450,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<7):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,1.5,15)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,-78+(k*25),450,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while k<7:
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,1.5,15)
         mBlock.SetLineColor(50)

         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   #  2 ___||____||_____||____||____||____||____||
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),78,525,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),-78,525,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,-78+(k*33),525,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(300,-78+(k*33),525,  TGeoRotation("r1",0,0,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   #        ||#  ||#  ||#  ||#
   
   #down
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),90,522,  TGeoRotation("r1",0,-45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<19):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300+(k*33.3),-90,522,  TGeoRotation("r1",0,45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-318,-78+(k*33.3),525,  TGeoRotation("r1",-90,45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<5):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 5,5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(318,-78+(k*33.3),525,  TGeoRotation("r1",90,45,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   # up
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),115,562,  TGeoRotation("r1",0,-115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),-115,562,  TGeoRotation("r1",0,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-340,-98+(k*13),565,  TGeoRotation("r1",-90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(340,-98+(k*13),565,  TGeoRotation("r1",90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   #up2
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),115,375,  TGeoRotation("r1",0,-115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),-115,375,  TGeoRotation("r1",0,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-340,-98+(k*13),375,  TGeoRotation("r1",-90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(50)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(340,-98+(k*13),375,  TGeoRotation("r1",90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   #up 3
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),115,568,  TGeoRotation("r1",0,-115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),-115,568,  TGeoRotation("r1",0,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-340,-98+(k*13),568,  TGeoRotation("r1",-90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(340,-98+(k*13),568,  TGeoRotation("r1",90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   #up4
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),115,385,  TGeoRotation("r1",0,-115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<50):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-320+(k*13),-115,385,  TGeoRotation("r1",0,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-340,-98+(k*13),385,  TGeoRotation("r1",-90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<17):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron, 2.5,2.5,20)
         mBlock.SetLineColor(44)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(340,-98+(k*13),385,  TGeoRotation("r1",90,115,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   # up kiwa
   #=========
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks,Iron, 270,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,0,620,  TGeoRotation("r1",0,0,0)))
   #===============#2
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks,Iron, 75,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(300,-50,600,  TGeoRotation("r1",0,20,-40)))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks,Iron, 75,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(300,50,600,  TGeoRotation("r1",0,-20,40)))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks,Iron, 75,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,50,600,  TGeoRotation("r1",0,-20,-40)))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks,Iron, 75,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,-50,600,  TGeoRotation("r1",0,20,40)))
   
   
   
   
   #===============#1
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 50,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(300,-80,413,  TGeoRotation("r1",0,20,-40)))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 50,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(300,80,413,  TGeoRotation("r1",0,-20,40)))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 50,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,80,413,  TGeoRotation("r1",0,-20,-40)))
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))
   mBlock = geom.MakeBox(nBlocks,Iron, 50,15,20)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-300,-80,413,  TGeoRotation("r1",0,20,40)))

   
   # _1_
   
   #front
   front=1 #60 Degrees 
   k=0
   i=0
   while i<7:
      while(k<44):
         nBlocks = sprintfPy(nBlocks,"ab{:d}_front{:d}",(N:=N+1), front )

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-280+(k*13),70+(i*12.5),425-(i*5),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   front=2 #120 Degrees 
   while i<7:
      while(k<44):
         nBlocks = sprintfPy(nBlocks,"ab{:d}_front{:d}",(N:=N+1),front)

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-280+(k*13),-70-(i*12.5),425-(i*5),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   #_2_
   
   front = 3 # 60 Degrees 
   k=0
   i=0
   while i<11:
      while(k<43):
         nBlocks = sprintfPy(nBlocks,"ab{:d}_front{:d}",(N:=N+1),front)
         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-270+(k*13),15+(i*12.5),620-(i*5),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   front = 4 # 120 Degrees 
   k=0
   i=0
   #BP Review 
   while i<11:
      while(k<43):
         nBlocks = sprintfPy(nBlocks,"ab{:d}_front{:d}", (N:=N+1), front)
         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-270+(k*13),-15-(i*12.5),620-(i*5),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
   
   
   
   
   ####left
   k=0
   i=0
   while i<6:
      while(k<11):

         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))
         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-335,81.25+(i*12.5),592.5-(i*2),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<7:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-322,69.75+(i*12.5),595-(i*2),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<8:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-309,56.25+(i*12.5),605-(i*4),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<9:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-296,50+(i*12.5),610-(i*4),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<10:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-283,37.5+(i*12.5),615-(i*4),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   k=0
   i=0
   
   while i<6:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-335,-81.25-(i*12.5),592.5-(i*2),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<7:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-322,-69.75-(i*12.5),595-(i*2),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<8:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-309,-56.25-(i*12.5),605-(i*4),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<9:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-296,-50-(i*12.5),610-(i*4),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<10:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-283,-37.5-(i*12.5),615-(i*4),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   ######/right
   
   
   
   k=0
   i=0
   
   while i<6:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(335,81.25+(i*12.5),592.5-(i*2),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<7:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(322,69.75+(i*12.5),595-(i*2),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<8:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(309,56.25+(i*12.5),605-(i*4),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<9:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(296,50+(i*12.5),610-(i*4),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<10:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(283,37.5+(i*12.5),615-(i*4),  TGeoRotation("r1",0,60,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   #
   
   
   
   
   
   k=0
   i=0
   
   while i<6:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(335,-81.25-(i*12.5),592.5-(i*2),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<7:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(322,-69.75-(i*12.5),595-(i*2),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<8:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(309,-56.25-(i*12.5),605-(i*4),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   k=0
   i=0
   
   while i<9:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(296,-50-(i*12.5),610-(i*4),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   k=0
   i=0
   
   while i<10:
      while(k<11):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeTubs(nBlocks,Iron, 3,6,6,10,170)
         mBlock.SetLineColor(13)
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(283,-37.5-(i*12.5),615-(i*4),  TGeoRotation("r1",0,120,0)))
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   #   ## || / / / / / / / || / / / / / / / / || / / / / / / / / / / /
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 330,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,-110,550,  TGeoRotation("r1",0,-45,0)))
   
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 330,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,110,550,  TGeoRotation("r1",0,45,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 110,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(335,0,550,  TGeoRotation("r1",90,-45,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 110,10,2)
   mBlock.SetLineColor(42)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-335,0,550,  TGeoRotation("r1",90,45,0)))
   
   
   
   #####################add box
   
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,2)
         mBlock.SetLineColor(8)
         
         # no indentatino block 
         
         #non indented block
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-270+(k*100),-111,550,  TGeoRotation("r1",0,-45,0)))

            
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   
   
   k=0
   i=0
   
   while i<5:
      while(k<6):
         nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

         mBlock = geom.MakeBox(nBlocks,Iron,18,10,2)
         mBlock.SetLineColor(8)
         # non indented block 
         top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-270+(k*100),111,550,  TGeoRotation("r1",0,45,0)))
           
            
         (k:=k+1)
         
      (i:=i+1)
      k=0
      
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,2)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(336,0,550,  TGeoRotation("r1",90,-45,0)))
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 18,10,2)
   mBlock.SetLineColor(8)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(-336,0,550,  TGeoRotation("r1",90,45,0)))
   
   
   
   
   #                  |           |           |            |           |
   
   
   
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 300,75,40)
   mBlock.SetLineColor(45)
   top.AddNodeOverlap(mBlock,1, TGeoCombiTrans(0,0,450,  TGeoRotation("r1",0,0,0)))
   
   
   
   #kiwa
   nBlocks = sprintfPy(nBlocks,"ab{:d}",(N:=N+1))

   mBlock = geom.MakeBox(nBlocks, Iron, 305,80,2.5)
   mBlock.SetLineColor(10)
   top.AddNodeOverlap(mBlock,1, TGeoTranslation(0,0,430))
   
   
   
   top.SetVisibility(False)
   geom.CloseGeometry()
   

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
         print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!



if __name__ == "__main__":
   south_gate()
