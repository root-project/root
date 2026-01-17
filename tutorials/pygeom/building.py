## \file
## \ingroup tutorial_geom
## Drawing a building where the Dept. of Physics is, using the ROOT geometry class TGeoManager.
##
## Reviewed by Sunman Kim (sunman98@hanmail.net)
## Supervisor: Prof. Inkyu Park (icpark@physics.uos.ac.kr)
##
## How to run: `%run building.py` in IPython interpreter, then use OpenGL
##
## This macro was created for the evaluation of Computational Physics course in 2006.
## We thank to Prof. Inkyu Park for his special lecture on ROOT and to everyone from the ROOT team.
##
## \image html geom_building.png width=800px
## \macro_code
##
## \author Hyung Ju Lee (laccalus@nate.com), Dept. of Physics, Univ. of Seoul
## \translator P. P.

import ROOT
import sys

TGeoManager = ROOT.TGeoManager

TGeoMaterial = ROOT.TGeoMaterial
TGeoMedium = ROOT.TGeoMedium
TGeoVolume = ROOT.TGeoVolume
TGeoTranslation = ROOT.TGeoTranslation
TGeoCombiTrans = ROOT.TGeoCombiTrans 
TGeoRotation = ROOT.TGeoRotation 

TCanvas = ROOT.TCanvas

std = ROOT.std
char = ROOT.char
sprintf = ROOT.std.sprintf


Declare = ROOT.gInterpreter.Declare
ProcessLine = ROOT.gInterpreter.ProcessLine

gGeoManager = ROOT.gGeoManager


def sprintfPy(buffer, FormatString, *args):
   buffer = FormatString.format(*args)
   return buffer

#def  building():
class building:
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
      
   global geom
   geom =  TGeoManager("geom","My First 3D Geometry")
   Declare('''
   //TGeoManager *geom = new TGeoManager("geom","My First 3D Geometry");
   ''')
   #geom = ROOT.geom
   
   # Materials
   Vacuum =  TGeoMaterial("vacuum",0,0,0)
   Fe =  TGeoMaterial("Fe",55.845,26,7.87)
   
   # Media
   Air =  TGeoMedium("Air",0,Vacuum)
   Iron =  TGeoMedium("Iron",0,Fe)
   
   # Volume
   Phy_Building = geom.MakeBox("top",Air,150,150,150)
   geom.SetTopVolume(Phy_Building)
   geom.SetTopVisible(False)
   # If you want to see the boundary, please input the number, 1 instead of 0.
   # Like this, geom.SetTopVisible(1)
   
   
   mBlocks = TGeoVolume()
   
   
   ##########################################################################################
   #####################         Front-Building        ####################/
   ##########################################################################################
   print("#####################         Front-Building        ") 
   i = 0
   F = 0 # Floor
   N = 0 # Block_no
   nW = 8 # Number of windows
   nF = 3 # Number of Floor
   no_Block = [char()]*100  # Name of Block #Not used. #New sprintfPy used instead.
   sP = 0 # Starting Phi of Tubs
   hP = 21 # Height of Tubs from Ground
   
   ################## Floors
   print("################## Floors")
   while (F<nF): 
      print("F", F, "nF", nF)
      N = 0
      i = 0
      sP = 0
      
      ################## Front of Building
      print("################## Front of Building")
      while (i<nW) :
         print("   N", N, "nW", nW)
         i+=1
         no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))                  # Windows (6.25)
         mBlocks = geom.MakeTubs(no_Block,Iron,21,29,1.8,sP,sP+6.25)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
         
         if i < nW:
            no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))               # Walls (8)
            mBlocks = geom.MakeTubs(no_Block,Iron,21,30,1.8,sP,sP+2.5)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
            
            no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
            mBlocks = geom.MakeTubs(no_Block,Iron,21,31,1.8,sP,sP+1)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
            
            no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
            mBlocks = geom.MakeTubs(no_Block,Iron,21,30,1.8,sP,sP+1)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
            
            no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
            mBlocks = geom.MakeTubs(no_Block,Iron,21,31,1.8,sP,sP+1)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
            
            no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
            mBlocks = geom.MakeTubs(no_Block,Iron,21,30,1.8,sP,sP+2.5)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
            
         
         if i>=nW:
            no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))               # Walls
            mBlocks = geom.MakeTubs(no_Block,Iron,21,30,1.8,sP,103)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP+(8*F)))
            
            
         
      
      #no_Block = sprintfPy(no_Block, "B1_F{:d}", ++F)                  # No Windows Floor
      no_Block = sprintfPy(no_Block, "B1_F{:d}", (F:=F+1))                  # No Windows Floor
      mBlocks = geom.MakeTubs(no_Block,Iron,21,30,2.2,0,103)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-12+(8*F)))
      
      ########################### Back of Building
      print("########################### Back of Building")
      no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
      mBlocks = geom.MakeTubs(no_Block,Iron,18.5,21,0.8,92,101)
      mBlocks.SetLineColor(12)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-9.4+(8*F)))
      
      if(F<nF):
         no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
         mBlocks = geom.MakeTubs(no_Block,Iron,18.5,21,3.2,92,102)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-5.4+(8*F)))
         
         
      
   #sys.exit() 
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))               # Walls
   mBlocks = geom.MakeTubs(no_Block,Iron,18.5,21,2,92,102)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-4))

   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.5,21,3.2,92,102)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-5.4+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,30,2,0,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-4.2+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,21,2,0,102)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-4.2+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,18.5,14,92,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,29))
   
   ################ Front of Building
   print("################ Front of Building")
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,29,2,0,97)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,13))
   
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,32,2,37,97)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,9))
   
   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,29,1.95,0,37)
   mBlocks.SetLineColor(30)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,9.05))

   no_Block = sprintfPy(no_Block, "B1_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,29,0.05,0,37)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,7.05))
   
   
   ################ Rooftop
   print("################ Rooftop")
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:= 0))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,29.5,0.2,0,102)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-2+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.5,21,0.2,0,101)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-2+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,30,0.7,102.9,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.8+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21.1,29.9,0.7,102,102.9)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.8+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21.1,21.5,0.5,98,102.9)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.8+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,21,21.1,0.7,98,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.8+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.6,21,0.7,101.9,102)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.8+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.6,21,0.7,101,101.9)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.8+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,29.5,29.9,0.5,0,102)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,29.9,30,0.5,0,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.1,18.5,0.5,-1,101.9)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,18.1,0.5,-0.5,102)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.1,18.4,0.5,101.9,102.9)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,18.1,0.5,102,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18.4,18.5,0.5,102,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   no_Block = sprintfPy(no_Block, "B1_RT{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,18.5,0.5,102.9,103)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,hP-1.7+(8*F)))
   
   
   ####################/ White Wall
   print("####################/ White Wall")
   no_Block = sprintfPy(no_Block, "B1_WW{:d}", (N:= 0))
   mBlocks = geom.MakeTubs(no_Block,Iron,20.8,31,19.5,sP,sP+1)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,26))
   
   no_Block = sprintfPy(no_Block, "B1_WW{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,26.8,31,5,sP,sP+1)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,2))
   
   no_Block = sprintfPy(no_Block, "B1_WW{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,23,24.3,5,sP,sP+1)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,2))
   
   no_Block = sprintfPy(no_Block, "B1_WW{:d}", (N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,20.8,21.3,5,sP,sP+1)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,2))
   
   
   
   ################# Zero Floor1
   print("################# Zero Floor1")
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=0))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,21,9,0,92)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,6))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,21,7.5,0,92)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,31.5))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,18,21,4.5,0,92)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,19.5))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,18,0.2,0,101)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,18.6))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,18,1.7,0,100)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,16.7))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,18,1.2,101,101.9)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,19.6))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,18,1.2,101.9,102)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,19.6))
   
   
   ################# Zero Floor2 
   print("################# Zero Floor2")
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,6.5,7,2.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7,10.75,13))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,6.5,7,3)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7,10.75,7.5))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,7,0.05,10)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7,17.95,7))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,6.9,0.20,10)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7,17.70,7))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.1,0.20,10)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-13.9,17.70,7))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.05,7,3.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-13.95,10.5,13.5))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.20,6.9,3.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-13.70,10.5,13.5))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.25,7,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-13.75,10.5,1))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,7,0.05,10)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7,3.55,7))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,6.9,0.20,10)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7,3.8,7))
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.1,0.20,10)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-13.9,3.8,7))
   
   
   ################# Zero Floor2 Continuation
   print("################# Zero Floor2 Continuation")
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5,5,1)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-5,23,-2))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5,0.25,1.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-5,28.25,-1.5))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.25,5.5,1.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-10.25,23,-1.5))
   
   no_Block = sprintfPy(no_Block, "B1_ZF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,3.5,5)
   mBlocks.SetLineColor(20)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-12.5,0,-4))
   
   
   ################# Ground
   print("################# Ground")
   no_Block = sprintfPy(no_Block, "B1_GRD{:d}",(N:=0))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,29,1,0,36.75)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,-2))
   
   no_Block = sprintfPy(no_Block, "B1_GRD{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,30.4,0.4,36.75,77.25)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,-2.7))
   
   no_Block = sprintfPy(no_Block, "B1_GRD{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,29.7,0.3,36.75,77.25)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,-2))
   
   no_Block = sprintfPy(no_Block, "B1_GRD{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,29,0.3,36.75,77.25)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,-1.3))
   
   no_Block = sprintfPy(no_Block, "B1_GRD{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,0,29,1,77.25,97)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,-2))
   
   
   ################### Pillars & fences
   print("################### Pillars & fences")
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=0))
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(29,4.2,6, TGeoRotation("r1",8.25,0,0)))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(24.2,16.5,6, TGeoRotation("r1",34.25,0,0)))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(14.5,25.4,6, TGeoRotation("r1",60.25,0,0)))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(1.9,29.2,6, TGeoRotation("r1",86.25,0,0)))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}",(N:=N+1))
   mBlocks = geom.MakeTubs(no_Block,Iron,29,30,2,0,36.75)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,0,-1))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,3,2,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-0.75,29.3,-1))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))      #장애인용
   mBlocks = geom.MakeBox(no_Block,Iron,0.25,4.3,1.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(6.5,30.6,-1.5))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.25,4.3,0.4)
   mBlocks.SetLineColor(10)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(1.125,30.6,-2.7))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,0.25,0.75)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(1.125,34.9,-2.25))
   
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeTrd1(no_Block,Iron,1.5,0,0.25,5.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(1.125,34.9,-1.5, TGeoRotation("r1",90,-90,90)))
   
   
   
   
   
   
   
   
   
   ##########################################################################################
   ##################### Second Part of Front-Building ####################/
   ##########################################################################################
   print("##################### Second Part of Front-Building ####################/")
   
   print("Second Part of Front-Building ")
   F=0
   while (F<nF) :
      print("F", F ,"nF", nF)
      N = 0
      i = 0
      nW = 7
      
      while (i<nW) :
         print("   i", i ,"nW", nW)
         #sys.exit()
         no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))      # Wall
         mBlocks = geom.MakeBox(no_Block,Iron,3.8,0.35,1.8)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,
         TGeoCombiTrans(23.38 + (21.65-6*i)*0.13,-21.2 + (21.65-6*i)*0.99,hP+(8*F),
         TGeoRotation("r1",-7.5,0,0)))
         no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,4.8,0.3,1.8)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,
         TGeoCombiTrans(23.38 + (21.0-6*i)*0.13,-21.2 + (21-6*i)*0.99,hP+(8*F),
         TGeoRotation("r1",-7.5,0,0)))
         no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,3.8,0.3,1.8)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,
         TGeoCombiTrans(23.38 + (20.4-6*i)*0.13,-21.2 + (20.4-6*i)*0.99,hP+(8*F),
         TGeoRotation("r1",-7.5,0,0)))
         no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,4.8,0.3,1.8)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,
         TGeoCombiTrans(23.38 + (19.7-6*i)*0.13,-21.2 + (19.7-6*i)*0.99,hP+(8*F),
         TGeoRotation("r1",-7.5,0,0)))
         no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,3.8,0.35,1.8)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,
         TGeoCombiTrans(23.38 + (19.05-6*i)*0.13,-21.2 + (19.05-6*i)*0.99,hP+(8*F),
         TGeoRotation("r1",-7.5,0,0)))
         
         
         no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))      # Windows
         mBlocks = geom.MakeBox(no_Block,Iron,3,1.4,1.8)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,
         TGeoCombiTrans(23.38 + (17.4-6*i)*0.13,-21.2 + (17.4-6*i)*0.99,hP+(8*F),
         TGeoRotation("r1",-7.5,0,0)))
         i+=1


         #BPwhile(i >= nW):
         if(i >= nW):
            print("i", i ,"nW", nW)
            #sys.exit()
            no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",F, (N:=N+1))      # Wall.
            mBlocks = geom.MakeBox(no_Block,Iron,5.8,1,1.8)
            mBlocks.SetLineColor(2)
            Phy_Building.AddNodeOverlap(mBlocks,1,
            TGeoCombiTrans(21.4 + (-21)*0.13,-21 + (-21)*0.99,hP+(8*F),
            TGeoRotation("r1",-7.5,0,0)))

      #sys.exit()
            
         
      
      #no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",++F, (N:=N+1))
      no_Block = sprintfPy(no_Block, "B12_F{:d}_B{:d}",(F:=F+1), (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,5.8,22,2.2)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.4,-21,hP-12+(8*F), TGeoRotation("r1",-7.5,0,0)))
      
      
   no_Block = sprintfPy(no_Block, "B12_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.8,22,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.4,-21,hP-4.2+(8*F), TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_F{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,2.8,22,14)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(18.43,-20.61,29, TGeoRotation("r1",-7.5,0,0)))
   
   
   ##############/ RoofTop
   print("##############/ RoofTop")
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=0))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,21.75,0.2)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.43,-20.75,hP-2+(8*F), TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.23,21.95,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(26.9,-21.72,hP-1.7+(8*F), TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.1,22,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(27.1,-21.75,hP-1.7+(8*F), TGeoRotation("r1",-7.5,0,0)))
   
   
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.23,3.6,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(13.65,-38.03,hP-1.7+(8*F), TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.02,3.8,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(13.3,-38.39,hP-1.7+(8*F), TGeoRotation("r1",-7.5,0,0)))
   
   
   
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.7,0.23,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(18.57,-42.48,hP-1.7+(8*F), TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.8,0.1,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(18.54,-42.71,hP-1.7+(8*F), TGeoRotation("r1",-7.5,0,0)))
   
   
   ################ Pillars & fences
   #BP
   print("################ Pillars & fences")
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=0))
   print(" no_Block")
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(28.32,-7.44,6, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   print(" no_Block")
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(26.75,-19.33,6, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   print(" no_Block")
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,9)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(25.19,-31.23,6, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   print(" no_Block")
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,11)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(23.75,-42.14,4, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   print(" no_Block")
   mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,11)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(13.84,-40.83,4, TGeoRotation("r1",-7.5,0,0)))
   
   
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.5,15.75,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(27.42,-15.48,-1, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.5,2,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(24.28,-39.27,-3, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,1.5,15.75,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(28.91,-15.68,-4, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_RT{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.8,0.5,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(18.8,-40.73,-3, TGeoRotation("r1",-7.5,0,0)))
   
   
   ############### Stair
   print("############### Stair")
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,3,0.5,3.25)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(28.33,-31.49,-2.75, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.5,6.25,1.625)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(30.56,-37.58,-4.375, TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B1_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeTrd1(no_Block,Iron,3.25,0,0.5,6.25)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(30.56,-37.58,-2.75, TGeoRotation("r1",-7.5,90,90)))
   
   
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,3,3,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(27.37,-34.89,-2.5, TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,2.5,3,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(27.74,-35.95,-3.5, TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,2.5,3,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(27.61,-36.94,-4.5, TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B12_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,2.5,3,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(27.48,-37.93,-5.5, TGeoRotation("r1",-7.5,0,0)))
   
   
   
   ################ Ground
   print("################ Ground")
   no_Block = sprintfPy(no_Block, "B12_GR{:d}", (N:=0))
   mBlocks = geom.MakeBox(no_Block,Iron,4.8,21,1)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.53,-20.1,-2, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_GR{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.8,18,9)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(12.86,-16.62,6, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_GR{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.8,22,2)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.4,-21,13, TGeoRotation("r1",-7.5,0,0)))
   
   no_Block = sprintfPy(no_Block, "B12_GR{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.8,22,1.95)
   mBlocks.SetLineColor(30)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.4,-21,9.05, TGeoRotation("r1",-7.5,0,0)))
   no_Block = sprintfPy(no_Block, "B12_GR{:d}_{:d}", F, (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.8,22,0.05)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoCombiTrans(21.4,-21,7.05, TGeoRotation("r1",-7.5,0,0)))
   
   
   
   #sys.exit()
   
   
   
   
   ##########################################################################################
   #####################         Bridge-Building       ####################/
   ##########################################################################################
   print("#####################         Bridge-Building       ####################/")
   F=1
   N = 0
   nF = 4
   
   
   no_Block = sprintfPy(no_Block, "B2_F{:d}", 6)
   mBlocks = geom.MakeBox(no_Block,Iron,7,17.5,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(12,-17.5,41))

   #sys.exit()
   
   while ((F:=F+1) <nF) :
      print("F", F,"nF", nF)
      ################ Front
      print("################ Front")
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,0.8,4,4)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(10,-4,-5 +(F*8)))
      
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.1,3.5,1)
      mBlocks.SetLineColor(12)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(11.9,-4,-2 +(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.1,4.5,0.2)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(11.9,-4,-3.2+(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.1,4,2.8)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(11.9,-4,-6.2+(F*8)))
      
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,0.7,4,4)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(13.6,-4,-5 +(F*8)))
      
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.1,3.5,1)
      mBlocks.SetLineColor(12)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(15.4,-4,-2 +(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.1,4.5,0.2)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(15.4,-4,-3.2+(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.1,4,2.8)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(15.4,-4,-6.2+(F*8)))
      
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,0.7,4,4)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(17.1,-4,-5 +(F*8)))
      
      
      ##################/ Back
      print("##################/ Back")
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.3,13.5,1.5)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(6.8,-21.5,-2.5 +(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.3,14,0.2)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(6.8,-21.5,-4.2+(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.3,13.5,2.3)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(6.8,-21.5,-6.8+(F*8)))
      
      
      
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,3.7,13,1.5)
      mBlocks.SetLineColor(12)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(11.8,-21.5,-2.5 +(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,3.7,14,0.2)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(11.8,-21.5,-4.2+(F*8)))
      no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,3.7,13.5,2.3)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(11.8,-21.5,-6.8+(F*8)))
      
      
      
    
   #sys.exit()
   
   no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}", 0,1)
   mBlocks = geom.MakeBox(no_Block,Iron,5,13.5,6)
   mBlocks.SetLineColor(30)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(10,-21.5,-15))
   no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}", 0,2)
   mBlocks = geom.MakeBox(no_Block,Iron,5,13.5,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(10,-21.5,-5))
   no_Block = sprintfPy(no_Block, "B2_F{:d}_{:d}", 0,3)
   mBlocks = geom.MakeBox(no_Block,Iron,5,13.5,4)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(10,-21.5,3))
   
   
   
   ################## RoofTop
   print("################## RoofTop")
   no_Block = sprintfPy(no_Block, "B2_F{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,7,17.4,0.1)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(12,-17.5,43.1))
   
   no_Block = sprintfPy(no_Block, "B2_F{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,0.2,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(9.5,-34.7,43.5))
   no_Block = sprintfPy(no_Block, "B2_F{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,0.05,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(9.5,-34.95,43.5))
   
   no_Block = sprintfPy(no_Block, "B2_F{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.75,0.2,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(13.75,-0.3,43.5))
   no_Block = sprintfPy(no_Block, "B2_F{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.55,0.05,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(13.55,-0.05,43.5))
   
   
   
   
   
   
   
   
   
   ##########################################################################################
   #####################         Building 3            ####################/
   ##########################################################################################
   print("#####################         Building 3            ####################/")
   F=0
   N = 0
   nF = 4
   nW = 6
   
   #sys.exit()
   
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 7)
   mBlocks = geom.MakeBox(no_Block,Iron,3,36,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,49))
   
   while ((F:=F+1) <nF) :
      print("F", F,"nF", nF)
      i=0
      N=0
      
      no_Block = sprintfPy(no_Block, "B3_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,4,36,0.2)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,14.7 +(F*8)))
      
      while ((i:=i+1) <nW) :
         print("i", i ,"nW", nW)
         #sys.exit()
         no_Block = sprintfPy(no_Block, "B3_F{:d}_{:d}",F, (N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,2.5,5,1.8)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-6 -(i*12),12.8 +(F*8)))
         
         no_Block = sprintfPy(no_Block, "B3_F{:d}_{:d}",F, (N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,2.8,1,1.8)
         mBlocks.SetLineColor(18)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-12 -(i*12),12.8 +(F*8)))
         
         
      
      no_Block = sprintfPy(no_Block, "B3_F{:d}_{:d}",F, (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,3,36,2)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,9.2 +(F*8)))
      
      
   
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 1)
   mBlocks = geom.MakeBox(no_Block,Iron,2.8,36,2)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,13))
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 2)
   mBlocks = geom.MakeBox(no_Block,Iron,2.8,36,2)
   mBlocks.SetLineColor(30)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,9))
   
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 3)
   mBlocks = geom.MakeBox(no_Block,Iron,2.8,36,4)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,3))
   
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 4)
   mBlocks = geom.MakeBox(no_Block,Iron,2.8,36,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,-5))
   
   # Steping 55 Boxes--rooms-- to speed up the process.
   """
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 5)
   mBlocks = geom.MakeBox(no_Block,Iron,2.8,36,6)
   mBlocks.SetLineColor(30)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-48,-15))
   """
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 61)
   mBlocks = geom.MakeBox(no_Block,Iron,3,8,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-88,49))
   
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 62)
   mBlocks = geom.MakeBox(no_Block,Iron,0.5,8,24)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.9,-88,23))
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 63)
   mBlocks = geom.MakeBox(no_Block,Iron,2,7,24)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-88,23))
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 64)
   mBlocks = geom.MakeBox(no_Block,Iron,0.5,8,24)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-0.1,-88,23))
   
   no_Block = sprintfPy(no_Block, "B3_F0{:d}", 65)
   mBlocks = geom.MakeBox(no_Block,Iron,3,8,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-88,-5))
   
   ####################/ Left-Side
   print("####################/ Left-Side")
   nF = 6
   nW = 6
   
   no_Block = sprintfPy(no_Block, "B3_F2{:d}",7)
   mBlocks = geom.MakeBox(no_Block,Iron,7,40.5,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-43.5,49))
   
   for F in range(nF):
      print("F", F, "nF", nF)
      N=0
      for i in range(nW):
         print("i", i, "nW", nW)
         #sys.exit()
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,6,2.35,2)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-14.35-(12*i),5 + (8*F)))
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,6.5,0.3,2)
         mBlocks.SetLineColor(18)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-17-(12*i),5 + (8*F)))
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,6,2.35,2)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-19.65-(12*i),5 + (8*F)))
         
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,7,1,2)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-23-(12*i),5 + (8*F)))
         
      
      no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,6.8,36,0.3)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-48,3.3 + (8*F)))
      
      no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,7,36,2)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-48,1 + (8*F)))
      
      for i in range(4):
         print("i", i, "range", 4)
         #sys.exit()
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,7,0.5,1.4)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-3.5,5.6 + (8*F)))
         
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,6,0.7,1.4)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-4.7,5.6 + (8*F)))
         
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,7,1.6,1.4)
         mBlocks.SetLineColor(2)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-7,5.6 + (8*F)))
         
         no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
         mBlocks = geom.MakeBox(no_Block,Iron,6,0.7,1.4)
         mBlocks.SetLineColor(12)
         Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-9.3,5.6 + (8*F)))
         
      
      no_Block = sprintfPy(no_Block, "B3_F2{:d}_{:d}",F,(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,7,3.5,2.6)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-6.5,1.6 + (8*F)))
      
   
   no_Block = sprintfPy(no_Block, "B3_F2{:d}",71)
   mBlocks = geom.MakeBox(no_Block,Iron,7,40.5,4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-43.5,-5))
   
   no_Block = sprintfPy(no_Block, "B3_F2{:d}",72)
   mBlocks = geom.MakeBox(no_Block,Iron,7,2,30)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-86,21))
   
   no_Block = sprintfPy(no_Block, "B3_F2{:d}",73)
   mBlocks = geom.MakeBox(no_Block,Iron,7,1,30)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.4,-11,21))
   
   
   
   ######################## Rooftop
   print("######################## Rooftop")
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=0))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,7,42.25,0.1)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.15,-45.5,51.1))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,2.75,41.75,0.1)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-54,51.1))
   
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,0.24,41.99,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(5.15,-53.99,51.5))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,0.01,42,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(5.4,-54,51.5))
   
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,0.24,3.99,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-0.35,-92,51.5))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,0.01,4,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-0.6,-92,51.5))
   
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,2.99,0.24,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-95.79,51.5))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}", (N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,3,0.01,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(2.4,-96.04,51.5))
   
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,0.24,42.49,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-14.14,-45.5,51.5))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=N+1))
   print( no_Block)
   mBlocks = geom.MakeBox(no_Block,Iron,0.01,42.5,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-14.39,-45.5,51.5))
   
   
   ############### Stair
   print("############### Stair")
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,6.99,0.24,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.15,-3.25,51.5))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,7,0.01,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.15,-3,51.5))
   
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,7,0.25,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.15,-87.74,51.5))
   no_Block = sprintfPy(no_Block, "B3_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,7,0.01,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-7.15,-87.99,51.5))
   
   
   
   ####################/ Pillars
   print("####################/ Pillars")
   N=0
   for i in range(6):
      print("i", i, "range", 6)
      #sys.exit()
      
      no_Block = sprintfPy(no_Block, "B3_PF{:d}", (N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,1.2,1.5,12)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.6,-12-(12*i),3))
      
   no_Block = sprintfPy(no_Block, "B3_PF{:d}", (N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,1.5,40,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(7,-56,-5))
   
   
   #################### Stair
   print("#################### Stair")
   no_Block = sprintfPy(no_Block, "B3_ST{:d}",(N:=0))
   mBlocks = geom.MakeBox(no_Block,Iron,0.5,7,5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-6.5,-88,-2))
   
   for i in range(5):
      print("i", i, "range", 5)
      #sys.exit()
      no_Block = sprintfPy(no_Block, "B3_ST{:d}",(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,3,5,0.5)
      mBlocks.SetLineColor(18)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3,-86-(0.7*i),-2-(1*i)))
      
   
   
   ##########################################################################################
   #####################           Mid-Building         ####################/
   ##########################################################################################
   print("#####################           Mid-Building         ")
   
   ######################### Left-Side
   print("######################### Left-Side")
   
   for F in range(5):
      print("R", F, "range", 5)
      N=0
      no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",F,(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,5.5,12.5,2.3)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,-7.5,9.6+(8*F)))
      
      no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",F,(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,5.5,2,1.7)
      mBlocks.SetLineColor(2)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,3,13.6+(8*F)))
      
      no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",F,(N:=N+1))
      mBlocks = geom.MakeBox(no_Block,Iron,5,10.5,1.7)
      mBlocks.SetLineColor(12)
      Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,-9.5,13.6+(8*F)))
      
   
   no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",9,(N:=0))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,12.5,6)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,-7.5,53))
   
   no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",9,(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,2,4.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,3,3))
   no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",9,(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5,10.5,4.5)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,-9.5,3))
   
   no_Block = sprintfPy(no_Block, "B4_LF{:d}_{:d}",9,(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,12.5,5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,-7.5,-3))
   
   
   
   
   ########################/ Right-Side
   print("########################/ Right-Side")
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.25,11,24)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.25,-9,19))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.25,4,32)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(8.75,2,27))
   
   
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,2,1.8)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,0,44.8))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,3.5,5)
   mBlocks.SetLineColor(20)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-12.5,0,-4))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,6,2,0.3)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-4,46.3))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4,2,1.5)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-4,44.5))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,7,1.8)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-13,44.8))
   
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,11,1.8)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-9,48.4))
   
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.25,1.5,2)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-0,52.2))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4,2,2)
   mBlocks.SetLineColor(12)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-4,52.2))
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,7,2)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-13,52.2))
   
   
   no_Block = sprintfPy(no_Block, "B4_RS{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,4.5,11,2.4)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-9,56.6))
   
   ###################### RoofTop
   print("###################### RoofTop")
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=0))
   mBlocks = geom.MakeBox(no_Block,Iron,4.25,10.9,0.2)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(4.5,-9,59))
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.25,12.4,0.2)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,-7.5,59))
   
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.24,12.4,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-8.79,-7.5,59.3))
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.01,12.4,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-9.06,-7.5,59.3))
   
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.24,13,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(8.75,-7,59.3))
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,0.01,13,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(9,-7,59.3))
   
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,8.75,0.24,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,-19.75,59.3))
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,8.75,0.01,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(0,-20.01,59.3))
   
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.25,0.24,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.5,4.55,59.3))
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,5.5,0.01,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(-3.75,5.1,59.3))
   
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,3.5,0.24,0.5)
   mBlocks.SetLineColor(18)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(5,1.55,59.3))
   no_Block = sprintfPy(no_Block, "B4_RT{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,3.5,0.01,0.5)
   mBlocks.SetLineColor(2)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(5,2.1,59.3))
   
   ##########################################################################################
   #####################             GROUND             ####################/
   ##########################################################################################
   print("#####################             GROUND             ####################/")
   no_Block = sprintfPy(no_Block, "GRD{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,40,90,2)
   mBlocks.SetLineColor(30)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(5,-20,-9))
   
   no_Block = sprintfPy(no_Block, "GRD{:d}",(N:=N+1))
   mBlocks = geom.MakeBox(no_Block,Iron,30,30,2)
   mBlocks.SetLineColor(41)
   Phy_Building.AddNodeOverlap(mBlocks,1,TGeoTranslation(5,30,-5))


   geom.CloseGeometry()
   
   
   ################# Draw
   print("################# Draw")
   Phy_Building.SetVisibility(False)

   # OpenGL or X3D

   ################# OpenGL  # Has problem in pyroot compiled with root v-6-30-06 
   # ROOT Team is currently debuggin this. 
   #print("Drawing with OpenGl option")
   #Phy_Building.Draw("ogl")
   ##BP
   #input("Press Enter to quit Physics Building")

   ################# X3D --not commonly used-- Deprecated: 
   # Zoom in j-key
   # Zoom out k-key
   # View options :
   #                w-key : Pure Lines (Full Vision Inside)
   #                e-key : Contour Lines 
   #                r-key : Solid Volume
   # Rotate:
   #         y-key : clockwise 
   #         a-key : up-down // Perpendicular to Screen (x-key)
   #         z-key : left-right // Perpendicular to Screen (c-key)
   # Picture, Update : 
   #         d-key : 
   #         f-key : Reconstruction, Update, Re-Modeling (q-key)(t-key)
   
   #
   # Moving in space :
   #                   u-key : up 
   #                   i-key : down 

   # Note: Drawing with OpenGL generates problems. Use x3d instead.
   """
   Error in <TGLLockable::TakeLock>: 'TGLViewerBase' unable to take SelectLock, already SelectLock
   """

   print("Drawing with x3d Option.")
   Phy_Building.Draw("x3d")
   #Phy_Building.Draw("ogl")
   #Phy_Building.Draw("")
   
   
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
  building()
