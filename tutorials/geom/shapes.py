## \file
## \ingroup tutorial_geom
## The old geometry shapes (see script geodemo.C)
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

TGeometry = ROOT.TGeometry 
TGeoManager = ROOT.TGeoManager 

TBRIK = ROOT.TBRIK
TTRD1 = ROOT.TTRD1
TTRD2 = ROOT.TTRD2
TTRAP = ROOT.TTRAP
TPARA = ROOT.TPARA
TGTRA = ROOT.TGTRA
TTUBE = ROOT.TTUBE
TTUBS = ROOT.TTUBS
TCONE = ROOT.TCONE
TCONS = ROOT.TCONS
TSPHE = ROOT.TSPHE
TPCON = ROOT.TPCON
TPGON = ROOT.TPGON

kRed = ROOT.kRed
kBlack = ROOT.kBlack
kYellow = ROOT.kYellow
kBlue = ROOT.kBlue

TNode = ROOT.TNode

TCanvas = ROOT.TCanvas

gGeoManager = ROOT.gGeoManager
ProcessLine = ROOT.gInterpreter.ProcessLine

#def shapes():
class shapes:
 def __init__(self):
   print("Running shapes class ...")
   
   ## Setting-up shapes-class
   # This method work for gGeoManager objects like TGeoVolume, TGeoMaterial, TGeo---------.
   # Here we will not use it because we different objects like TBRIK, TCONE, T----.
   # Check deep-down-below to check a different aproach to delete previous geometries with gROOT.Remove().
   #global gGeoManager
   #if gGeoManager :
   #   ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #   gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   #gGeoManager = ROOT.gGeoManager
   #gGeoManager = ROOT.gGeoManager
    
   global c1 
   ROOT.gROOT.GetListOfCanvases().Clear()
   c1 = TCanvas("glc1","Geometry Shapes",200,10,700,500) # TCanvas 
   #ProcessLine('''
   #gROOT->GetListOfCanvases()->Clear();
   #TCanvas *c1 = new TCanvas("glc1","Geometry Shapes",10,10,500,500);
   #''')
   #c1 = ROOT.c1
   
   # In shapes.C, when we are compiling with clang, we need this line. 
   # However, since pyroot already load all the libraries, we won't need it here.
   #ROOT.gSystem.Load("libGeom")  

   ## Delete previous geometry-objects in case this script is re-executed.
   ## or Cleaning gGeoManager, in case you plot many-many figures.
   
   global gGeoManager
   gGeoManager = ROOT.gGeoManager
   if gGeoManager:
      gGeoManager.GetListOfNodes().Clear()
      gGeoManager.GetListOfShapes().Clear()
      gGeoManager.GetListOfNodes().Delete()
      gGeoManager.GetListOfShapes().Delete()
   # NOTE: Be careful by using this method of cleaning and deleting.
   # TBRIK and similar classes don't save objects on gGeoManager.
   # We have to delete it manually. 
     
      
   global brik, trd1, trd2, trap, para, gtra, tube, tubs
   global cone, cons, sphe, sphe1, sphe2  
   global pcon, pgon
   #  Define some volumes
   brik = TBRIK("BRIK","BRIK","void",200,150,150)
   trd1 = TTRD1("TRD1","TRD1","void",200,50,100,100)
   trd2 = TTRD2("TRD2","TRD2","void",200,50,200,50,100)
   trap = TTRAP("TRAP","TRAP","void",190,0,0,60,40,90,15,120,80,180,15)
   para = TPARA("PARA","PARA","void",100,200,200,15,30,30)
   gtra = TGTRA("GTRA","GTRA","void",390,0,0,20,60,40,90,15,120,80,180,15)
   tube = TTUBE("TUBE","TUBE","void",150,200,400)
   tubs = TTUBS("TUBS","TUBS","void",80,100,100,90,235)
   cone = TCONE("CONE","CONE","void",100,50,70,120,150)
   cons = TCONS("CONS","CONS","void",50,100,100,200,300,90,270)
   sphe = TSPHE("SPHE","SPHE","void",25,340, 45,135, 0,270)
   sphe1 = TSPHE("SPHE1","SPHE1","void",0,140, 0,180, 0,360)
   sphe2 = TSPHE("SPHE2","SPHE2","void",0,200, 10,120, 45,145)
   
   pcon = TPCON("PCON","PCON","void",180,270,4)
   pcon.DefineSection(0,-200,50,100)
   pcon.DefineSection(1,-50,50,80)
   pcon.DefineSection(2,50,50,80)
   pcon.DefineSection(3,200,50,100)
   
   pgon = TPGON("PGON","PGON","void",180,270,8,4)
   pgon.DefineSection(0,-200,50,100)
   pgon.DefineSection(1,-50,50,80)
   pgon.DefineSection(2,50,50,80)
   pgon.DefineSection(3,200,50,100)
   
   #  Set shapes attributes
   brik.SetLineColor(1)
   trd1.SetLineColor(2)
   trd2.SetLineColor(3)
   trap.SetLineColor(4)
   para.SetLineColor(5)
   gtra.SetLineColor(7)
   tube.SetLineColor(6)
   tubs.SetLineColor(7)
   cone.SetLineColor(2)
   cons.SetLineColor(3)
   pcon.SetLineColor(6)
   pgon.SetLineColor(2)
   sphe.SetLineColor(kRed)
   sphe1.SetLineColor(kBlack)
   sphe2.SetLineColor(kBlue)
   
   
   #  Build the geometry hierarchy
   global node1, node2, node3, node4, node5, node6, node7, node8
   global node9, node10, node11, node12, node13, node14, node15
   node1 = TNode("NODE1","NODE1","BRIK")
   node1.cd()
   
   node2 = TNode("NODE2","NODE2","TRD1",0,0,-1000)
   node3 = TNode("NODE3","NODE3","TRD2",0,0,1000)
   node4 = TNode("NODE4","NODE4","TRAP",0,-1000,0)
   node5 = TNode("NODE5","NODE5","PARA",0,1000,0)
   node6 = TNode("NODE6","NODE6","TUBE",-1000,0,0)
   node7 = TNode("NODE7","NODE7","TUBS",1000,0,0)
   node8 = TNode("NODE8","NODE8","CONE",-300,-300,0)
   node9 = TNode("NODE9","NODE9","CONS",300,300,0)
   node10 = TNode("NODE10","NODE10","PCON",0,-1000,-1000)
   node11 = TNode("NODE11","NODE11","PGON",0,1000,1000)
   node12 = TNode("NODE12","NODE12","GTRA",0,-400,700)
   node13 = TNode("NODE13","NODE13","SPHE",10,-400,500)
   node14 = TNode("NODE14","NODE14","SPHE1",10, 250,300)
   node15 = TNode("NODE15","NODE15","SPHE2",10,-100,-200)
   
   
   # Draw this geometry in the current canvas
   node1.cd()
   node1.Draw("gl")
   c1.Update()
   c1.Draw()
   #
   #  Draw the geometry using the OpenGL viewer.
   #  Note that this viewer may also be invoked from the "View" menu in
   #  the canvas tool bar
   #
   # once in the viewer, select the Help button
   # For example typing r will show a solid model of this geometry.
   
   # If you have memory problems use this:
   #raise RuntimeError("To Keep open the Canvas without crashing memory")
   # But it is unlikely to happen.
   
   
   #  Cleaning and Deleting some volumes
   #  # This method doesn´t work. It Deletes from python isntead of gROOT. 
   #brik.Clear() 
   #brik.Delete()
   #trd1.Clear() 
   #trd1.Delete()
   #trd2.Clear() 
   #trd2.Delete()
   #trap.Clear() 
   #trap.Delete()
   #para.Clear() 
   #para.Delete()
   #gtra.Clear() 
   #gtra.Delete()
   #tube.Clear() 
   #tube.Delete()
   #tubs.Clear() 
   #tubs.Delete()
   #cone.Clear() 
   #cone.Delete()
   #cons.Clear() 
   #cons.Delete()

   # Using Remove instead
   ROOT.gROOT.Remove( brik ) 
   ROOT.gROOT.Remove( trd1 ) 
   ROOT.gROOT.Remove( trd2 ) 
   ROOT.gROOT.Remove( trap ) 
   ROOT.gROOT.Remove( para ) 
   ROOT.gROOT.Remove( gtra ) 
   ROOT.gROOT.Remove( tube ) 
   ROOT.gROOT.Remove( tubs ) 
   ROOT.gROOT.Remove( cone ) 
   ROOT.gROOT.Remove( cons ) 


   
   # Cleaning and Deleting some nodes
   # This method, doesn´t work either.
   #node1.Clear() 
   #node1.Delete()
   #node2.Clear() 
   #node2.Delete()
   #node3.Clear() 
   #node3.Delete()
   #node4.Clear() 
   #node4.Delete()
   #node5.Clear() 
   #node5.Delete()
   #node6.Clear() 
   #node6.Delete()
   #node7.Clear() 
   #node7.Delete()
   #node8.Clear() 
   #node8.Delete()
   #node9.Clear() 
   #node9.Delete()
   #node10.Clear() 
   #node10.Delete()
   #node11.Clear() 
   #node11.Delete()
   #node12.Clear() 
   #node12.Delete()
   #node13.Clear() 
   #node13.Delete()
   #node14.Clear() 
   #node14.Delete()
   #node15.Clear() 
   #node15.Delete()
   
   # Using ROOT.gROOT.Remove() instead
   ROOT.gROOT.Remove( node1 ) 
   ROOT.gROOT.Remove( node2 ) 
   ROOT.gROOT.Remove( node3 ) 
   ROOT.gROOT.Remove( node4 ) 
   ROOT.gROOT.Remove( node5 ) 
   ROOT.gROOT.Remove( node6 ) 
   ROOT.gROOT.Remove( node7 ) 
   ROOT.gROOT.Remove( node8 ) 
   ROOT.gROOT.Remove( node9 ) 
   ROOT.gROOT.Remove( node10 ) 
   ROOT.gROOT.Remove( node11 ) 
   ROOT.gROOT.Remove( node12 ) 
   ROOT.gROOT.Remove( node13 ) 
   ROOT.gROOT.Remove( node14 ) 
   ROOT.gROOT.Remove( node15 ) 

   # After many, many trials. This method should be included in pyroot as an standard rule.
   # Why ? Python manages memory differently than C++. It isn't efficient sometimes, but that 
   # doesn't matter when we deal with its philosophy of "scripting-your-idea-into-code".
   # This method allows run this script many, many times without crashing memory; and also using 
   # or leaving the canvas open to play with. 

   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   gb = [i for i in globals()] 
   myvars = [x for x in dir() if not x.startswith("__")]
   myvars += [x for x in gb if not x.startswith("__")]
   myvars += [x for x in vars(self) ] 
   for var in myvars: 
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         exec(f"ROOT.gROOT.Remove(self.{var})")
         print("deleting", var)
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!
   
   
      

if __name__ == "__main__":
   myshapes = shapes()
