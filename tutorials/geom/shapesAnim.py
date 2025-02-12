## \file
## \ingroup tutorial_geom
## Macro illustrating how to animate a geometry picture using a Timer
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

TTimer = ROOT.TTimer 
TRandom3 = ROOT.TRandom3 

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
TSPHE = ROOT.TSPHE
TSPHE = ROOT.TSPHE
TPCON = ROOT.TPCON
TPGON = ROOT.TPGON
 
TNode = ROOT.TNode

kRed = ROOT.kRed
kBlack = ROOT.kBlack
kYellow = ROOT.kYellow
kBlue = ROOT.kBlue

ProcessLine = ROOT.gInterpreter.ProcessLine
gPad = ROOT.gPad
gGeoManager = ROOT.gGeoManager

# Spherical Coordinates 
pi = 0 
theta = 0
phi = 30

# void
#Not to use: def shapesAnim() :
class shapesAnim :
      
   def __init__(self):

      #Note: Delete previous geometry objects in case this script is re-executed
      #      Notice the word script. If you call the class again as in shapesAnim()
      #      no memory problems araise. If you run again %run shapesAnim.py problems 
      #      araise if you don't delete your previous geometries.
      # Delete all variables stores in gROOT.
      # Including your geometries.
      ROOT.gROOT.Clear()
      global gGeoManager
      if gGeoManager:
         gGeoManager.GetListOfNodes().Delete()
         gGeoManager.GetListOfShapes().Delete()
   
      ROOT.gROOT.GetListOfCanvases().Clear()
      # c1 = TCanvas("c1","Geometry Shapes",10,10,500,500)
      ProcessLine('''
      TCanvas *c1 = new TCanvas("c1","Geometry Shapes with Animation",10,10,500,500);
      ''')
      global c1
      self.c1 = ROOT.c1 
      
      global gPad
      self.gPad = ROOT.gPad
      
      #  Define some volumes
      self.brik = TBRIK("BRIK","BRIK","void",200,150,150)
      self.trd1 = TTRD1("TRD1","TRD1","void",200,50,100,100)
      self.trd2 = TTRD2("TRD2","TRD2","void",200,50,200,50,100)
      self.trap = TTRAP("TRAP","TRAP","void",190,0,0,60,40,90,15,120,80,180,15)
      self.para = TPARA("PARA","PARA","void",100,200,200,15,30,30)
      self.gtra = TGTRA("GTRA","GTRA","void",390,0,0,20,60,40,90,15,120,80,180,15)
      self.tube = TTUBE("TUBE","TUBE","void",150,200,400)
      self.tubs = TTUBS("TUBS","TUBS","void",80,100,100,90,235)
      self.cone = TCONE("CONE","CONE","void",100,50,70,120,150)
      self.cons = TCONS("CONS","CONS","void",50,100,100,200,300,90,270)
      self.sphe = TSPHE("SPHE","SPHE","void",25,340, 45,135, 0,270)
      self.sphe1 = TSPHE("SPHE1","SPHE1","void",0,140, 0,180, 0,360)
      self.sphe2 = TSPHE("SPHE2","SPHE2","void",0,200, 10,120, 45,145)
      
      self.pcon = TPCON("PCON","PCON","void",180,270,4)
      self.pcon.DefineSection(0,-200,50,100)
      self.pcon.DefineSection(1,-50,50,80)
      self.pcon.DefineSection(2,50,50,80)
      self.pcon.DefineSection(3,200,50,100)
      
      self.pgon = TPGON("PGON","PGON","void",180,270,8,4)
      self.pgon.DefineSection(0,-200,50,100)
      self.pgon.DefineSection(1,-50,50,80)
      self.pgon.DefineSection(2,50,50,80)
      self.pgon.DefineSection(3,200,50,100)
      
      #  Set shapes attributes
      self.brik.SetLineColor(1)
      self.trd1.SetLineColor(2)
      self.trd2.SetLineColor(3)
      self.trap.SetLineColor(4)
      self.para.SetLineColor(5)
      self.gtra.SetLineColor(7)
      self.tube.SetLineColor(6)
      self.tubs.SetLineColor(7)
      self.cone.SetLineColor(2)
      self.cons.SetLineColor(3)
      self.pcon.SetLineColor(6)
      self.pgon.SetLineColor(2)
      self.sphe.SetLineColor(kRed)
      self.sphe1.SetLineColor(kBlack)
      self.sphe2.SetLineColor(kBlue)
      
      
      #  Build the geometry hierarchy
      global node1
      self.node1 = TNode("NODE1","NODE1","BRIK")
      self.node1.cd()
      
      self.node2 = TNode("NODE2","NODE2","TRD1",0,0,-1000)
      self.node3 = TNode("NODE3","NODE3","TRD2",0,0,1000)
      self.node4 = TNode("NODE4","NODE4","TRAP",0,-1000,0)
      self.node5 = TNode("NODE5","NODE5","PARA",0,1000,0)
      self.node6 = TNode("NODE6","NODE6","TUBE",-1000,0,0)
      self.node7 = TNode("NODE7","NODE7","TUBS",1000,0,0)
      self.node8 = TNode("NODE8","NODE8","CONE",-300,-300,0)
      self.node9 = TNode("NODE9","NODE9","CONS",300,300,0)
      self.node10 = TNode("NODE10","NODE10","PCON",0,-1000,-1000)
      self.node11 = TNode("NODE11","NODE11","PGON",0,1000,1000)
      self.node12 = TNode("NODE12","NODE12","GTRA",0,-400,700)
      self.node13 = TNode("NODE13","NODE13","SPHE",10,-400,500)
      self.node14 = TNode("NODE14","NODE14","SPHE1",10, 250,300)
      self.node15 = TNode("NODE15","NODE15","SPHE2",10,-100,-200)
      
      
      # Draw this geometry in the current canvas
      self.node1.cd()
      self.node1.Draw()
      self.c1.Update()
      
      
      #start a Timer
      ProcessLine('''
      //TTimer *timer = new TTimer(2);    // Really Fast
      //TTimer *timer = new TTimer(20);
      //TTimer *timer = new TTimer(200);
      //TTimer *timer = new TTimer(2000); // Really slow
      ''')
      #self.timer = ROOT.timer
      self.timer = TTimer(2)
   
      #Set your python function name
      FuncName = "shapesAnim.Animate(my_shapesAnim_obj)" # Random Rotation
      #FuncName = "Animate2()" # Random Rotation
      #FuncName = "Animate('N')" # Normal Rotation
      #FuncName = "Animate('R')" # Random Rotation
      global PyFuncName
      PyFuncName = 'TPython::Exec("{}")'.format(FuncName)
      self.timer.SetCommand(PyFuncName)
      #Not to use: timer.SetCommand(FuncName)
     
      self.timer.TurnOn()
      #timer.TurnOff()
    
      #raise RuntimeError("Anticipate Error to Avoid Crash: After Closing Window")
       
      #Before we close the functions, we'll define all its variables as global.
      # Otherwise we would have have to write 'global' many many times.
      # We can try to minimize code by using dir() and iterate over all variables.
      #valid_vars = [var for var in dir() if var.isidentifier()] 
      #for var in valid_vars:
      #   exec(f"global {var}")
      
      # The alternative is to define a class instead of a function. After an instance 
      # of the class is declared, its elements remain intact. That's what we need here.

      # #############################################################
      # If you don´t use it, after closing the-canvas-window storms in
      # your-ipython-interpreter will happen. By that I mean crashing 
      # memory iteratively. Since the timer is 'On', it repeats the 
      # process again-and-again.  

      #print("Deleting objs from gROOT")
      myvars = [x for x in vars(self)]
      for var in myvars: 
         if not var.startswith("__") :
            #print("deleting", var)
            exec(f"ROOT.gROOT.Remove(self.{var})")
      # Now, it works!!!
   
   
   def Animate(self, option = "R"):
      global theta, phi, pi
      if option == "R":
         r3 = TRandom3()
         theta += 10 * r3.Rndm()
         phi += 10 * r3.Rndm()
         pi += 10 * r3.Rndm()
   
      if option == "N":
         theta += 2
         phi += 2
         pi += 0
      #print(theta, phi, pi)

      # Getting out of TTimer after closing the Canvas.
      if not self.gPad :
         print("gPad not found! Setting Timer Off.")
         self.timer.TurnOff() 
         return # Keep this. Otherwise, self.gPad.GetView will raise Error.

      self.gPad.GetView().RotateView(theta,phi)
      self.gPad.Modified()
      self.gPad.Update()
      self.gPad.Draw()


# In case you need it:
def avoidMemoryCrash():
   raise RuntimeError("Avoid Memory Crash at Closing Window")
# It helped me at translating this C-tutorial into Python.



if __name__ == "__main__":
   #shapesAnim()
   my_shapesAnim_obj = shapesAnim()
   #my_shapesAnim_obj.timer.TurnOff() # In case you need it.


   
 
 
