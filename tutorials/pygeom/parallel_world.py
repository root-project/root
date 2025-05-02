## \file
## \ingroup tutorial_geom
## Misaligning geometry generate in many cases overlaps, due to the idealization
## of the design and the fact that in real life movements of the geometry volumes
## have constraints and are correlated.
##
## This typically generates inconsistent
## response of the navigation methods, leading to inefficiencies during tracking,
## errors in the material budget calculations, and so on. Among those, there are
## dangerous cases when the hidden volumes are sensitive.
## This macro demonstrates how to use the "parallel world" feature to assign
## highest navigation priority to some physical paths in geometry.
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
TGeoBBox = ROOT.TGeoBBox
TGeoVolume = ROOT.TGeoVolume
TGeoVolumeAssembly = ROOT.TGeoVolumeAssembly
TGeoPhysicalNode = ROOT.TGeoPhysicalNode

TString = ROOT.TString
TCanvas = ROOT.TCanvas
TStopwatch = ROOT.TStopwatch

#k = ROOT.k
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kTRUE = ROOT.kTRUE
kFALSE = ROOT.kFALSE
kGreen = ROOT.kGreen

#_t = ROOT._t
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t

#g= ROOT.g
gPad = ROOT.gPad
gGeoManager = ROOT.gGeoManager

ProcessLine = ROOT.gInterpreter.ProcessLine 
gGeoManager = ROOT.gGeoManager

def DelROOTObjs(self):
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   #myvars = [x for x in dir() if not x.startswith("__")]
   #myvars = [x for x in globals() if not x.startswith("_")]
   myvars = [x for x in vars(self) ] 
   for var in myvars: 
      #exec(f"ROOT.gROOT.Remove(self.{var})")
      #print(var)
      try:
         exec(f"ROOT.gROOT.Remove(self.{var})")
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!

#______________________________________________________________________________
# void
#def parallel_world(usepw=kTRUE, useovlp=kTRUE) :
class parallel_world:
 #______________________________________________________________________________
 # void
 def align(self) :
   # Aligning 2 sensors so they will overlap with the support. One sensor is positioned
   # normally while the other using the shared matrix
   global node, pw
   self.node = TGeoPhysicalNode()
   self.pw = self.gGeoManager.GetParallelWorld()
   self.sag = Double_t()
   #    for (Int_t i=0; i<10; i++)
   for  i in range(0, 10): 
      #Not to Use: self.node = self.gGeoManager.MakePhysicalNode(TString.Format("/TOP_1/chip_{:d}".format(i+1)))
      self.node = self.gGeoManager.MakePhysicalNode(("/TOP_1/chip_{:d}".format(i+1)))
      self.sag = 8. - 0.494*(i-4.5)*(i-4.5)
      self.tr = TGeoTranslation(0., -225.+50.*i, 10-self.sag)
      self.node.Align(self.tr)
      if (self.pw): self.pw.AddNode(("/TOP_1/chip_{:d}".format(i+1)))
   DelROOTObjs(self) 


 def __init__(self, usepw=kTRUE, useovlp=kTRUE) :
   print("running parllel_world class")
   global gGeoManager
   if gGeoManager:
      ROOT.gGeoManager = ROOT.MakeNullPointer("TGeoManager")
   gGeoManager = ROOT.gGeoManager
   self.gGeoManager = ROOT.gGeoManager

   #self.geom = TGeoManager("parallel_world", "Showcase for prioritized physical paths")
   ProcessLine('''
   TGeoManager *geom = new TGeoManager("parallel_world", "Showcase for prioritized physical paths");
   ''')
   global geom
   geom = ROOT.geom
   self.geom = ROOT.geom

   self.matV = TGeoMaterial("Vac", 0,0,0)
   self.medV = TGeoMedium("MEDVAC",1,self.matV)
   self.matAl = TGeoMaterial("Al", 26.98,13,2.7)
   self.medAl = TGeoMedium("MEDAL",2,self.matAl)
   self.matSi = TGeoMaterial("Si", 28.085,14,2.329)
   self.medSi = TGeoMedium("MEDSI",3,self.matSi)
   global top
   self.top = self.gGeoManager.MakeBox("TOP",self.medV,100,400,1000)
   self.gGeoManager.SetTopVolume(self.top)
   
   # Shape for the support block
   self.sblock = TGeoBBox("sblock", 20,10,2)
   # The volume for the support
   self.support = TGeoVolume("block", self.sblock, self.medAl)
   self.support.SetLineColor(kGreen)
   
   # Shape for the sensor to be prioritized in case of overlap
   self.ssensor = TGeoBBox("sensor", 19,9,0.2)
   # The volume for the sensor
   self.sensor = TGeoVolume("sensor",self.ssensor, self.medSi)
   self.sensor.SetLineColor(kRed)
   
   # Chip assembly of support+sensor
   self.chip = TGeoVolumeAssembly("chip")
   self.chip.AddNode(self.support, 1)
   self.chip.AddNode(self.sensor,1, TGeoTranslation(0,0,-2.1))
   
   # A ladder that normally sags
   self.sladder = TGeoBBox("sladder", 20,300,5)
   # The volume for the ladder
   self.ladder = TGeoVolume("ladder",self.sladder, self.medAl)
   self.ladder.SetLineColor(kBlue)
   
   # Add nodes
   self.top.AddNode(self.ladder,1)
   #    for (Int_t i=0; i<10; i++)
   for  i in range(0, 10):
      self.top.AddNode(self.chip, i+1, TGeoTranslation(0, -225.+50.*i, 10))
   
   self.gGeoManager.CloseGeometry()
   self.pw = 0
   self.pw = self.gGeoManager.CreateParallelWorld("priority_sensors")
   # Align chips
   parallel_world.align(self) 
   if usepw:
      if (useovlp): self.pw.AddOverlap(self.ladder)
      self.pw.CloseGeometry()
      self.gGeoManager.SetUseParallelWorldNav(True)
      
   self.cname = TString()
   self.cname = "cpw" if usepw else "cnopw"
   global c 
   self.c = ROOT.gROOT.GetListOfCanvases().FindObject(self.cname) # (TCanvas)
   if (self.c): self.c.cd()
   self.c = TCanvas(self.cname, "",800,600)
   self.top.Draw()
   #   top.RandomRays(0,0,0,0, sensor.GetName())
   # Track random "particles" coming from the block side and draw only the tracklets
   # actually crossing one of the sensors. Note that some of the tracks coming
   # from the outer side may see the full sensor, while the others only part of it.
   self.timer = TStopwatch()
   self.timer.Start()
   self.top.RandomRays(100000,0,0,-30,self.sensor.GetName())
   self.timer.Stop()
   self.timer.Print()
   
   global gPad
   self.gPad = ROOT.gPad 
   global view
   self.view = gPad.GetView() # TView3D
   self.view.SetParallel()
   self.view.Side()
   self.view.Draw()
   if (usepw): self.pw.PrintDetectedOverlaps()
   
   DelROOTObjs(self)
   
   # if pyroot crashes. we use raise error to keep the canvas up.
   #raise RuntimeError("Press Enter to exit parallel_world() function")
   #input("Press Enter to quit parallel_world() function")
   # This can be solved by defining a class instead of functions, therefrom we can save all important objects.
   # However, in terms of readability, it doesn´t have a good look. Alternatives are to def f(): and raise 
   # error at its end; or we can define all inner variables as global. Choose which one you preffer better.    
   # In this script you can replace all self--variables by just variables eliminating --self.-- from them. 
   # That shouldn´t be huge task.
   
   # You can still view the Canvas after the running of parallel_word(), if you create an instance of the class.
   # ` myPW = parallel_world()`
   # Otherwise it'll open and close withou crashing memory; which is good but it is not the purpose.
   # Create instances of as many parallel_world()'s as you want and inspect for its members.
   # If you close the Canvas, you are destroying gPad, gGeoManager; therefore you cannot re-draw it with:
   # myPW.view.Draw() # Doesn´t work, if you closed the Canvas. 
   # You'll have to re-run again :
   #  myPW = parallel_wordl()
   # to see its geometry.
 
   #DelROOTObjs(self) 
   # #############################################################
   # If you don´t use it, after closing the-canvas-window storms in
   # your-ipython-interpreter will happen. By that I mean crashing 
   # memory iteratively. Since the timer is 'On', it repeats the 
   # process again-and-again.  
   #  
   #print("Deleting objs from gROOT")
   myvars = [x for x in dir() if not x.startswith("__")]
   myvars += [x for x in globals() if not x.startswith("_")]
   #myvars = [x for x in vars(self) ] 
   for var in myvars: 
      #print(var)
      try:
         exec(f"ROOT.gROOT.Remove({var})")
         exec(f"ROOT.gROOT.Remove(self.{var})")
         #print("deleting", var, "from gROOT")
         #Improve: Not to use exec, consumes too much memory. Try without exec.
         #TODO: replace much by toomuch. English language error.
      except :
         pass 
   # Now, it works!!!
      
   


if __name__ == "__main__":
   myPW = parallel_world()
