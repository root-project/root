## \file
## \ingroup tutorial_geom
## Creates and runs a simple iterator plugin connected to TGeoPainter iterator.
##
## It demonstrates the possibility to dynamically change the color of drawn
## volumes according some arbitrary criteria *WITHOUT* changing the color of the
## same volume drawn on branches that do not match the criteria.
##
## ~~~{.cpp}
## To run:
## IP[1]   %run runplugin.py
## IP[2]   select(2,kMagenta)
## IP[3]   select(3,kBlue)
## ...
## ~~~
##
## \macro_code
##
## \author Andrei Gheata
## \translator P. P.


import ROOT

gGeoManager = ROOT.gGeoManager

nullptr = ROOT.nullptr
kGreen = ROOT.kGreen
kMagenta = ROOT.kMagenta
kBlue = ROOT.kBlue

# 
ROOT.gROOT.LoadMacro("iterplugin.cxx+")
iterplugin = ROOT.iterplugin

plugin = nullptr 
def runplugin() :
   
   tutdir = str(ROOT.gROOT.GetTutorialDir())
  
   ROOT.gROOT.ProcessLine(".x " + tutdir + "/geom/rootgeom.C")
   #ROOT.TPython.Exec("import rootgeom")
   #ROOT.TPython.Exec("rootgeom.rootgeom()")

   global plugin
   plugin = iterplugin()
   gGeoManager.GetGeomPainter().SetIteratorPlugin(plugin)

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
         print("deleting", var)
         #Improve: Not to use exec, consumes much memory. Try without exec.
      except :
         pass 
   # Now, it works!!!
   

def select(replica=1, color=kGreen) :
   # Change current color. Replica range: 1-4
   global plugin, gGeoManager
   plugin.Select(replica, color)
   gGeoManager.GetGeomPainter().ModifiedPad()
   


if __name__ == "__main__":

   runplugin()

   # Improve: Make loop show canvas in between.
   #while(True):
   for i in range(1):
      #ROOT.gROOT.FindObject("c1").Close()

      r = input("\n\nselect(replica = [1,2,3,4], color =[kGreen, kMagenta, kBlue]) " +
            "\n... choose\n"+
            "\nreplica = ? int ")
      c = input("select(replica = [1,2,3,4], color =[kGreen, kMagenta, kBlue]) " +
            "\n... choose\n"+
            "\ncolor = ? string or int  ")
      try: 
         #select(int(r), locals()[c]) 
         if int(r) not in range(1,5): raise RuntimeError

         if   r.isalpha() and c.isalpha() : raise RuntimeError#select(int(r),locals()[c])
         elif r.isdigit() and c.isdigit() : select(int(r), int(c)) 
         elif r.isalpha() and c.isdigit() : raise RuntimeError#select(int(r), int(c) )
         elif r.isdigit() and c.isalpha() : select(int(r), locals()[c]) 
         print(20*">"+" Look your new style") 
         
      except :
         print("something went wrong. Type Correctly, Please!")



