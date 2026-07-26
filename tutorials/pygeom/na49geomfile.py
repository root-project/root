## \file
## \ingroup tutorial_geom
##  Before executing this macro, the file na49.C or na49.py must have been executed
##  The important thing is that ROOT knows what "na49" is.  
##  In Python, you can run it after you've just ran na49.py; so you can save 
##  its geometry on na49.root file. Such a file would help you later to run na49view.py
##  to display two sides of the na49-geometry on a single TCanvas. Happy coding:)
##
## \macro_code
##
## \author Andrei Gheata
## \translator P. P.


import ROOT

gBenchmark = ROOT.gBenchmark
TGeometry = ROOT.TGeometry
TFile = ROOT.TFile

# void
def na49geomfile() :
   global gBenchMark
   gBenchmark.Start("geometry")
   n49 = ROOT.gROOT.FindObject("na49") #TGeometry
   if n49:
      print("n49-object was found. It is going to be saved on na49.root")
      na = TFile("na49.root","RECREATE")
      n49.Write()
      na.Write()
      
   else: 
      print("n49-object was NOT found.")
      print("Please, run first na49.py")
 
   gBenchmark.Show("geometry")
   


if __name__ == "__main__":
   na49geomfile()
