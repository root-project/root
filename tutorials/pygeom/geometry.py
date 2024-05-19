## \file
## \ingroup tutorial_geom
## Example of the old geometry package (now obsolete)
#
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
TString = ROOT.TString

# void
def geometry() :
   Dir = ROOT.gSystem.UnixPathName(__file__)
   Dir = TString(Dir)
   Dir.ReplaceAll("geometry.py","")
   Dir.ReplaceAll("/./","/")
   ROOT.gROOT.Macro(("{:s}/na49.C".format(Dir.Data())))
   ROOT.gROOT.Macro(("{:s}/na49geomfile.C".format(Dir.Data())))
   #Not to use: ROOT.gROOT.Macro(("{:s}/na49.py".format(Dir.Data())))
   #Not to use: ROOT.gROOT.Macro(("{:s}/na49geomfile.py".format(Dir.Data())))
   # In Python we simply import and execute.
   #import na49
   #na49()
   #import na49geomfile
   #na49geomfile()
   
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
   geometry()
