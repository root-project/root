# \file
# \ingroup FOAM's_python_tutorials
# \notebook -nodraw
# This simple macro demonstrates persistency of FOAM object.
# First run macro foam_demo.py to create file foam_demo.root with FOAM object.
#
# Next type `ipytohn3 foam_demopers.py` from shell command line
#
# \macro_code
#
# \author Stascek Jadach
# \translator P. P.



import ROOT
import foam_demo
import ctypes


TFile = 		 ROOT.TFile
TFoam = 		 ROOT.TFoam
TROOT = 		 ROOT.TROOT
TSystem = 		 ROOT.TSystem
TFoamIntegrand = 		 ROOT.TFoamIntegrand

TFDISTR_Py = foam_demo.TFDISTR_Py 
TFDISTR = foam_demo.TFDISTR 
 
Double_t = ROOT.Double_t
c_double = ctypes.c_double

ProcessLine = ROOT.gInterpreter.ProcessLine

def foam_demopers():

   
   # need to load the foam_demo tutorial for the definition of the function rho
   macroName = ROOT.gROOT.GetTutorialDir()
   macroName.Append("foam/foam_demo.py")
   # ProcessLine(" .L {:s}+".format( macroName.Data() )[:-3]+".C" )
   # This was already been done at `import foam_demo` .
   # However, it can be done independently using `root[] .L foam_demo.C ` as shown above.
   # Alternatively, we can use the actual `foam_demo.py` script from `root` itself as follows.
   # ProcessLine(" TPython::LoadMacro('{:s}')".format( macroName.Data() ) ) 
   # But it has a few problems with the dictionaries, for now. Anyway, it works fine.
   
   #******************************************
   print("================================")
   fileA = TFile("foam_demo.root")
   fileA.cd()
   print("------------------------------------------------------------------")
   fileA.ls()
   print("------------------------------------------------------------------")
   fileA.Map()
   print("------------------------------------------------------------------")
   fileA.ShowStreamerInfo()
   print("------------------------------------------------------------------")
   fileA.GetListOfKeys().Print()
   print("------------------------------------------------------------------")
   #*******************************************
   FoamX = fileA.Get("FoamX")
   #*******************************************
   #  FoamX.PrintCells()
   FoamX.CheckAll(1)
   
   # N.B. the integrand functions need to be reset
   # because cannot be made persistent
   # If you used the `.C` class use the next. 
   # rho = ProcessLine("return  TFDISTR()")
   # If you used the `.py` class you can use 
   rho = TFDISTR_Py()
   # or 
   # rho = TFDISTR() # either way. 
   FoamX.SetRho(rho)
   
   MCvect = [Double_t()]*2 # 2-dim vector generated in the MC run
   c_MCvect = (c_double*2)(*MCvect) 
   
   for loop in range( 50000 ):
      FoamX.MakeEvent()            # generate MC event
      FoamX.GetMCvect( c_MCvect)   # get generated vector (x,y)
      x = c_MCvect[0]
      y = c_MCvect[1]
      if loop < 10 :
         print(f"loop at {loop} with (x,y) = ( ", x ,", ", y ," )")
      if loop % 10000 == 0:
         print(f"loop at {loop} with (x,y) = ( ", x ,", ", y ," )")
         
   # loop's end 
   #
   IntNorm, Errel, MCresult, MCerror = [c_double() for i in range(4)]
   FoamX.Finalize(   IntNorm, Errel)     # final printout
   FoamX.GetIntegMC( MCresult, MCerror)  # get MC integral, should be one
   print(f" MCresult= {MCresult.value:6f} +- {MCerror.value:6f}" )
   print("===================== TestPers FINISHED =======================")
   return 0
   
#_____________________________________________________________________________
#

if __name__ == "__main__":
   foam_demopers()
