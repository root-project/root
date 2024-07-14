## \file
## \ingroup tutorial_graphics
## \notebook
##
## An example on how to display PS, EPS, PDF files in canvas.
##
## So, to load a PS file in a TCanvas-object, the ghostscript program needs to be install.
## - On most unix systems it is installed by default.
## - On Windows it has to be installed from http://pages.cs.wisc.edu/~ghost/
##   also the place where gswin32c.exe sits should be added in the PATH. One
##   way to do it is:
##     1. Start the Control Panel
##     2. Double click on System
##     3, Open the "Advanced" tab
##     4. Click on the "Environment Variables" button
##     5. Find "Path" in "System variable list", click on it.
##     6. Click on the "Edit" button.
##     7. In the "Variable value" field add the path of gswin32c
##        (after a ";") it should be something like:
##        "C:\Program Files\gs\gs8.13\bin"
##     8. click "OK" as much as needed.
##
## \macro_code
##
## \author Valeriy Onoutchin
## \translator P. P.


import ROOT

#classes
TROOT = ROOT.TROOT 
TCanvas = ROOT.TCanvas
TPad = ROOT.TPad
TLatex = ROOT.TLatex
TImage = ROOT.TImage 

#globals
gPad = ROOT.gPad
SetBatch = ROOT.gROOT.SetBatch




# void
def psview() :

   # set to batch mode -> do not display graphics
   ROOT.gROOT.SetBatch(1)
   
   # create a PostScript file
   global Dir
   Dir = ROOT.gROOT.GetTutorialDir()
   Dir.Append("/graphics/feynman.C")
   ROOT.gROOT.Macro( Dir.Data() )
   gPad.Print("feynman.eps")
   
   # back to graphics mode
   ROOT.gROOT.SetBatch(0)
   
   # create an image from PS file
   global ps
   ps = TImage.Open("feynman.eps")
   
   if not ps:
      print("GhostScript (gs) program must be installed\n")
      return
      
   
   global mycanvas, tex
   mycanvas = TCanvas("psexam", "Example how to display PS file in canvas", 600, 400)
   tex = TLatex(0.06,0.9,"The picture below has been loaded from a PS file:")
   tex.Draw()
   
   global eps
   eps = TPad("eps", "eps", 0., 0., 1., 0.75)
   eps.Draw()
   eps.cd()
   ps.Draw("xxx")
   


if __name__ == "__main__":
   psview()
