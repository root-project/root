
## \ingroup tutorial_graphics
## \notebook -js
##
## This scripts show how to use TExec-class to handle keyboard events and TComplex to draw the Mandelbrot
## set.
##
## Pressing the keys 'z' and 'u' will zoom and unzoom the picture
## near the mouse location, 'r' will reset to the default view.
##
## Try it (in compiled mode!) with:   `root mandelbrot.C+`
##
## \macro_image (tcanvas_js)
##
## ### Details
##
##    when a mouse event occurs the myexec() function is called (by
##    using AddExec). Depending on the pressed key, the mygenerate()
##    function is called, with the proper arguments. Note the
##    last_x and last_y variables that are used in myexec() to store
##    the last pointer coordinates (px is not a pointer position in
##    kKeyPress events).
##
## \macro_code
##
## \author Luigi Bardelli <bardelli@fi.infn.it>
## \translator P. P.


import ROOT

TCanvas = ROOT.TCanvas 
TComplex = ROOT.TComplex 
TH2 = ROOT.TH2 
TH2F = ROOT.TH2F 
TComplex = ROOT.TComplex
TROOT = ROOT.TROOT 
TStyle = ROOT.TStyle 
TVirtualPad = ROOT.TVirtualPad 
TPyDispatcher = ROOT.TPyDispatcher

#constants
kKeyPress = ROOT.kKeyPress 
#types
nullptr = ROOT.nullptr
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t

#system utils
Remove = ROOT.gROOT.Remove 

#globals
gPad = ROOT.gPad
gStyle = ROOT.gStyle


#utils
def printf( string , *args):
   print( string % args , end="") 


last_histo = TH2F() #nullptr

# TH2F 
def mygenerate(factor : Double_t, cen_x : Double_t, cen_y : Double_t) :


   printf("Regenerating...\n")
   
   # resize histo:
   global last_histo 
   if factor > 0:

      # Double_t    
      dx = \
      last_histo.GetXaxis().GetXmax() - last_histo.GetXaxis().GetXmin()
      # Double_t    
      dy = \
      last_histo.GetYaxis().GetXmax() - last_histo.GetYaxis().GetXmin()

      last_histo.SetBins(last_histo.GetNbinsX(), cen_x - factor * dx / 2,
         cen_x + factor * dx / 2, last_histo.GetNbinsY(),
         cen_y - factor * dy / 2, cen_y + factor * dy / 2
      )
      last_histo.Reset()
      
   else:

      if (last_histo) :
         Remove( last_histo )

      # allocate first view...
      last_histo = TH2F(
        "h2",
        "Mandelbrot [MOVE-MOUSE-FIRST and  press z to zoom, u to unzoom, r to reset\n  Take your time.]",
        200, -2, 2, 200, -2, 2
      )
      last_histo.SetStats(False)
      
   max_iter = 50
   #for(int bx=1; bx<=last_histo->GetNbinsX(); bx++)
   #for(int by=1; by<=last_histo->GetNbinsY(); by++) {
   global z, point
   for bx in range(1, last_histo.GetNbinsX(), 1):
      for by in range(1, last_histo.GetNbinsY(), 1):

         # Double_t
         x = last_histo.GetXaxis().GetBinCenter(bx)
         # Double_t
         y = last_histo.GetYaxis().GetBinCenter(by)
   
         z = TComplex(x, y)
         point = TComplex(x, y)
         Iter = 0
         while (z.Rho() < 2) :
            z = z * z + point
            last_histo.Fill(x, y)
            Iter += 1
            if (Iter > max_iter):
               break
         
      
   last_histo.SetContour(99)
   last_histo.Draw("colz")
   gPad.Modified()
   gPad.Update()
   printf("Done.\n")
   return last_histo
   

# void
def myexec() :
   #print("myexec()")
   # get event information
   global event
   event = gPad.GetEvent()

   global px, py
   px = gPad.GetEventX()
   py = gPad.GetEventY()
   #print("px , py : ", px, py)
   
   # some magic to get the coordinates...
   global xd, yd, y, x
   xd = gPad.AbsPixeltoX(px)
   yd = gPad.AbsPixeltoY(py)
   x = gPad.PadtoX(xd)
   y = gPad.PadtoY(yd)
   
   global last_x, last_y
   last_x = Float_t()
   last_y = Float_t()
   
   if (event != kKeyPress) :
      last_x = x
      last_y = y
      return
      
   
   Z = 2.
   #switch (px) {
   px = chr(px)
   match (px):
      case 'z': # ZOOM
         last_histo = mygenerate(1. / Z, last_x, last_y)
         #break
      case 'u': # UNZOOM
         last_histo = mygenerate(Z, last_x, last_y)
         #break
      case 'r': # RESET
         last_histo = mygenerate(-1, last_x, last_y)
         #break
      
   

# void
def mandelbrot() :
   # cosmetics...
   gStyle.SetPadGridX(True)
   gStyle.SetPadGridY(True)
 
   global canvas
   canvas = TCanvas("canvas", "View Mandelbrot set")

   # this generates and draws the first view...
   last_histo = mygenerate(-1, 0, 0)
   
   # add exec
   gPad.AddExec("myexec", """
      TPython::Exec( \"myexec() \") ; 
      //TPython::Exec( \"gPad.Modified() \") ; 
      //TPython::Exec( \"gPad.Update() \") ; 
   """)


if __name__ == "__main__":
   mandelbrot()
