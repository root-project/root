## \file
## \ingroup tutorial_graphics
## \notebook
##
## Example on how to define a canvas with partition.
##
## Well, ... 
## Sometimes the Divide() method is not appropriate to divide a Canvas.
## For instance: in a 4x4 division, the upper left subcanvas(or pad) is not the
## the same as the lower right subcanvas. 
## This happens because the left and right margins of all the pads do not have the
## same width and height. 
## So, CanvasPartition(this script) does that-- divide properly--. 
## This example also ensure that the axis labels and titles have the same
## sizes and that the tick-marks-length, we let them being uniform.
## In addition, the XtoPad and YtoPad functions(which also we define here)
## allow to emplace graphics objects, like text,
## in the right place and in each sub-pad previously defined.
##
## Enjoy TCanving, TPadding, and TDoing.
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT

import cppyy
#ctypes = cppyy.ctypes # Wait to cppyy3.11 to implement ctypes.
import ctypes

#classes
TCanvas = ROOT.TCanvas
TH1F = ROOT.TH1F
TText = ROOT.TText
TPad = ROOT.TPad

#types
Float_t = ROOT.Float_t
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t
c_double = ctypes.c_double


#globals
gStyle = ROOT.gStyle
gROOT = ROOT.gROOT
gPad = ROOT.gPad

#system-utils
FindObject = ROOT.gROOT.FindObject
Remove = ROOT.gROOT.Remove


#pre-definition
#def CanvasPartition(C : TCanvas = TCanvas(), Nx : Int_t = 2, Ny : Int_t = 2,
#                     lMargin : Float_t = 0.15, rMargin : Float_t = 0.05,
#                     bMargin : Float_t = 0.15, tMargin : Float_t = 0.05): pass
#def toPad(x : Double_t): pass
#def YtoPad(x : Double_t): pass


# void
def canvas2() :
   gStyle.SetOptStat(0)
   
   global C
   C = ROOT.gROOT.FindObject("C") # TCanvas
   if (C) : 
      ROOT.gROOT.Remove(C) 
      del C
   C = TCanvas("C","canvas",1024,640)
   C.SetFillStyle(4000)
   
   # Number of PADS
   Nx = 5 # Ok.
   Ny = 5 # Ok.
   # Number of PADS: Variations
   #Nx = 3 # Ok.
   #Ny = 3 # Ok.
   #Nx = 10 # Ok.
   #Ny = 3 # Ok.
   #Nx = 5 # Ok.
   #Ny = 7 # Ok.
   
   # Margins
   lMargin = 0.12
   rMargin = 0.05
   bMargin = 0.15
   tMargin = 0.05
   
   # Canvas setup
   C = CanvasPartition(C,Nx,Ny,lMargin,rMargin,bMargin,tMargin) # TCanvas
   
   # Dummy histogram.
   global h
   h = ROOT.gROOT.FindObject("histo") # TH1F
   if (h) :
      ROOT.gROOT.Remove(h)
      del h
   h = TH1F("histo","",100,-5.0,5.0)
   h.FillRandom("gaus",10000)
   h.GetXaxis().SetTitle("x axis")
   h.GetYaxis().SetTitle("y axis")
   
   #From C: TPad *pad[Nx][Ny];
   #To Python:
   #TPad pad[Nx][Ny]
   global spads
   pads = [ [ TPad() for y in range(Ny) ] for x in range(Nx) ]
   
   global hFrame_list, text_list
   hFrame_list = []
   text_list = []
   #for (Int_t i = 0; i < Nx; i++) {
   #   for (Int_t j = 0; j < Ny; j++) {
   for i in range(0, Nx, 1):
      for j in range(0, Ny, 1):
         C.cd(0)
         
         # Get the padss previously created.
         pads[i][j] = C.FindObject( "pad_%d_%d"%(i,j) ) # TPad
         pads[i][j].Draw()
         pads[i][j].SetFillStyle(4000)
         pads[i][j].SetFrameFillStyle(4000)
         pads[i][j].cd()
         
         # Size factors
         xFactor = pads[0][0].GetAbsWNDC()/pads[i][j].GetAbsWNDC() # Float_t
         yFactor = pads[0][0].GetAbsHNDC()/pads[i][j].GetAbsHNDC() # Float_t
         
         #Debug: global hFrame_list
         hFrame = h.Clone( "h_%d_%d"%(i,j) ) # TH1F
         
         # y axis range
         hFrame.SetMinimum(0.0001); # do not show 0
         hFrame.SetMaximum(1.2*h.GetMaximum())
         
         # Format for y axis
         hFrame.GetYaxis().SetLabelFont(43)
         #hFrame.GetYaxis().SetLabelSize(13) # Two divisions.
         hFrame.GetYaxis().SetLabelSize(16) # Four division.
         hFrame.GetYaxis().SetLabelOffset(0.02)
         hFrame.GetYaxis().SetTitleFont(43)
         hFrame.GetYaxis().SetTitleSize(16)
         hFrame.GetYaxis().SetTitleOffset(2)
         
         hFrame.GetYaxis().CenterTitle()
         hFrame.GetYaxis().SetNdivisions(505)
         
         # TICKS Y Axis
         hFrame.GetYaxis().SetTickLength(xFactor*0.04/yFactor)
         
         # Format for x axis
         hFrame.GetXaxis().SetLabelFont(43)
         hFrame.GetXaxis().SetLabelSize(16)
         hFrame.GetXaxis().SetLabelOffset(0.02)
         hFrame.GetXaxis().SetTitleFont(43)
         hFrame.GetXaxis().SetTitleSize(16)
         hFrame.GetXaxis().SetTitleOffset(1)
         hFrame.GetXaxis().CenterTitle()
         hFrame.GetXaxis().SetNdivisions(505)
         
         # TICKS X Axis
         hFrame.GetXaxis().SetTickLength(yFactor*0.06/xFactor)
         
         # Draw cloned histogram with individual settings
         hFrame.Draw()
         hFrame_list.append( hFrame )
         
         #global text
         text = TText()
         text.SetTextAlign(31)
         text.SetTextFont(43)
         text.SetTextSize(10)

         #Debug: print( gPad.GetName() )
         text.DrawTextNDC(XtoPad(0.9), YtoPad(0.8), gPad.GetName())
         text_list.append( text )
         
      
   C.cd()
   



# TCanvas
def CanvasPartition(C : TCanvas, Nx : Int_t, Ny : Int_t,
                    lMargin : Float_t, rMargin : Float_t,
                    bMargin : Float_t, tMargin : Float_t) :
   if (not C): return
   
   # Setup Pad layout:
   vSpacing = 0.0
   vStep = (1.- bMargin - tMargin - (Ny-1) * vSpacing) / Ny
   
   hSpacing = 0.0
   hStep = (1.- lMargin - rMargin - (Nx-1) * hSpacing) / Nx
   
   vposd,vposu,vmard,vmaru,vfactor = [ Float_t() ] *5
   hposl,hposr,hmarl,hmarr,hfactor = [ Float_t() ] *5
   
   #for (Int_t i=0; i<Nx; i++) {
   for i in range(0, Nx, 1):
      
      if i==0:
         hposl = 0.0
         hposr = lMargin + hStep
         hfactor = hposr-hposl
         hmarl = lMargin / hfactor
         hmarr = 0.0
         
      elif (i == Nx-1):
         hposl = hposr + hSpacing
         hposr = hposl + hStep + rMargin
         hfactor = hposr-hposl
         hmarl = 0.0
         hmarr = rMargin / (hposr-hposl)
         
      else:
         hposl = hposr + hSpacing
         hposr = hposl + hStep
         hfactor = hposr-hposl
         hmarl = 0.0
         hmarr = 0.0
         
      
      #for (Int_t j=0; j<Ny; j++) {
      for j in range(0, Ny, 1):
         
         if j==0:
            vposd = 0.0
            vposu = bMargin + vStep
            vfactor = vposu-vposd
            vmard = bMargin / vfactor
            vmaru = 0.0
            
         elif (j == Ny-1):
            vposd = vposu + vSpacing
            vposu = vposd + vStep + tMargin
            vfactor = vposu-vposd
            vmard = 0.0
            vmaru = tMargin / (vposu-vposd)
            
         else:
            vposd = vposu + vSpacing
            vposu = vposd + vStep
            vfactor = vposu-vposd
            vmard = 0.0
            vmaru = 0.0
            
         
         C.cd(0)
         
         global name, pad_ij
         name = "pad_%d_%d"%(i,j)
         pad_ij = C.FindObject(name) # TPad
         if (pad_ij) : 
            exec("""ROOT.gROOT.Remove({name})""")
            del pad_ij
            pass


         pad_ij = TPad(name,"",hposl,vposd,hposr,vposu)
         #Debug: print( "pad_ij", pad_ij.GetName() ) 
         pad_ij.SetLeftMargin(hmarl)
         pad_ij.SetRightMargin(hmarr)
         pad_ij.SetBottomMargin(vmard)
         pad_ij.SetTopMargin(vmaru)
         
         pad_ij.SetFrameBorderMode(0)
         pad_ij.SetBorderMode(0)
         pad_ij.SetBorderSize(0)
         
         C.cd(0)
         pad_ij.Draw()
         C.Update()
         C.Draw()
         #Debug: ROOT.gROOT.SetBatchMode()
         #Debug: ROOT.gApplication.Run()
         
   return C # TCanvas   
   

# double
def XtoPad(x : Double_t ) :
 
   xl,yl,xu,yu = [ c_double() for _ in range(4) ]

   global gPad
   #Debug: gPad = ROOT.gPad
   gPad.GetPadPar(xl,yl,xu,yu)
   #Debug: print("all values, ", xl.value ,  yl.value ,  xu.value ,  yu.value  )
   #Debug: print(xu.value , xl.value , "values.value  x.value ")
   xl,  yl,  xu,  yu = xl.value, yl.value, xu.value, yu.value
   pw = xu-xl
   lm = gPad.GetLeftMargin()
   rm = gPad.GetRightMargin()
   fw = pw-pw*lm-pw*rm
   return (x*fw+pw*lm)/pw
   #Debug: return (x*fw+pw*lm)/(1+pw)
   

# double
def YtoPad(y : Double_t) :
   
   xl,yl,xu,yu = [ c_double() for _ in range(4) ]

   global gPad
   #Debug: gPad = ROOT.gPad 
   gPad.GetPadPar(xl,yl,xu,yu)
   xl,  yl,  xu,  yu = xl.value, yl.value, xu.value, yu.value
   ph = yu-yl
   #Debug: print(yu, yl, "values y")
   tm = gPad.GetTopMargin()
   bm = gPad.GetBottomMargin()
   fh = ph-ph*bm-ph*tm
   return (y*fh+bm*ph)/ph
   #Debug: return (y*fh+bm*ph)/(ph+1)
   


if __name__ == "__main__":
   canvas2()
