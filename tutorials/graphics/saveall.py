## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script creates 100 canvases and stores them in different images files 
## by using the TCanvas.SaveAll() method.
## It also demonstrates how different output formats can be used in batch mode.
##
## \macro_code
##
## \author Sergey Linev
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TString = ROOT.TString 
TH1I = ROOT.TH1I
TFile = ROOT.TFile
TVirtualPad = ROOT.TVirtualPad
TPad = ROOT.TPad

TPaveLabel = ROOT.TPaveLabel
TPavesText = ROOT.TPavesText
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow
TWbox = ROOT.TWbox
TPad = ROOT.TPad
TBox = ROOT.TBox
TPad = ROOT.TPad

#standard library
std = ROOT.std
vector = std.vector

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t

nullptr = ROOT.nullptr

c_double = ctypes.c_double

#utils
def to_c( ls ):
   return (c_double * len(ls) )( * ls )
def printf(string, *args):
   print( string % args )
def sprintf(buffer, string, *args):
   buffer = string % args 
   return buffer

#constants
kBlue = ROOT.kBlue
kRed = ROOT.kRed
kGreen = ROOT.kGreen

#globals
gStyle = ROOT.gStyle
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gBenchmark = ROOT.gBenchmark
gROOT = ROOT.gROOT

#def TCanvas_SaveAll( pads: vector["TPad"], name : str = None ):
def TCanvas_SaveAll( pads: list[TPad], name : str = None ):

   ## #if isinstance(pads, vector["TPad"] ): pass
   ## if all( isinstance( item, TPad ) for item in pads ): pass
   ## else : 
   ##    #raise TypeError(pads, "pads has to be a vector[\"TPads\"] object ")
   ##    raise TypeError(pads, "`pads` has to be a list[ TPads, ...]. ")
   ##    return

   # SaveAll on a pdf-file.
   if name == None or name.endswith(".pdf"):
      #pdf_file = TFile("SaveAll.pdf", "RECREATE")
      if len(pads) == 0 : return
      elif len(pads) == 1 : 
         print(pads[0])
         pads[0].SaveAs("SaveAll.pdf")
      elif len(pads) == 2 : 
         pads[0].SaveAs("SaveAll.pdf[")
         pads[0].SaveAs("SaveAll.pdf")
         pads[1].SaveAs("SaveAll.pdf")
         pads[1].SaveAs("SaveAll.pdf]")
      elif len(pads) > 2 : 
         pads[0].SaveAs("SaveAll.pdf[")
         #for p in pads[1:-1] :
         for p in pads:
            p.SaveAs("SaveAll.pdf")
         pads[-1].SaveAs("SaveAll.pdf]")

      print("All canvases saved on SaveAll.pdf")
            
   # SaveAll on a root-file.
   elif name.endswith(".root"):
      pdf_file = TFile(name, "RECREATE")
      for p in pads :
         p.Write(name)
      print(f"All canvases saved on {name}")
      pdf_file.Close()
            
   # SaveAll on many *.png( or *.svg) files.
   elif name.endswith(".png") or name.endswith(".svg") :
 
      if "%03d" in name[:-4]:
         n = 1
         for p in pads :
            p.SaveAs( name%n )
            n += 1
         print(f"All canvases saved with {name}")
      else :
         new_name = name[:-4] + "%03d" + name[-4:]
         n = 1
         for p in pads :
            p.SaveAs( new_name%n )
            n += 1
         print(f"All canvases saved with {new_name}")
         

   else :
      print("Nothing saved. Please enter a valid name")

#In case some version of ROOT didn't implemented the TCanvas::SaveAll method.
#Loading SaveAll method to TCanvas
TCanvas.SaveAll = TCanvas_SaveAll


# void
def saveall() :

   #Important:
   # Enforce batch mode to avoid appearance of multiple windows.
   ROOT.gROOT.SetBatch(True); 
   #Debug: ROOT.gROOT.SetBatch(False); 
   
   N = 100 # Number of Canvases
   #N = 3 # Number of Canvases
   #N = 2 # Number of Canvases
   #N = 1 # Number of Canvases

   ##Not to use:
   #global pads
   #pads = vector["TPad"](N)
   #pads = vector["TCanvas"](N)
   #pads = vector["TCanvas"](N)
   #pads = vector["TH1I"](N)
   #Note: 
   #      std.vector is still not completely developed, it has some issues. Maybe on a 
   #      future version > cppyy3.11
   
   global c, h1, h1_list, c_list, pad_list
   h1_list = []
   c_list = []
   pad_list = []
   #for(int n = 0; n < 100; ++n) {
   for n in range(0, N, 1): 
      
      c = TCanvas("canvas%d"%n, "Canvas with histogram")
      h1 = TH1I("hist%d"%n, "Histogram with random data #%d"%n, 100, -5., 5)
      h1.SetDirectory(0)
      h1.FillRandom("gaus", 10000)
      
      c.cd(0)
      h1.Draw("SAME")
      c.Update()

      h1_list.append( h1 )
      c_list.append( c )
      
      #Not to use:  
      #pads.push_back(c) 
      #pads[n] = tmp_pad 
      #pads.insert( pads.begin() + n , tmp_pad)
      #Note:
      #      RuntimeError. Call to delete constructor of TPad not very well implemened.
      #      c = TCanvas("c")
      #      pads = vector["TPad"]()
      #      pads.push_back(c)
      #Instead:
      global tmp_pad
      tmp_pad = TPad()
      gPad.Copy(tmp_pad)
      pad_list.append( tmp_pad )
      #But, anyway, it is not useful. The useful information is contained in the canvases.
      #Specifically, here, in c_list. So, that's what we are passing as the input in TCanvas.SaveAll.
      #If you decide to use the TCanvas.SaveAll from ROOT.TCanvas.SaveAll, again, you'll have
      #to pass a list of canvases instead of std::vector<TPad>() { ...}. 
      # Anyway, both methods function allright.


   #Using ROOT-SaveAll-function as defined in ROOT.
   #ROOT.TCanvas.SaveAll(c_list, "image%03d.png"); # create 100 PNG images
   #ROOT.TCanvas.SaveAll(c_list, "image.svg"); # create 100 SVG images, %d pattern will be automatically append
   #ROOT.TCanvas.SaveAll(c_list, "images.root"); # create single ROOT file with all canvases
   #ROOT.TCanvas.SaveAll(c_list, "allcanvases.pdf"); # save all existing canvases in allcanvases.pdf file
   
   #Using SaveAll-function defined here in this script. 
   TCanvas.SaveAll(c_list, "image%03d.png"); # create 100 PNG images
   
   TCanvas.SaveAll(c_list, "image.svg"); # create 100 SVG images, %d pattern will be automatically append
   
   TCanvas.SaveAll(c_list, "images.root"); # create single ROOT file with all canvases
   
   TCanvas.SaveAll(c_list, "allcanvases.pdf"); # save all existing canvases in allcanvases.pdf file
   


if __name__ == "__main__":
   saveall()
