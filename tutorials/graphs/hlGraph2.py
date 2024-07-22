## \file
## \ingroup tutorial_graphs
##
## This tutorial demonstrates how to use the highlight mode --of TCanvas.Highlighted
## -- on a graph.
##
## \macro_code
##
## \date March 2018
## \author Jan Musinsky
## \translator P. P.


import ROOT

#classes
TVirtualPad = ROOT.TVirtualPad
TObject = ROOT.TObject

TNtuple = ROOT.TNtuple
TMath = ROOT.TMath
TH1F = ROOT.TH1F

TCanvas = ROOT.TCanvas
TString = ROOT.TString
TText = ROOT.TText
TBox = ROOT.TBox

TPyDispatcher = ROOT.TPyDispatcher

#types
Int_t = ROOT.Int_t
nullptr = ROOT.nullptr

#utils
def TString_Format(buffer, *args):
   buffer = buffer % args
   return buffer
TString.Format = TString_Format

#constants
kBlack = ROOT.kBlack
kRed = ROOT.kRed
#
kCannotPick = ROOT.kCannotPick
kCanDelete = ROOT.kCanDelete

#globals
gInterpreter = ROOT.gInterpreter
gPad = ROOT.gPad
gRandom = ROOT.gRandom
gROOT = ROOT.gROOT
gInterpreter = ROOT.gInterpreter

#system utils
ProcessLineFast = gROOT.ProcessLineFast
ProcessLine = gInterpreter.ProcessLine
BindObject = ROOT.BindObject
Remove = gROOT.Remove


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#variables
ntuple = TNtuple() # nullptr

# void
def HighlightBinId(pad : TVirtualPad, obj : TObject, ihp : Int_t, y : Int_t = 0):

   global Canvas2
   Canvas2 = gROOT.GetListOfCanvases().FindObject("Canvas2") # TCanvas

   if ( not Canvas2) : return

   global histo
   histo = Canvas2.FindObject("histo") # TH1F

   if ( not histo) : return
   
   global px, py, pz
   global i, p, hbin
   px = ntuple.GetV1()[ihp]
   py = ntuple.GetV2()[ihp]
   pz = ntuple.GetV3()[ihp]
   i = ntuple.GetV4()[ihp]
   p = TMath.Sqrt(px*px + py*py + pz*pz)
   hbin = histo.FindBin(p)
   
   global bh
   bh = Canvas2.FindObject("TBox") # TBox
   redraw = False

   if not bh:
      bh = TBox()
      bh.SetFillColor(kBlack)
      bh.SetFillStyle(3001)
      bh.SetBit(kCannotPick)
      bh.SetBit(kCanDelete)
      redraw = True
      
   bh.SetX1(histo.GetBinLowEdge(hbin))
   bh.SetY1(histo.GetMinimum())
   bh.SetX2(histo.GetBinWidth(hbin) + histo.GetBinLowEdge(hbin))
   bh.SetY2(histo.GetBinContent(hbin))
   
   global th
   th = Canvas2.FindObject("TText") # TText

   if not th:
      th = TText()
      th.SetName("TText")
      th.SetTextColor(bh.GetFillColor())
      th.SetBit(kCanDelete)
      redraw = True
      
   th.SetText(histo.GetXaxis().GetXmax()*0.75, histo.GetMaximum()*0.5,
      TString.Format("id = %d", i))
   
   if (ihp == -1):  # after highlight disabled
      #
      Remove( bh )
      #del bh
      #
      Remove( th )
      #del th
      

   Canvas2.Modified()
   Canvas2.Update()

   if ( not redraw) : return
   
   global savepad
   savepad = gPad

   Canvas2.cd()
   bh.Draw()
   th.Draw()
   Canvas2.Update()
   savepad.cd()
   

# void
def hlGraph2() :

   global Dir
   Dir = gROOT.GetTutorialDir()
   Dir.Append("/hsimple.C")
   Dir.ReplaceAll("/./","/")

   if ( not gInterpreter.IsLoaded(Dir.Data())) : 
      gInterpreter.LoadMacro(Dir.Data())
   

   global file
   file = ProcessLineFast("hsimple(1)") # int -> TFile
   file = BindObject(file, "TFile") # 

   if ( not file) : return
   
   #ntuple = ...
   file.GetObject["TNtuple"]("ntuple", ntuple) # proxy_call 

   if ( not ntuple) : return
   

   global Canvas1
   Canvas1 = TCanvas("Canvas1", "Canvas1", 0, 0, 500, 500)
   #Not to use:
   #Canvas1.HighlightConnect("HighlightBinId(TVirtualPad*,TObject*,Int_t,Int_t)") # Not implemented.
   #Instead:

   global PyD_HighlightConnect
   PyD_HighlightConnect = TPyDispatcher( HighlightBinId )
   Canvas1.Connect("Highlighted(TVirtualPad*,TObject*,Int_t,Int_t)", 
                   "TPyDispatcher", 
                   PyD_HighlightConnect, 
                   "Dispatch(TVirtualPad*,TObject*,Int_t)"
                   )
   
   cut = "pz > 3.0"
   ntuple.Draw("px:py", cut)
   graph = gPad.FindObject("Graph") # TGraph

   

   global info
   info = TText(0.0, 4.5, "please move the mouse over the graph")
   info.SetTextAlign(22)
   info.SetTextSize(0.03)
   info.SetTextColor(kRed+1)
   info.SetBit(kCannotPick)
   info.Draw()
   
   graph.SetHighlight()
   

   global Canvas2 # Same Canvas.
   Canvas2 = TCanvas("Canvas2", "Canvas2", 505, 0, 600, 400)
   ntuple.Draw("TMath::Sqrt(px*px + py*py + pz*pz)>>histo(100, 0, 15)", cut)
   
   # Must be last
   ntuple.Draw("px:py:pz:i", cut, "goff")
   


if __name__ == "__main__":
   hlGraph2()
