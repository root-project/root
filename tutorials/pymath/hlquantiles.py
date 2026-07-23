## \file
## \ingroup tutorial_math
## Demo for quantiles (with highlight mode)
##
## \macro_image
## \macro_code
##
## \authors Rene Brun, Eddy Offermann, Jan Musinsky
## \translator P. P.


import ROOT
import cppyy
ctypes = cppyy.ctypes


TVirtualPad = ROOT.TVirtualPad # abstract class
TObject = ROOT.TObject
TGraph = ROOT.TGraph
TH1F = ROOT.TH1F
TCanvas = ROOT.TCanvas
TString = ROOT.TString
TText = ROOT.TText
TLegend = ROOT.TLegend

#slot communication 
TPyDispatcher = ROOT.TPyDispatcher

#math
Math = ROOT.Math
#IFunction = Math.IFunction 

#types
Double_t = ROOT.Double_t
Float_t = ROOT.Float_t
Int_t = ROOT.Int_t
nullptr  = ROOT.nullptr
c_double = ctypes.c_double

#constants colors
kWhite = ROOT.kWhite
kRed = ROOT.kRed
kBlue = ROOT.kBlue
kMagenta = ROOT.kMagenta
#constants options
kCannotPick = ROOT.kCannotPick

#globals
gPad = ROOT.gPad

#utils
def to_c(ls):
   return (c_double * len(ls) )( * ls )

def to_py(c_ls):
   return list(c_ls) 

# variables
# Before:
#lq = nullptr # TList # TList is deprecated and will be removed in ROOT v6.34.
# Now:
global lq, gr
lq = [] # python-list
gr = nullptr # TGraph


def HighlightQuantile_test(pad : TVirtualPad = super(TVirtualPad),
                      obj : TObject     = TObject(),
                      ihp : Int_t       = Int_t(),
                      y   : Int_t       = Int_t() 
                          ) :
   print(10*">> "+"first")
   print( type(ihp), int(y))
   print( (ihp), (y))
   print(10*">> "+"first")
   global lq
   print(lq[ihp])
   lq[ihp].Draw("alp")
   return 

# void
def HighlightQuantile(pad : TVirtualPad = super(TVirtualPad),
                      obj : TObject     = TObject(),
                      ihp : Int_t       = Int_t(),
                      y   : Int_t       = Int_t() 
                          ) :
   global gr, lq, gPad
   # show the evolution of all quantiles in the bottom pad
   if (obj != gr) : return
   if (ihp == -1) : return
   
   global savepad
   savepad = gPad

   pad.GetCanvas().cd(3)
   #BP: lq.At(ihp).Draw("alp")
   #Not to use: lq.At(ihp).Draw("alp") # lq is not a TList.
   global lq
   lq[ihp].Draw("alp")
   gPad.Update()

   if (savepad): savepad.cd()
   


# void
def hlquantiles() :
   nq = 100
   nshots = 10
   # position where to compute the quantiles in [0,1]
   xq = [ Double_t() for _ in range(nq) ]  
   # array to contain the quantiles
   yq = [ Double_t() for _ in range(nq) ]  
 
   #for (Int_t i=0; i<nq; i++) 
   for i in range(0, nq, 1): xq[i] = Float_t(i+1)/nq
   xq = to_c(xq)
   
   global gr70, gr90, gr98
   gr70 = TGraph(nshots)
   gr90 = TGraph(nshots)
   gr98 = TGraph(nshots)
   
   global grq
   grq = [ TGraph() for _ in range(nq) ]

   #for (ig = 0; ig < nq; ig++)
   ig = 0
   while( ig < nq ):
      grq[ig] = TGraph(nshots)
      ig += 1

   global h
   h = TH1F("h","demo quantiles",50,-3,3)
    
   yq = to_c(yq) 
   #for (Int_t shot=0; shot<nshots; shot++) {
   for shot in range(0, nshots, 1): 
 
      h.FillRandom("gaus",50)
      h.GetQuantiles(nq,yq,xq)
      gr70.SetPoint(shot,shot+1,yq[70])
      gr90.SetPoint(shot,shot+1,yq[90])
      gr98.SetPoint(shot,shot+1,yq[98])

      #for (ig = 0; ig < nq; ig++)
      ig = 0
      while( ig < nq ):
         grq[ig].SetPoint(shot,shot+1,yq[ig])
         ig += 1
      
   
   #show the original histogram in the top pad
   global c1
   c1 = TCanvas("c1","demo quantiles",10,10,600,900)

   #DOING
   #BP: 
   #Note: 
   #  For HighlightConnect.
   #      TCanvas::HighlightConnect is a method wich only works in .C. 
   #      It is a short method for .Connect( signal = "", 0,0, slot="ourfunction(...) )
   #      In Python, we have to use necessarily Connect and TPyDispatcher as is shown below.

   #Not to use: c1.HighlightConnect("HighlightQuantile(TVirtualPad*,TObject*,Int_t,Int_t)")
   #
   signal = "Highlighted(TVirtualPad*,TObject*,Int_t,Int_t)"
   PyD_HighlightQuantile = TPyDispatcher( HighlightQuantile )
   #Note: 
   #  For Dispatcher.
   #     The dispatcher only has a definition for three arguments: TVirtualPad * ,TObject*, and Int_t
   #     Putting a fourth argument gives error. 
   #     For the purposes of highlighting the quantiles, we won't need a fourth argument; it is already 
   #     defined in our previous Python-definition:
   #        def HighlightQuantile(pad = TVirtualPad(), obj=TObject(), ihp, y =Int_t()) : ...

   c1.Connect(signal, "TPyDispatcher", PyD_HighlightQuantile, "Dispatch(TVirtualPad *, TObject*, Int_t )")
 
   c1.SetFillColor(41)
   c1.Divide(1,3)
   c1.cd(1)
   h.SetFillColor(38)
   h.Draw()
   
   # show the final quantiles in the middle pad
   c1.cd(2)
   gPad.SetFrameFillColor(33)
   gPad.SetGrid()
   
   global gr
   gr = TGraph(nq,xq,yq)
   gr.SetTitle("final quantiles")
   gr.SetMarkerStyle(21)
   gr.SetMarkerColor(kRed)
   gr.SetMarkerSize(0.3)
   gr.Draw("ap")
   
   # prepare quantiles
   # Before : 
   # lq = TList() # Deprecated 
   # Now: 
   global lq
   lq = [] 
   #for (Int_t ig = 0; ig < nq; ig++) {
   ig = 0
   while ( ig < nq ): 
      grq[ig].SetMinimum(gr.GetYaxis().GetXmin())
      grq[ig].SetMaximum(gr.GetYaxis().GetXmax())
      grq[ig].SetMarkerStyle(23)
      grq[ig].SetMarkerColor(ig%100)
      grq[ig].SetTitle( str( TString.Format("q%02d"%ig) ) )

      #lq.Add( grq[ig]) # TList  way.
      lq.append( grq[ig]) # Python way.
      
      ig += 1

      
   global info   
   info = TText(0.1, 2.4, "please move the mouse over the graph")
   info.SetTextSize(0.08)
   info.SetTextColor(gr.GetMarkerColor())
   info.SetBit(kCannotPick)
   info.Draw()
   
   gr.SetHighlight()
   
   # show the evolution of some  quantiles in the bottom pad
   c1.cd(3)
   gPad.SetFrameFillColor(17)
   gPad.DrawFrame(0,0,nshots+1,3.2)
   gPad.SetGrid()
   gr98.SetMarkerStyle(22)
   gr98.SetMarkerColor(kRed)
   gr98.Draw("lp")
   gr90.SetMarkerStyle(21)
   gr90.SetMarkerColor(kBlue)
   gr90.Draw("lp")
   gr70.SetMarkerStyle(20)
   gr70.SetMarkerColor(kMagenta)
   gr70.Draw("lp")
  
   # add a legend
   global legend
   legend = TLegend(0.85,0.74,0.95,0.95)
   legend.SetTextFont(72)
   legend.SetTextSize(0.05)
   legend.AddEntry(gr98," q98","lp")
   legend.AddEntry(gr90," q90","lp")
   legend.AddEntry(gr70," q70","lp")
   legend.Draw()
   



if __name__ == "__main__":
   hlquantiles()
