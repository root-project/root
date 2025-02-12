## \file
## \ingroup tutorial_graphics
## \notebook -js
##
## This script shows how ATLAS-Style looks like. 
## Its style is based on another style-file from BaBar-Experiment.
##
## \macro_image
## \macro_code
##
## \author  M.Sutton
## \translator P. P.


import ROOT
#import cppyy
#ctypes = cppyy.ctypes
import ctypes

#classes
TCanvas = ROOT.TCanvas
TGraphErrors = ROOT.TGraphErrors
TGraphAsymmErrors = ROOT.TGraphAsymmErrors
TLatex = ROOT.TLatex
TLegend = ROOT.TLegend
TFile = ROOT.TFile

#math functions
sqrt = ROOT.sqrt

#types
Int_t = ROOT.Int_t
Double_t = ROOT.Double_t
nullptr = ROOT.nullptr
c_double = ctypes.c_double

#globals
gRandom = ROOT.gRandom


#variables
GMAX = Int_t(864)

nren = 3
mur = [1.0,0.25,4.0]
muf = [1.0,0.25,4.0]
NUMPDF = 41

#utils
def sprintf(buffer, string, *args):
   buffer = string % args
   return buffer
def to_c( x ):
   return c_double( x )
def to_py( c_x ):
   return c_x.value 


#prototype functions
def GetGraph(ir : Int_t, ifs : Int_t,icut : Int_t, ipdf : Int_t): pass # TGraphErrors
def AddtoBand(g1 : TGraphErrors, g2 : TGraphAsymmErrors): pass # void
def MakeBand(g0 : TGraphErrors, g1 : TGraphErrors, g2 : TGraphErrors): pass # TGraphAsymmErrors

# void
def AtlasExample() :

   ROOT.gROOT.SetStyle("ATLAS")
   
   icol1=5# Int_t
   icol2=5# Int_t
   
   global canvas
   canvas = TCanvas("canvas","single inclusive jets",50,50,600,600)
   canvas.SetLogy()
   
   ymin = 1.e-3
   ymax = 2e7
   xmin = 60.00
   xmax = 3500.
   global frame
   frame = canvas.DrawFrame(xmin,ymin,xmax,ymax) # TFrame
   frame.SetYTitle("d#sigma_{jet}/dE_{T,jet} [fb/GeV]")
   frame.SetXTitle("E_{T,jet}  [GeV]")
   frame.GetYaxis().SetTitleOffset(1.4)
   frame.GetXaxis().SetTitleOffset(1.4)
   
   ncut = 1
   global data, g1
   data = [ TGraphErrors() for _ in range(ncut) ]
   g1   = [ [ TGraphErrors() for _ in range(ncut) ] for _ in range(nren) ]
   
   #for (Int_t icut=0; icut<ncut; icut++)  # loop over cuts
   # loop over cuts
   for icut in range(0, ncut, 1):
      #TGraphErrors g1[nren][ncut]
      #for (Int_t ir=0; ir<nren; ir++)  # loop over ren scale
      #loop over ren scale
      for ir in range(0, nren, 1): 
         g1[ir][icut] = GetGraph(ir,ir,icut,0)
         if not g1[ir][icut]:
            print(f" g1 not found ")
            return
            
         g1[ir][icut].SetLineColor(1)
         g1[ir][icut].SetMarkerStyle(0)
         
      
      daname = " "*100
      daname = sprintf(daname,"data_%d",icut)
      data[icut] = g1[0][icut].Clone(daname) # TGraphErrors
      data[icut].SetMarkerStyle(20)
      data[icut].SetMarkerColor(1)
      
      # Just invent some data
      #for (Int_t i=0; i< data[icut]->GetN(); i++) {
      for i in range(0, data[icut].GetN(), 1):
         x1 = y1 = e = dx1 = 0. # Double_t

         c_x1 = to_c(x1)
         c_y1 = to_c(y1)
         data[icut].GetPoint(i, c_x1, c_y1)
         x1 = to_py(c_x1)
         y1 = to_py(c_y1)

         r1 = 0.4*(gRandom.Rndm(1)+2)
         r2 = 0.4*(gRandom.Rndm(1)+2)
         y = Double_t()
         if (icut==0) : y = r1*y1 + r1*r2*r2*x1/50000.
         else : y = r1*y1
         e = sqrt(y*1000)/200
         data[icut].SetPoint(i, x1,y)
         data[icut].SetPointError(i,dx1,e)
         
      
      global scale, scalepdf
      scale = [ TGraphAsymmErrors() for _ in range(ncut) ]
      scalepdf = [ TGraphAsymmErrors() for _ in range(ncut) ]
      
      scale[icut] = MakeBand(g1[0][icut], g1[1][icut], g1[2][icut]) # TGraphAssymErrors
      scalepdf[icut] = scale[icut].Clone("scalepdf") # TGraphAsymmErrors
      
      
      global gpdf 
      gpdf = [ [ TGraphErrors() for _ in range(ncut) ] for _ in range(NUMPDF) ] 
      #Elements go in this order : gpdf[0:NUMPDF][0:ncut]

      #for (Int_t ipdf=0; ipdf<NUMPDF; ipdf++) {
      for ipdf in range(0, NUMPDF, 1):
         gpdf[ipdf][icut] = GetGraph(0,0,icut,ipdf)
         if not gpdf[ipdf][icut]:
            print(f" gpdf not  found ")
            return
            
         gpdf[ipdf][icut].SetLineColor(2)
         gpdf[ipdf][icut].SetLineStyle(1)
         gpdf[ipdf][icut].SetMarkerStyle(0)
         gpdf[ipdf][icut], scalepdf[icut] = AddtoBand(gpdf[ipdf][icut], scalepdf[icut]) # void() -> g1, g2
         
      
      scalepdf[icut].SetFillColor(icol2)
      scalepdf[icut].Draw("zE2")
      scalepdf[icut].SetLineWidth(3)
      scale[icut].SetFillColor(icol1)
      scale[icut].Draw("zE2")
      g1[0][icut].SetLineWidth(3)
      g1[0][icut].Draw("z")
      data[icut].Draw("P")
      
   
   global t
   t = TLatex()
   t.SetNDC()
   t.DrawLatex(0.3,  0.85, "#sqrt{s}= 14 TeV")
   t.DrawLatex(0.57, 0.85, "|#eta_{jet}|<0.5")
   
   global l
   l = TLegend(0.45,0.65,0.8,0.8,"","NDC")
   l.SetBorderSize(0)
   l.SetTextFont(42)
   l.AddEntry("data_0", "Data 2009", "ep")
   l.AddEntry("scalepdf", "NLO QCD", "lf")
   l.Draw()
    
   canvas.Draw()
   canvas.Update()
   

# TGraphErrors
def GetGraph(ir : Int_t, ifs : Int_t,icut : Int_t, ipdf : Int_t) :
   cuts = [ \
      "0.0 <= |eta| < 0.5",
      "0.5 <= |eta| < 1.0",
      "1.0 <= |eta| < 1.5",
      "1.5 <= |eta| < 2.0",
      "2.0 <= |eta| < 3.0"
   ]
      
   
   mur = [ 1.0,0.25,4.0 ]
   muf = [ 1.0,0.25,4.0 ]
   
   TFile.SetCacheFileDir(".")
   global file
   file = TFile.Open("http://root.cern.ch/files/AtlasGraphs.root", "CACHEREAD")
   
   gname = " "*100 # char
   tname = " "*100 # char
   
   if (ipdf>=0):
      tname = sprintf(tname," E_T (mu_r=%g, mu_f=%g);%s Pdf: %d",mur[ir],muf[ifs],cuts[icut],ipdf)
   else:
      tname = sprintf(tname," E_T %s Ms= %d",cuts[icut],-ipdf)
   
   g1 = 0
   
   #for (int i=1; i<=GMAX; i++) {
   for i in range(1, GMAX+1, 1):   
      gname = sprintf(gname,"full_%d",i)
      g1 = file.Get(gname) # TGraphError 
      if not g1:
         print(gname ," not found ")
         return nullptr

         
      
      title = g1.GetTitle()
      
      #if (strcmp(title,tname)==0) : break
      if (title == tname) : break
      g1 = 0
      
     
   if (not g1): return nullptr
   return g1
   

# TGraphAsymmErrors
def MakeBand(g0 : TGraphErrors, g1 : TGraphErrors, g2 : TGraphErrors) :
   
   global g3
   g3 = TGraphAsymmErrors()
   
   x1 = 0.; y1 = 0.; x2 = 0.; y2 = 0.; y0 = 0; x3 = 0.
   dum = Double_t()
   #for (Int_t i=0; i<g1->GetN(); i++) {
   for i in range(0, g1.GetN(), 1):
      c_x1 = to_c(x1)
      c_y0 = to_c(y0)
      c_y1 = to_c(y1)
      c_y2 = to_c(y2)
      g0.GetPoint(i, c_x1, c_y0)
      g1.GetPoint(i, c_x1, c_y1)
      g2.GetPoint(i, c_x1, c_y2)
      x1 = to_py(c_x1)
      y0 = to_py(c_y0)
      y1 = to_py(c_y1)
      y2 = to_py(c_y2)
      
      if (i==g1.GetN()-1) :x2=x1
      else:                
         c_x2 = to_c(x2)
         c_dum = to_c(dum)
         g2.GetPoint(i+1,c_x2,c_dum)
         x2 =  to_py(c_x2)
         dum = to_py(c_dum)
      
      if (i==0) :           x3=y2
      else:                 
         c_x3 = to_c(x3)
         c_dum = to_c(dum)
         g2.GetPoint(i-1,c_x3,c_dum)
         x3  = to_py(c_x3)
         dum = to_py(c_dum)
      
      tmp = y2
      if y1 < y2:
         y2 = y1
         y1 = tmp
         
      g3.SetPoint(i,x1,y0)
      
      binwl = (x1-x3)/2.
      binwh = (x2-x1)/2.
      if (binwl == 0.) : binwl = binwh
      if (binwh == 0.) : binwh = binwl
      g3.SetPointError(i, binwl, binwh, y0-y2, y1-y0)
      
      
   return g3
   

# void
# g1, g2
def AddtoBand(g1 : TGraphErrors, g2 : TGraphAsymmErrors) :
   
   x1=0.; y1=0.; y2=0.; y0=0 # Double_t
   
   if (g1.GetN() != g2.GetN()) :
      print(f" graphs don't have the same number of elements ")
   
   EYhigh = g2.GetEYhigh()
   EYlow = g2.GetEYlow()
   
   #for (Int_t i=0; i<g1->GetN(); i++) {
   for i in range(0, g1.GetN(), 1): 
      c_x1 = to_c(x1)
      c_y1 = to_c(y1)
      c_y2 = to_c(y2)
      g1.GetPoint(i, c_x1, c_y1)
      g2.GetPoint(i, c_x1, c_y2)
      x1 = to_py(c_x1)
      y1 = to_py(c_y1)
      y2 = to_py(c_y2)
      
      if ( y1==0 or y2==0 ):
         raise RuntimeError("check these points very carefully : AddtoBand() : point ", i)
      
      eyh=0.; eyl=0. # Double_t 
      
      y0 = y1-y2
      if (y0 != 0) :
         if y0 > 0:
            eyh = EYhigh[i]
            eyh = sqrt( eyh*eyh + y0*y0 )
            g2.SetPointEYhigh(i, eyh)
            
         else:
            eyl = EYlow[i]
            eyl = sqrt( eyl*eyl + y0*y0 )
            g2.SetPointEYlow(i, eyl)

   return g1, g2                
         
      
   


if __name__ == "__main__":
   AtlasExample()
