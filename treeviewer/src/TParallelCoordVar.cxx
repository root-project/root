// @(#)root/treeviewer:$Name:  $:$Id: TParallelCoordVar.cxx,v 1.1 2007/07/24 20:00:46 brun Exp $
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TParallelCoordVar.h"
#include "TParallelCoord.h"
#include "TParallelCoordRange.h"

#include <Riostream.h>
#include <TROOT.h>
#include "TLatex.h"
#include "TLine.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TH1.h"
#include "TStyle.h"
#include "TBox.h"
#include "TCollection.h"
#include "TList.h"
#include "TFrame.h"
#include "TCanvas.h"
#include "TMarker.h"

ClassImp(TParallelCoordVar)


//______________________________________________________________________________
TParallelCoordVar::TParallelCoordVar()
   :TNamed(), TAttLine(), TAttFill(kRed,0)
{
   // Default constructor.

   Init();
}


//______________________________________________________________________________
TParallelCoordVar::~TParallelCoordVar()
{
   //Destructor.

   if (fHistogram) delete fHistogram;
   if (fRanges){
      TIter next(fRanges);
      while(TParallelCoordRange* range = (TParallelCoordRange*)next()) fParallel->CleanUpSelections(range);
      fRanges->Delete();
      delete fRanges;
   }
   if (fVal) delete [] fVal;
}


//______________________________________________________________________________
TParallelCoordVar::TParallelCoordVar(Double_t *val, const char* title, Int_t id, TParallelCoord* parallel)
   :TNamed("TParallelCoordVar",title), TAttLine(1,1,1), TAttFill(kRed,1001)
{
   // Normal constructor.

   Init();
   fId = id;
   fParallel      = parallel;
   fRanges        = new TList();
   fNentries       = fParallel->GetNentries();
   
   fVal = new Double_t[fParallel->GetNentries()];
   
   Double_t ave = 0;
   
   for(Long64_t ui = 0;ui<fParallel->GetNentries();++ui) fVal[ui]=val[ui];
   
   Double_t min,max;
   min = FLT_MAX;
   max = -FLT_MAX;
   for(Long64_t li=0; li<parallel->GetNentries();++li){
      if(val[li]<min) min = val[li];
      if(val[li]>max) max = val[li];
      ave+=val[li];
   }
   
   fMean = ave/((Double_t)fNentries);
   fMinInit    = min;
   fMinCurrent = fMinInit;
   fMaxInit    = max;
   fMaxCurrent = fMaxInit;
   GetHistogram();
   GetQuantiles();
}


//______________________________________________________________________________
void TParallelCoordVar::AddRange()
{
   // Add a range to the current selection on the axis.

   TParallelCoordSelect *select = fParallel->GetCurrentSelection();
   TParallelCoordRange *range = new TParallelCoordRange(this,select);
   range->Draw();
}


//______________________________________________________________________________
void TParallelCoordVar::DeleteVariable()
{
   // Delete variables.

   fParallel->RemoveVariable(this);
   delete this;
}


//______________________________________________________________________________
Int_t TParallelCoordVar::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Computes the distance from the axis.

   if(!gPad) return 9999;
   Double_t xx = gPad->AbsPixeltoX(px);
   Double_t yy = gPad->AbsPixeltoY(py);
   Double_t dist = 9999;
   
   if (fX1==fX2) {
      if (yy>fY1+0.01 && yy<fY2-0.01) dist = TMath::Abs(xx - fX1);
   } else {
      if (xx>fX1+0.01 && xx<fX2-0.01) dist = TMath::Abs(yy - fY1);
   }
   if (dist<=0.005) return 0;
   else             return 9999;
}


//______________________________________________________________________________
void TParallelCoordVar::Draw(Option_t *option)
{
   // Draw the axis.

   AppendPad(option);
}


//______________________________________________________________________________
Bool_t TParallelCoordVar::Eval(Long64_t evtidx, TParallelCoordSelect *select)
{
   // Check if the entry is within the range(s) of "select". Two ranges on a single
   // axis are conjugated as a "or": to be selected, the entry must be in one of
   // the ranges.

   if (fRanges->GetSize() > 0){
      TIter next(fRanges);
      Bool_t inarange = kFALSE;
      Bool_t noOwnedRange = kTRUE;
      while (TParallelCoordRange *range = (TParallelCoordRange*)next()){
         if(select->Contains(range)) {
            noOwnedRange = kFALSE;
            if(range->IsIn(fVal[evtidx])) inarange = kTRUE;
         }
      }
      if (noOwnedRange) return kTRUE;
      else return inarange;
   }
   else return kTRUE;
}


//______________________________________________________________________________
void TParallelCoordVar::ExecuteEvent(Int_t entry, Int_t px, Int_t py)
{
   // Execute the corresponding entry.

   if (!gPad) return;
   if (!gPad->IsEditable() && entry!=kMouseEnter) return;

   static Int_t pxold, pyold;
   static Int_t zoom; // -1:nothing zoomed or translated, 0:translating the axis, 1:zooming
   static Int_t pzoomold;
   static Bool_t first = kTRUE;

   Int_t px1,px2,py1,py2,n=-1;
   px1 = gPad->XtoAbsPixel(fX1);
   px2 = gPad->XtoAbsPixel(fX2);
   py1 = gPad->YtoAbsPixel(fY1);
   py2 = gPad->YtoAbsPixel(fY2);

   if(fX1 == fX2) {
      if(gPad->AbsPixeltoX(px)-fX1 > 0) gPad->SetCursor(kArrowVer);
      else                              gPad->SetCursor(kArrowHor);
   } else {
      if(gPad->AbsPixeltoY(py)-fY1 > 0) gPad->SetCursor(kArrowHor);
      else                              gPad->SetCursor(kArrowVer);
   }

   gVirtualX->SetLineColor(-1);
   switch (entry) {
      case kButton1Down:
         if (fX1==fX2){
            ((TCanvas*)gPad)->Selected(gPad,fParallel,1);
            if(gPad->AbsPixeltoX(px)-fX1 > 0){
               zoom = 1;
               gVirtualX->DrawLine(gPad->XtoAbsPixel(fX1-0.05),py,gPad->XtoAbsPixel(fX1+0.05),py);
               first = kTRUE;
               pzoomold = py;
            } else {
               zoom = 0;
               gVirtualX->DrawLine(px,py1,px,py2);
            }
         } else {
            if(gPad->AbsPixeltoY(py)-fY1 > 0){
               zoom = 1;
               gVirtualX->DrawLine(px,gPad->YtoAbsPixel(fY1-0.05),px,gPad->YtoAbsPixel(fY1+0.05));
               first=kTRUE;
               pzoomold = px;
            } else {
               zoom = 0;
               gVirtualX->DrawLine(px1,py,px2,py);
            }
         }
         pxold = px;
         pyold = py;
         break;
      case kButton1Up: {
         Double_t xx = gPad->AbsPixeltoX(px);
         Double_t yy = gPad->AbsPixeltoY(py);
         TFrame *frame = gPad->GetFrame();
         if (fX1==fX2) {
            if(zoom == 0){
               Double_t axisSpace = (frame->GetX2() - frame->GetX1())/(fParallel->GetNvar() - 1);
               Double_t pos = (xx - frame->GetX1())/axisSpace;
               if (pos < 0) n = -1;
               else         n = (Int_t)pos;
            } else {
               Double_t min = GetValuefromXY(xx,yy);
               Double_t max = GetValuefromXY(xx,gPad->AbsPixeltoY(pzoomold));
               if(TMath::Abs(min-max) < 0.00001) return;       // Avoid zooming if the axis is just clicked.
               if (fParallel->TestBit(TParallelCoord::kGlobalScale)) {
                  if (min>max) {
                     Double_t mem = min;
                     min = max; max = mem;
                  }
                  fParallel->SetGlobalMin(min);
                  fParallel->SetGlobalMax(max);
               } else {
                  SetCurrentLimits(min,max);
               }
            }
         } else {
            if(zoom == 0) {
               Double_t axisSpace = (frame->GetY2() - frame->GetY1())/(fParallel->GetNvar() - 1);
               Double_t pos = (yy-frame->GetY1())/axisSpace;
               if (pos < 0) n= -1;
               else         n = (Int_t)pos;
            } else {
               Double_t min = GetValuefromXY(xx,yy);
               Double_t max = GetValuefromXY(gPad->AbsPixeltoX(pzoomold),yy);
               SetCurrentLimits(min,max);
            }
         }
         if(zoom == 0){
            if (n>=0 && (UInt_t)n>=fParallel->GetNvar()) --n;
            else if (n<fParallel->GetVarList()->IndexOf(this)) ++n;
            fParallel->GetVarList()->Remove(this);
            fParallel->GetVarList()->AddAt(this,n);
         }
         gPad->Modified();
         break;
      }
      case kMouseMotion:
         pxold=px;
         pyold=py;
         break;
      case kButton1Motion:
         if(fX1==fX2){
            if(zoom==0){
               gPad->SetCursor(kArrowHor);
               gVirtualX->DrawLine(pxold,py1,pxold,py2);
               gVirtualX->DrawLine(px,py1,px,py2);
            } else if(zoom==1) {
               gPad->SetCursor(kArrowVer);
               if(!first) gVirtualX->DrawLine(gPad->XtoAbsPixel(fX1-0.05),pyold,gPad->XtoAbsPixel(fX1+0.05),pyold);
               gVirtualX->DrawLine(gPad->XtoAbsPixel(fX1-0.05),py,gPad->XtoAbsPixel(fX1+0.05),py);
               first = kFALSE;
            }
         } else {
            if(zoom==0){
               gPad->SetCursor(kArrowVer);
               gVirtualX->DrawLine(px1,pyold,px2,pyold);
               gVirtualX->DrawLine(px1,py,px2,py);
            } else if(zoom==1){
               gPad->SetCursor(kArrowHor);
               if(!first) gVirtualX->DrawLine(pxold,gPad->YtoAbsPixel(fY1-0.05),pxold,gPad->YtoAbsPixel(fY1+0.05));
               gVirtualX->DrawLine(px,gPad->YtoAbsPixel(fY1-0.05),px,gPad->YtoAbsPixel(fY1+0.05));
               first = kFALSE;
            }
         }
         pxold = px;
         pyold = py;
         break;
   }
}


//______________________________________________________________________________
void TParallelCoordVar::GetEntryXY(Long64_t n, Double_t & x, Double_t & y)
{
   // Get the position of the variable on the graph for the n'th entry.

   if(fX1==fX2){
      x = fX1;
      if (TestBit(kLogScale)) y = fY1 + (fY2 - fY1) *
                                  (TMath::Log10(fVal[n]/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
      else                    y = fY1 + (fY2 - fY1) *
                                  (fVal[n] - fMinCurrent) / (fMaxCurrent - fMinCurrent);
   } else {
      y = fY1;
      if (TestBit(kLogScale)) x = fX1 + (fX2 - fX1) *
                                  (TMath::Log10(fVal[n]/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
      else                    x = fX1 + (fX2 - fX1) *
                                  (fVal[n] - fMinCurrent) / (fMaxCurrent - fMinCurrent);
   }
}


//______________________________________________________________________________
Int_t TParallelCoordVar::GetEvtWeight(Long64_t evtidx)
{
   Int_t bin = 1 + (Int_t)((fVal[evtidx] - fMinCurrent)/((fMaxCurrent-fMinCurrent)/fNbins));
   return (Int_t)fHistogram->GetBinContent(bin);
}


//______________________________________________________________________________
TH1F* TParallelCoordVar::GetHistogram()
{
   // Create or recreate the histogram.

   if(fHistogram) delete fHistogram;
   fHistogram = NULL;
   fHistogram = new TH1F("hpa", "hpa", fNbins, fMinCurrent, fMaxCurrent);
   fHistogram->SetDirectory(0);
   for(Long64_t li=0; li<fParallel->GetNentries();++li) {
      if(fVal[li] > fMinCurrent && fVal[li] < fMaxCurrent) fHistogram->Fill(fVal[li]);
   }
   return fHistogram;
}


//______________________________________________________________________________
char* TParallelCoordVar::GetObjectInfo(Int_t px, Int_t py) const
{
   // Returns info about this axis.

   if (!gPad) return "";
   static char info[128];
   Double_t xx = gPad->AbsPixeltoX(px);
   Double_t yy = gPad->AbsPixeltoY(py);
   if (fX1 == fX2) {
      if (yy<fY1) {
         sprintf(info,"%s = %f", GetTitle(), fMinCurrent);
      } else if (yy>fY2) {
         sprintf(info,"%s = %f", GetTitle(), fMaxCurrent);
      } else {
         Double_t axislength = fY2-fY1;
         Double_t pos = (yy-fY1)/axislength;
         sprintf(info,"%s = %f", GetTitle(), fMinCurrent + pos*(fMaxCurrent-fMinCurrent));
      }
   } else {
      if (xx<fX1) {
         sprintf(info,"%s = %f", GetTitle(), fMinCurrent);
      } else if(xx>fX2) {
         sprintf(info,"%s = %f", GetTitle(), fMaxCurrent);
      } else {
         Double_t axislength = fX2-fX1;
         Double_t pos = (xx-fX1)/axislength;
         sprintf(info,"%s = %f", GetTitle(), pos*(fMaxCurrent-fMinCurrent));
      }
   }
   return info;
}


//______________________________________________________________________________
void TParallelCoordVar::GetQuantiles()
{
   // Get the box plot values (quantiles).

   Double_t *quantiles = new Double_t[3];
   Double_t *prob = new Double_t[3];
   prob[0]=0.25; prob[1]=0.5; prob[2] = 0.75;
   if (!TestBit(kLogScale)) TMath::Quantiles(fNentries,3,fVal,quantiles,prob,kFALSE);
   else {
      Double_t* logval = new Double_t[fNentries];
      Int_t selected = 0;
      if(fMinInit<=0) {
         for (Long64_t n=0;n<fNentries;++n) {
            if (fVal[n] >= fMinCurrent) {
               logval[selected] = TMath::Log10(fVal[n]);
               ++selected;
            }
         }
      } else {
         for (Long64_t n=0;n<fNentries;++n) {
            logval[n] = TMath::Log10(fVal[n]);
         }
         selected = fNentries;
      }
      TMath::Quantiles(selected,3,logval,quantiles,prob,kFALSE);
      delete [] logval;
   }
   fQua1 = quantiles[0];
   fMed = quantiles[1];
   fQua3 = quantiles[2];
   delete [] quantiles;
   delete [] prob;
}


//______________________________________________________________________________
Double_t TParallelCoordVar::GetValuefromXY(Double_t x,Double_t y)
{
   // Get the value corresponding to the posiiton.

   Double_t pos;
   if (fX1 == fX2) {
      if (y<=fY1)      pos = fMinCurrent;
      else if (y>=fY2) pos = fMaxCurrent;
      else             pos = fMinCurrent + ((y-fY1)/(fY2-fY1))*(fMaxCurrent-fMinCurrent);
   } else {
      if (x<=fX1)      pos = fMinCurrent;
      else if (x>=fX2) pos = fMaxCurrent;
      else             pos = fMinCurrent + ((x-fX1)/(fX2-fX1))*(fMaxCurrent-fMinCurrent);
   }
   return pos;
}


//______________________________________________________________________________
void TParallelCoordVar::GetXYfromValue(Double_t value, Double_t & x, Double_t & y)
{
   // Get a position corresponding to the value on the axis.

   if(value < fMinCurrent || value > fMaxCurrent) return;
   
   if (fX1==fX2) {
      x = fX1;
      if (TestBit(kLogScale)) y = fY1 + (fY2 - fY1) *
                                  (TMath::Log10(value/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
      else                    y = fY1 + (fY2 - fY1) *
                                  (value - fMinCurrent) / (fMaxCurrent - fMinCurrent);
   } else {
      y = fY1;
      if (TestBit(kLogScale)) x = fX1 + (fX2 - fX1) *
                                  (TMath::Log10(value/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
      else                    x = fX1 + (fX2 - fX1) *
                                  (value - fMinCurrent) / (fMaxCurrent - fMinCurrent);
   }
}


//______________________________________________________________________________
void TParallelCoordVar::Init()
{
   // Initialise the TParallelVar variables.

   fX1         = 0;
   fX2         = 0;
   fY1         = 0;
   fY2         = 0;
   fId         = 0;
   fMean= 0;
   fMinInit    = 0;
   fMinCurrent = 0;
   fMaxInit    = 0;
   fMaxCurrent = 0;
   fMed = 0;
   fQua1= 0;
   fQua3= 0;
   fNentries    = 0;
   fParallel   = NULL;
   fHistogram  = NULL;
   fNbins      = 100;
   fHistoLW    = 2;
   fHistoHeight     = 0;
   fRanges     = NULL;
   SetBit(kLogScale,kFALSE);
   SetBit(kShowBox,kFALSE);
   SetBit(kShowBarHisto,kFALSE);
}


//______________________________________________________________________________
void TParallelCoordVar::Paint(Option_t* /*option*/)
{
   // Paint the axis.

   PaintHistogram();
   if (TestBit(kShowBox)) PaintBoxPlot();
   PaintLabels();
}


//______________________________________________________________________________
void TParallelCoordVar::PaintBoxPlot()
{
   // Paint the boxes in the case of a cnadle chart.

   TLine *line = new TLine();
   line->SetLineColor(GetLineColor());
   line->SetLineWidth(1);
   TBox *box = new TBox();
   box->SetLineWidth(1);
   box->SetLineColor(GetLineColor());
   box->SetLineStyle(1);
   box->SetFillStyle(0);
   
   TFrame* frame = gPad->GetFrame();
   
   Double_t boxSize;
   if (fParallel->GetNvar() > 1) {
      if (fX1==fX2) boxSize = fHistoHeight*((frame->GetY2()-frame->GetY1())/(fParallel->GetNvar()-1));
      else          boxSize = fHistoHeight*((frame->GetX2()-frame->GetX1())/(fParallel->GetNvar()-1));
      if (boxSize >= 0.03) boxSize = 0.03;
   }
   else boxSize = 0.03;
   
   Double_t qua1,med,qua3,max,min;
   Double_t a,b,maxinit,mininit;
   if (TestBit(kLogScale)) {
      a = TMath::Log10(fMinCurrent);
      b = TMath::Log10(fMaxCurrent/fMinCurrent);
      if(fMinInit > 0) mininit = TMath::Log10(fMinInit);
      else             mininit = TMath::Log10(fMinCurrent);
      maxinit = TMath::Log10(fMaxInit);
   } else {
      a = fMinCurrent;
      b = fMaxCurrent-fMinCurrent;
      mininit = fMinInit;
      maxinit = fMaxInit;
   }
   if(fX1==fX2) {
      qua1 = fY1 + ((fQua1-a)/b)*(fY2-fY1);
      qua3 = fY1 + ((fQua3-a)/b)*(fY2-fY1);
      med  = fY1 + ((fMed-a)/b)*(fY2-fY1);
      max  = fY1 + ((maxinit-a)/b)*(fY2-fY1);
      min  = fY1 + ((mininit-a)/b)*(fY2-fY1);
   } else {
      qua1 = fX1 + ((fQua1-a)/b)*(fX2-fX1);
      qua3 = fX1 + ((fQua3-a)/b)*(fX2-fX1);
      med  = fX1 + ((fMed-a)/b)*(fX2-fX1);
      max  = fX1 + ((maxinit-a)/b)*(fX2-fX1);
      min  = fX1 + ((mininit-a)/b)*(fX2-fX1);
   }
   
   // min and max lines.
   if (fX1==fX2) {
      line->PaintLine(fX1-boxSize,min,fX1+boxSize,min);
      line->PaintLine(fX2-boxSize,max,fX2+boxSize,max);
   } else {
      line->PaintLine(min,fY1-boxSize,min,fY1+boxSize);
      line->PaintLine(max,fY2-boxSize,max,fY2+boxSize);
   }
   
   // lines from min and max to the box.
   line->SetLineStyle(7);
   if (fX1==fX2) {
      if (min<frame->GetY1()) min = frame->GetY1();
      if (max>frame->GetY2()) max = frame->GetY2();
      line->PaintLine(fX1,min,fX1,qua1);
      line->PaintLine(fX1,qua3,fX1,max);
   } else {
      if (min<frame->GetX1()) min = frame->GetX1();
      if (max>frame->GetX2()) max = frame->GetX2();
      line->PaintLine(min,fY1,qua1,fY2);
      line->PaintLine(qua3,fY1,max,fY2);
   }
   
   // Box
   if(fX1==fX2) box->PaintBox(fX1-boxSize,qua1,fX1+boxSize,qua3);
   else box->PaintBox(qua1,fY1-boxSize,qua3,fY1+boxSize);
   
   // Median line
   line->SetLineStyle(1);
   if(fX1==fX2) line->PaintLine(fX1-boxSize,med,fX1+boxSize,med);
   else line->PaintLine(med,fY1-boxSize,med,fY1+boxSize);
   
   // Paint average
   if (!TestBit(kLogScale) || (TestBit(kLogScale) && fMean > 0)) {
      Double_t mean;
      if (TestBit(kLogScale)) mean = TMath::Log10(fMean);
      else mean = fMean;
      TMarker *mark = NULL;
      if(fX1==fX2) mark = new TMarker(fX1,fY1 + ((mean-a)/b)*(fY2-fY1),24);
      else         mark = new TMarker(fX1 + ((mean-a)/b)*(fY2-fY1),fY1,24);
      mark->Paint();
      delete mark;
   }

   delete line;
   delete box;
}


//______________________________________________________________________________
void TParallelCoordVar::PaintHistogram()
{
   // Paint the histogram on the axis.

   Int_t i;
   
   TFrame *frame = gPad->GetFrame();
   
   // Paint the axis body.
   if (fHistoHeight!=0 && TestBit(kShowBarHisto)) {
      // Paint the axis body using bar chart.
      TBox *b = new TBox();
      b->SetFillStyle(GetFillStyle());
      b->SetFillColor(GetFillColor());
      b->SetLineStyle(GetLineStyle());
      b->SetLineColor(GetFillColor());
      b->SetLineWidth(GetLineWidth());
      Double_t hmin = fHistogram->GetMinimum();
      Double_t hmax = fHistogram->GetMaximum();
      if (fX1 == fX2) {
         // Vertical case.
         Double_t dy = (fY2-fY1)/fNbins;
         Double_t dv = (fMaxCurrent - fMinCurrent)/fNbins;
         Double_t v = fMinCurrent;
         Double_t y1 = fY1,x2,y2;
         for (i=1; i<=fNbins; i++) {
            x2 = fX1+((fHistogram->GetBinContent(i)-hmin)/(hmax-hmin))*fHistoHeight*((frame->GetX2()-frame->GetX1())/(fParallel->GetNvar()-1));
            if(TestBit(kLogScale)) y2 = fY1 + (fY2-fY1)*(TMath::Log10((v+dv)/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
            else y2=y1+dy;
            b->PaintBox(fX1,y1,x2,y2,"l");
            y1=y2;
            v += dv;
         }
      } else {
         // Horizontal case.
         Double_t dx = (fX2-fX1)/fNbins;
         Double_t dv = (fMaxCurrent - fMinCurrent)/fNbins;
         Double_t v = fMinCurrent;
         Double_t x1 = fX1,x2,y2;
         for (i=1; i<=fNbins; i++) {
            y2 = fY1+((fHistogram->GetBinContent(i)-hmin)/(hmax-hmin))*fHistoHeight*((frame->GetY2()-frame->GetY1())/(fParallel->GetNvar()-1));
            if(TestBit(kLogScale)) x2 = fX1 + (fX2-fX1)*(TMath::Log10((v+dv)/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
            else x2=x1+dx;
            b->PaintBox(x1,fY1,x2,y2,"l");
            x1=x2;
            v+=dv;
         }
      }
      delete b;
   }
   if (fHistoLW==0 && !TestBit(kShowBox)) {
      // Paint the axis body as a simple line.
      TLine* l = new TLine(fX1,fY1,fX2,fY2);
      l->SetLineWidth(GetLineWidth());
      l->SetLineColor(GetLineColor());
      l->SetLineStyle(GetLineColor());
      l->Paint();
      delete l;
   } else if (fHistoLW!=0){
      // Paint the axis body using the color palette.
      TLine *lb = new TLine();
      lb->SetLineWidth(fHistoLW);
      Double_t hmin = fHistogram->GetMinimum();
      Double_t hmax = fHistogram->GetMaximum();
      Int_t theColor;
      Int_t ncolors = gStyle->GetNumberOfColors();
      if (fX1 == fX2) {
         // Vertical case.
         Double_t dy = (fY2-fY1)/fNbins;
         Double_t y1 = fY1,y2;
         Double_t dv = (fMaxCurrent - fMinCurrent)/fNbins;
         Double_t v = fMinCurrent;
         for (i=1; i<=fNbins; i++) {
            theColor = (Int_t)( ((fHistogram->GetBinContent(i)-hmin)/(hmax-hmin))*(ncolors-1) );
            if(TestBit(kLogScale)) y2 = fY1 + (fY2-fY1)*(TMath::Log10((v+dv)/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
            else y2=y1+dy;
            lb->SetLineColor(gStyle->GetColorPalette(theColor));
            lb->PaintLine(fX1,y1,fX1,y2);
            y1=y2;
            v+=dv;
         }
      } else {
         // Horizontal case.
         Double_t dx = (fX2-fX1)/fNbins;
         Double_t dv = (fMaxCurrent - fMinCurrent)/fNbins;
         Double_t v = fMinCurrent;
         Double_t x1 = fX1,x2;
         for (i=1; i<=fNbins; i++) {
            theColor = (Int_t)( ((fHistogram->GetBinContent(i)-hmin)/(hmax-hmin))*(ncolors-1) );
            lb->SetLineColor(gStyle->GetColorPalette(theColor));
            if(TestBit(kLogScale)) x2 = fX1 + (fX2-fX1)*(TMath::Log10((v+dv)/fMinCurrent)) / (TMath::Log10(fMaxCurrent/fMinCurrent));
            else x2=x1+dx;
            lb->PaintLine(x1,fY1,x2,fY1);
            x1=x2;
            v+=dv;
         }
      }
      delete lb;
   }
}


//______________________________________________________________________________
void TParallelCoordVar::PaintLabels()
{
   // Paint the axis labels and titles.

   TLatex* t = new TLatex();
   TFrame *frame = gPad->GetFrame();
   t->SetTextSize(0.03);
   if (fX1==fX2) {
      t->SetText(fX1,frame->GetY1() - 0.04 - t->GetTextSize(),GetTitle());
      Double_t tlength = t->GetXsize();
      if (fX1-0.5*tlength<0.01) {
         t->SetTextAlign(11);
         t->SetText(0.01, frame->GetY1() - 0.04 - t->GetTextSize(), GetTitle());
         t->Paint();
      } else if (fX1+0.5*tlength > 0.99) {
         t->SetTextAlign(31);
         t->SetText(0.99,frame->GetY1() - 0.04 - t->GetTextSize(),GetTitle());
         t->Paint();
      } else {
         t->SetTextAlign(21);
         t->PaintLatex(fX1,frame->GetY1() - 0.04 - t->GetTextSize(),0,0.03,GetTitle());
      }
      if (!fParallel->TestBit(TParallelCoord::kCandleChart)) {
         t->SetTextAlign(21);
         t->PaintLatex(fX1,frame->GetY2() + 0.005,0,0.025,Form("%6.4f",fMaxCurrent));
         t->SetTextAlign(23);
         t->PaintLatex(fX1,frame->GetY1() - 0.005,0,0.025,Form("%6.4f",fMinCurrent));
      }
   } else {
      t->SetText(fX1-0.04,fY1+0.02,GetTitle());
      t->SetTextSize(0.03);
      Double_t tlength = t->GetXsize();
      if (fX1-0.04-tlength<0.01) {
         t->SetTextAlign(12);
         t->SetText(0.01,fY1+0.02,GetTitle());
         t->Paint();
      } else {
         t->SetTextAlign(32);
         t->PaintLatex(fX1-0.04,fY1+0.02,0,0.03,GetTitle());
      }
      if (!fParallel->TestBit(TParallelCoord::kCandleChart)) {
         t->SetTextAlign(12);
         t->PaintLatex(0.01,fY1-0.02,0,0.025,Form("%6.4f",fMinCurrent));
         t->SetTextAlign(32);
         t->PaintLatex(0.99,fY1-0.02,0,0.025,Form("%6.4f",fMaxCurrent));
      }
   }
   delete t;
}


//______________________________________________________________________________
void TParallelCoordVar::Print(Option_t* /*option*/) const
{
   // Print the axis main datas.

   printf("**************variable #%d**************\n",fParallel->GetVarList()->IndexOf(this));
   printf("at x1=%f, y1=%f, x2=%f, y2=%f.\n",fX1,fY1,fX2,fY2);
   printf("min = %f, Q1 = %f, Med = %f, Q3 = %f, Max = %f\n", fMinInit, fQua1, fMed, fQua3, fMaxInit);
}


//______________________________________________________________________________
void TParallelCoordVar::SetBoxPlot(Bool_t box)
{
   // Set the axis to display a candle.

   SetBit(kShowBox,box);
   if (box) SetHistogramHeight(0.5);
   else {
      SetHistogramHeight(0);
      SetHistogramLineWidth(2);
   }
}


//______________________________________________________________________________
void TParallelCoordVar::SetHistogramBinning(Int_t n)
{
   // Set the histogram binning.

   if (n < 0 || n == fNbins) return;
   fNbins = n;
   GetHistogram();
}


//______________________________________________________________________________
void TParallelCoordVar::SetHistogramHeight(Double_t h)
{
   // Set the height of the bar histogram.

   fHistoHeight = h;
   if (!fParallel->TestBit(TParallelCoord::kCandleChart)){
      if(h!=0) SetBit(kShowBarHisto,kTRUE);
      else SetBit(kShowBarHisto,kFALSE);
   }
}


//______________________________________________________________________________
void TParallelCoordVar::SetCurrentMin(Double_t min)
{
   // Set the current minimum of the axis.

   fMinCurrent = min;
}


//______________________________________________________________________________
void TParallelCoordVar::SetCurrentMax(Double_t max)
{
   // Set the current maximum of the axis.

   fMaxCurrent = max;
}


//______________________________________________________________________________
void TParallelCoordVar::SetCurrentLimits(Double_t min, Double_t max)
{
   // Set the limits within which one the entries must be painted.

   if (min>max) {
      Double_t mem = min;
      min = max;
      max = mem;
   }
   if(TestBit(kLogScale) && max<=0) return;
   if(TestBit(kLogScale) && min<=0) min = 0.00001*max;
   fMinCurrent = min;
   fMaxCurrent = max;
   
   delete fHistogram;
   fHistogram = NULL;
   GetHistogram();
   
   if (fParallel->TestBit(TParallelCoord::kGlobalScale)) {
      fParallel->SetGlobalMin(min);
      fParallel->SetGlobalMax(max);
   }
}


//______________________________________________________________________________
void TParallelCoordVar::SetLiveRangesUpdate(Bool_t on)
{
   // If true, the pad is updated while the motion of a dragged range.

   TIter next(fRanges);
   while (TParallelCoordRange* range = (TParallelCoordRange*)next()) range->SetBit(TParallelCoordRange::kLiveUpdate,on);
}


//______________________________________________________________________________
void TParallelCoordVar::SetLogScale(Bool_t log)
{
   // Set the axis in logscale.

   if (log == TestBit (kLogScale)) return;
   if (fMaxInit < 0)             SetBit(kLogScale,kFALSE);
   else if (log) {
      if (fMaxCurrent < 0 ) fMaxCurrent = fMaxInit;
      if (fMinCurrent < 0 ) fMinCurrent = 0.00001*fMaxCurrent;
      SetBit(kLogScale,kTRUE);
      SetCurrentMin(fMinCurrent);
      SetCurrentMax(fMaxCurrent);
   } else {
      SetBit(kLogScale,kFALSE);
      SetCurrentMin(fMinInit);
      SetCurrentMax(fMaxInit);
   }
   GetQuantiles();
   GetHistogram();
}


//______________________________________________________________________________
void TParallelCoordVar::SetX(Double_t x, Bool_t gl)
{
   // Set the X position of the axis in the case of a vertical axis.
   // and rotate the axis if it was horizontal.

   TFrame *frame = gPad->GetFrame();
   if (!gl) {
      fY1 = frame->GetY1();
      fY2 = frame->GetY2();
   } else {
      Double_t gmin = fParallel->GetGlobalMin();
      Double_t gmax = fParallel->GetGlobalMax();
      fY1 = frame->GetY1() + ((fMinCurrent-gmin)/(gmax-gmin))*(frame->GetY2()-frame->GetY1());
      fY2 = frame->GetY1() + ((fMaxCurrent-gmin)/(gmax-gmin))*(frame->GetY2()-frame->GetY1());
   }
   fX1 = fX2 = x;
}


//______________________________________________________________________________
void TParallelCoordVar::SetY(Double_t y, Bool_t gl)
{
   // Set the Y position of the axis in the case of a horizontal axis.
   // and rotate the axis if it was vertical.

   TFrame *frame = gPad->GetFrame();
   if (!gl) {
      fX1 = frame->GetX1();
      fX2 = frame->GetX2();
   } else {
      Double_t gmin = fParallel->GetGlobalMin();
      Double_t gmax = fParallel->GetGlobalMax();
      fX1 = frame->GetX1() + ((fMinCurrent-gmin)/(gmax-gmin))*(frame->GetX2()-frame->GetX1());
      fX2 = frame->GetX1() + ((fMaxCurrent-gmin)/(gmax-gmin))*(frame->GetX2()-frame->GetX1());
   }
   fY1 = fY2 = y;
}
