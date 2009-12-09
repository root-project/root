// @(#)root/gpad:$Id: TCreatePrimitives.cxx,v 1.0

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCreatePrimitives                                                    //
//                                                                      //
// Creates new primitives.                                              //
//                                                                      //
// The functions in this static class are called by TPad::ExecuteEvent  //
// to create new primitives in gPad from the TPad toolbar.              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TArrow.h"
#include "TPavesText.h"
#include "TPaveLabel.h"
#include "TCurlyArc.h"
#include "TArc.h"
#include "TLatex.h"
#include "TMarker.h"
#include "TDiamond.h"
#include "TGroupButton.h"
#include "TVirtualPad.h"
#include "TCreatePrimitives.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TMath.h"

//______________________________________________________________________________
TCreatePrimitives::TCreatePrimitives()
{
   // TCreatePrimitives default constructor
}


//______________________________________________________________________________
TCreatePrimitives::~TCreatePrimitives()
{
   // TCreatePrimitives destructor
}


//______________________________________________________________________________
void TCreatePrimitives::Ellipse(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   //  Create a new arc/ellipse in this gPad
   //
   //  Click left button to indicate arrow starting position.
   //  Release left button to terminate the arrow.
   //

   static Double_t x0, y0, x1, y1;

   static Int_t pxold, pyold;
   static Int_t px0, py0;
   static Int_t linedrawn;
   Double_t xc,yc,r1,r2;
   TEllipse *el = 0;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      px0   = px; py0   = py;
      pxold = px; pyold = py;
      linedrawn = 0;
      break;

   case kButton1Motion:
      if (linedrawn) gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      break;

   case kButton1Up:
      x1 = gPad->AbsPixeltoX(px);
      y1 = gPad->AbsPixeltoY(py);
      if (gPad->GetLogx()) {
         x0 = TMath::Power(10,x0);
         x1 = TMath::Power(10,x1);
      }
      if (gPad->GetLogy()) {
         y0 = TMath::Power(10,y0);
         y1 = TMath::Power(10,y1);
      }
      xc = 0.5*(x0+x1);
      yc = 0.5*(y0+y1);
      if (mode == kArc) {
         r1 = 0.5*TMath::Abs(x1-x0);
         el = new TArc(xc, yc, r1);
      }
      if (mode == kEllipse) {
         r1 = 0.5*TMath::Abs(x1-x0);
         r2 = 0.5*TMath::Abs(y1-y0);
         el = new TEllipse(xc, yc, r1, r2);
      }
      gPad->GetCanvas()->FeedbackMode(kFALSE);
      gPad->Modified(kTRUE);
      if (el) el->Draw();
      gPad->GetCanvas()->Selected((TPad*)gPad, el, event);
      gROOT->SetEditorMode();
      break;
   }
}


//______________________________________________________________________________
void TCreatePrimitives::Line(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   // Create a new line/arrow in this gPad
   //
   //  Click left button to indicate arrow starting position.
   //  Release left button to terminate the arrow.
   //

   static Double_t x0, y0, x1, y1;

   static Int_t pxold, pyold;
   static Int_t px0, py0;
   static Int_t linedrawn;
   TLine *line;
   TArrow *arrow;
   TCurlyLine *cline;
   Double_t radius, phimin,phimax;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      px0   = px; py0   = py;
      pxold = px; pyold = py;
      linedrawn = 0;
      break;

   case kButton1Motion:
      if (linedrawn) gVirtualX->DrawLine(px0, py0, pxold, pyold);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawLine(px0, py0, pxold, pyold);
      break;

   case kButton1Up:
      if (px == px0 && py == py0) break;
      x1 = gPad->AbsPixeltoX(px);
      y1 = gPad->AbsPixeltoY(py);
      gPad->Modified(kTRUE);
      if (gPad->GetLogx()) {
         x0 = TMath::Power(10,x0);
         x1 = TMath::Power(10,x1);
      }
      if (gPad->GetLogy()) {
         y0 = TMath::Power(10,y0);
         y1 = TMath::Power(10,y1);
      }
      if (mode == kLine) {
         line = new TLine(x0,y0,x1,y1);
         line->Draw();
         gPad->GetCanvas()->Selected((TPad*)gPad, line, event);
      }
      if (mode == kArrow) {
         arrow = new TArrow(x0,y0,x1,y1
                            , TArrow::GetDefaultArrowSize()
                            , TArrow::GetDefaultOption());
         arrow->Draw();
         gPad->GetCanvas()->Selected((TPad*)gPad, arrow, event);
      }
      if (mode == kCurlyLine) {
         cline = new TCurlyLine(x0,y0,x1,y1
                                , TCurlyLine::GetDefaultWaveLength()
                                , TCurlyLine::GetDefaultAmplitude());
         cline->Draw();
         gPad->GetCanvas()->Selected((TPad*)gPad, cline, event);
      }
      if (mode == kCurlyArc) {
         //calculate radius in pixels and convert to users x
         radius = gPad->PixeltoX((Int_t)(TMath::Sqrt((Double_t)((px-px0)*(px-px0) + (py-py0)*(py-py0)))))
                 - gPad->PixeltoX(0);
         phimin = 0;
         phimax = 360;
         cline = new TCurlyArc(x0,y0,radius,phimin,phimax
                                , TCurlyArc::GetDefaultWaveLength()
                                , TCurlyArc::GetDefaultAmplitude());
         cline->Draw();
         gPad->GetCanvas()->Selected((TPad*)gPad, cline, event);
      }
      gROOT->SetEditorMode();
      break;
   }
}


//______________________________________________________________________________
void TCreatePrimitives::Pad(Int_t event, Int_t px, Int_t py, Int_t)
{
   // Create a new pad in gPad
   //
   //  Click left button to indicate one corner of the pad
   //  Click left button to indicate the opposite corner
   //
   //  The new pad is inserted in the pad where the first point is selected.
   //

   static Int_t px1old, py1old, px2old, py2old;
   static Int_t px1, py1, px2, py2, pxl, pyl, pxt, pyt;
   static Bool_t boxdrawn;
   static TPad *padsav;
   Double_t xlow, ylow, xup, yup;
   TPad * newpad;

   Int_t  n = 0;
   TObject *obj;
   TIter next(gPad->GetListOfPrimitives());

   while ((obj = next())) {
      if (obj->InheritsFrom(TPad::Class())) {
         n++;
      }
   }

   switch (event) {

   case kButton1Down:
      padsav = (TPad*)gPad;
      gPad->cd();
      gVirtualX->SetLineColor(-1);
      px1 = gPad->XtoAbsPixel(gPad->GetX1());
      py1 = gPad->YtoAbsPixel(gPad->GetY1());
      px2 = gPad->XtoAbsPixel(gPad->GetX2());
      py2 = gPad->YtoAbsPixel(gPad->GetY2());
      px1old = px; py1old = py;
      boxdrawn = 0;
      break;

   case kButton1Motion:
      if (boxdrawn) gVirtualX->DrawBox(pxl, pyl, pxt, pyt, TVirtualX::kHollow);
      px2old = px;
      px2old = TMath::Max(px2old, px1);
      px2old = TMath::Min(px2old, px2);
      py2old = py;
      py2old = TMath::Max(py2old, py2);
      py2old = TMath::Min(py2old, py1);
      pxl = TMath::Min(px1old, px2old);
      pxt = TMath::Max(px1old, px2old);
      pyl = TMath::Max(py1old, py2old);
      pyt = TMath::Min(py1old, py2old);
      boxdrawn = 1;
      gVirtualX->DrawBox(pxl, pyl, pxt, pyt, TVirtualX::kHollow);
      break;

   case kButton1Up:
      gPad->Modified(kTRUE);
      gPad->SetDoubleBuffer(1);   // Turn on double buffer mode
      gVirtualX->SetDrawMode(TVirtualX::kCopy);       // set drawing mode back to normal (copy) mode
      xlow = (Double_t(pxl) - Double_t(px1))/(Double_t(px2) - Double_t(px1));
      ylow = (Double_t(py1) - Double_t(pyl))/(Double_t(py1) - Double_t(py2));
      xup  = (Double_t(pxt) - Double_t(px1))/(Double_t(px2) - Double_t(px1));
      yup  = (Double_t(py1) - Double_t(pyt))/(Double_t(py1) - Double_t(py2));
      gROOT->SetEditorMode();
      boxdrawn = 0;
      if (xup <= xlow || yup <= ylow) return;
      newpad = new TPad(Form("%s_%d",gPad->GetName(),n+1),"newpad",xlow, ylow, xup, yup);
      if (newpad->IsZombie()) break;
      newpad->SetFillColor(gStyle->GetPadColor());
      newpad->Draw();
      gPad->GetCanvas()->Selected((TPad*)gPad, newpad, event);
      padsav->cd();
      break;
   }
}


//______________________________________________________________________________
void TCreatePrimitives::Pave(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   // Create a new pavetext in gPad
   //
   //  Click left button to indicate one corner of the pavelabel.
   //  Release left button at the opposite corner.
   //

   static Double_t x0, y0, x1, y1;

   static Int_t pxold, pyold;
   static Int_t px0, py0;
   static Int_t linedrawn;
   const Int_t kTMAX=100;
   Int_t i,pxl,pyl;
   Double_t temp;
   Double_t xp0,xp1,yp0,yp1;
   static char atext[kTMAX];
   TObject *pave = 0;

   if (mode == kPaveLabel)
      ((TPad *)gPad)->EventPave();

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      px0   = px; py0   = py;
      pxold = px; pyold = py;
      linedrawn = 0;
      break;

   case kButton1Motion:
      if (linedrawn) gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawBox(px0, py0, pxold, pyold, TVirtualX::kHollow);
      break;

   case kButton1Up:
      if (px == px0) px = px0+10;
      if (py == py0) py = py0-10;
      x1 = gPad->AbsPixeltoX(px);
      y1 = gPad->AbsPixeltoY(py);

      if (x1 < x0) {temp = x0; x0 = x1; x1 = temp;}
      if (y1 < y0) {temp = y0; y0 = y1; y1 = temp;}
      xp0 = gPad->PadtoX(x0);
      xp1 = gPad->PadtoX(x1);
      yp0 = gPad->PadtoY(y0);
      yp1 = gPad->PadtoY(y1);
      if (mode == kPave)      pave = new TPave(xp0,yp0,xp1,yp1);
      if (mode == kPaveText ) pave = new TPaveText(xp0,yp0,xp1,yp1);
      if (mode == kPavesText) pave = new TPavesText(xp0,yp0,xp1,yp1);
      if (mode == kDiamond)   pave = new TDiamond(x0,y0,x1,y1);
      if (mode == kPaveLabel || mode == kButton) {
         ((TPad *)gPad)->StartEditing();
         gSystem->ProcessEvents();
         pxl = (px0 + px)/2;
         pyl = (py0 + py)/2;
         for (i=0;i<kTMAX;i++) atext[i] = ' ';
         atext[kTMAX-1] = 0;
         gVirtualX->RequestString(pxl, pyl, atext);
         for (i=kTMAX-2;i>=0;i--) {
            if ((i==0) || (atext[i] != ' ')) {
               atext[i+1] = 0;
               break;
            }
         }
         if (mode == kPaveLabel) { 
            pave = new TPaveLabel(xp0,yp0,xp1,yp1,atext);
            gSystem->ProcessEvents();
            ((TPad *)gPad)->RecordPave(pave);
         }
         if (mode == kButton) pave = new TButton(atext,"",
                              (x0-gPad->GetX1())/(gPad->GetX2() - gPad->GetX1()),
                              (y0-gPad->GetY1())/(gPad->GetY2() - gPad->GetY1()),
                              (x1-gPad->GetX1())/(gPad->GetX2() - gPad->GetX1()),
                              (y1-gPad->GetY1())/(gPad->GetY2() -
                              gPad->GetY1()));
      }
      gPad->GetCanvas()->FeedbackMode(kFALSE);
      gPad->Modified(kTRUE);
      if (pave) pave->Draw();
      gPad->GetCanvas()->Selected((TPad*)gPad, pave, event);
      gROOT->SetEditorMode();
      gPad->Update();
      break;
   }
}


//______________________________________________________________________________
void TCreatePrimitives::PolyLine(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   // Create a new PolyLine in gPad
   //
   //  Click left button to indicate a new point
   //  Click left button at same place or double click to close the polyline
   //

   static Int_t pxold, pyold, px1old, py1old;
   static Int_t npoints = 0;
   static Int_t linedrawn = 0;
   Int_t dp;
   static TGraph *gr = 0;

   switch (event) {

   case kButton1Double:
   case kButton1Down:
      if (npoints == 0) {
         gVirtualX->SetLineColor(-1);
      } else {
         gPad->GetCanvas()->FeedbackMode(kFALSE);
         gVirtualX->DrawLine(px1old, py1old, pxold, pyold);
      }
      // stop collecting new points if new point is close ( < 5 pixels) of previous point
      if (event == kButton1Double) {
         px = px1old;
         py = py1old;
      }
      dp = TMath::Abs(px-px1old) +TMath::Abs(py-py1old);
      if (npoints && dp < 5) {
         gPad->Modified(kTRUE);
         if (mode == kCutG && gr) {
            gr->Set(gr->GetN() + 1);
            Double_t x0, y0;
            gr->GetPoint(0, x0, y0);
            gr->SetPoint(npoints, x0, y0);
         }
         npoints = 0;
         linedrawn = 0;
         gr = 0;
         gROOT->SetEditorMode();
         break;
      }
      if (npoints == 1 && gr == 0) {
         if (mode == kPolyLine) {
            gr = new TGraph(2);
            gr->ResetBit(TGraph::kClipFrame);

         } else {
            gr = (TGraph*)gROOT->ProcessLineFast(
                 Form("new %s(\"CUTG\",%d",
                      gROOT->GetCutClassName(),2));
         }
         gr->SetPoint(0, gPad->PadtoX(gPad->AbsPixeltoX(px1old)),
                         gPad->PadtoY(gPad->AbsPixeltoY(py1old)));
         gr->SetPoint(1, gPad->PadtoX(gPad->AbsPixeltoX(px)),
                         gPad->PadtoY(gPad->AbsPixeltoY(py)));
         npoints = 2;
         gr->Draw("L");
         gPad->GetCanvas()->Selected((TPad*)gPad, gr, event);
      } else if (npoints > 1) {
         gr->Set(gr->GetN() + 1);
         gr->SetPoint(npoints, gPad->PadtoX(gPad->AbsPixeltoX(px)),
                         gPad->PadtoY(gPad->AbsPixeltoY(py)));
         npoints ++;
         gPad->Modified();
         gPad->Update();
      } else {
         npoints = 1;
      }
      px1old = px; py1old = py;
      pxold  = px; pyold  = py;
      linedrawn = 0;
      break;

   case kMouseMotion:
   case kButton1Motion:
   case kButton1Up:
      if (npoints < 1) return;
      gPad->GetCanvas()->FeedbackMode(kTRUE);
      if (linedrawn) {
         gVirtualX->DrawLine(px1old, py1old, pxold, pyold);
      }
      pxold = px;
      pyold = py;
      linedrawn = 1;
      gVirtualX->DrawLine(px1old, py1old, pxold, pyold);
      break;
   }
}


//______________________________________________________________________________
void TCreatePrimitives::Text(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   // Create a new TLatex at the cursor position in gPad
   //
   // Click left button to indicate the text position
   //

   const Int_t kTMAX=100;
   static char atext[kTMAX];
   Int_t i, lentext;
   TLatex *newtext;
   TMarker *marker;
   Double_t x, y;

   switch (event) {

   case kButton1Down:
      x = gPad->AbsPixeltoX(px);
      y = gPad->AbsPixeltoY(py);
      if (gPad->GetLogx()) x = TMath::Power(10,x);
      if (gPad->GetLogy()) y = TMath::Power(10,y);
      if (mode == kMarker) {
         marker = new TMarker(x,y,gStyle->GetMarkerStyle());
         marker->Draw();
         gPad->GetCanvas()->Selected((TPad*)gPad, marker, event);
         gROOT->SetEditorMode();
         break;
      }
      ((TPad *)gPad)->StartEditing();
      gSystem->ProcessEvents();
      for (i=0;i<kTMAX;i++) atext[i] = ' ';
      atext[kTMAX-1] = 0;
      lentext = kTMAX;
      newtext = new TLatex();
      gVirtualX->SetLineColor(-1);
      newtext->TAttText::Modify();
      gVirtualX->RequestString(px, py, atext);
      lentext = strlen(atext);
      for (i=lentext-1;i>=0;i--) {
         if (atext[i] != ' ') {
            atext[lentext] = 0;
            break;
         }
         lentext--;
      }
      if (!lentext) break;
      TLatex copytext(x, y, atext);
      gSystem->ProcessEvents();
      ((TPad *)gPad)->RecordLatex(&copytext);
      newtext->DrawLatex(x, y, atext);
      gPad->Modified(kTRUE);
      gPad->GetCanvas()->Selected((TPad*)gPad, newtext, event);
      gROOT->SetEditorMode();
      gPad->Update();
      break;
   }
}
