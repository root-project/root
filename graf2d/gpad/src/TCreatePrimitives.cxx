// @(#)root/gpad:$Id: TCreatePrimitives.cxx,v 1.0

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TCreatePrimitives
\ingroup gpad

Creates new primitives.

The functions in this static class are called by TPad::ExecuteEvent
to create new primitives in gPad from the TPad toolbar.
*/

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TMarker.h"
#include "TGroupButton.h"
#include "TVirtualPad.h"
#include "TCreatePrimitives.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TMath.h"
#include "KeySymbols.h"
#include "TCutG.h"

TPave *TCreatePrimitives::fgPave = nullptr;
TPaveText *TCreatePrimitives::fgPaveText = nullptr;
TPavesText *TCreatePrimitives::fgPavesText = nullptr;
TDiamond *TCreatePrimitives::fgDiamond = nullptr;
TPaveLabel *TCreatePrimitives::fgPaveLabel = nullptr;
TGraph *TCreatePrimitives::fgPolyLine = nullptr;
TBox *TCreatePrimitives::fgPadBBox = nullptr;

////////////////////////////////////////////////////////////////////////////////
/// TCreatePrimitives default constructor.

TCreatePrimitives::TCreatePrimitives()
{
}

////////////////////////////////////////////////////////////////////////////////
/// TCreatePrimitives destructor.

TCreatePrimitives::~TCreatePrimitives()
{
}

////////////////////////////////////////////////////////////////////////////////
/// @deprecated
/// Create a new arc/ellipse in this gPad.

void TCreatePrimitives::Ellipse(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   if (auto pad = dynamic_cast<TPad *>(gPad))
      ExecuteLine(*pad, mode, event, px, py);
}

/////////////////////////////////////////////
/// Internal class implementing tlatex editing
class TLineEditInteractive : public TVirtualPad::TInteractive {
   public:

   TLineEditInteractive(Int_t mode, Double_t x0, Double_t y0, Int_t px0, Int_t py0)
   {
      fMode = mode;
      fX0 = x0;
      fY0 = y0;
      fPX0 = px0;
      fPY0 = py0;
   }

   template<typename T, typename... Rest>
   T *Make(TPad &parent, Rest&&... rest)
   {
      if (fObject) {
         auto res = dynamic_cast<T *>(fObject);
         if (!res)
            fObject = nullptr; // failure, interactive will be deleted
         return res;
      }
      auto res2 = new T(std::forward<Rest>(rest)...);
      parent.Add(res2);
      fObject = res2;
      return res2;
   }


   Int_t fMode = 0;
   Double_t fX0 = 0, fY0 = 0;
   Int_t fPX0 = 0, fPY0 = 0;
   TObject *fObject = nullptr;
};


////////////////////////////////////////////////////////////////////////////////
/// Create a new line/arrow in the specified pad
///
///  - Click left button to indicate arrow starting position.
///  - Move mouse pointer when left button clicked to specify second position
///  - Release left button to terminate the arrow.

void TCreatePrimitives::ExecuteLine(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py)
{
   Double_t x = parent.PadtoX(parent.AbsPixeltoX(px));
   Double_t y = parent.PadtoY(parent.AbsPixeltoY(py));

   // first mouse click
   if (event == kButton1Down) {
      parent.MakeInteractive<TLineEditInteractive>(&parent, mode, x, y, px, py);
      return;
   }

   auto inter = parent.GetInteractive<TLineEditInteractive>(&parent);
   if (!inter)
      return;

   // verify that object is still there or mode is changed
   if ((inter->fObject && !parent.FindObject(inter->fObject)) || (mode != inter->fMode)) {
      parent.FreeInteractive(&parent);
      return;
   }

   // when mouse released - finish interactive
   if (event == kButton1Up) {
      if (inter->fObject)
         parent.GetCanvas()->Selected(&parent, inter->fObject, kButton1Down);
      parent.FreeInteractive(&parent);
      gROOT->SetEditorMode();
      return;
   }

   // ignore all other events
   if (event != kButton1Motion)
      return;

   switch (mode) {
   case kLine:
      if (auto line = inter->Make<TLine>(parent, inter->fX0, inter->fY0, x, y)) {
         line->SetX2(x);
         line->SetY2(y);
      }
      break;
   case kArrow:
      if (auto arrow = inter->Make<TArrow>(parent, inter->fX0, inter->fY0, x, y, TArrow::GetDefaultArrowSize(),
                                           TArrow::GetDefaultOption())) {
         arrow->SetX2(x);
         arrow->SetY2(y);
      }
      break;
   case kCurlyLine:
      if (auto cline = inter->Make<TCurlyLine>(parent, inter->fX0, inter->fY0, x, y, TCurlyLine::GetDefaultWaveLength(),
                                               TCurlyLine::GetDefaultAmplitude()))
         cline->SetEndPoint(x, y);
      break;
   case kCurlyArc:
      if (auto carc = inter->Make<TCurlyArc>(parent, inter->fX0, inter->fY0, 1., 0., 360.,
                                             TCurlyArc::GetDefaultWaveLength(), TCurlyArc::GetDefaultAmplitude())) {
         auto pradius = TMath::Sqrt((px - inter->fPX0) * (px - inter->fPX0) + (py - inter->fPY0) * (py - inter->fPY0));
         auto radius = parent.PixeltoX((Int_t)pradius) - parent.PixeltoX(0);
         carc->SetRadius(radius);
      }
      break;
   case kArc:
      if (auto arc = inter->Make<TArc>(parent, 1., 1., 1, 45, 315)) {
         arc->SetR1(0.5 * TMath::Abs(x - inter->fX0));
         arc->SetR2(0.5 * TMath::Abs(y - inter->fY0));
         arc->SetX1(0.5*(x + inter->fX0));
         arc->SetY1(0.5*(y + inter->fY0));
      }
      break;
   case kEllipse:
      if (auto ellipse = inter->Make<TEllipse>(parent, 1., 1., 1, 1)) {
         ellipse->SetR1(0.5 * TMath::Abs(x - inter->fX0));
         ellipse->SetR2(0.5 * TMath::Abs(y - inter->fY0));
         ellipse->SetX1(0.5*(x + inter->fX0));
         ellipse->SetY1(0.5*(y + inter->fY0));
      }
      break;

   default: break;
   }

   if (inter->fObject) {
      parent.Modified(kTRUE);
      parent.Update();
   } else
      parent.FreeInteractive(&parent);
}

////////////////////////////////////////////////////////////////////////////////
/// @deprecated
/// Create a new line/arrow in this gPad.

void TCreatePrimitives::Line(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   if (auto pad = dynamic_cast<TPad *>(gPad))
      ExecuteLine(*pad, mode, event, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Create a new pad in gPad.
///
///  - Click left button to indicate one corner of the pad.
///  - Click left button to indicate the opposite corner.
///
/// The new pad is inserted in the pad where the first point is selected.

void TCreatePrimitives::Pad(Int_t event, Int_t px, Int_t py, Int_t)
{
   static Int_t px1old, py1old, px2old, py2old;
   static Int_t px1, py1, px2, py2, pxl, pyl, pxt, pyt;
   static TVirtualPad *padsav = nullptr;
   Double_t xlow, ylow, xup, yup;

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
      padsav = gPad;
      gPad->cd();
      px1 = gPad->XtoAbsPixel(gPad->GetX1());
      py1 = gPad->YtoAbsPixel(gPad->GetY1());
      px2 = gPad->XtoAbsPixel(gPad->GetX2());
      py2 = gPad->YtoAbsPixel(gPad->GetY2());
      px1old = px; py1old = py;
      break;

   case kButton1Motion:
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

      if (fgPadBBox) {
         fgPadBBox->SetX1(gPad->AbsPixeltoX(pxl));
         fgPadBBox->SetY1(gPad->AbsPixeltoY(pyl));
         fgPadBBox->SetX2(gPad->AbsPixeltoX(pxt));
         fgPadBBox->SetY2(gPad->AbsPixeltoY(pyt));
      } else {
         fgPadBBox = new TBox(pxl, pyl, pxt, pyt);
         fgPadBBox->Draw("l");
      }
      gPad->Modified(kTRUE);
      gPad->Update();
      break;

   case kButton1Up:
      fgPadBBox->Delete();
      fgPadBBox = nullptr;
      xlow = (Double_t(pxl) - Double_t(px1))/(Double_t(px2) - Double_t(px1));
      ylow = (Double_t(py1) - Double_t(pyl))/(Double_t(py1) - Double_t(py2));
      xup  = (Double_t(pxt) - Double_t(px1))/(Double_t(px2) - Double_t(px1));
      yup  = (Double_t(py1) - Double_t(pyt))/(Double_t(py1) - Double_t(py2));

      gROOT->SetEditorMode();
      if (xup <= xlow || yup <= ylow) return;
      auto newpad = new TPad(TString::Format("%s_%d",gPad->GetName(),n+1).Data(),"newpad",xlow, ylow, xup, yup);
      if (newpad->IsZombie()) break;
      newpad->SetFillColor(gStyle->GetPadColor());
      newpad->Draw();
      auto canvas = gPad->GetCanvas();
      if (canvas) canvas->Selected((TPad*)gPad, newpad, kButton1Down);
      if (padsav) {
         padsav->cd();
         padsav = nullptr;
      }
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new pavetext in gPad.
///
///  - Click left button to indicate one corner of the pavelabel.
///  - Release left button at the opposite corner.

void TCreatePrimitives::Pave(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   static Double_t x0, y0;
   Double_t xp0,xp1,yp0,yp1,xold,yold;

   if (mode == kPaveLabel)
      ((TPad *)gPad)->EventPave();

   switch (event) {

   case kKeyPress:
      if (mode == kPaveLabel) {
         if ((py == kKey_Return) || (py == kKey_Enter)) {
            TString s(fgPaveLabel->GetTitle());
            Int_t l = s.Length();
            s.Remove(l-1);
            fgPaveLabel->SetLabel(s.Data());
            gSystem->ProcessEvents();
            gPad->Modified(kTRUE);
            gROOT->SetEditorMode();
            gPad->Update();
            fgPaveLabel = nullptr;
         } else if (py == kKey_Backspace) {
            TString s(fgPaveLabel->GetTitle());
            Int_t l = s.Length();
            if (l>1) {
               s.Replace(l-2, 2, "<");
               fgPaveLabel->SetLabel(s.Data());
               gPad->Modified(kTRUE);
               gPad->Update();
            }
         } else if (isprint(py)) {
            TString s(fgPaveLabel->GetTitle());
            Int_t l = s.Length();
            s.Insert(l-1,(char)py);
            fgPaveLabel->SetLabel(s.Data());
            gPad->Modified(kTRUE);
            gPad->Update();
         }
      }
      break;

   case kButton1Down:
      x0 = gPad->AbsPixeltoX(px);
      y0 = gPad->AbsPixeltoY(py);
      break;

   case kButton1Motion:
      xold = gPad->AbsPixeltoX(px);
      yold = gPad->AbsPixeltoY(py);

      xp0 = gPad->PadtoX(x0);
      xp1 = gPad->PadtoX(xold);
      yp0 = gPad->PadtoY(y0);
      yp1 = gPad->PadtoY(yold);

      if (mode == kPave) {
         if (fgPave) {
            if (xold < x0) {
               fgPave->SetX1(xold);
               fgPave->SetX2(x0);
            } else {
               fgPave->SetX1(x0);
               fgPave->SetX2(xold);
            }
            if (yold < y0) {
               fgPave->SetY1(yold);
               fgPave->SetY2(y0);
            } else {
               fgPave->SetY1(y0);
               fgPave->SetY2(yold);
            }
         } else {
            fgPave = new TPave(xp0,yp0,xp1,yp1);
            fgPave->Draw();
         }
         gPad->Modified(kTRUE);
         gPad->Update();
      }

      if (mode == kPaveText ) {
         if (fgPaveText){
            if (xold < x0) {
               fgPaveText->SetX1(xold);
               fgPaveText->SetX2(x0);
            } else {
               fgPaveText->SetX1(x0);
               fgPaveText->SetX2(xold);
            }
            if (yold < y0) {
               fgPaveText->SetY1(yold);
               fgPaveText->SetY2(y0);
            } else {
               fgPaveText->SetY1(y0);
               fgPaveText->SetY2(yold);
            }
         } else {
            fgPaveText = new TPaveText(xp0,yp0,xp1,yp1);
            fgPaveText->Draw();
         }
         gPad->Modified(kTRUE);
         gPad->Update();
      }

      if (mode == kPavesText) {
         if (fgPavesText){
            if (xold < x0) {
               fgPavesText->SetX1(xold);
               fgPavesText->SetX2(x0);
            } else {
               fgPavesText->SetX1(x0);
               fgPavesText->SetX2(xold);
            }
            if (yold < y0) {
               fgPavesText->SetY1(yold);
               fgPavesText->SetY2(y0);
            } else {
               fgPavesText->SetY1(y0);
               fgPavesText->SetY2(yold);
            }
         } else {
            fgPavesText = new TPavesText(xp0,yp0,xp1,yp1);
            fgPavesText->Draw();
         }
         gPad->Modified(kTRUE);
         gPad->Update();
      }

      if (mode == kPaveLabel) {
         if (fgPaveLabel){
            if (xold < x0) {
               fgPaveLabel->SetX1(xold);
               fgPaveLabel->SetX2(x0);
            } else {
               fgPaveLabel->SetX1(x0);
               fgPaveLabel->SetX2(xold);
            }
            if (yold < y0) {
               fgPaveLabel->SetY1(yold);
               fgPaveLabel->SetY2(y0);
            } else {
               fgPaveLabel->SetY1(y0);
               fgPaveLabel->SetY2(yold);
            }
         } else {
            fgPaveLabel = new TPaveLabel(xp0,yp0,xp1,yp1,">");
            fgPaveLabel->Draw();
         }
         gPad->Modified(kTRUE);
         gPad->Update();
      }

      if (mode == kDiamond) {
         if (fgDiamond){
            if (xold < x0) {
               fgDiamond->SetX1(xold);
               fgDiamond->SetX2(x0);
            } else {
               fgDiamond->SetX1(x0);
               fgDiamond->SetX2(xold);
            }
            if (yold < y0) {
               fgDiamond->SetY1(yold);
               fgDiamond->SetY2(y0);
            } else {
               fgDiamond->SetY1(y0);
               fgDiamond->SetY2(yold);
            }
         } else {
            fgDiamond = new TDiamond(x0,y0,xold,yold);
            fgDiamond->Draw();
         }
         gPad->Modified(kTRUE);
         gPad->Update();
      }
      break;

   case kButton1Up:
      gPad->GetCanvas()->Selected(gPad, fgPave, kButton1Down);
      if (mode == kPave) {
         gPad->GetCanvas()->Selected(gPad, fgPave, kButton1Down);
         fgPave = nullptr;
      }
      if (mode == kPaveText ) {
         gPad->GetCanvas()->Selected(gPad, fgPaveText, kButton1Down);
         fgPaveText = nullptr;
      }
      if (mode == kPavesText) {
         gPad->GetCanvas()->Selected(gPad, fgPavesText, kButton1Down);
         fgPavesText = nullptr;
      }
      if (mode == kDiamond)   {
         gPad->GetCanvas()->Selected(gPad, fgDiamond, kButton1Down);
         fgDiamond = nullptr;
      }
      if (mode == kPaveLabel) {
         gPad->GetCanvas()->Selected(gPad, fgPaveLabel, kButton1Down);
         ((TPad *)gPad)->StartEditing();
         gSystem->ProcessEvents();
         if (mode == kPaveLabel) {
            gPad->Modified(kTRUE);
            gPad->Update();
            break;
         }
      }
      gROOT->SetEditorMode();
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new PolyLine in gPad.
///
///  - Click left button to indicate a new point.
///  - Click left button at same place or double click to close the polyline.

void TCreatePrimitives::PolyLine(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   static Int_t pxnew, pynew, pxold, pyold, dp;
   Double_t xnew, ynew, xold, yold;
   static Int_t npoints = 0;

   switch (event) {

   case kButton1Down:
      pxnew = px;
      pynew = py;
      npoints++;
      if (fgPolyLine) {
         fgPolyLine->Set(fgPolyLine->GetN() +1);
         fgPolyLine->SetPoint(npoints, gPad->PadtoX(gPad->AbsPixeltoX(pxnew)),
                                       gPad->PadtoY(gPad->AbsPixeltoY(pynew)));
         // stop collecting new points if new point is close to previous point
         if (npoints > 1) {
            xnew = gPad->PadtoX(gPad->AbsPixeltoX(pxnew));
            ynew = gPad->PadtoY(gPad->AbsPixeltoY(pynew));
            fgPolyLine->GetPoint(fgPolyLine->GetN()-3, xold, yold);
            pxold = gPad->XtoAbsPixel(xold);
            pyold = gPad->YtoAbsPixel(yold);
            dp   = TMath::Abs(pxnew-pxold) +TMath::Abs(pynew-pyold);
            if (dp < 7) {
               if (mode == kPolyLine) {
                  fgPolyLine->Set(npoints-1);
               } else {
                  fgPolyLine->GetPoint(0, xnew, ynew);
                  fgPolyLine->SetPoint(npoints, xnew, ynew);
               }
               gPad->GetCanvas()->Selected(gPad, fgPolyLine, kButton1Down);
               fgPolyLine = nullptr;
               npoints    = 0;
               gPad->Modified();
               gPad->Update();
               gROOT->SetEditorMode();
            }
         }
      } else {
         if (mode == kPolyLine) {
            fgPolyLine = new TGraph(1);
            fgPolyLine->ResetBit(TGraph::kClipFrame);
         } else { // TCutG case
            fgPolyLine = (TGraph*) new TCutG("CUTG",1);
         }
         fgPolyLine->SetPoint(0, gPad->PadtoX(gPad->AbsPixeltoX(pxnew)),
         gPad->PadtoY(gPad->AbsPixeltoY(pynew)));
         fgPolyLine->Draw("L");
      }
      break;

   case kButton1Double:
      if (fgPolyLine) {
         if (mode == kPolyLine) {
            fgPolyLine->Set(npoints);
         } else {
            fgPolyLine->GetPoint(0, xnew, ynew);
            fgPolyLine->SetPoint(npoints, xnew, ynew);
         }
         gPad->GetCanvas()->Selected(gPad, fgPolyLine, kButton1Down);
         fgPolyLine = nullptr;
         npoints = 0;
         gPad->Modified();
         gPad->Update();
         gROOT->SetEditorMode();
      }
      fgPolyLine = nullptr;
      break;

   case kMouseMotion:
      pxnew = px;
      pynew = py;
      if (fgPolyLine) {
         fgPolyLine->SetPoint(npoints, gPad->PadtoX(gPad->AbsPixeltoX(pxnew)),
                                       gPad->PadtoY(gPad->AbsPixeltoY(pynew)));
         gPad->Modified();
         gPad->Update();

      }
      break;
   }
}


/////////////////////////////////////////////
/// Internal class implementing tlatex editing
class TTextEditInteractive : public TVirtualPad::TInteractive {
   public:

   TTextEditInteractive(TLatex *ltx)
   {
      fLatex = ltx;
   }

   void FinishEditing()
   {
      TString s = fLatex->GetTitle();
      Int_t l = s.Length();
      if (l > 1) {
         // keep dummy symbol to let select it afterwards
         s.Remove(l-1);
         fLatex->SetTitle(s.Data());
      }
   }

   bool PressBackspace()
   {
      TString s(fLatex->GetTitle());
      Int_t l = s.Length();
      if (l < 2)
         return false;
      s.Replace(l-2, 2, "<");
      fLatex->SetTitle(s.Data());
      return true;
   }

   void AppendChar(char c)
   {
      TString s = fLatex->GetTitle();
      Int_t l = s.Length();
      s.Insert(l-1, c);
      fLatex->SetTitle(s.Data());
   }

   TLatex *fLatex = nullptr;
};

////////////////////////////////////////////////////////////////////////////////
/// Create a new TLatex or TMarker at the cursor position in the provided pad
///
///  - Click left button to indicate the text position.
///  - Type text
///  - Press Enter or click mouse once again to stop editing

void TCreatePrimitives::ExecuteText(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py)
{
   auto inter = (mode == kEditor) ? parent.GetInteractive<TTextEditInteractive>(&parent) : nullptr;
   if (inter && !parent.FindObject(inter->fLatex)) {
      parent.FreeInteractive(&parent);
      inter = nullptr;
   }

   switch (event) {

   case kKeyPress: {
      // only modify in editor mode
      if (!inter)
         break;

      if ((py == kKey_Return) || (py == kKey_Enter)) {
         inter->FinishEditing();
         gSystem->ProcessEvents();
         parent.Modified(kTRUE);
         gROOT->SetEditorMode();
         parent.Update();
         parent.GetCanvas()->Selected(gPad, inter->fLatex, kButton1Down);
         parent.FreeInteractive(&parent);
      } else if (py == kKey_Backspace) {
         if (inter->PressBackspace()) {
            parent.Modified(kTRUE);
            parent.Update();
         }
      } else if (isprint(py)) {
         inter->AppendChar((char)py);
         parent.Modified(kTRUE);
         parent.Update();
      }
      break;
   }

   case kButton1Down: {
      if (inter) {
         // second click during editing will finalize editor
         inter->FinishEditing();
         gSystem->ProcessEvents();
         parent.Modified(kTRUE);
         gROOT->SetEditorMode();
         parent.Update();
         parent.GetCanvas()->Selected(gPad, inter->fLatex, kButton1Down);
         parent.FreeInteractive(&parent);
         break;
      }

      Double_t x = parent.PadtoX(parent.AbsPixeltoX(px));
      Double_t y = parent.PadtoY(parent.AbsPixeltoY(py));

      if (mode == kMarker) {
         auto marker = new TMarker(x, y, 28);
         parent.GetCanvas()->Selected(&parent, marker, kButton1Down);
         parent.Add(marker);
         gROOT->SetEditorMode();
      } else if (mode == kText) {
         ((TPad *)&parent)->StartEditing();
         gSystem->ProcessEvents();
         auto ltx = new TLatex(x,y,"<");
         parent.GetCanvas()->Selected(gPad, ltx, kButton1Down);
         parent.Add(ltx);
         parent.Modified(kTRUE);
         parent.Update();
         gROOT->SetEditorMode("Editor"); // switch to text editing
         parent.MakeInteractive<TTextEditInteractive>(&parent, ltx);
      }
      break;
   }
   }
}


////////////////////////////////////////////////////////////////////////////////
/// @deprectaed
/// Create a new TLatex or TMarker at the cursor position in gPad.

void TCreatePrimitives::Text(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   if (auto pad = dynamic_cast<TPad *>(gPad))
      ExecuteText(*pad, mode, event, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Handle edit mode in the TPad
/// Executed when editmode is not 0

void TCreatePrimitives::ExecuteEvent(TPad &pad, Int_t edit_mode, Int_t event, Int_t px, Int_t py)
{
   switch (edit_mode) {
   case kPad: Pad(event, px, py, 0); break;
   case kMarker:
   case kText:
   case kEditor: ExecuteText(pad, edit_mode, event, px, py); break;
   case kLine:
   case kArrow:
   case kCurlyLine:
   case kCurlyArc:
   case kArc:
   case kEllipse: ExecuteLine(pad, edit_mode, event, px, py); break;
   case kPolyLine:
   case kCutG: PolyLine(event, px, py, edit_mode); break;
   case kButton:
   case kPave:
   case kPaveLabel:
   case kPaveText:
   case kPavesText:
   case kDiamond: Pave(event, px, py, edit_mode); break;
   default: break;
   }
}
