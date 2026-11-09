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
      ExecuteTwoPoint(*pad, mode, event, px, py);
}

/////////////////////////////////////////////
/// Internal class implementing mouse click/move/relase editing
/// Used for many classes like line, arrow, arc,
class TTwoPointInteractive : public TVirtualPad::TInteractive {
   public:

   TTwoPointInteractive(Int_t mode, Double_t x0, Double_t y0, Int_t px0, Int_t py0)
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

   void SetPaveCoord(TPave *pave, Double_t x, Double_t y)
   {
      pave->SetX1(TMath::Min(x, fX0));
      pave->SetX2(TMath::Max(x, fX0));
      pave->SetY1(TMath::Min(y, fY0));
      pave->SetY2(TMath::Max(y, fY0));
   }

   Int_t fMode = 0;
   Double_t fX0 = 0, fY0 = 0;
   Int_t fPX0 = 0, fPY0 = 0;
   TObject *fObject = nullptr;
};


/////////////////////////////////////////////
/// Internal class implementing text editing
class TTextEditInteractive : public TVirtualPad::TInteractive {
   public:

   TTextEditInteractive(TLatex *ltx)
   {
      fLatex = ltx;
   }

   TTextEditInteractive(TPaveLabel *lbl)
   {
      fLabel = lbl;
   }

   TObject *GetObject() const
   {
      if (fLatex)
         return fLatex;
      if (fLabel)
         return fLabel;
      return nullptr;
   }

   TString GetText() const
   {
      if (fLatex)
         return fLatex->GetTitle();
      if (fLabel)
         return fLabel->GetLabel();
      return "";
   }

   void SetText(const TString &s)
   {
      if (fLatex)
         fLatex->SetTitle(s);
      else if (fLabel)
         return fLabel->SetLabel(s);
   }

   void FinishEditing()
   {
      auto s = GetText();
      Int_t l = s.Length();
      if (l > 1) {
         s.Remove(l-1);
         SetText(s);
      }
   }

   bool ProcessKeyPress(TPad &parent, Int_t py)
   {
      if ((py == kKey_Return) || (py == kKey_Enter)) {
         FinishEditing();
         gSystem->ProcessEvents();
         parent.Modified(kTRUE);
         gROOT->SetEditorMode();
         parent.Update();
         parent.GetCanvas()->Selected(&parent, GetObject(), kButton1Down);
         return true;
      }

      auto s = GetText();
      Int_t l = s.Length();

      if (py == kKey_Backspace) {
         if (l > 1) {
            s.Replace(l-2, 2, "<");
            SetText(s);
            parent.Modified(kTRUE);
            parent.Update();
         }
      } else if (isprint(py)) {
         s.Insert(l - 1, (char)py);
         SetText(s);
         parent.Modified(kTRUE);
         parent.Update();
      }

      return false;
   }

   TLatex *fLatex = nullptr;
   TPaveLabel *fLabel = nullptr;
};


////////////////////////////////////////////////////////////////////////////////
/// Create a new line/arrow in the specified pad
///
///  - Click left button to indicate arrow starting position.
///  - Move mouse pointer when left button clicked to specify second position
///  - Release left button to terminate the arrow.

void TCreatePrimitives::ExecuteTwoPoint(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py)
{
   // special branch for text editing
   if (mode == kPaveLabel) {
      parent.EventPave(); // TODO: check if this is really needed
      auto inter_edit = parent.GetInteractive<TTextEditInteractive>(&parent);
      if (inter_edit) {
         if ((event == kKeyPress) && inter_edit->ProcessKeyPress(parent, py))
            parent.FreeInteractive(&parent);
         return;
      }
   }

   Double_t x = parent.PadtoX(parent.AbsPixeltoX(px));
   Double_t y = parent.PadtoY(parent.AbsPixeltoY(py));

   // first mouse click
   if (event == kButton1Down) {
      parent.MakeInteractive<TTwoPointInteractive>(&parent, mode, x, y, px, py);
      return;
   }

   auto inter = parent.GetInteractive<TTwoPointInteractive>(&parent);
   if (!inter)
      return;

   // verify that object is still there or mode is changed
   if ((inter->fObject && !parent.FindObject(inter->fObject)) || (mode != inter->fMode)) {
      parent.FreeInteractive(&parent);
      return;
   }

   // when mouse released - finish interactive
   if (event == kButton1Up) {
      auto obj = inter->fObject;
      if (obj)
         parent.GetCanvas()->Selected(&parent, obj, kButton1Down);

      parent.FreeInteractive(&parent);

      if ((mode == kPaveLabel) && obj)
         parent.MakeInteractive<TTextEditInteractive>(&parent, (TPaveLabel *)obj);
      else
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
   case kPave:
      if (auto pave = inter->Make<TPave>(parent, x, y, x, y))
         inter->SetPaveCoord(pave, x, y);
      break;
   case kPaveLabel:
      if (auto pave = inter->Make<TPaveLabel>(parent, x, y, x, y, ">"))
         inter->SetPaveCoord(pave, x, y);
      break;
   case kPaveText:
      if (auto pave = inter->Make<TPaveText>(parent, x, y, x, y))
         inter->SetPaveCoord(pave, x, y);
      break;
   case kPavesText:
      if (auto pave = inter->Make<TPavesText>(parent, x, y, x, y))
         inter->SetPaveCoord(pave, x, y);
      break;
   case kDiamond:
      if (auto diamond = inter->Make<TDiamond>(parent, x, y, x, y))
         inter->SetPaveCoord(diamond, x, y);
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
      ExecuteTwoPoint(*pad, mode, event, px, py);
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
/// @deprecated
/// Create a new pave-derived object in gPad.

void TCreatePrimitives::Pave(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   if (auto pad = dynamic_cast<TPad *>(gPad))
      ExecuteTwoPoint(*pad, mode, event, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Interactive object for TGraph and TCutG editing
class TPolyLineEditInteractive : public TVirtualPad::TInteractive {
   public:

   TPolyLineEditInteractive(Int_t mode, const char *name, Double_t x0, Double_t y0, Int_t px0, Int_t py0)
   {
      fX0 = x0;
      fY0 = y0;
      fPX0 = px0;
      fPY0 = py0;
      if (mode == kPolyLine) {
         fGraph = new TGraph(2);
         fGraph->SetName(name);
         fGraph->ResetBit(TGraph::kClipFrame);
      } else {
         fGraph = new TCutG(name, 2);
      }
      fGraph->SetPoint(0, x0, y0);
      fGraph->SetPoint(1, x0, y0);
      fNPoints = 1;
   }

   Double_t fX0 = 0, fY0 = 0;
   Int_t fPX0 = 0, fPY0 = 0;
   TGraph *fGraph = nullptr;
   Int_t fNPoints = 0;
};


////////////////////////////////////////////////////////////////////////////////
/// Create a new TGraph or TCutG in the pad
///
///  - Click left button to indicate a new point.
///  - Click left button at same place or double click to close the polyline.

void TCreatePrimitives::ExecutePolyLine(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py)
{
   Double_t x = parent.PadtoX(parent.AbsPixeltoX(px));
   Double_t y = parent.PadtoY(parent.AbsPixeltoY(py));

   auto inter = parent.GetInteractive<TPolyLineEditInteractive>(&parent);

   switch (event) {

   case kButton1Down:
      if (inter) {
         inter->fGraph->SetPoint(inter->fNPoints, x, y);
         // if new point close to original point - just stop editing
         if ((inter->fNPoints > 1) && (TMath::Abs(px - inter->fPX0) + TMath::Abs(py - inter->fPY0) < 7)) {
            inter->fGraph->SetPoint(inter->fNPoints, inter->fX0, inter->fY0);
            parent.GetCanvas()->Selected(&parent, inter->fGraph, kButton1Down);
            parent.FreeInteractive(&parent);
            parent.Modified();
            parent.Update();
            gROOT->SetEditorMode();
         } else {
            // add new point
            inter->fNPoints++;
            inter->fGraph->Set(inter->fNPoints + 1);
            inter->fGraph->SetPoint(inter->fNPoints, x, y);
         }
      } else {
         TString name = mode == kPolyLine ? "graph" : "CUTG";
         Int_t cnt = 0;
         while (parent.FindObject(name.Data()) && (cnt++ < 1000))
            name.Form("%s%d", mode == kPolyLine ? "gr_" : "cutg_", cnt);

         inter = parent.MakeInteractive<TPolyLineEditInteractive>(&parent, mode, name.Data(), x, y, px, py);
         parent.Add(inter->fGraph, "L");
      }
      break;

   case kButton1Double:
      if (inter) {
         if (inter->fNPoints < (mode == kCutG ? 3 : 2)) {
            // too few points - primitive is not complete
            parent.Remove(inter->fGraph);
            delete inter->fGraph;
         } else if (mode == kCutG) {
            // force same first and last point
            inter->fGraph->SetPoint(inter->fNPoints, inter->fX0, inter->fY0);
         } else {
            // just remove last point which only was visisble during painting
            inter->fGraph->Set(inter->fNPoints);
         }
         parent.GetCanvas()->Selected(&parent, inter->fGraph, kButton1Down);

         parent.FreeInteractive(&parent);
         parent.Modified();
         parent.Update();
         gROOT->SetEditorMode();
      }
      break;

   case kMouseMotion:
      if (inter) {
         inter->fGraph->SetPoint(inter->fNPoints, x, y);
         parent.Modified();
         parent.Update();
      }
      break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// @deprecated
/// Create a new PolyLine in gPad.

void TCreatePrimitives::PolyLine(Int_t event, Int_t px, Int_t py, Int_t mode)
{
   if (auto pad = dynamic_cast<TPad *>(gPad))
      ExecutePolyLine(*pad, mode, event, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Create a new TLatex or TMarker at the cursor position in the provided pad
///
///  - Click left button to indicate the text position.
///  - Type text
///  - Press Enter or click mouse once again to stop editing

void TCreatePrimitives::ExecuteText(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py)
{
   auto inter = (mode == kText) ? parent.GetInteractive<TTextEditInteractive>(&parent) : nullptr;
   if (inter && !parent.FindObject(inter->GetObject())) {
      parent.FreeInteractive(&parent);
      inter = nullptr;
   }

   switch (event) {

   case kKeyPress:
      // only modify in editor mode
      if (inter && inter->ProcessKeyPress(parent, py))
         parent.FreeInteractive(&parent);
      break;

   case kButton1Down: {
      if (inter) {
         // second click during editing will finalize editor
         inter->FinishEditing();
         gSystem->ProcessEvents();
         parent.Modified(kTRUE);
         gROOT->SetEditorMode();
         parent.Update();
         parent.GetCanvas()->Selected(&parent, inter->fLatex, kButton1Down);
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
         parent.StartEditing();
         gSystem->ProcessEvents();
         auto ltx = new TLatex(x,y,"<");
         parent.GetCanvas()->Selected(gPad, ltx, kButton1Down);
         parent.Add(ltx);
         parent.Modified(kTRUE);
         parent.Update();
         parent.MakeInteractive<TTextEditInteractive>(&parent, ltx);
      }
      break;
   }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// @deprecated
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
   case kText: ExecuteText(pad, edit_mode, event, px, py); break;
   case kLine:
   case kArrow:
   case kCurlyLine:
   case kCurlyArc:
   case kArc:
   case kEllipse:
   case kPave:
   case kPaveLabel:
   case kPaveText:
   case kPavesText:
   case kDiamond: ExecuteTwoPoint(pad, edit_mode, event, px, py); break;
   case kPolyLine:
   case kCutG: ExecutePolyLine(pad, edit_mode, event, px, py); break;
   default: break;
   }
}
