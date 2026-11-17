// @(#)root/gpad:$Id: TCreatePrimitives.cxx

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
to create new primitives from the TCanvas toolbar.
*/

#include "TCreatePrimitives.h"

#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TMarker.h"
#include "TGroupButton.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TMath.h"
#include "KeySymbols.h"
#include "TCutG.h"
#include "TLatex.h"
#include "TCurlyArc.h"
#include "TArrow.h"
#include "TArc.h"
#include "TPavesText.h"
#include "TPaveLabel.h"
#include "TDiamond.h"
#include "TGraph.h"


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
/// Used for many classes like line, arrow, arc, pad, paves, ...
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
         // check that object of expected type
         auto res = dynamic_cast<T *>(fObject);
         // if not match - just "forget" about it, pad will take care about correct cleanup
         if (!res)
            fObject = nullptr; // failure, interactive will be deleted
         return res;
      }
      auto res2 = new T(std::forward<Rest>(rest)...);
      // add created object to pad list of primitives, delegating ownership to the pad
      parent.Add(res2);
      fObject = res2;
      return res2;
   }

   template<class T>
   void SetCoord(T *obj, Double_t x, Double_t y)
   {
      obj->SetX1(TMath::Min(x, fX0));
      obj->SetX2(TMath::Max(x, fX0));
      obj->SetY1(TMath::Min(y, fY0));
      obj->SetY2(TMath::Max(y, fY0));
   }

   Int_t fMode = 0;               ///< edit mode when interactive action started
   Double_t fX0 = 0, fY0 = 0;     ///< initial mouse pointer position in user coordiantes
   Int_t fPX0 = 0, fPY0 = 0;      ///< initial mouse pointer position in pixel coordiantes
   TObject *fObject = nullptr;    ///< object created by interactive action, owned by pad
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

   TTextEditInteractive(TPaveText *txt)
   {
      fPaveText = txt;
   }

   TObject *GetObject() const
   {
      if (fLatex)
         return fLatex;
      if (fLabel)
         return fLabel;
      if (fPaveText)
         return fPaveText;
      return nullptr;
   }

   TString GetText() const
   {
      if (fLatex)
         return fLatex->GetTitle();
      if (fLabel)
         return fLabel->GetLabel();
      if (fPaveText)
         return fPaveText->GetSize() > 0 ? fPaveText->GetLine(0)->GetTitle() : "";
      return "";
   }

   void SetText(const TString &s)
   {
      if (fLatex)
         fLatex->SetTitle(s);
      else if (fLabel)
         return fLabel->SetLabel(s);
      else if (fPaveText && (fPaveText->GetSize() > 0))
         fPaveText->GetLine(0)->SetTitle(s);
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

   TLatex *fLatex = nullptr;       ///< editing text in TLatex
   TPaveLabel *fLabel = nullptr;   ///< or in TPaveLabel
   TPaveText *fPaveText = nullptr; ///< or in TPave(s)Text
};


////////////////////////////////////////////////////////////////////////////////
/// Create a new line/arrow/pave/subpad in the specified pad
///
///  - Click left button to indicate object start position.
///  - Move mouse pointer when left button pressed to specify second position
///  - Release left button to stop position/size change
///  - for TPaveText, TPavesText, TPaveLabel text editing started until Enter key is pressed

void TCreatePrimitives::ExecuteTwoPoint(TPad &parent, Int_t mode, Int_t event, Int_t px, Int_t py)
{
   if (mode == kPaveLabel)
      parent.EventPave(); // TODO: check if this is really needed

   // special branch for text editing
   if ((mode == kPaveLabel) || (mode == kPaveText) || (mode == kPavesText)) {
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
      if (mode == kPad) {
         parent.Remove(obj);
         Int_t  npads = 0;
         TIter next(parent.GetListOfPrimitives());
         while (auto pobj = next()) {
            if (pobj->InheritsFrom(TPad::Class()))
               npads++;
         }

         auto box = (TBox *) obj;
         Double_t xlow = (parent.XtoPad(box->GetX1()) - parent.GetX1())/(parent.GetX2() - parent.GetX1());
         Double_t ylow = (parent.YtoPad(box->GetY1()) - parent.GetY1())/(parent.GetY2() - parent.GetY1());
         Double_t xup = (parent.XtoPad(box->GetX2()) - parent.GetX1())/(parent.GetX2() - parent.GetX1());
         Double_t yup = (parent.YtoPad(box->GetY2()) - parent.GetY1())/(parent.GetY2() - parent.GetY1());

         // TBox object no longer needed
         delete obj;
         obj = nullptr;

         // to preserve gPad when new pad is created
         TVirtualPad::TContext padtxt(true);

         auto newpad = new TPad(TString::Format("%s_%d",parent.GetName(),npads+1).Data(),"newpad", xlow, ylow, xup, yup);
         if (!newpad->IsZombie()) {
            newpad->UseCurrentStyle();
            parent.Add(newpad);
            obj = newpad;
         }
      }

      if (obj)
         parent.GetCanvas()->Selected(&parent, obj, kButton1Down);

      parent.FreeInteractive(&parent);

      if (((mode == kPaveText) || (mode == kPavesText)) && obj)
         parent.MakeInteractive<TTextEditInteractive>(&parent, (TPaveText *)obj);
      else if ((mode == kPaveLabel) && obj)
         parent.MakeInteractive<TTextEditInteractive>(&parent, (TPaveLabel *)obj);
      else
         gROOT->SetEditorMode();
      return;
   }

   // ignore all other events
   if (event != kButton1Motion)
      return;

   bool had_object = (inter->fObject != nullptr);

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
         inter->SetCoord<TPave>(pave, x, y);
      break;
   case kPaveLabel:
      if (auto pave = inter->Make<TPaveLabel>(parent, x, y, x, y, ">"))
         inter->SetCoord<TPave>(pave, x, y);
      break;
   case kPaveText:
      if (auto pave = inter->Make<TPaveText>(parent, x, y, x, y)) {
         if (!had_object)
            pave->AddText(">");
         inter->SetCoord<TPave>(pave, x, y);
      }
      break;
   case kPavesText:
      if (auto pave = inter->Make<TPavesText>(parent, x, y, x, y)) {
         if (!had_object)
            pave->AddText(">");
         inter->SetCoord<TPave>(pave, x, y);
      }
      break;
   case kDiamond:
      if (auto diamond = inter->Make<TDiamond>(parent, x, y, x, y))
         inter->SetCoord<TPave>(diamond, x, y);
      break;
   case kPad:
      if (auto box = inter->Make<TBox>(parent, x, y, x, y)) {
         box->SetFillStyle(1); // to paint only border
         inter->SetCoord<TBox>(box, x, y);
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
      ExecuteTwoPoint(*pad, mode, event, px, py);
}

////////////////////////////////////////////////////////////////////////////////
/// @deprecated
/// Create a new pad in gPad.

void TCreatePrimitives::Pad(Int_t event, Int_t px, Int_t py, Int_t)
{
   if (auto pad = dynamic_cast<TPad *>(gPad))
      ExecuteTwoPoint(*pad, kPad, event, px, py);
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
         parent.GetCanvas()->Selected(&parent, ltx, kButton1Down);
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
   case kDiamond:
   case kPad: ExecuteTwoPoint(pad, edit_mode, event, px, py); break;
   case kPolyLine:
   case kCutG: ExecutePolyLine(pad, edit_mode, event, px, py); break;
   default: gROOT->SetEditorMode(); break; // any other values dismissed
   }
}
