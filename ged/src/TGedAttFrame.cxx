// @(#)root/ged:$Name:  $:$Id: TGedAttFrame.cxx,v 1.7 2004/04/06 21:06:13 rdm Exp $
// Author: Marek Biskup, Ilka Antcheva   22/07/03

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGedAttFrame, TGedAttNameFrame, TGedAttFillFrame,                   //
//  TGedAttLineFrame, TGedAttTextFrame, TGedAttMarkerFrame              //
//                                                                      //
//  Frames with object attributes, just like on TAttCanvases.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGedAttFrame.h"
#include "TGColorSelect.h"
#include "TGedMarkerSelect.h"
#include "TGClient.h"
#include "TGMsgBox.h"
#include "TGGC.h"
#include "TGedPatternSelect.h"
#include "TGColorDialog.h"
#include "TGResourcePool.h"
#include "TG3DLine.h"
#include "TColor.h"
#include "TGToolTip.h"
#include "TGButton.h"
#include "TCint.h"
#include "TCanvas.h"
#include "TVirtualMutex.h"
#include "TVirtualPad.h"
#include "TGToolTip.h"
#include "TGLabel.h"
#include "TGComboBox.h"

#include "Api.h"
#include "TGLabel.h"
#include "TPaveLabel.h"
#include <snprintf.h>

ClassImp(TGedAttFrame)
ClassImp(TGedAttFillFrame)
ClassImp(TGedAttLineFrame)
ClassImp(TGedAttTextFrame)
ClassImp(TGedAttMarkerFrame)

enum {
   kPATTERN,
   kCOLOR,
   kLINE_WIDTH,
   kLINE_STYLE,
   kFONT_SIZE,
   kFONT_STYLE,
   kFONT_ALIGN,
   kMARKER,
   kMARKER_SIZE
};

//______________________________________________________________________________
TGedAttFrame::TGedAttFrame(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGCompositeFrame(p, width, height, options, back), TGWidget(id)
{
   fPad = 0;
   fModel = 0;

   Associate(p);
}

//______________________________________________________________________________
long TGedAttFrame::ExecuteInt(TObject *obj, const char *method, const char *params)
{
   R__LOCKGUARD(gCINTMutex);
   void       *address;
   long        offset;
   G__CallFunc func;

   // set pointer to interface method and arguments
   func.SetFunc(obj->IsA()->GetClassInfo(), method, params, &offset);

   // call function
   address = (void*)((Long_t)obj + offset);
   return func.ExecInt(address);
}

//______________________________________________________________________________
Float_t TGedAttFrame::ExecuteFloat(TObject *obj, const char *method,
                                   const char *params)
{
   R__LOCKGUARD(gCINTMutex);
   void       *address;
   long        offset;
   G__CallFunc func;

   // set pointer to interface method and arguments
   func.SetFunc(obj->IsA()->GetClassInfo(), method, params, &offset);
   // call function
   address = (void*)((Long_t)obj + offset);
   return func.ExecDouble(address);
}

//______________________________________________________________________________
TGCompositeFrame* TGedAttFrame::MakeTitle(const char *p)
{
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 128, 40,
                               kHorizontalFrame | kLHintsExpandX | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, p), new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   f1->AddFrame(new TGHorizontal3DLine(f1),
                new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop));
   return f1;
}

//______________________________________________________________________________
void TGedAttFrame::SetActive(Bool_t active)
{
   if (active)
      ((TGCompositeFrame*)GetParent())->ShowFrame(this);
   else
      ((TGCompositeFrame*)GetParent())->HideFrame(this);

   ((TGMainFrame*)GetMainFrame())->Layout();
}

//______________________________________________________________________________
void TGedAttFrame::Refresh()
{
   SetModel(fPad, fModel, 0);
}

//______________________________________________________________________________
void TGedAttFrame::ConnectToCanvas(TCanvas* c)
{
   TQObject::Connect(c, "Selected(TPad*,TObject*,Int_t)", "TGedAttFrame",
                     this, "SetModel(TPad*,TObject*,Int_t)");
}

//______________________________________________________________________________
TGedAttNameFrame::TGedAttNameFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   MakeTitle("Name");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fLabel = new TGLabel(f2, "");
   f2->AddFrame(fLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
}

//______________________________________________________________________________
void TGedAttNameFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   // Slot connected to Selected() signal of TCanvas.

   fModel = obj;
   fPad = pad;

   if (obj == 0) {
      SetActive(kFALSE);
      return;
   }

   TString string;
   string.Append(fModel->GetName());
   string.Append("::");
   string.Append(fModel->ClassName());

   fLabel->SetText(new TGString(string));
   SetActive();
}

//______________________________________________________________________________
TGedAttFillFrame::TGedAttFillFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   MakeTitle("Fill");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);
   fPatternSelect = new TGedPatternSelect(f2, 1, kPATTERN);
   f2->AddFrame(fPatternSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fPatternSelect->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
}

//______________________________________________________________________________
void TGedAttFillFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAttFill")) {

      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   Color_t c = ExecuteInt(fModel, "GetFillColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   Style_t s = (Style_t) ExecuteInt(fModel, "GetFillStyle", "");

   fPatternSelect->SetPattern(s);
   fColorSelect->SetColor(p);
   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttFillFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   if(!fModel)
      return kTRUE;
   bool b = false;
   switch (GET_MSG(msg)) {
      case kC_PATTERNSEL:
         switch (GET_SUBMSG(msg)) {
            case kPAT_SELCHANGED:
               {
                  char a[100];
                  snprintf(a, 100, "%ld",  parm2);
                  fModel->Execute("SetFillStyle", a, 0);
                  b = true;
                  SendMessage(fMsgWindow,
                        MK_MSG(kC_PATTERNSEL, kPAT_SELCHANGED), parm1, parm2);
               }
               break;
            default:
               break;
         }
         break;
      case kC_COLORSEL:
         switch (GET_SUBMSG(msg)) {
            case kCOL_SELCHANGED:
               {
                  char a[100];
                  snprintf(a, 100, "%d", TColor::GetColor(parm2));
                  fModel->Execute("SetFillColor", a, 0);
                  b = true;
                  SendMessage(fMsgWindow, msg, parm1, parm2);
               }
         }
   }
   if (b)
   {
      fPad->Modified();
      gPad->Modified();
      gPad->Update();
   }
   return kTRUE;
}

//______________________________________________________________________________
TGedAttLineFrame::TGedAttLineFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{

   MakeTitle("Line");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);

   fStyleCombo = new TGLineStyleComboBox(this, kLINE_STYLE);
   fStyleCombo->Resize(126, 20);
   AddFrame(fStyleCombo, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));

   fWidthCombo = new TGLineWidthComboBox(f2, kLINE_WIDTH);
   f2->AddFrame(fWidthCombo, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fWidthCombo->Resize(80, 20);
   fWidthCombo->Associate(this);
}

//______________________________________________________________________________
void TGedAttLineFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAttLine")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   Int_t s = ExecuteInt(fModel, "GetLineStyle", "");
   fStyleCombo->Select(s);

   Int_t w = ExecuteInt(fModel, "GetLineWidth", "");
   fWidthCombo->Select(w);

   Color_t c = ExecuteInt(fModel, "GetLineColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p);

   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttLineFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   if(!fModel)
      return kTRUE;
   char a[100];
   bool b = false;
   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED)
   {
      snprintf(a, 100, "%d", TColor::GetColor(parm2));
      fModel->Execute("SetLineColor", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }
   if (GET_MSG(msg) == kC_COMMAND && GET_SUBMSG(msg) == kCM_COMBOBOX) {

      if (parm1 == kLINE_WIDTH) {
         snprintf(a, 100, "%ld", parm2);
         fModel->Execute("SetLineWidth", a, 0);
         b = true;
      } else if (parm1 == kLINE_STYLE) {
         snprintf(a, 100, "%ld", parm2);
         fModel->Execute("SetLineStyle", a, 0);
         b = true;
      }
   }
   if (b) {
      fPad->Modified();
      fPad->Update();
   }
   return kTRUE;
}

//______________________________________________________________________________
TGedAttTextFrame::TGedAttTextFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   MakeTitle("Text");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1,1, 1));
   fColorSelect->Associate(this);
   fSizeCombo = BuildFontSizeComboBox(f2, kFONT_SIZE);
   f2->AddFrame(fSizeCombo, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fSizeCombo->Resize(80, 20);
   fSizeCombo->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
   fTypeCombo = new TGFontTypeComboBox(this, kFONT_STYLE);
   fTypeCombo->Resize(126, 20);
   AddFrame(fTypeCombo, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fAlignCombo = BuildTextAlignComboBox(this, kFONT_ALIGN);
   fAlignCombo->Resize(126, 20);
   AddFrame(fAlignCombo, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
}

//______________________________________________________________________________
void TGedAttTextFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAttText")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   fTypeCombo->Select(ExecuteInt(fModel, "GetTextFont", "") / 10);

   Float_t s = ExecuteFloat(fModel, "GetTextSize", "");
   Float_t dy;

   if (obj->InheritsFrom("TPaveLabel")) {
      TBox *pl = (TBox*)obj;
      dy = s * (pl->GetY2() - pl->GetY1());
   }
   else
      dy = s * (fPad->GetY2() - fPad->GetY1());

   Int_t size = fPad->YtoPixel(0.0) - fPad->YtoPixel(dy);
   if (size > 50) size = 50;
   if (size < 0)  size = 0;
   fSizeCombo->Select(size);

   fAlignCombo->Select(ExecuteInt(fModel, "GetTextAlign", ""));

   Color_t c = ExecuteInt(fModel, "GetTextColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p);

   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttTextFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   if(!fModel) return kTRUE;

   char a[100];
   bool b = false;

   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED) {
      snprintf(a, 100, "%d", TColor::GetColor(parm2));
      fModel->Execute("SetTextColor", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }

   if (GET_MSG(msg) == kC_COMMAND && GET_SUBMSG(msg) == kCM_COMBOBOX) {

      if (parm1 == kFONT_SIZE) {

         Float_t dy = fPad->AbsPixeltoY(0) - fPad->AbsPixeltoY(parm2);
         Float_t TextSize;

         if (fModel->InheritsFrom("TPaveLabel")) {
            TBox *pl = (TBox*) fModel;
            TextSize = dy/(pl->GetY2() - pl->GetY1());
         }
         else
            TextSize = dy/(fPad->GetY2() - fPad->GetY1());

         snprintf(a, 100, "%f", TextSize);
         fModel->Execute("SetTextSize", a, 0);
         b = true;
      } else if (parm1 == kFONT_STYLE) {
         snprintf(a, 100, "%ld", parm2 * 10);
         fModel->Execute("SetTextFont", a, 0);
         b = true;
      } else if (parm1 == kFONT_ALIGN) {
         snprintf(a, 100, "%ld", parm2);
         fModel->Execute("SetTextAlign", a, 0);
         b = true;
      }
   }

   if (b) {
      fPad->Modified();
      fPad->Update();
   }
   return kTRUE;
}

//______________________________________________________________________________
TGComboBox* TGedAttTextFrame::BuildFontSizeComboBox(TGFrame* parent, Int_t id)
{
   char a[100];
   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("Default", 0);
   for (int i = 1; i <= 50; i++) {
      snprintf(a, 100, "%d", i);
      c->AddEntry(a, i);
   }

   return c;
}

//______________________________________________________________________________
TGComboBox* TGedAttTextFrame::BuildTextAlignComboBox(TGFrame* parent, Int_t id)
{
   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("13 Top, Left", 13);
   c->AddEntry("23 Top, Middle", 23);
   c->AddEntry("33 Top, Right", 33);
   c->AddEntry("12 Middle, Left", 12);
   c->AddEntry("22 Middle, Middle", 22);
   c->AddEntry("32 Middle, Right", 32);
   c->AddEntry("11 Bottom, Left", 11);
   c->AddEntry("21 Bottom, Middle", 21);
   c->AddEntry("31 Bottom, Right", 31);

   return c;
}

//______________________________________________________________________________
TGedAttMarkerFrame::TGedAttMarkerFrame(const TGWindow *p, Int_t id, Int_t width,
                                       Int_t height,UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   MakeTitle("Marker");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1,1, 1));
   fColorSelect->Associate(this);

   fMarkerSelect = new TGedMarkerSelect(f2, 1, kMARKER);
   f2->AddFrame(fMarkerSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fMarkerSelect->Associate(this);

   fSizeCombo = BuildMarkerSizeComboBox(f2, kMARKER_SIZE);
   f2->AddFrame(fSizeCombo, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fSizeCombo->Resize(40, 20);
   fSizeCombo->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
}

//______________________________________________________________________________
void TGedAttMarkerFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAttMarker"))
   {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   Float_t s = ExecuteFloat(fModel, "GetMarkerSize", "");
   s = TMath::Nint(s * 5);

   if (s > 15) s = 15;

   if (s < 1)  s = 1;

   fSizeCombo->Select((int) s);

   fMarkerSelect->SetMarkerStyle((Style_t) ExecuteInt(fModel, "GetMarkerStyle", ""));

   Color_t c = ExecuteInt(fModel, "GetMarkerColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p);

   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttMarkerFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   if(!fModel) return kTRUE;

   char a[100];
   bool b = false;

   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED) {

      snprintf(a, 100, "%d", TColor::GetColor(parm2));
      fModel->Execute("SetMarkerColor", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   } else if (GET_MSG(msg) == kC_COMMAND && GET_SUBMSG(msg) == kCM_COMBOBOX) {
      if (parm1 == kMARKER_SIZE) {
         snprintf(a, 100, "%f", 0.2 * parm2);
         fModel->Execute("SetMarkerSize", a, 0);
         b = true;
      }
   } else if (GET_MSG(msg) == kC_MARKERSEL && GET_SUBMSG(msg) == kMAR_SELCHANGED) {
      snprintf(a, 100, "%d", (int) parm2);
      fModel->Execute("SetMarkerStyle", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }

   if (b) {
      fPad->Modified();
      fPad->Update();
   }

   return kTRUE;
}

//______________________________________________________________________________
TGComboBox* TGedAttMarkerFrame::BuildMarkerSizeComboBox(TGFrame* parent, Int_t id)
{
   char a[100];
   TGComboBox *c = new TGComboBox(parent, id);

   for (int i = 1; i <= 15; i++) {
      snprintf(a, 100, "%.1f", 0.2*i);
      c->AddEntry(a, i);
   }

   return c;
}

