// @(#)root/ged:$Name:  $:$Id: TGedAttFrame.cxx,v 1.9 2004/04/22 16:28:28 brun Exp $
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
#include "TGButtonGroup.h"
#include "TGNumberEntry.h"
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
ClassImp(TGedAttAxisFrame)
ClassImp(TGedAttAxisTitle)
ClassImp(TGedAttAxisLabel)

enum {
   kPATTERN,
   kCOLOR,
   kLINE_WIDTH,
   kLINE_STYLE,
   kFONT_SIZE,
   kFONT_STYLE,
   kFONT_ALIGN,
   kMARKER,
   kMARKER_SIZE,
   kAXIS_TICKS,
   kAXIS_DIV1,
   kAXIS_DIV2,
   kAXIS_DIV3,
   kAXIS_OPTIM,
   kAXIS_LOG,
   kAXIS_TITSIZE,
   kAXIS_TITOFFSET,
   kAXIS_CENTERED,
   kAXIS_ROTATED,
   kAXIS_LBLSIZE,
   kAXIS_LBLOFFSET,
   kAXIS_TICKSBOTH,
   kAXIS_LBLLOG,
   kAXIS_LBLEXP,
   kAXIS_LBLDIR,
   kAXIS_LBLSORT
};

//______________________________________________________________________________
TGedAttFrame::TGedAttFrame(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGCompositeFrame(p, width, height, options, back), TGWidget(id)
{
   // Constructor of the base GUI attribute frame.
   fCanvas = 0;
   fPad    = 0;
   fModel  = 0;

   Associate(p);
}

//______________________________________________________________________________
Long_t TGedAttFrame::ExecuteInt(TObject *obj, const char *method, const char *params)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD(gCINTMutex);
   void       *address;
   Long_t      offset;
   G__CallFunc func;

   // set pointer to interface method and arguments
   func.SetFunc(obj->IsA()->GetClassInfo(), method, params, &offset);

   // call function
   address = (void*)((Long_t)obj + offset);
   return func.ExecInt(address);
}

//______________________________________________________________________________
char *TGedAttFrame::ExecuteChar(TObject *obj, const char *method, const char *params)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD(gCINTMutex);
   void       *address;
   Long_t      offset;
   G__CallFunc func;

   // set pointer to interface method and arguments
   func.SetFunc(obj->IsA()->GetClassInfo(), method, params, &offset);

   // call function
   address = (void*)((Long_t)obj + offset);
   return (char *)func.ExecInt(address);
}

//______________________________________________________________________________
Float_t TGedAttFrame::ExecuteFloat(TObject *obj, const char *method,
                                   const char *params)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD(gCINTMutex);
   void       *address;
   Long_t      offset;
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
   // Create attribute frame title.

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 145, 10,
                               kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(new TGLabel(f1, p), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1),
                new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop));

   return f1;
}

//______________________________________________________________________________
void TGedAttFrame::SetActive(Bool_t active)
{
   // Set active GUI attribute frames related to the selected object.

   if (active)
      ((TGCompositeFrame*)GetParent())->ShowFrame(this);
   else
      ((TGCompositeFrame*)GetParent())->HideFrame(this);

   ((TGMainFrame*)GetMainFrame())->Layout();
}

//______________________________________________________________________________
void TGedAttFrame::Refresh()
{
   // Refresh the GUI info about the object attributes.

   SetModel(fPad, fModel, 0);
}

//______________________________________________________________________________
void TGedAttFrame::Update()
{
   // Update the current pad when an attribute is changed via GUI.

      fPad->Modified();
      fPad->Update();
}

//______________________________________________________________________________
void TGedAttFrame::ConnectToCanvas(TCanvas* c)
{
   // Connect the GUI attribute frames to the selected object in the canvas.

   TQObject::Connect(c, "Selected(TPad*,TObject*,Int_t)", "TGedAttFrame",
                     this, "SetModel(TPad*,TObject*,Int_t)");
}

//______________________________________________________________________________
TGedAttNameFrame::TGedAttNameFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Create the frame of the selected object info.

   MakeTitle("Name");
   
   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fLabel = new TGLabel(f2, "");
   f2->AddFrame(fLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
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

   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   fLabel->SetTextColor(color, kTRUE);
   fLabel->SetText(new TGString(string));
   SetActive();
}

//______________________________________________________________________________
TGedAttFillFrame::TGedAttFillFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of fill attributes GUI.

   MakeTitle("Fill");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);
   fPatternSelect = new TGedPatternSelect(f2, 1, kPATTERN);
   f2->AddFrame(fPatternSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fPatternSelect->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}

//______________________________________________________________________________
void TGedAttFillFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   // Pick up the used fill attributes.

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
   // Process message.

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
   // Constructor of line attributes GUI.

   MakeTitle("Line");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);

   fStyleCombo = new TGLineStyleComboBox(this, kLINE_STYLE);
   fStyleCombo->Resize(137, 20);
   AddFrame(fStyleCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));

   fWidthCombo = new TGLineWidthComboBox(f2, kLINE_WIDTH);
   f2->AddFrame(fWidthCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fWidthCombo->Resize(91, 20);
   fWidthCombo->Associate(this);
}

//______________________________________________________________________________
void TGedAttLineFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   // Pick up the used line attributes.

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
   // Process message.

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
   if (b) Update();

   return kTRUE;
}

//______________________________________________________________________________
TGedAttTextFrame::TGedAttTextFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of text attributes GUI.

   MakeTitle("Text");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);
   fSizeCombo = BuildFontSizeComboBox(f2, kFONT_SIZE);
   f2->AddFrame(fSizeCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fSizeCombo->Resize(91, 20);
   fSizeCombo->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fTypeCombo = new TGFontTypeComboBox(this, kFONT_STYLE);
   fTypeCombo->Resize(137, 20);
   AddFrame(fTypeCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fAlignCombo = BuildTextAlignComboBox(this, kFONT_ALIGN);
   fAlignCombo->Resize(137, 20);
   AddFrame(fAlignCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
}

//______________________________________________________________________________
void TGedAttTextFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   // Pick up the values of used text attributes.

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
   // Process message.

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

   if (b) Update();

   return kTRUE;
}

//______________________________________________________________________________
TGComboBox* TGedAttTextFrame::BuildFontSizeComboBox(TGFrame* parent, Int_t id)
{
   // Create text size combo box.

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
   // Create text align combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("13 Top, Left",   13);
   c->AddEntry("23 Top, Middle", 23);
   c->AddEntry("33 Top, Right",  33);
   c->AddEntry("12 Middle, Left",   12);
   c->AddEntry("22 Middle, Middle", 22);
   c->AddEntry("32 Middle, Right",  32);
   c->AddEntry("11 Bottom, Left",   11);
   c->AddEntry("21 Bottom, Middle", 21);
   c->AddEntry("31 Bottom, Right",  31);

   return c;
}

//______________________________________________________________________________
TGedAttMarkerFrame::TGedAttMarkerFrame(const TGWindow *p, Int_t id, Int_t width,
                                       Int_t height,UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of marker attributes GUI.

   MakeTitle("Marker");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColorSelect = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fColorSelect->Associate(this);

   fMarkerSelect = new TGedMarkerSelect(f2, 1, kMARKER);
   f2->AddFrame(fMarkerSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fMarkerSelect->Associate(this);

   fSizeCombo = BuildMarkerSizeComboBox(f2, kMARKER_SIZE);
   f2->AddFrame(fSizeCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   fSizeCombo->Resize(50, 20);
   fSizeCombo->Associate(this);
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}

//______________________________________________________________________________
void TGedAttMarkerFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   // Pick up the values of used marker attributes.

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
   // Process message.

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

   if (b) Update();

   return kTRUE;
}

//______________________________________________________________________________
TGComboBox* TGedAttMarkerFrame::BuildMarkerSizeComboBox(TGFrame* parent, Int_t id)
{
   // Marker size combobox.

   char a[100];
   TGComboBox *c = new TGComboBox(parent, id);

   for (int i = 1; i <= 15; i++) {
      snprintf(a, 100, "%.1f", 0.2*i);
      c->AddEntry(a, i);
   }

   return c;
}

//______________________________________________________________________________
TGedAttAxisFrame::TGedAttAxisFrame(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of axis attributes GUI.
   
   MakeTitle("Axis");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fAxisColor = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fAxisColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fAxisColor->Associate(this);
   TGLabel *fTicksLabel = new TGLabel(f2, "Ticks:");
   f2->AddFrame(fTicksLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 
                                               3, 0, 1, 1));
   fTickLength = new TGNumberEntry(f2, 0.03, 5, kAXIS_TICKS, 
                                       TGNumberFormat::kNESRealTwo,
                                       TGNumberFormat::kNEAAnyNumber,
                                       TGNumberFormat::kNELLimitMinMax,-1.,1.);
   fTickLength->Connect("ValueSet(Long_t)", "TGedAttAxisFrame", this, 
                        "DoTickLength()");
   f2->AddFrame(fTickLength, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fTicksBoth = new TGCheckButton(f4, "\"+-\"", kAXIS_TICKSBOTH);
   fTicksBoth->Connect("Toggled(Bool_t)","TGedAttAxisFrame",this,
                       "DoTicks()");
   fTicksBoth->SetToolTipText("Set ticks on both axis sides if selected");
   f4->AddFrame(fTicksBoth, new TGLayoutHints(kLHintsLeft | kLHintsBottom, 
                                              3, 1, 1, 0));
   fOptimize = new TGCheckButton(f4, "Optimize", kAXIS_OPTIM);
   fOptimize->SetState(kButtonDown);
   fOptimize->Connect("Toggled(Bool_t)","TGedAttAxisFrame",this,"DoDivisions()");
   f4->AddFrame(fOptimize, new TGLayoutHints(kLHintsTop, 18, 1, 1, 0));
   AddFrame(f4, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fLogAxis = new TGCheckButton(f5, "Log", kAXIS_LOG);
   fLogAxis->Connect("Toggled(Bool_t)","TGedAttAxisFrame",this,"DoLogAxis()");
   f5->AddFrame(fLogAxis, new TGLayoutHints(kLHintsLeft | kLHintsBottom, 3, 1, 0, 0));
   fLogAxis->SetToolTipText("Set Log scale if selected");

   fMoreLog = new TGCheckButton(f5, "MoreLog", kAXIS_LBLLOG);
   fMoreLog->Connect("Toggled(Bool_t)","TGedAttAxisFrame",this,"DoMoreLog()");
   f5->AddFrame(fMoreLog, new TGLayoutHints(kLHintsLeft, 19, 1, 0, 0));
   fMoreLog->SetToolTipText("Set more Log labels if selected");

   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   // axis divisions as three number entry widgets 
   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fDiv3 = new TGNumberEntry(f3, 10, 2,kAXIS_DIV1, TGNumberFormat::kNESInteger,
                                       TGNumberFormat::kNEANonNegative, 
                                       TGNumberFormat::kNELLimitMinMax, 0, 99);
   fDiv3->GetNumberEntry()->SetToolTipText("Tertiary axis divisions");
   fDiv3->Connect("ValueSet(Long_t)", "TGedAttAxisFrame", this, "DoDivisions()");
   f3->AddFrame(fDiv3, new TGLayoutHints(kLHintsLeft, 27, 0, 1, 1));
   fDiv2 = new TGNumberEntry(f3, 5, 2, kAXIS_DIV2, TGNumberFormat::kNESInteger,
                                       TGNumberFormat::kNEANonNegative, 
                                       TGNumberFormat::kNELLimitMinMax, 0, 99);
   fDiv2->GetNumberEntry()->SetToolTipText("Secondary axis divisions");
   fDiv2->Connect("ValueSet(Long_t)", "TGedAttAxisFrame", this, "DoDivisions()");
   f3->AddFrame(fDiv2, new TGLayoutHints(kLHintsLeft, 1, 0, 1, 1));
   fDiv1 = new TGNumberEntry(f3, 0, 2, kAXIS_DIV3, TGNumberFormat::kNESInteger,
                                       TGNumberFormat::kNEANonNegative, 
                                       TGNumberFormat::kNELLimitMinMax, 0, 99);
   fDiv1->GetNumberEntry()->SetToolTipText("Primary axis divisions");
   fDiv1->Connect("ValueSet(Long_t)", "TGedAttAxisFrame", this,"DoDivisions()");
   f3->AddFrame(fDiv1, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 1, 1, 1, 1));
   AddFrame(f3, new TGLayoutHints(kLHintsLeft, 1, 1, 4, 0));

   fTicksFlag = 1;
}

//______________________________________________________________________________
void TGedAttAxisFrame::SetModel(TPad* pad, TObject* obj, Int_t)
{
   // Pick up the used values of axis attributes.
   
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAxis")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   Color_t c = ExecuteInt(fModel, "GetAxisColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   fAxisColor->SetColor(p);

   Float_t ticks = ExecuteFloat(fModel, "GetTickLength", "");
   fTickLength->SetNumber(ticks);
   
   Int_t div = ExecuteInt(fModel, "GetNdivisions", "");
   fDiv3->SetNumber(div % 100);
   fDiv2->SetNumber((div/100) % 100);
   fDiv1->SetNumber((div/10000) % 100);
   
   if ((!strcmp(fModel->GetName(),"xaxis") && gPad->GetLogx()) ||
       (!strcmp(fModel->GetName(),"yaxis") && gPad->GetLogy()) ||
       (!strcmp(fModel->GetName(),"zaxis") && gPad->GetLogz())) 

      fLogAxis->SetState(kButtonDown);
   else fLogAxis->SetState(kButtonUp);

   if (fLogAxis->GetState() == kButtonUp) {
      fMoreLog->SetState(kButtonDisabled);
   } else {
      Int_t morelog = ExecuteInt(fModel, "GetMoreLogLabels", "");
      if (morelog) fMoreLog->SetState(kButtonDown);
      else         fMoreLog->SetState(kButtonUp);
   }
   
   char *both = ExecuteChar(fModel, "GetTicks", "");
   if (!strcmp(both,"+-")) {
      fTicksBoth->SetState(kButtonDown);
   } else {
      fTicksBoth->SetState(kButtonUp);
      if (!strcmp(both,"-")) fTicksFlag = -1;
      if (!strcmp(both,"+")) fTicksFlag =  1;
   }

   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttAxisFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{

   // Process message.
   
   if(!fModel)
      return kTRUE;

   char a[100];
   bool b = false;

   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED) {

      snprintf(a, 100, "%d", TColor::GetColor(parm2));
      fModel->Execute("SetAxisColor", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }
   if (b) Update();

   return kTRUE;
}

//______________________________________________________________________________
void TGedAttAxisFrame::DoTickLength()
{
   // Slot connected to the tick length settings.

   char a[100];
   snprintf(a, 100, "%f", (Float_t)fTickLength->GetNumber());
   fModel->Execute("SetTickLength", a, 0);
   Update();
   if (fTickLength->GetNumber() < 0) fTicksFlag = -1;
   else fTicksFlag = 1;
}

//______________________________________________________________________________
void TGedAttAxisFrame::DoTicks()
{
   // Slot connected to the ticks draw settings.

   char a[100];
   if (fTicksBoth->GetState() == kButtonDown) {
      snprintf(a, 100, "\"+-\"");
   } else {
      if (fTicksFlag == -1) snprintf(a, 100, "\"-\"");
      else snprintf(a, 100, "\"\"");
  }
   fModel->Execute("SetTicks", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisFrame::DoDivisions()
{
   // Slot connected to the number of divisions.

   char a[100];

   // the number of divisions are used 3 number entry widgets
   Int_t div = (Int_t)(fDiv3->GetNumber() + fDiv2->GetNumber()  * 100 
                                          + fDiv1->GetNumber() * 10000);
   snprintf(a, 100, "%d,%d", div, fOptimize->GetState());
   fModel->Execute("SetNdivisions", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisFrame::DoLogAxis()
{
   // Slot for Log scale setting.

   if (fLogAxis->GetState()) {

      if (!strcmp(fModel->GetName(),"xaxis")) gPad->SetLogx(1);
      if (!strcmp(fModel->GetName(),"yaxis")) gPad->SetLogy(1);
      if (!strcmp(fModel->GetName(),"zaxis")) gPad->SetLogz(1);

      Int_t morelog = ExecuteInt(fModel, "GetMoreLogLabels", "");
      if (!morelog) fMoreLog->SetState(kButtonDown);
      else          fMoreLog->SetState(kButtonUp);

      fOptimize->SetState(kButtonDisabled);

   } else {
      if (!strcmp(fModel->GetName(),"xaxis")) gPad->SetLogx(0);
      if (!strcmp(fModel->GetName(),"yaxis")) gPad->SetLogy(0);
      if (!strcmp(fModel->GetName(),"zaxis")) gPad->SetLogz(0);
      
      fMoreLog->SetState(kButtonDisabled);
      fOptimize->SetState(kButtonDown);
   }
   gPad->Modified();
   gPad->Update();
}

//______________________________________________________________________________
void TGedAttAxisFrame::DoMoreLog()
{
   // Slot connected to more Log labels flag

   char a[100];
   snprintf(a, 100, "%d", fMoreLog->GetState());
   fModel->Execute("SetMoreLogLabels", a, 0);
   Update();
}

//______________________________________________________________________________
TGedAttAxisTitle::TGedAttAxisTitle(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{

   // Constructor of axis title attributes GUI.

   MakeTitle("Title");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fTitleColor = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fTitleColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fTitleColor->Associate(this);
   TGLabel *fSizeLbl = new TGLabel(f2, "Size:");                              
   f2->AddFrame(fSizeLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 
                                            6, 1, 1, 1));
   fTitleSize = new TGNumberEntry(f2, 0.05, 5, kAXIS_TITSIZE, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 0., 1.);
   fTitleSize->Connect("ValueSet(Long_t)", "TGedAttAxisTitle", this, 
                       "DoTitleSize()");
   f2->AddFrame(fTitleSize, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   fTitleFont = new TGFontTypeComboBox(this, kFONT_STYLE);
   fTitleFont->Resize(137, 20);
   fTitleFont->Connect("Selected(Int_t)", "TGedAttAxisTitle", this, 
                       "DoTitleFont()"); 
   AddFrame(fTitleFont, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 1));
   fPrecision = 0;

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fCentered = new TGCheckButton(f3, "Centered", kAXIS_CENTERED);
   fCentered->Connect("Toggled(Bool_t)","TGedAttAxisTitle",this,
                      "DoTitleCentered()");
   f3->AddFrame(fCentered, new TGLayoutHints(kLHintsTop, 3, 1, 1, 0));
   TGLabel *fOffsetLbl = new TGLabel(f3, "Offset:");                              
   f3->AddFrame(fOffsetLbl, new TGLayoutHints(kLHintsLeft, 23, 1, 3, 0));
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fRotated = new TGCheckButton(f4, "Rotated", kAXIS_ROTATED);
   fRotated->SetState(kButtonDown);
   fRotated->Connect("Toggled(Bool_t)","TGedAttAxisTitle",this,
                     "DoTitleRotated()");
   f4->AddFrame(fRotated, new TGLayoutHints(kLHintsTop, 3, 1, 6, 0));
   fTitleOffset = new TGNumberEntry(f4, 1.00, 6, kAXIS_TITOFFSET, 
                                        TGNumberFormat::kNESRealTwo,
                                        TGNumberFormat::kNEAAnyNumber, 
                                        TGNumberFormat::kNELLimitMinMax, 0.1, 10.);
   fTitleOffset->Connect("ValueSet(Long_t)", "TGedAttAxisTitle", this, 
                         "DoTitleOffset()");
   f4->AddFrame(fTitleOffset, new TGLayoutHints(kLHintsLeft, 6, 1, 0, 0));
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}

//______________________________________________________________________________
void TGedAttAxisTitle::SetModel(TPad* pad, TObject* obj, Int_t)
{

   // Pick up the used values of axis title.
   
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAxis") || strlen(obj->GetTitle()) == 0) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   Color_t c = ExecuteInt(fModel, "GetTitleColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   fTitleColor->SetColor(p);

   Float_t size = ExecuteFloat(fModel, "GetTitleSize", "");
   fTitleSize->SetNumber(size);

   Style_t font = ExecuteInt(fModel, "GetTitleFont", "");
   fTitleFont->Select(font / 10);
   fPrecision = (Int_t)(font % 10);

   Float_t offset = ExecuteFloat(fModel, "GetTitleOffset", "");
   fTitleOffset->SetNumber(offset);

   Int_t centered = (Int_t)ExecuteInt(fModel, "GetCenterTitle", "");
   if (centered) fCentered->SetState(kButtonDown);
   else          fCentered->SetState(kButtonUp);
   
   Int_t rotated = (Int_t)ExecuteInt(fModel, "GetRotateTitle", "");
   if (rotated) fRotated->SetState(kButtonDown);
   else         fRotated->SetState(kButtonUp);

   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttAxisTitle::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process message.

   if(!fModel)
      return kTRUE;

   char a[100];
   bool b = false;

   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED) {

      snprintf(a, 100, "%d", TColor::GetColor(parm2));
      fModel->Execute("SetTitleColor", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }
   if (b) Update();

   return kTRUE;
}

//______________________________________________________________________________
void TGedAttAxisTitle::DoTitleSize()
{
   // Slot connected to the title font size.

   char a[100];
   snprintf(a, 100, "%f", (Float_t)fTitleSize->GetNumber());
   fModel->Execute("SetTitleSize", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisTitle::DoTitleFont()
{
   // Slot connected to the title font.

   char a[100];
   snprintf(a, 100, "%d", fTitleFont->GetSelected() * 10 + fPrecision);
   fModel->Execute("SetTitleFont", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisTitle::DoTitleOffset()
{
   // Slot connected to the title offset.

   char a[100];
   snprintf(a, 100, "%f", (Float_t)fTitleOffset->GetNumber());
   fModel->Execute("SetTitleOffset", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisTitle::DoTitleCentered()
{
   // Slot connected to centered title option.

   char a[100];
   snprintf(a, 100, "%d", fCentered->GetState());
   fModel->Execute("CenterTitle", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisTitle::DoTitleRotated()
{
   // Slot connected to the title rotation.

   char a[100];
   snprintf(a, 100, "%d", fRotated->GetState());
   fModel->Execute("RotateTitle", a, 0);
   Update();
}

//______________________________________________________________________________
TGedAttAxisLabel::TGedAttAxisLabel(const TGWindow *p, Int_t id, Int_t width,
                                   Int_t height, UInt_t options, Pixel_t back)
   : TGedAttFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of axis label attributes GUI.

   MakeTitle("Labels");

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fLabelColor = new TGColorSelect(f2, 0, kCOLOR);
   f2->AddFrame(fLabelColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fLabelColor->Associate(this);
   TGLabel *fSizeLbl = new TGLabel(f2, "Size:");                              
   f2->AddFrame(fSizeLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 
                                            1, 0, 1, 1));
   fLabelSize = new TGNumberEntry(f2, 0.05, 6, kAXIS_LBLSIZE, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 0., 1.);
   fLabelSize->Connect("ValueSet(Long_t)", "TGedAttAxisLabel", this, 
                       "DoLabelSize()");
   f2->AddFrame(fLabelSize, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fNoExponent = new TGCheckButton(f3, "NoExp", kAXIS_LBLEXP);
   fNoExponent->Connect("Toggled(Bool_t)","TGedAttAxisLabel",this,"DoNoExponent()");
   fNoExponent->SetToolTipText("Set no exponent labels if selected");
   f3->AddFrame(fNoExponent, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 
                                               3, 1, 8, 0));
   fLabelOffset = new TGNumberEntry(f3, 0.005, 6, kAXIS_LBLOFFSET, 
                                        TGNumberFormat::kNESRealThree,
                                        TGNumberFormat::kNEAAnyNumber, 
                                        TGNumberFormat::kNELLimitMinMax,-1.,1.);
   fLabelOffset->Connect("ValueSet(Long_t)", "TGedAttAxisLabel", this, 
                         "DoLabelOffset()");
   fLabelOffset->GetNumberEntry()->SetToolTipText("Labels' offset");
   f3->AddFrame(fLabelOffset, new TGLayoutHints(kLHintsLeft, 11, 1, 3, 0));
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   fLabelFont = new TGFontTypeComboBox(this, kFONT_STYLE);
   fLabelFont->Resize(137, 20);
   fLabelFont->Connect("Selected(Int_t)", "TGedAttAxisLabel", this, 
                       "DoLabelFont()"); 
   AddFrame(fLabelFont, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 0));
}

//______________________________________________________________________________
void TGedAttAxisLabel::SetModel(TPad* pad, TObject* obj, Int_t)
{

   // Pick up the used values of the axis labels.
   
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TAxis")) {
      SetActive(kFALSE);
      return;
   }

   fModel  = obj;
   fPad    = pad;

   Color_t c = ExecuteInt(fModel, "GetLabelColor", "");
   Pixel_t p = TColor::Number2Pixel(c);
   fLabelColor->SetColor(p);

   Float_t size = ExecuteFloat(fModel, "GetLabelSize", "");
   fLabelSize->SetNumber(size);

   Style_t font = ExecuteInt(fModel, "GetLabelFont", "");
   fLabelFont->Select(font / 10);
   fPrecision = (Int_t)(font % 10);

   Float_t offset = ExecuteFloat(fModel, "GetLabelOffset", "");
   fLabelOffset->SetNumber(offset);

   Int_t noexp = (Int_t)ExecuteInt(fModel, "GetNoExponent", "");
   if (noexp) fNoExponent->SetState(kButtonDown);
   else       fNoExponent->SetState(kButtonUp);

   SetActive();
}

//______________________________________________________________________________
Bool_t TGedAttAxisLabel::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{

   // Process message.

   if(!fModel)
      return kTRUE;

   char a[100];
   bool b = false;

   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED) {

      snprintf(a, 100, "%d", TColor::GetColor(parm2));
      fModel->Execute("SetLabelColor", a, 0);
      b = true;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }
   if (b) Update();

   return kTRUE;
}

//______________________________________________________________________________
void TGedAttAxisLabel::DoLabelSize()
{
   // Slot connected to the label size.

   char a[100];
   snprintf(a, 100, "%f", (Float_t)fLabelSize->GetNumber());
   fModel->Execute("SetLabelSize", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisLabel::DoLabelFont()
{
   // Slot connected to the label font.

   char a[100];
   snprintf(a, 100, "%d", fLabelFont->GetSelected() * 10 + fPrecision);
   fModel->Execute("SetLabelFont", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisLabel::DoLabelOffset()
{
   // Slot connected to the label offset.

   char a[100];
   snprintf(a, 100, "%f", (Float_t)fLabelOffset->GetNumber());
   fModel->Execute("SetLabelOffset", a, 0);
   Update();
}

//______________________________________________________________________________
void TGedAttAxisLabel::DoNoExponent()
{
   // Slot connected to the labels' exponent flag.

   char a[100];
   snprintf(a, 100, "%d", fNoExponent->GetState());
   fModel->Execute("SetNoExponent", a, 0);
   Update();
}

