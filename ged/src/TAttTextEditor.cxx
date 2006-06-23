// @(#)root/ged:$Name:  $:$Id: TAttTextEditor.cxx,v 1.8 2006/03/20 21:43:41 pcanal Exp $
// Author: Ilka Antcheva   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TAttTextEditor                                                      //
//                                                                      //
//  Implements GUI for editing text attributes.                         //
//          color, font, font size and alignment                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TAttTextEditor.gif">
*/
//End_Html


#include "TAttTextEditor.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"
#include "TGComboBox.h"
#include "TGClient.h"
#include "TColor.h"
#include "TPaveLabel.h"
#include "TVirtualPad.h"
#include "snprintf.h"
#include "TClass.h"

ClassImp(TAttTextEditor)

enum ETextWid {
   kCOLOR,
   kFONT_SIZE,
   kFONT_STYLE,
   kFONT_ALIGN
};

//______________________________________________________________________________
TAttTextEditor::TAttTextEditor(const TGWindow *p, Int_t id, Int_t width,
                               Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of text attributes GUI.

   fAttText = 0;

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

   TClass *cl = TAttText::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);

}

//______________________________________________________________________________
TAttTextEditor::~TAttTextEditor()
{
   // Destructor of text editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________
void TAttTextEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the values of used text attributes.

   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom(TAttText::Class())) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   fAttText = dynamic_cast<TAttText *>(fModel);
   fAvoidSignal = kTRUE;

   fTypeCombo->Select(fAttText->GetTextFont() / 10);

   Float_t s = fAttText->GetTextSize();
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
   fSizeCombo->Select(size, kFALSE);

   fAlignCombo->Select(fAttText->GetTextAlign(), kFALSE);

   Color_t c = fAttText->GetTextColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fColorSelect->SetColor(p, kFALSE);

   SetActive();
   fAvoidSignal = kFALSE;
}

//______________________________________________________________________________
Bool_t TAttTextEditor::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process message.

   if(!fModel) return kTRUE;

   Bool_t b = kFALSE;

   if (GET_MSG(msg) == kC_COLORSEL && GET_SUBMSG(msg) == kCOL_SELCHANGED) {
      fAttText->SetTextColor(TColor::GetColor(parm2));
      b = kTRUE;
      SendMessage(fMsgWindow, msg, parm1, parm2);
   }

   if (GET_MSG(msg) == kC_COMMAND && GET_SUBMSG(msg) == kCM_COMBOBOX) {

      if (parm1 == kFONT_SIZE) {

         Float_t dy = fPad->AbsPixeltoY(0) - fPad->AbsPixeltoY(parm2);
         Float_t textSize;

         if (fModel->InheritsFrom("TPaveLabel")) {
            TBox *pl = (TBox*) fModel;
            textSize = dy/(pl->GetY2() - pl->GetY1());
         }
         else
            textSize = dy/(fPad->GetY2() - fPad->GetY1());

         fAttText->SetTextSize(textSize);
         
         b = kTRUE;
      } else if (parm1 == kFONT_STYLE) {
         Int_t fontPrec = fAttText->GetTextFont()%10;
         fAttText->SetTextFont(parm2 * 10 + fontPrec);
         b = kTRUE;
      } else if (parm1 == kFONT_ALIGN) {
         fAttText->SetTextAlign(parm2);
         b = true;
      }
   }

   if (b && !fAvoidSignal) Update();

   return kTRUE;
}

//______________________________________________________________________________
TGComboBox* TAttTextEditor::BuildFontSizeComboBox(TGFrame* parent, Int_t id)
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
TGComboBox* TAttTextEditor::BuildTextAlignComboBox(TGFrame* parent, Int_t id)
{
   // Create text align combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("11 Bottom, Left",   11);
   c->AddEntry("21 Bottom, Middle", 21);
   c->AddEntry("31 Bottom, Right",  31);
   c->AddEntry("12 Middle, Left",   12);
   c->AddEntry("22 Middle, Middle", 22);
   c->AddEntry("32 Middle, Right",  32);
   c->AddEntry("13 Top, Left",   13);
   c->AddEntry("23 Top, Middle", 23);
   c->AddEntry("33 Top, Right",  33);

   return c;
}
