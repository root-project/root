// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldEditor.cxx,v 1.6 2004/10/07 09:56:53 rdm Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldEditor                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiBldEditor.h"
#include "TGuiBldHintsEditor.h"
#include "TGResourcePool.h"
#include "TGTab.h"
#include "TGLabel.h"
#include "TGButtonGroup.h"
#include "TGNumberEntry.h"
#include "TG3DLine.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"


ClassImp(TGuiBldEditor)


////////////////////////////////////////////////////////////////////////////////
class TGuiBldNameFrame : public TGCompositeFrame {

private:
   TGLabel         *fLabel;
   TGTextEntry     *fFrameName;
   TGuiBldEditor   *fEditor;

public:
   TGuiBldNameFrame(const TGWindow *p, TGuiBldEditor *editor);
   virtual ~TGuiBldNameFrame() { }

   void ChangeSelected(TGFrame *frame);
};

//______________________________________________________________________________
TGuiBldNameFrame::TGuiBldNameFrame(const TGWindow *p, TGuiBldEditor *editor) :
                  TGCompositeFrame(p, 1, 1)
{
   //

   fEditor = editor;
   fEditDisabled = kTRUE;
   SetCleanup(kDeepCleanup);
   TGFrame *frame = fEditor->GetSelected();

   TGCompositeFrame *f = new TGHorizontalFrame(this);
   f->AddFrame(new TGLabel(f, "Name"), new TGLayoutHints(kLHintsLeft, 0, 1, 0, 0));
   f->AddFrame(new TGHorizontal3DLine(f), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   f = new TGHorizontalFrame(this);
   AddFrame(f, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 1, 1, 0, 0));

   TString name = "";
   if (frame) {
      frame->ClassName();
      name += "::";
   }

   fLabel = new TGLabel(f, name.Data());
   f->AddFrame(fLabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 10, 1, 0, 0));
   fFrameName = new TGTextEntry(f, frame ? frame->GetName() : "noname");
   fFrameName->SetAlignment(kTextLeft);
   fFrameName->Resize(80, fFrameName->GetHeight());
   f->AddFrame(fFrameName, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,1));
   fFrameName->SetEnabled(kFALSE);

   //Pixel_t color;
   //fClient->GetColorByName("#ff0000", color);
   //fLabel->SetTextColor(color, kFALSE);
   //fFrameName->SetTextColor(color, kFALSE);
}

//______________________________________________________________________________
void TGuiBldNameFrame::ChangeSelected(TGFrame *frame)
{
   //

   fFrameName->Disconnect();

   if (!frame) return;

   TString name = frame->ClassName();
   name += "::";

   fLabel->SetText(name.Data());
   fFrameName->SetText(frame->GetName());
   fFrameName->Connect("TextChanged(char*)", frame->ClassName(), frame, "SetName(char*)");
   Resize();
   DoRedraw();
}

////////////////////////////////////////////////////////////////////////////////
class TGuiBldGeometryFrame : public TGVerticalFrame {

private:
   TGuiBldEditor   *fEditor;

public:
   TGuiBldGeometryFrame(const TGWindow *p, TGuiBldEditor *editor);
   virtual ~TGuiBldGeometryFrame() { }

};

//______________________________________________________________________________
TGuiBldGeometryFrame::TGuiBldGeometryFrame(const TGWindow *p, TGuiBldEditor *editor) :
                        TGVerticalFrame(p, 1, 1)
{
   //

   fEditor = editor;
   fEditDisabled = kTRUE;
   SetCleanup(kDeepCleanup);

   TGCompositeFrame *f = new TGHorizontalFrame(this);
   f->AddFrame(new TGLabel(f, "Geometry"), new TGLayoutHints(kLHintsNormal, 1, 1));
   f->AddFrame(new TGHorizontal3DLine(f), new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 5, 5));
   AddFrame(f, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   // composite frame
   TGCompositeFrame *frame826 = new TGCompositeFrame(this,275,69,kHorizontalFrame);

   // vertical frame
   TGVerticalFrame *frame928 = new TGVerticalFrame(frame826,59,64,kVerticalFrame);
   TGLabel *frame834 = new TGLabel(frame928,"      ");
   frame928->AddFrame(frame834, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,2,2,2,2));
   TGLabel *frame833 = new TGLabel(frame928,"Width");
   frame928->AddFrame(frame833,  new TGLayoutHints(kLHintsLeft | kLHintsCenterY,5,3,2,2));
   TGLabel *frame839 = new TGLabel(frame928,"Height");
   frame928->AddFrame(frame839, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,5,3,2,2));
   frame826->AddFrame(frame928);
   frame928->MoveResize(2,2,59,64);

   // vertical frame
   TGVerticalFrame *frame24 = new TGVerticalFrame(frame826,68,67,kVerticalFrame);
   TGLabel *frame25 = new TGLabel(frame24,"      ");
   frame24->AddFrame(frame25, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   TGNumberEntry *frame26 = new TGNumberEntry(frame24, (Double_t) 0,5,-1,(TGNumberFormat::EStyle) 0);
   frame24->AddFrame(frame26, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   TGNumberEntry *frame30 = new TGNumberEntry(frame24, (Double_t) 0,5,-1,(TGNumberFormat::EStyle) 0);
   frame24->AddFrame(frame30, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   frame826->AddFrame(frame24);
   frame24->MoveResize(65,2,68,67);

   // vertical frame
   TGVerticalFrame *frame14 = new TGVerticalFrame(frame826,68,67,kVerticalFrame);
   TGLabel *frame15 = new TGLabel(frame14,"Min");
   frame14->AddFrame(frame15, new TGLayoutHints(kLHintsLeft | kLHintsTop | kLHintsCenterX,0,0,1,0));
   TGNumberEntry *frame16 = new TGNumberEntry(frame14, (Double_t) 0,5,-1,(TGNumberFormat::EStyle) 0);
   frame14->AddFrame(frame16, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   TGNumberEntry *frame20 = new TGNumberEntry(frame14, (Double_t) 0,5,-1,(TGNumberFormat::EStyle) 0);
   frame14->AddFrame(frame20, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   frame826->AddFrame(frame14);
   frame14->MoveResize(137,2,68,67);

   // vertical frame
   TGVerticalFrame *frame4 = new TGVerticalFrame(frame826,68,67,kVerticalFrame);
   TGLabel *frame5 = new TGLabel(frame4,"Max");
   frame4->AddFrame(frame5, new TGLayoutHints(kLHintsLeft | kLHintsTop  | kLHintsCenterX,0,0,1,0));
   TGNumberEntry *frame6 = new TGNumberEntry(frame4, (Double_t) 0,5,-1,(TGNumberFormat::EStyle) 0);
   frame4->AddFrame(frame6, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   TGNumberEntry *frame10 = new TGNumberEntry(frame4, (Double_t) 0,5,-1,(TGNumberFormat::EStyle) 0);
   frame4->AddFrame(frame10, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));
   frame826->AddFrame(frame4);
   frame4->MoveResize(207,0,68,67);

   AddFrame(frame826, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,1,0));

   MapSubwindows();
   Resize();
}

////////////////////////////////////////////////////////////////////////////////
class TGuiBldBorderFrame : public TGHorizontalFrame {

private:
 enum  EBldBorderFrameMode { kBldBorderNone, kBldBorderSunken,
                             kBldBorderPlain, kBldBorderRaised, kBldBorderDouble };

private:
   TGuiBldEditor   *fEditor;
   TGFrame         *fSelected;
   TGButtonGroup   *fBtnGroup;
   TGColorSelect   *fBgndFrame;
   TGColorSelect   *fFgndFrame;

public:
   TGuiBldBorderFrame(const TGWindow *p, TGuiBldEditor *editor);
   virtual ~TGuiBldBorderFrame() { }

   void  ChangeSelected(TGFrame*);
};

//______________________________________________________________________________
TGuiBldBorderFrame::TGuiBldBorderFrame(const TGWindow *p, TGuiBldEditor *editor) :
             TGHorizontalFrame(p, 1, 1)
{
   //

   fEditor = editor;
   fEditDisabled = kTRUE;
   fBgndFrame = 0;
   fFgndFrame = 0;

   SetCleanup(kDeepCleanup);

   fBtnGroup = new TGButtonGroup(this,"Border Mode",kVerticalFrame | kFitWidth);

   TGRadioButton *frame299 = new TGRadioButton(fBtnGroup," Sunken",kBldBorderSunken);
   frame299->SetToolTipText("Set a sunken border of the frame");
   TGRadioButton *frame302 = new TGRadioButton(fBtnGroup," Plain",kBldBorderPlain);
   frame302->SetToolTipText("Set no border of the frame");
   TGRadioButton *frame305 = new TGRadioButton(fBtnGroup," Raised",kBldBorderRaised);
   frame305->SetState(kButtonDown);
   frame305->SetToolTipText("Set a raised border of the frame");
   TGCheckButton *frame300 = new TGCheckButton(fBtnGroup," Double",kBldBorderDouble);
   frame300->SetToolTipText("Set double border of the frame");

   fBtnGroup->SetRadioButtonExclusive(kTRUE);
   fBtnGroup->Resize(136,86);
   AddFrame(fBtnGroup);
   fBtnGroup->Connect("Pressed(Int_t)", "TGuiBldEditor", fEditor, "UpdateBorder(Int_t)");
   frame300->Connect("Pressed()", "TGuiBldEditor", fEditor, "UpdateBorder(=4)");
   frame300->Connect("Released()", "TGuiBldEditor", fEditor, "UpdateBorder(=5)");
/*
   TGCompositeFrame *f = new TGGroupFrame(this,"Palette",kVerticalFrame | kFitWidth);
   TGHorizontalFrame *hf = new TGHorizontalFrame(f ,1, 1);
   f->AddFrame(hf);
   fBgndFrame = new TGColorSelect(hf, 0, 1);
   fBgndFrame->SetEditDisabled();
   fBgndFrame->SetColor(GetDefaultFrameBackground());
   fBgndFrame->Connect("ColorSelected(Pixel_t)", "TGuiBldEditor", fEditor, "UpdateBackground(Pixel_t)");
   hf->AddFrame(fBgndFrame);
   TGLabel *bl = new TGLabel(hf, "Backgrnd");
   hf->AddFrame(bl);

   hf = new TGHorizontalFrame(f ,1, 1);
   f->AddFrame(hf);
   fFgndFrame = new TGColorSelect(hf, 0, 1);
   fFgndFrame->SetEditDisabled();
   fFgndFrame->SetColor(GetBlackPixel());
   fFgndFrame->Connect("ColorSelected(Pixel_t)", "TGuiBldEditor", fEditor, "UpdateForeground(Pixel_t)");
   hf->AddFrame(fFgndFrame);
   bl = new TGLabel(hf, "Foregrnd");
   hf->AddFrame(bl);

   f->Resize(44,86);
   AddFrame(f);
*/
}

//______________________________________________________________________________
void TGuiBldBorderFrame::ChangeSelected(TGFrame *frame)
{
   //

   fSelected = frame;

   UInt_t opt = fSelected->GetOptions();

   fBtnGroup->SetButton(kBldBorderDouble, opt & kDoubleBorder);
   fBtnGroup->SetButton(kBldBorderSunken, opt & kSunkenFrame);
   fBtnGroup->SetButton(kBldBorderRaised, opt & kRaisedFrame);
   fBtnGroup->SetButton(kBldBorderPlain, !(opt & kRaisedFrame) && !(opt & kSunkenFrame));

   if (fBgndFrame) fBgndFrame->SetColor(fSelected->GetBackground());
   if (fFgndFrame) fFgndFrame->SetColor(fSelected->GetForeground());
}

////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGuiBldEditor::TGuiBldEditor(const TGWindow *p) : TGCompositeFrame(p, 1, 1)
{
   //

   fSelected = 0;
   fEditDisabled = kTRUE;

   SetCleanup(kDeepCleanup);

   TGTab *tab = new TGTab(this, 80, 40);

   AddFrame(tab, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));
   TGCompositeFrame *tabcont = tab->AddTab("Properties");

   fNameFrame  = new TGuiBldNameFrame(tabcont, this);
   tabcont->AddFrame(fNameFrame,  new TGLayoutHints(kLHintsNormal | kLHintsExpandX,2,2,2,2));

   fHintsFrame = 0;

   fHintsFrame = new TGuiBldHintsEditor(tabcont, this);
   tabcont->AddFrame(fHintsFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,2,2,2,2));

   //TGFrame *frame = new TGuiBldGeometryFrame(tabcont, this);
   //tabcont->AddFrame(frame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,2,2,2,2));

   fBorderFrame = new TGuiBldBorderFrame(tabcont, this);
   tabcont->AddFrame(fBorderFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX,2,2,2,2));

   MapSubwindows();
   Resize(239, 357);
   SetWindowName("Frame Property Editor");

   fEmbedded = kFALSE;
}

//______________________________________________________________________________
TGuiBldEditor::~TGuiBldEditor()
{
   //

}

//______________________________________________________________________________
void  TGuiBldEditor::Hide()
{
   //

   UnmapWindow();
}

//______________________________________________________________________________
void  TGuiBldEditor::ChangeSelected(TGFrame *frame)
{
   //

   if (!frame) return;

   fSelected = frame;

   if (fNameFrame) fNameFrame->ChangeSelected(fSelected);

   if (frame->GetFrameElement()) {
      if (fHintsFrame) {
         fHintsFrame->MapWindow();
         fHintsFrame->ChangeSelected(fSelected);
      }
   } else {
      if (fHintsFrame) fHintsFrame->UnmapWindow();
   }

   if (fBorderFrame) fBorderFrame->ChangeSelected(fSelected);

   Emit("ChangeSelected(TGFrame*)", (long)fSelected);

   Resize(GetDefaultSize());
   MapRaised();
}

//______________________________________________________________________________
void  TGuiBldEditor::UpdateSelected(TGFrame *)
{
   //

   Emit("UpdateSelected(TGFrame*)", (long)fSelected);
}

//______________________________________________________________________________
void  TGuiBldEditor::UpdateBorder(Int_t b)
{
   //

   if (!fSelected) return;

   UInt_t opt = fSelected->GetOptions();

   switch (b) {
   case 1:
      opt &= ~kRaisedFrame;
      opt |= kSunkenFrame;
      break;
   case 2:
      opt &= ~kSunkenFrame;
      opt &= ~kRaisedFrame;
      break;
   case 3:
      opt &= ~kSunkenFrame;
      opt |= kRaisedFrame;
      break;
   case 4:
      opt |= kDoubleBorder;
      break;
   case 5:
      opt &= ~kDoubleBorder;
      break;
   default:
      return;
   }

   fSelected->ChangeOptions(opt);
   fClient->NeedRedraw(fSelected, kTRUE);
}

//______________________________________________________________________________
void  TGuiBldEditor::UpdateBackground(Pixel_t col)
{
   //

   if (!fSelected) return;

   fSelected->SetBackgroundColor(col);
   fClient->NeedRedraw(fSelected, kTRUE);
}

//______________________________________________________________________________
void  TGuiBldEditor::UpdateForeground(Pixel_t col)
{
   //

   if (!fSelected) return;

   fSelected->SetForegroundColor(col);
   fClient->NeedRedraw(fSelected, kTRUE);
}

