// @(#)root/ged:$Name:  TPadEditor.cxx
// Author: Ilka Antcheva   24/06/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TPadEditor                                                          //
//                                                                      //
//  Editor of pad/canvas objects.                                       //
//       color and fill style,                                          //             
//      'Edit' check box sets pad/canvad editable,                      //
//      'Crosshair' sets a cross hair on the pad,                       //
//      'Fixed aspect ratio' can be set when resizing the pad           //
//      'TickX' and 'TickY' set ticks along the X and Y axis            //
//      'GridX' and 'GridY' set a grid along the X and Y axis           //
//       pad/canvas border size can be set if a sinken or a raised      //
//       border mode is selected; no border mode can be set to          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TPadEditor.gif">
*/
//End_Html

#include "TPadEditor.h"
#include "TGClient.h"
#include "TGButton.h"
#include "TGComboBox.h"
#include "TGButtonGroup.h"
#include "TGLabel.h"
#include "TColor.h"
#include "TCanvas.h"

ClassImp(TGedFrame)
ClassImp(TPadEditor)

enum {
   kCOLOR,
   kPAD_FAR,
   kPAD_EDIT,
   kPAD_CROSS,
   kPAD_GRIDX,
   kPAD_GRIDY,
   kPAD_LOGX,
   kPAD_LOGY,
   kPAD_LOGZ,
   kPAD_TICKX,
   kPAD_TICKY,
   kPAD_BSIZE,
   kPAD_BMODE
};


//______________________________________________________________________________
TPadEditor::TPadEditor(const TGWindow *p, Int_t id, Int_t width,
                       Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of TPad editor GUI.

   fPadPointer = 0;
   
   MakeTitle("Pad/Canvas");

   fFixedAR = new TGCheckButton(this, "Fixed aspect ratio", kPAD_FAR);
   fFixedAR->SetToolTipText("Set fixed aspect ratio");
   AddFrame(fFixedAR, new TGLayoutHints(kLHintsTop, 4, 1, 2, 1));

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fCrosshair = new TGCheckButton(f2, "Crosshair", kPAD_CROSS);
   fCrosshair->SetToolTipText("Set crosshair");
   f2->AddFrame(fCrosshair, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fEditable = new TGCheckButton(f2, "Edit", kPAD_EDIT);
   fEditable->SetToolTipText("Set editable mode");
   f2->AddFrame(fEditable, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fGridX = new TGCheckButton(f3, "GridX", kPAD_GRIDX);
   fGridX->SetToolTipText("Set grid along X");
   f3->AddFrame(fGridX, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fGridY = new TGCheckButton(f3, "GridY", kPAD_GRIDY);
   fGridY->SetToolTipText("Set grid along Y");
   f3->AddFrame(fGridY, new TGLayoutHints(kLHintsTop, 24, 1, 1, 1));
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fTickX = new TGCheckButton(f4, "TickX", kPAD_TICKX);
   fTickX->SetToolTipText("Set tick marks along X");
   f4->AddFrame(fTickX, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fTickY = new TGCheckButton(f4, "TickY", kPAD_TICKY);
   fTickY->SetToolTipText("Set tick marks along Y");
   f4->AddFrame(fTickY, new TGLayoutHints(kLHintsTop, 24, 1, 1, 1));
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 0, 2));

   MakeTitle("Log Scale");

   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fLogX = new TGCheckButton(f5, ":X", kPAD_LOGX);
   fLogX->SetToolTipText("Set logarithmic scale along X");
   f5->AddFrame(fLogX, new TGLayoutHints(kLHintsTop, 4, 1, 1, 1));
   fLogY = new TGCheckButton(f5, ":Y", kPAD_LOGY);
   fLogY->SetToolTipText("Set logarithmic scale along Y");
   f5->AddFrame(fLogY, new TGLayoutHints(kLHintsTop, 15, 1, 1, 1));
   fLogZ = new TGCheckButton(f5, ":Z", kPAD_LOGZ);
   fLogZ->SetToolTipText("Set logarithmic scale along Z");
   f5->AddFrame(fLogZ, new TGLayoutHints(kLHintsTop, 15, 1, 1, 1));
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGButtonGroup *bgr = new TGButtonGroup(f6,3,1,3,0, "Border Mode");
   bgr->SetRadioButtonExclusive(kTRUE);
   fBmode = new TGRadioButton(bgr, " Sunken border", 77);
   fBmode->SetToolTipText("Set a sinken border of the pad/canvas");
   fBmode0 = new TGRadioButton(bgr, " No border", 78);
   fBmode0->SetToolTipText("Set no border of the pad/canvas");
   fBmode1 = new TGRadioButton(bgr, " Raised border", 79);
   fBmode1->SetToolTipText("Set a raised border of the pad/canvas");
   bgr->SetButton(79, kTRUE);
   bgr->SetLayoutHints(new TGLayoutHints(kLHintsLeft, 0,0,3,0), fBmode);
   bgr->Show();
   bgr->ChangeOptions(kFitWidth|kChildFrame|kVerticalFrame);
   f6->AddFrame(bgr, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4, 1, 0, 0));
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   f7 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fSizeLbl = new TGLabel(f7, "Size:");                              
   f7->AddFrame(fSizeLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 6, 1, 0, 0));
   fBsize = new TGLineWidthComboBox(f7, kPAD_BSIZE);
   fBsize->Connect("Selected(Int_t)", "TPadEditor", this, "DoBorderSize(Int_t)"); 
   fBsize->Resize(92, 20);
   f7->AddFrame(fBsize, new TGLayoutHints(kLHintsLeft, 13, 1, 0, 0));
   fBsize->Associate(this);
   AddFrame(f7, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TPad::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
   
   fInit = kTRUE;
}

//______________________________________________________________________________
TPadEditor::~TPadEditor()
{ 
   // Destructor of fill editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup(); 
}

//______________________________________________________________________________
void TPadEditor::ConnectSignals2Slots()
{ 
   // Connect signals to slots.

   fFixedAR->Connect("Toggled(Bool_t)","TPadEditor",this,"DoFixedAspectRatio(Bool_t)");
   fCrosshair->Connect("Toggled(Bool_t)","TPadEditor",this,"DoCrosshair(Bool_t)");
   fEditable->Connect("Toggled(Bool_t)","TPadEditor",this,"DoEditable(Bool_t)");
   fGridX->Connect("Toggled(Bool_t)","TPadEditor",this,"DoGridX(Bool_t)");
   fGridY->Connect("Toggled(Bool_t)","TPadEditor",this,"DoGridY(Bool_t)");
   fTickX->Connect("Toggled(Bool_t)","TPadEditor",this,"DoTickX(Bool_t)");
   fTickY->Connect("Toggled(Bool_t)","TPadEditor",this,"DoTickY(Bool_t)");
   fLogX->Connect("Toggled(Bool_t)","TPadEditor",this,"DoLogX(Bool_t)");
   fLogY->Connect("Toggled(Bool_t)","TPadEditor",this,"DoLogY(Bool_t)");
   fLogZ->Connect("Toggled(Bool_t)","TPadEditor",this,"DoLogZ(Bool_t)");
   fBmode->Connect("Toggled(Bool_t)","TPadEditor",this,"DoBorderMode()");
   fBmode0->Connect("Toggled(Bool_t)","TPadEditor",this,"DoBorderMode()");
   fBmode1->Connect("Toggled(Bool_t)","TPadEditor",this,"DoBorderMode()");
   
   fInit = kFALSE;
}

//______________________________________________________________________________
void TPadEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used fill attributes.

   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TPad")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   fPadPointer = (TPad *)fModel;
   
   Bool_t on;

   on = fPadPointer->HasFixedAspectRatio();
   if (on) fFixedAR->SetState(kButtonDown);
   else fFixedAR->SetState(kButtonUp);
   
   on = fPadPointer->HasCrosshair();
   if (on) fCrosshair->SetState(kButtonDown);
   else fCrosshair->SetState(kButtonUp);
   
   on = fPadPointer->IsEditable();
   if (on) fEditable->SetState(kButtonDown);
   else fEditable->SetState(kButtonUp);

   on = fPadPointer->GetGridx();
   if (on) fGridX->SetState(kButtonDown);
   else fGridX->SetState(kButtonUp);
   
   on = fPadPointer->GetGridy();
   if (on) fGridY->SetState(kButtonDown);
   else fGridY->SetState(kButtonUp);

   Int_t par;
   par = fPadPointer->GetLogx();
   if (par) fLogX->SetState(kButtonDown);
   else fLogX->SetState(kButtonUp);
   
   par = fPadPointer->GetLogy();
   if (par) fLogY->SetState(kButtonDown);
   else fLogY->SetState(kButtonUp);

   par = fPadPointer->GetLogz();
   if (par) fLogZ->SetState(kButtonDown);
   else fLogZ->SetState(kButtonUp);

   par = fPadPointer->GetTickx();
   if (par) fTickX->SetState(kButtonDown);
   else fTickX->SetState(kButtonUp);

   par = fPadPointer->GetTicky();
   if (par) fTickY->SetState(kButtonDown);
   else fTickY->SetState(kButtonUp);

   par = fPadPointer->GetBorderMode();
   if (par == -1) fBmode->SetState(kButtonDown);
   else if (par == 1) fBmode1->SetState(kButtonDown);
   else fBmode0->SetState(kButtonDown);

   par = fPadPointer->GetBorderSize();
   if (par < 1) par = 1;
   if (par > 16) par = 16;
   fBsize->Select(par);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TPadEditor::DoEditable(Bool_t on)
{
   // Slot connected to the check box 'Editable'.

   fPadPointer->SetEditable(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoCrosshair(Bool_t on)
{
   // Slot connected to the check box 'Crosshair'.

   fPadPointer->SetCrosshair(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoFixedAspectRatio(Bool_t on)
{
   // Slot connected to the check box 'Fixed aspect ratio'.

   fPadPointer->SetFixedAspectRatio(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoGridX(Bool_t on)
{
   // Slot connected to the check box 'GridX'.

   fPadPointer->SetGridx(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoGridY(Bool_t on)
{
   // Slot connected to the check box 'GridY'.

   fPadPointer->SetGridy(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoLogX(Bool_t on)
{
   // Slot connected to the check box 'LogX'.

   fPadPointer->SetLogx(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoLogY(Bool_t on)
{
   // Slot connected to the check box 'LogY'.

   fPadPointer->SetLogy(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoLogZ(Bool_t on)
{
   // Slot connected to the check box 'LogZ'.

   fPadPointer->SetLogz(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoTickX(Bool_t on)
{
   // Slot connected to the check box 'TickX'.

   fPadPointer->SetTickx(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoTickY(Bool_t on)
{
   // Slot connected to the check box 'TickY'.

   fPadPointer->SetTicky(on);
   Update();
}

//______________________________________________________________________________
void TPadEditor::DoBorderMode()
{
   // Slot connected to the border mode settings.
   
   Int_t mode = 0;
   if (fBmode->GetState() == kButtonDown) mode = -1;
   else if (fBmode0->GetState() == kButtonDown) mode = 0;
   else mode = 1;

   if (!mode) HideFrame(f7);
   else ShowFrame(f7);
   Layout();
   
   fPadPointer->SetBorderMode(mode);
   Update();
   gPad->Modified();
   gPad->Update();
}

//______________________________________________________________________________
void TPadEditor::DoBorderSize(Int_t size)
{
   // Slot connected to the border size settings.
   
   fPadPointer->SetBorderSize(size);
   Update();
}
