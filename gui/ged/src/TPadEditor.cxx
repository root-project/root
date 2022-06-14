// @(#)root/ged:$Id$
// Author: Ilka Antcheva   24/06/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TPadEditor
 \ingroup ged

Editor of pad/canvas objects.
   color and fill style,
  'Edit' check box sets pad/canvad editable,
  'Crosshair' sets a cross hair on the pad,
  'Fixed aspect ratio' can be set when resizing the pad
  'TickX' and 'TickY' set ticks along the X and Y axis
  'GridX' and 'GridY' set a grid along the X and Y axis
   pad/canvas border size can be set if a sinken or a raised
   border mode is selected; no border mode can be set to

*/


#include "TPadEditor.h"
#include "TGedEditor.h"
#include "TGComboBox.h"
#include "TGButtonGroup.h"
#include "TGLabel.h"
#include "TCanvas.h"

ClassImp(TPadEditor);

enum EPadWid {
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


////////////////////////////////////////////////////////////////////////////////
/// Constructor of TPad editor GUI.

TPadEditor::TPadEditor(const TGWindow *p, Int_t width,
                       Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   fPadPointer = 0;

   MakeTitle("Pad/Canvas");

   fFixedAR = new TGCheckButton(this, "Fixed aspect ratio", kPAD_FAR);
   fFixedAR->SetToolTipText("Set fixed aspect ratio");
   AddFrame(fFixedAR, new TGLayoutHints(kLHintsTop, 4, 1, 2, 1));

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGCompositeFrame *f3 = new TGCompositeFrame(f2, 40, 20, kVerticalFrame);
   fCrosshair = new TGCheckButton(f3, "Crosshair", kPAD_CROSS);
   fCrosshair->SetToolTipText("Set crosshair");
   f3->AddFrame(fCrosshair, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fGridX = new TGCheckButton(f3, "GridX", kPAD_GRIDX);
   fGridX->SetToolTipText("Set grid along X");
   f3->AddFrame(fGridX, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fTickX = new TGCheckButton(f3, "TickX", kPAD_TICKX);
   fTickX->SetToolTipText("Set tick marks along X");
   f3->AddFrame(fTickX, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   f2->AddFrame(f3, new TGLayoutHints(kLHintsTop, 0, 1, 0, 0));

   TGCompositeFrame *f4 = new TGCompositeFrame(f2, 40, 20, kVerticalFrame);
   fEditable = new TGCheckButton(f4, "Edit", kPAD_EDIT);
   fEditable->SetToolTipText("Set editable mode");
   f4->AddFrame(fEditable, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fGridY = new TGCheckButton(f4, "GridY", kPAD_GRIDY);
   fGridY->SetToolTipText("Set grid along Y");
   f4->AddFrame(fGridY, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   fTickY = new TGCheckButton(f4, "TickY", kPAD_TICKY);
   fTickY->SetToolTipText("Set tick marks along Y");
   f4->AddFrame(fTickY, new TGLayoutHints(kLHintsTop, 3, 1, 1, 1));
   f2->AddFrame(f4, new TGLayoutHints(kLHintsTop, 0, 1, 0, 0));

   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

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
   fBgroup = new TGButtonGroup(f6,3,1,3,0, "Border Mode");
   fBgroup->SetRadioButtonExclusive(kTRUE);
   fBmode = new TGRadioButton(fBgroup, " Sunken border", 77);
   fBmode->SetToolTipText("Set a sinken border of the pad/canvas");
   fBmode0 = new TGRadioButton(fBgroup, " No border", 78);
   fBmode0->SetToolTipText("Set no border of the pad/canvas");
   fBmode1 = new TGRadioButton(fBgroup, " Raised border", 79);
   fBmode1->SetToolTipText("Set a raised border of the pad/canvas");
   fBmodelh = new TGLayoutHints(kLHintsLeft, 0,0,3,0);
   fBgroup->SetLayoutHints(fBmodelh, fBmode);
   fBgroup->ChangeOptions(kFitWidth|kChildFrame|kVerticalFrame);
   f6->AddFrame(fBgroup, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4, 1, 0, 0));
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   TGCompositeFrame *f7 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fSizeLbl = new TGLabel(f7, "Size:");
   f7->AddFrame(fSizeLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 6, 1, 0, 0));
   fBsize = new TGLineWidthComboBox(f7, kPAD_BSIZE);
   fBsize->Resize(92, 20);
   f7->AddFrame(fBsize, new TGLayoutHints(kLHintsLeft, 13, 1, 0, 0));
   fBsize->Associate(this);
   AddFrame(f7, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   fInit = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor of fill editor.

TPadEditor::~TPadEditor()
{
   // children of TGButonGroup are not deleted
   delete fBmode;
   delete fBmode0;
   delete fBmode1;
   delete fBmodelh;
}

////////////////////////////////////////////////////////////////////////////////
/// Connect signals to slots.

void TPadEditor::ConnectSignals2Slots()
{
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
   fBgroup->Connect("Clicked(Int_t)","TPadEditor",this,"DoBorderMode()");
   fBsize->Connect("Selected(Int_t)", "TPadEditor", this, "DoBorderSize(Int_t)");
   fInit = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Pick up the used fill attributes.

void TPadEditor::SetModel(TObject* obj)
{
   if (!obj || !obj->InheritsFrom("TPad"))
      return;
   fPadPointer = (TPad *)obj;
   fAvoidSignal = kTRUE;
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
   if (par == -1) {
      fBgroup->SetButton(77, kTRUE);
      fBsize->SetEnabled(kTRUE);
   } else if (par == 1) {
      fBgroup->SetButton(79, kTRUE);
      fBsize->SetEnabled(kTRUE);
   } else {
      fBgroup->SetButton(78, kTRUE);
      fBsize->SetEnabled(kFALSE);
   }
   par = fPadPointer->GetBorderSize();
   if (par < 1) par = 1;
   if (par > 16) par = 16;
   fBsize->Select(par);

   if (fInit) ConnectSignals2Slots();

   fAvoidSignal = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Exclude TAttLineEditor from this interface.

void TPadEditor::ActivateBaseClassEditors(TClass* cl)
{
   fGedEditor->ExcludeClassEditor(TAttLine::Class());
   TGedFrame::ActivateBaseClassEditors(cl);
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'Editable'.

void TPadEditor::DoEditable(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetEditable(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'Crosshair'.

void TPadEditor::DoCrosshair(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetCrosshair(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'Fixed aspect ratio'.

void TPadEditor::DoFixedAspectRatio(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetFixedAspectRatio(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'GridX'.

void TPadEditor::DoGridX(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetGridx(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'GridY'.

void TPadEditor::DoGridY(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetGridy(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'LogX'.

void TPadEditor::DoLogX(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetLogx(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'LogY'.

void TPadEditor::DoLogY(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetLogy(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'LogZ'.

void TPadEditor::DoLogZ(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetLogz(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'TickX'.

void TPadEditor::DoTickX(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetTickx(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the check box 'TickY'.

void TPadEditor::DoTickY(Bool_t on)
{
   if (fAvoidSignal) return;
   fPadPointer->SetTicky(on);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the border mode settings.

void TPadEditor::DoBorderMode()
{
   if (fAvoidSignal) return;
   Int_t mode = 0;
   if (fBmode->GetState() == kButtonDown) mode = -1;
   else if (fBmode0->GetState() == kButtonDown) mode = 0;
   else mode = 1;

   if (!mode) {
      fBsize->SetEnabled(kFALSE);
   } else {
      fBsize->SetEnabled(kTRUE);
   }
   fPadPointer->SetBorderMode(mode);
   Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot connected to the border size settings.

void TPadEditor::DoBorderSize(Int_t size)
{
   if (fAvoidSignal) return;
   fPadPointer->SetBorderSize(size);
   Update();
}
