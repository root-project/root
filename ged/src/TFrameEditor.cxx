// @(#)root/ged:$Name:  TFrameEditor.cxx
// Author: Ilka Antcheva   08/03/05

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TFrameEditor                                                        //
//                                                                      //
//  Editor of frame objects.                                            //
//                                                                      //
//  Frame border can be set to sunken, raised or no border.             //
//  Border size can be set for sunken or rized frames (1-15 pixels).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TFrameEditor.gif">
*/
//End_Html

#include "TFrameEditor.h"
#include "TGClient.h"
#include "TGButton.h"
#include "TGComboBox.h"
#include "TGButtonGroup.h"
#include "TGLabel.h"
#include "TFrame.h"
#include "TVirtualPad.h"

ClassImp(TFrameEditor)

enum {
   kFR_BSIZE,
   kFR_BMODE
};


//______________________________________________________________________________
TFrameEditor::TFrameEditor(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of TFrame editor GUI.

   fFrame = 0;
   
   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGButtonGroup *bgr = new TGButtonGroup(f2,3,1,3,0, "Frame Border Mode");
   bgr->SetRadioButtonExclusive(kTRUE);
   fBmode = new TGRadioButton(bgr, " Sunken", 77);
   fBmode->SetToolTipText("Set a sunken border of the frame");
   fBmode0 = new TGRadioButton(bgr, " No border", 78);
   fBmode0->SetToolTipText("Set no border of the frame");
   fBmode1 = new TGRadioButton(bgr, " Raised", 79);
   fBmode1->SetToolTipText("Set a raised border of the frame");
   bgr->SetButton(79, kTRUE);
   bgr->SetLayoutHints(new TGLayoutHints(kLHintsLeft, 0,0,3,0), fBmode);
   bgr->Show();
   bgr->ChangeOptions(kFitWidth|kChildFrame|kVerticalFrame);
   f2->AddFrame(bgr, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fSizeLbl = new TGLabel(f3, "Size:");                              
   f3->AddFrame(fSizeLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 6, 1, 0, 0));
   fBsize = new TGLineWidthComboBox(f3, kFR_BSIZE);
   fBsize->Resize(92, 20);
   f3->AddFrame(fBsize, new TGLayoutHints(kLHintsLeft, 13, 1, 0, 0));
   fBsize->Associate(this);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   MapSubwindows();
   Layout();
   MapWindow();

   TClass *cl = TFrame::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
   
   fInit = kTRUE;
}

//______________________________________________________________________________
TFrameEditor::~TFrameEditor()
{ 
   // Destructor of frame editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup(); 
}

//______________________________________________________________________________
void TFrameEditor::ConnectSignals2Slots()
{ 
   // Connect signals to slots.

   fBmode->Connect("Toggled(Bool_t)","TFrameEditor",this,"DoBorderMode()");
   fBmode0->Connect("Toggled(Bool_t)","TFrameEditor",this,"DoBorderMode()");
   fBmode1->Connect("Toggled(Bool_t)","TFrameEditor",this,"DoBorderMode()");
   fBsize->Connect("Selected(Int_t)", "TFrameEditor", this, "DoBorderSize(Int_t)"); 
   
   fInit = kFALSE;
}

//______________________________________________________________________________
void TFrameEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the frame attributes.

   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TFrame")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;

   fFrame = (TFrame *)fModel;
   
   Int_t par;

   par = fFrame->GetBorderMode();
   if (par == -1) fBmode->SetState(kButtonDown);
   else if (par == 1) fBmode1->SetState(kButtonDown);
   else fBmode0->SetState(kButtonDown);

   par = fFrame->GetBorderSize();
   if (par < 1) par = 1;
   if (par > 16) par = 16;
   fBsize->Select(par);
   
   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________
void TFrameEditor::DoBorderMode()
{
   // Slot connected to the border mode settings.
   
   Int_t mode = 0;
   if (fBmode->GetState() == kButtonDown) mode = -1;
   else if (fBmode0->GetState() == kButtonDown) mode = 0;
   else mode = 1;

   if (!mode) HideFrame(f3);
   else ShowFrame(f3);
   Layout();
   
   fFrame->SetBorderMode(mode);
   Update();
   gPad->Modified();
   gPad->Update();
}

//______________________________________________________________________________
void TFrameEditor::DoBorderSize(Int_t size)
{
   // Slot connected to the border size settings.
   
   fFrame->SetBorderSize(size);
   Update();
}
