// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldNameFrame.cxx,v 1.3 2006/05/28 20:15:09 brun Exp $
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
// TGuiNameFrame                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TGuiBldNameFrame.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TGuiBldEditor.h"
#include "TGLayout.h"
#include "TG3DLine.h"
#include "TColor.h"
#include "TROOT.h"
#include "TRootGuiBuilder.h"

//______________________________________________________________________________
TGuiBldNameFrame::TGuiBldNameFrame(const TGWindow *p, TGuiBldEditor *editor) :
                  TGCompositeFrame(p, 1, 1)
{
   // Constructor.

   fEditor = editor;
   fEditDisabled = kEditDisable;
   SetCleanup(kDeepCleanup);
   TGFrame *frame = fEditor->GetSelected();

   TGCompositeFrame *f = fTitleFrame = new TGHorizontalFrame(this);
   f->AddFrame(new TGLabel(f, "Name"), new TGLayoutHints(kLHintsLeft, 0, 1, 0, 0));
   f->AddFrame(new TGHorizontal3DLine(f), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   f = new TGVerticalFrame(this);
   AddFrame(f, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 1, 1, 0, 0));

   TString name = "";
   if (frame) {
      frame->ClassName();
   }

   fLabel = new TGLabel(f, name.Data());
   f->AddFrame(fLabel, new TGLayoutHints(kLHintsCenterX, 10, 1, 0, 0));
   fFrameName = new TGTextEntry(f, frame ? frame->GetName() : "noname");
   fFrameName->SetAlignment(kTextLeft);
   fFrameName->Resize(120, fFrameName->GetHeight());
   f->AddFrame(fFrameName, new TGLayoutHints(kLHintsCenterX,1));
   fFrameName->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TGuiBldNameFrame::DoRedraw()
{
   // Redraw frame (just a prototype).

   //TColor *col = gROOT->GetColor(29);
   //TRootGuiBuilder::SetBgndColor(fTitleFrame, col->GetPixel());
   TGCompositeFrame::DoRedraw();
}

//______________________________________________________________________________
void TGuiBldNameFrame::Reset()
{
   // Reset name frame.

   fFrameName->SetText("");
   fLabel->SetText("");
   DoRedraw();
}

//______________________________________________________________________________
void TGuiBldNameFrame::ChangeSelected(TGFrame *frame)
{
   // Change selected frame.

   fFrameName->Disconnect();

   if (!frame) {
      Reset();
      return;
   }

   TString name = frame->ClassName();

   fLabel->SetText(name.Data());
   fFrameName->SetText(frame->GetName());
   ///fFrameName->Connect("TextChanged(char*)", frame->ClassName(), frame, "SetName(char*)");
   Resize();
   DoRedraw();
}
