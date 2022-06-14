// @(#)root/ged:$Id$
// Author: Ilka Antcheva   10/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGedFrame
    \ingroup ged

Base frame for implementing GUI - a service class.

*/

#include "TGedFrame.h"
#include "TGedEditor.h"
#include "TG3DLine.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TGLabel.h"
#include "TGToolTip.h"
#include "TGCanvas.h"
#include "TGScrollBar.h"


ClassImp(TGedFrame);

////////////////////////////////////////////////////////////////////////////////
/// Constructor of the base GUI attribute frame.

TGedFrame::TGedFrame(const TGWindow *p, Int_t width,
                     Int_t height, UInt_t options, Pixel_t back)
      : TGCompositeFrame(p, width, height, options, back),
        fInit(kTRUE),
        fGedEditor(0),
        fModelClass(0),
        fAvoidSignal(kFALSE),
        fExtraTabs(0),
        fPriority(50)
{
   fName = "";
   fGedEditor = TGedEditor::GetFrameCreator();
   SetCleanup(kDeepCleanup);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor of the base GUI attribute frame.

TGedFrame::~TGedFrame()
{
   if (fExtraTabs) {
      TGedSubFrame* sf;
      TIter next(fExtraTabs);
      while ((sf = (TGedSubFrame*) next()) != 0) {
         delete sf->fFrame;
         fExtraTabs->Remove(sf);
         delete sf;
      }
      delete fExtraTabs;
   }

   // Destructor of TGCompositeFrame will do the rest.
}

////////////////////////////////////////////////////////////////////////////////
/// Update the current pad when an attribute is changed via GUI.

void TGedFrame::Update()
{
   fGedEditor->Update(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Get draw options of the selected object.

Option_t *TGedFrame::GetDrawOption() const
{
   if (!fGedEditor->GetPad()) return "";

   TListIter next(fGedEditor->GetPad()->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next())) {
      if (obj == fGedEditor->GetModel()) return next.GetOption();
   }
   return "";
}

////////////////////////////////////////////////////////////////////////////////
/// Create attribute frame title.

void TGedFrame::MakeTitle(const char *title)
{
   TGCompositeFrame *f1 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame |
                                                              kFitWidth |
                                                              kFixedWidth |
                                                              kOwnBackground);
   f1->AddFrame(new TGLabel(f1, title),
                new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1),
                new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
}

////////////////////////////////////////////////////////////////////////////////
/// Adds tab container to list of extra tabs.

void TGedFrame::AddExtraTab(TGedSubFrame* sf)
{
   if (fExtraTabs == 0) fExtraTabs = new TList();
   fExtraTabs->Add(sf);
   sf->fFrame->SetCleanup(kDeepCleanup);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a vertical frame to be used by 'owner' in extra tab 'name'.
/// The new frame is registered into the sub-frame list.

TGVerticalFrame* TGedFrame::CreateEditorTabSubFrame(const char* name)
{
   TGCompositeFrame* tabcont  = fGedEditor->GetEditorTab(name);

   TGVerticalFrame* newframe = new TGVerticalFrame(tabcont);
   AddExtraTab(new TGedFrame::TGedSubFrame(TString(name), newframe));
   return newframe;
}

////////////////////////////////////////////////////////////////////////////////
/// Refresh the GUI info about the object attributes.

void TGedFrame::Refresh(TObject* model)
{
   SetModel(model);
}

////////////////////////////////////////////////////////////////////////////////
/// Set drawing option for object. This option only affects
/// the drawing style and is stored in the option field of the
/// TObjOptLink supporting a TPad's primitive list (TList).

void TGedFrame::SetDrawOption(Option_t *option)
{
   if (!fGedEditor->GetPad() || !option) return;

   TListIter next(fGedEditor->GetPad()->GetListOfPrimitives());
   delete fGedEditor->GetPad()->FindObject("Tframe");
   TObject *obj;
   while ((obj = next())) {
      if (obj == fGedEditor->GetModel()) {
         next.SetOption(option);
         fGedEditor->GetPad()->Modified();
         fGedEditor->GetPad()->Update();
         return;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Provide list of editors for base-classes.
/// In this class we return all classed with editors found via recursive
/// descent into list of base classes.
/// Override to control which editors are actually shown (see TH2Editor).

void TGedFrame::ActivateBaseClassEditors(TClass* cl)
{
   // printf("%s::FillListOfBaseEditors %s\n", IsA()->GetName(), cl->GetName());
   if (cl->GetListOfBases()->IsEmpty() == kFALSE) {
      fGedEditor->ActivateEditors(cl->GetListOfBases(), kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create the frame containing the selected object name.

TGedNameFrame::TGedNameFrame(const TGWindow *p, Int_t width,
                              Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   fPriority = 0;

   f1 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame |
                                            kFixedWidth      |
                                            kOwnBackground);
   f1->AddFrame(new TGLabel(f1,"Name"),
                new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1),
                new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop));

   f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame | kFixedWidth);
   fLabel = new TGLabel(f2, "");
   f2->AddFrame(fLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   // Set red color for the name.
   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   fLabel->SetTextColor(color, kFALSE);

   // create tool tip with delay 300 ms
   fTip = new TGToolTip(fClient->GetDefaultRoot(), this, "TGedNameFrame", 500);

   AddInput(kEnterWindowMask | kLeaveWindowMask | kKeyPressMask | kButtonPressMask);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGedNameFrame::~TGedNameFrame()
{
   delete fTip;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse crossing event for tooltip.

Bool_t TGedNameFrame::HandleCrossing(Event_t *event)
{
   if (event->fType == kEnterNotify)
      fTip->Reset();
   else
      fTip->Hide();

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Handle mouse button event.

Bool_t TGedNameFrame::HandleButton(Event_t * /*event*/)
{
   if (fTip) fTip->Hide();

   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets text for the label.

void TGedNameFrame::SetModel(TObject* obj)
{
   TString string;

   if (obj == 0) {
      fLabel->SetText(new TGString("Object not selected"));
      return;
   }
   string.Append(obj->GetName());
   string.Append("::");
   string.Append(obj->ClassName());

   fLabel->SetText(new TGString(string));
   string = TString::Format("Name: %s\nTitle:   %s\nClass: %s",
                            obj->GetName(), obj->GetTitle(), obj->ClassName());
   fTip->SetText(string);

   // Resize label-frame to a reasonable width.
   {
      TGCanvas     *canvas = fGedEditor->GetTGCanvas();
      TGVScrollBar *vsb    = canvas->GetVScrollbar();

      Int_t hscrollw = (vsb && vsb->IsMapped()) ? vsb->GetWidth() : 0;
      Int_t labwidth = TMath::Min(fLabel->GetDefaultSize().fWidth,
                                  canvas->GetWidth() - 10 - hscrollw);
      f2->SetWidth(TMath::Max(labwidth, 80));
   }
}
