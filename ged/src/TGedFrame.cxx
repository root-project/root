// @(#)root/ged:$Name:  $:$Id: TGedFrame.cxx,v 1.9 2005/12/06 17:32:15 rdm Exp $
// Author: Ilka Antcheva   10/05/04

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGedFrame                                                           //
//                                                                      //
//  Base frame for implementing GUI - a service class.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGedFrame.h"
#include "TGClient.h"
#include "TG3DLine.h"
#include "TCanvas.h"
#include "TGLabel.h"
#include "TGTab.h"
#include <snprintf.h>


ClassImp(TGedFrame)


//______________________________________________________________________________
TGedFrame::TGedFrame(const TGWindow *p, Int_t id, Int_t width,
                           Int_t height, UInt_t options, Pixel_t back)
   : TGCompositeFrame(p, width, height, options, back), TGWidget(id)
{
   // Constructor of the base GUI attribute frame.

   fPad    = 0;
   fModel  = 0;
   fInit   = kTRUE;

   Associate(p);
   fTab = (TGTab*)p->GetParent()->GetParent();

//   gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
TGedFrame::TGedFrame(const TGedFrame& gf) :
  TGCompositeFrame(gf),
  TGWidget(gf),
  fModel(gf.fModel),
  fPad(gf.fPad),
  fInit(gf.fInit),
  fTab(gf.fTab)
{ }

//______________________________________________________________________________
TGedFrame& TGedFrame::operator=(const TGedFrame& gf)
{
  if(this!=&gf) {
    TGCompositeFrame::operator=(gf);
    TGWidget::operator=(gf);
    fModel=gf.fModel;
    fPad=gf.fPad;
    fInit=gf.fInit;
    fTab=gf.fTab;
  } return *this;
}

//______________________________________________________________________________
TGedFrame::~TGedFrame()
{
   // Destructor of the base GUI attribute frame.

//   gROOT->GetListOfCleanups()->Remove(this);

}

//______________________________________________________________________________
Option_t *TGedFrame::GetDrawOption() const
{
   // Get draw options of the selected object.

   if (!fPad) return "";

   TListIter next(fPad->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next())) {
      if (obj == fModel) return next.GetOption();
   }
   return "";
}

//______________________________________________________________________________
void TGedFrame::MakeTitle(const char *title)
{
   // Create attribute frame title.

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame |
                                                              kLHintsExpandX | 
                                                              kFixedWidth | 
                                                              kOwnBackground);
   f1->AddFrame(new TGLabel(f1, title), 
                new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1),
                new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
}

//______________________________________________________________________________
void TGedFrame::SetActive(Bool_t active)
{
   // Set active GUI attribute frames related to the selected object.

   if (active)
      ((TGCompositeFrame*)GetParent())->ShowFrame(this);
   else
      ((TGCompositeFrame*)GetParent())->HideFrame(this);

   ((TGMainFrame*)GetMainFrame())->Layout();

   // to avoid that the user changes options on a deactivated Tab
   if (fTab->IsEnabled(fTab->GetCurrent()))
      fTab->SetTab(fTab->GetCurrent());
   else
      fTab->SetTab(0);
}

//______________________________________________________________________________
void TGedFrame::RecursiveRemove(TObject* /*obj*/)
{
   // Remove references to fModel in case the fModel is being deleted
   // Deactivate attribute frames if they point to obj

//   if (fModel != obj ) return;
//      SetModel(fPad,0,0);
}

//______________________________________________________________________________
void TGedFrame::Refresh()
{
   // Refresh the GUI info about the object attributes.

   SetModel(fPad, fModel, 0);
}

//______________________________________________________________________________
void TGedFrame::SetDrawOption(Option_t *option)
{
   // Set drawing option for object. This option only affects
   // the drawing style and is stored in the option field of the
   // TObjOptLink supporting a TPad's primitive list (TList).

   if (!fPad || !option) return;

   TListIter next(fPad->GetListOfPrimitives());
   delete fPad->FindObject("Tframe");
   TObject *obj;
   while ((obj = next())) {
      if (obj == fModel) {
         next.SetOption(option);
         fPad->Modified();
         fPad->Update();
         return;
      }
   }
}

//______________________________________________________________________________
void TGedFrame::Update()
{
   // Update the current pad when an attribute is changed via GUI.

   if (fPad) {
      fPad->Modified();
      fPad->Update();
   }
}

//______________________________________________________________________________
TGedNameFrame::TGedNameFrame(const TGWindow *p, Int_t id, Int_t width,
                             Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Create the frame containing the selected object name.

   f1 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame | 
                                            kFixedWidth      | 
                                            kOwnBackground);
   f1->AddFrame(new TGLabel(f1,"Name"), 
                new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), 
                new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f1, new TGLayoutHints(kLHintsTop));

   f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fLabel = new TGLabel(f2, "");
   f2->AddFrame(fLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   // Set red color for the name.
   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   fLabel->SetTextColor(color, kFALSE);
}

//______________________________________________________________________________
TGedNameFrame::TGedNameFrame(const TGedNameFrame& nf) :
  TGedFrame(nf),
  fLabel(nf.fLabel),
  f1(nf.f1),
  f2(nf.f2)
{ }

//______________________________________________________________________________
TGedNameFrame& TGedNameFrame::operator=(const TGedNameFrame& nf)
{
  if(this!=&nf) {
    TGedFrame::operator=(nf);
    fLabel=nf.fLabel;
    f1=nf.f1;
    f2=nf.f2;
  } return *this;
}

//______________________________________________________________________________
TGedNameFrame::~TGedNameFrame()
{
   // Destructor of the name frame.

   TGFrameElement *el;
   TIter next(GetList());

   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________
void TGedNameFrame::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Slot connected to Selected() signal of TCanvas.

   fModel = obj;
   fPad = pad;

   TString string;

   if (obj == 0) {
      SetActive(kFALSE);
      return;
   }

   string.Append(fModel->GetName());
   string.Append("::");
   string.Append(fModel->ClassName());
   fLabel->SetText(new TGString(string));

   SetActive();
}
