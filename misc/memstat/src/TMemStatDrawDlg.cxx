// @(#)root/memstat:$Name$:$Id$
// Author: Anar Manafov (A.Manafov@gsi.de) 31/05/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// STD
#include <sstream>
#include <algorithm>
// ROOT
#include "TGLabel.h"
#include "TGComboBox.h"
#include "TGNumberEntry.h"
#include "TGButton.h"
#include "TCanvas.h"
#include "TRootEmbeddedCanvas.h"
// MemStat
#include "TMemStat.h"
#include "TMemStatDrawDlg.h"
#include "TMemStatResource.h"

using namespace std;

//______________________________________________________________________________
struct SFill : public binary_function<string, TGComboBox*, bool>
{
   bool operator()(const string &val, TGComboBox* box) const
   {
      if (!box)
         return false;//parametr is a NULL pointer"

      box->AddEntry(val.c_str(), box->GetNumberOfEntries());
      return true;
   }
};

//______________________________________________________________________________
TMemStatDrawDlg::TMemStatDrawDlg(const TGWindow *p, const TGWindow *main, TMemStat *MemStat):
      fMain(NULL),
      fMemStat(MemStat),
      fboxOrder(NULL),
      fboxSortStat(NULL),
      fboxSortStamp(NULL),
      fNmbStackDeep(NULL),
      fNmbSortDeep(NULL),
      fNmbMaxLength(NULL),
      fEc(NULL)
{
   // Create a dialog window. A dialog window pops up with respect to its
   // "main" window.

   fMain = new TGTransientFrame(p, main, 800, 400, kVerticalFrame | kFixedSize);
   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);
   fMain->Connect("CloseWindow()", "TMemStatDrawDlg", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.
   PlaceCtrls(fMain);

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->MapSubwindows();
   fMain->Resize();

   //TODO: Change the title of the dialog
   fMain->SetWindowName("Draw TMemStat");
   fMain->MapWindow();
}

//______________________________________________________________________________
TMemStatDrawDlg::~TMemStatDrawDlg()
{
   // Delete test dialog widgets.

   fMain->DeleteWindow();  // deletes fMain
}

//______________________________________________________________________________
void TMemStatDrawDlg::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

//______________________________________________________________________________
void TMemStatDrawDlg::PlaceCtrls(TGCompositeFrame *frame)
{
   TGCompositeFrame *cont = new TGCompositeFrame(frame, 800, 400, kHorizontalFrame | kFitWidth | kFitHeight);
   frame->AddFrame(cont, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY | kLHintsExpandX, 2, 2, 2, 2));

   TGCompositeFrame *contR = new TGCompositeFrame(cont, 200, 200, kVerticalFrame | kFitWidth | kFitHeight);
   cont->AddFrame(contR, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY, 2, 2, 2, 2));

   StringVector_t values;
   // Order
   values.push_back("increasing");
   values.push_back("decreasing");
   PlaceLBoxCtrl(contR, &fboxOrder, "Order: ", values, resCBoxOrder);
   // Sort Stat
   values.clear();
   values.push_back("TotalAllocCount");
   values.push_back("TotalAlocSize");
   values.push_back("AllocCount");
   values.push_back("AllocSize");
   PlaceLBoxCtrl(contR, &fboxSortStat, "Sort stat: ", values, resCBoxSortStat);
   // Sort Stamp
   values.clear();
   values.push_back("Current");
   values.push_back("MaxSize");
   values.push_back("MaxCount");
   PlaceLBoxCtrl(contR, &fboxSortStamp, "Sort stamp: ", values, resCBoxSortStamp);
   // sortdeep and stackdeep
   PlaceDeepCtrl(contR);

   // a Draw button
   TGTextButton *btnDraw = new TGTextButton(contR);
   btnDraw->Connect("Clicked()", "TMemStatDrawDlg", this, "HandleDrawMemStat()");
   btnDraw->SetText("Draw");
   contR->AddFrame(btnDraw,
                   new TGLayoutHints(kLHintsCenterX | kLHintsExpandX, 10, 10, 10, 10));

   // a Canvas
   PlaceEmbeddedCanvas(cont);
}

//______________________________________________________________________________
void TMemStatDrawDlg::PlaceLBoxCtrl(TGCompositeFrame *frame, TGComboBox **box,
                                    const string &Label, const StringVector_t &Vealues, Int_t resource)
{
   TGHorizontalFrame *horz = new TGHorizontalFrame(frame);
   frame->AddFrame(horz, new TGLayoutHints(kLHintsExpandX));
   // text description
   TGLabel *lbl = new TGLabel(horz, Label.c_str());
   horz->AddFrame(lbl,
                  new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));

   // a list box of stamps
   *box = new TGComboBox( horz, resource );
   (*box)->Resize(120, 20);
   horz->AddFrame((*box), new TGLayoutHints(kLHintsRight, 2, 2, 2, 2));

   // filling Combo box with values
   for_each(Vealues.begin(), Vealues.end(), bind2nd(SFill(), (*box)));

   (*box)->Select(0);
}

//______________________________________________________________________________
void TMemStatDrawDlg::PlaceDeepCtrl(TGCompositeFrame *frame)
{
   // create and layout "Deep" controls

   // deep of information
   TGGroupFrame *ContDeep = new TGGroupFrame(frame, "Deepnes", kVerticalFrame | kFitWidth | kFitHeight);
   frame->AddFrame(ContDeep, new TGLayoutHints(kLHintsExpandX));

   // A "Deep" frame
   TGHorizontalFrame *horz0 = new TGHorizontalFrame(ContDeep);
   ContDeep->AddFrame(horz0, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY));
   // ------ Stack Deep
   // text description
   TGLabel *lblStackDeep = new TGLabel(horz0, "Stack deep:");
   horz0->AddFrame(lblStackDeep, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
   // number entry box for specifying the stack deepness
   fNmbStackDeep = new TGNumberEntry(horz0, fMemStat->fStackDeep, 1, resNmbStackDeep, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 50);
   horz0->AddFrame(fNmbStackDeep, new TGLayoutHints( kLHintsRight, 2, 2, 2, 2));
   fNmbStackDeep->Resize(100, 20);

   // ------ Sort Deep
   TGHorizontalFrame *horz1 = new TGHorizontalFrame(ContDeep);
   ContDeep->AddFrame(horz1, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY));
   // text description
   TGLabel *LabSortDeep = new TGLabel(horz1, "Sort deep:");
   horz1->AddFrame(LabSortDeep, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
   // number entry box for specifying the number of stamps
   fNmbSortDeep = new TGNumberEntry(horz1, fMemStat->fStackDeep, 1, resNmbSortDeep, TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 50);
   horz1->AddFrame(fNmbSortDeep, new TGLayoutHints( kLHintsRight, 2, 2, 2, 2));
   fNmbSortDeep->Resize(100, 20);

   // ------ Max lenthg
   TGHorizontalFrame *horz2 = new TGHorizontalFrame(ContDeep);
   ContDeep->AddFrame(horz2, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY));
   // text description
   TGLabel *lbl = new TGLabel(horz2, "Max length:");
   horz2->AddFrame(lbl, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
   // number entry box for specifying the number of stamps
   fNmbMaxLength = new TGNumberEntry(horz2, fMemStat->fStackDeep, 1, resNmbMaxLength, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 500);
   horz2->AddFrame(fNmbMaxLength, new TGLayoutHints(kLHintsRight, 2, 2, 2, 2));
   fNmbMaxLength->Resize(100, 20);
}

//______________________________________________________________________________
void TMemStatDrawDlg::ReDraw()
{
   if (!fMemStat)
      return;

   //"order 0 sortstat 2 sortstamp 1 sortdeep 10 stackdeep 5 maxlength 50"
   ostringstream ss;
   // order
   if (fboxOrder)
      ss << "order " << fboxOrder->GetSelected();
   // sort stat
   if (fboxSortStat)
      ss << " sortstat " << fboxSortStat->GetSelected();
   // sort stamp
   if (fboxSortStamp)
      ss << " sortstamp " << fboxSortStamp->GetSelected();
   // sort deep
   if (fNmbStackDeep)
      ss << " sortdeep " << fNmbStackDeep->GetIntNumber();
   // stack deep
   if (fNmbSortDeep)
      ss << " stackdeep " << fNmbSortDeep->GetIntNumber();
   // max length
   if (fNmbMaxLength)
      ss << " maxlength " << fNmbMaxLength->GetIntNumber();

   fMemStat->Draw(ss.str().c_str());
   fEc->GetCanvas()->Modified();
   fEc->GetCanvas()->Update();
}

//______________________________________________________________________________
void TMemStatDrawDlg::HandleDrawMemStat()
{
   ReDraw();
}

//______________________________________________________________________________
void TMemStatDrawDlg::PlaceEmbeddedCanvas(TGCompositeFrame *frame)
{
   if (fEc)
      return;

   fEc = new TRootEmbeddedCanvas("ec", frame, 200, 200);
   frame->AddFrame(fEc, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   fEc->GetCanvas()->SetBorderMode(0);
}
