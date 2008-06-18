// @(#)root/memstat:$Name$:$Id$
// Author: M.Ivanov -- Anar Manafov (A.Manafov@gsi.de) 28/04/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  GUI for the TMemStat                                                     //
//  used for the memomry checker                                             //
//  Example usage:                                                           //
/*
  aliroot
  TMemStatViewer::ShowGUI("memstat.root")
*/
// - Resize windows - (BUG to BE FIXED -> ROOT bug)                          //
//                                                                           //
//////////////////////////////////////////////////////////////////////////////

// STD
#include <functional>
#include <stdexcept>
#include <algorithm>
// ROOT
#include "TGTextView.h"
#include "TGLabel.h"
#include "TGTab.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGSplitter.h"
#include "TGButtonGroup.h"
#include "TGComboBox.h"
#include "TObjArray.h"
// Memstat
#include "TMemStat.h"
#include "TMemStatViewerGUI.h"
#include "TMemStatResource.h"
#include "TMemStatDrawDlg.h"

ClassImp(TMemStatViewerGUI)

using namespace std;


//______________________________________________________________________________
struct SStringToListBox : public binary_function<string, TGComboBox*, bool> {
   bool operator()(string str, TGComboBox* box) const {
      if (!box)
         return false;

      box->AddEntry(str.c_str(), box->GetNumberOfEntries());
      return true;
   }
};

//______________________________________________________________________________
struct SFillListBox : public binary_function<TObject*, TGComboBox*, bool> {
   bool operator()(TObject *aObj, TGComboBox* box) const {
      if (!aObj || !box)
         return false; //TODO: need an assert "SFillListBox: parametr is a NULL pointer"

      if ((aObj->IsA() == TObjString::Class())) {
         TObjString *str(dynamic_cast<TObjString*>(aObj));
         if (!str)
            return false; // TODO: need an assert "SFillListBox: Container's element is not a TObjString object."

         SStringToListBox()(str->String().Data(), box);
      }

      return true;
   }
};

//______________________________________________________________________________
TMemStatViewerGUI::TMemStatViewerGUI(const TGWindow *p, UInt_t w, UInt_t h, Option_t* option):
      TGCompositeFrame(p, w, h),
      fViewer(NULL),
      fText(NULL),
      fNmbStackDeep(NULL),
      fNmbSortDeep(NULL)
{
   // TMemStatViewerGUI constructor; fileName specifies the ROOT tree used for drawing

   SetCleanup(kDeepCleanup);

   // ************************* content of this MainFrame *************************
   // top level container with horizontal layout
   // container for all GUI elements, vertical divided
   TGCompositeFrame *ContMain = new TGCompositeFrame(this, w, h, kVerticalFrame | kFixedWidth | kFixedHeight);
   AddFrame(ContMain, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // container for all GUI elements, horizontal divided
   TGCompositeFrame *ContLCR = new TGCompositeFrame(ContMain, w, h, kHorizontalFrame | kFixedWidth | kFixedHeight);
   ContMain->AddFrame(ContLCR, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // ************************* content of ContLCR *************************
   // container for GUI elements on left side
   TGCompositeFrame *ContLeft = new TGCompositeFrame(ContLCR, 160, 200, kVerticalFrame | kFixedWidth | kFitHeight);
   ContLCR->AddFrame(ContLeft, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY, 5, 3, 3, 3));

   // left vertical splitter
   TGVSplitter *splitLeft = new TGVSplitter(ContLCR);
   splitLeft->SetFrame(ContLeft, kTRUE);
   ContLCR->AddFrame(splitLeft, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   // container for GUI elements at the center
   TGCompositeFrame *ContCenter = new TGCompositeFrame(ContLCR, 150, 200, kVerticalFrame | kFixedWidth | kFitHeight);
   ContLCR->AddFrame(ContCenter, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   fText = new TGTextView(ContCenter);
   ContCenter->AddFrame(fText, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // Display everything
   Initialize(option);

   MakeStampList(ContLeft);
   MakeSelection(ContLeft);
   MakeContSortStat(ContLeft);   // Make content for Sort Statistic
   MakeContSortStamp(ContLeft);  // make constent for sort Stamps
   MakeContDeep(ContLeft);       // make constent for sort Stamps
   MakeDrawButton(ContLeft);

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   // Default View
   fViewer->fSortStat = TMemStat::kTotalAllocCount;
   fViewer->fSortStamp = TMemStat::kCurrent;
   MakePrint();
}

//______________________________________________________________________________
TMemStatViewerGUI::~TMemStatViewerGUI()
{
   Cleanup();
   if (fViewer)
      fViewer->Delete();
}

//______________________________________________________________________________
void TMemStatViewerGUI::Initialize(Option_t* option)
{
   // initializes the GUI with default settings and opens tree for drawing

   delete fViewer;
   fViewer = new TMemStat(option);
   fViewer->Report();
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeContSortStat(TGCompositeFrame *frame)
{
   // make windows for Sorting *STAT* selection

   // sorting statistic option
   TGVButtonGroup *SortStatGroup = new TGVButtonGroup(frame, "Statistic type");
   frame->AddFrame(SortStatGroup, new TGLayoutHints(kLHintsExpandX));
   new TGRadioButton(SortStatGroup, "Total Alloc Count", rbtnTotalAllocCount);
   new TGRadioButton(SortStatGroup, "Total Alloc Size", rbtnTotalAllocSize);
   new TGRadioButton(SortStatGroup, "Alloc Count", rbtnAllocCount);
   new TGRadioButton(SortStatGroup, "Alloc Size", rbtnAllocSize);
   SortStatGroup->SetButton(rbtnTotalAllocCount);
   SortStatGroup->Connect("Pressed(Int_t)", "TMemStatViewerGUI", this, "HandleButtonsSortStat(Int_t)");
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeContSortStamp(TGCompositeFrame *frame)
{
   // make windows for Sorting *STAMP* selection

   // sorting stamp option
   TGVButtonGroup *SortStampGroup = new TGVButtonGroup(frame, "Sorting stamp");
   frame->AddFrame(SortStampGroup, new TGLayoutHints(kLHintsExpandX));
   new TGRadioButton(SortStampGroup, "Current", rbtnCurrent);
   new TGRadioButton(SortStampGroup, "Max Size", rbtnMaxSize);
   new TGRadioButton(SortStampGroup, "Max Count", rbtnMaxCount);
   SortStampGroup->SetButton(rbtnCurrent);
   SortStampGroup->Connect("Pressed(Int_t)", "TMemStatViewerGUI", this, "HandleButtonsSortStamp(Int_t)");
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeStampList(TGCompositeFrame *frame)
{
   // make STAMPs list box

   if (!fViewer)
      return;

   const TObjArray *StampList = fViewer->GetStampList();
   if (!StampList)
      return;

   TGHorizontalFrame *horz = new TGHorizontalFrame(frame);
   frame->AddFrame(horz, new TGLayoutHints(kLHintsExpandX, 2, 2, 10, 2));
   // text description
   TGLabel *lblName = new TGLabel(horz, "Stamp name:");
   horz->AddFrame(lblName,
                  new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));

   // a list box of stamps
   TGComboBox *StampListBox = new TGComboBox(horz, lstStamps);
   StampListBox->Resize(100, 20);
   horz->AddFrame(StampListBox, new TGLayoutHints(kLHintsExpandX));
   StampListBox->Connect("Selected(const char*)", "TMemStatViewerGUI", this, "HandleStampSelect(const char*)");

   // filling Combo box of stamps
   TIter iter(StampList);
   for_each(iter.Begin(), TIter::End(),
            bind2nd(SFillListBox(), StampListBox));

   const Int_t count(StampListBox->GetNumberOfEntries());
   if (count <= 0)
      return;
   // Selecting the last stamp
   StampListBox->Select(count - 1);

   TObjString *str(dynamic_cast<TObjString*>(StampList->At(StampListBox->GetSelected())));
   if (!str)
      return;

   fViewer->SetCurrentStamp(*str);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleStampSelect(const char* value)
{
   fViewer->SetCurrentStamp(value);
   MakePrint();
}

//______________________________________________________________________________
void  TMemStatViewerGUI::MakeContDeep(TGCompositeFrame *frame)
{
   // create and layout "Deep" controls

   TGGroupFrame *ContDeep = new TGGroupFrame(frame, "Deepnes");
   ContDeep->SetLayoutManager(new TGMatrixLayout(ContDeep, 0, 2, 5));
   frame->AddFrame(ContDeep, new TGLayoutHints(kLHintsExpandX));

   // ------ Stack Deep
   // text description
   TGLabel *lblStackDeep = new TGLabel(ContDeep, "Stack Deep:");
   ContDeep->AddFrame(lblStackDeep, 0);
   // number entry box for specifying the stack deepness
   fNmbStackDeep = new TGNumberEntry(ContDeep, fViewer->fStackDeep, 1, -1, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 50);
   ContDeep->AddFrame(fNmbStackDeep, 0);
   fNmbStackDeep->Connect("ValueSet(Long_t)", "TMemStatViewerGUI", this, "HandleDeep(Long_t)");
   fNmbStackDeep->Resize(60, 20);

   // ------ Sort Deep
   // text description
   TGLabel *LabSortDeep = new TGLabel(ContDeep, "Sort Deep:");
   ContDeep->AddFrame(LabSortDeep, 0);
   // number entry box for specifying the number of stamps
   fNmbSortDeep = new TGNumberEntry(ContDeep, fViewer->fStackDeep, 1, -1, TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 50);
   ContDeep->AddFrame(fNmbSortDeep, 0);
   fNmbSortDeep->Connect("ValueSet(Long_t)", "TMemStatViewerGUI", this, "HandleDeep(Long_t)");
   fNmbSortDeep->Resize(60, 20);
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeDrawButton(TGCompositeFrame *frame)
{
   // Creats a "Draw TMemStat" button

   // TODO: Move this to Menu. Make it a main menu item instead of a button

   TGHorizontalFrame *horz = new TGHorizontalFrame(frame);
   frame->AddFrame(horz, new TGLayoutHints(kLHintsExpandX));
   // text description
   TGTextButton *btnDraw = new TGTextButton(horz);
   btnDraw->SetText("&Draw TMemStat");
   horz->AddFrame(btnDraw,
                  new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 10, 10));
   btnDraw->Connect("Clicked()", "TMemStatViewerGUI", this, "HandleDrawMemStat()");
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleDrawMemStat()
{
   new TMemStatDrawDlg(gClient->GetRoot(), this, fViewer);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleButtonsSortStat(Int_t id)
{
   // handles mutual radio button exclusions - set sort stat type

   HandleRButtons(id, rbtnTotalAllocCount, &fViewer->fSortStat);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleButtonsSortStamp(Int_t id)
{
   // handles mutual radio button exclusions - set sort stat type

   HandleRButtons(id, rbtnCurrent, &fViewer->fSortStamp);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleDeep(Long_t /*id*/)
{
   // handles stack deep

   fViewer->fStackDeep = fNmbStackDeep->GetIntNumber();
   fViewer->fSortDeep = fNmbSortDeep->GetIntNumber();
   MakePrint();
}

//______________________________________________________________________________
void TMemStatViewerGUI::ShowGUI()
{
   // initialize and show GUI for presentation

   TGMainFrame* frmMain = new TGMainFrame(gClient->GetRoot(), 800, 600);
   frmMain->SetWindowName("TMemStat analysis console");
   frmMain->SetCleanup(kDeepCleanup);

   TMemStatViewerGUI* calibViewer1 = new TMemStatViewerGUI(frmMain, 800, 600, "read");
   frmMain->AddFrame(calibViewer1, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   frmMain->MapSubwindows();
   frmMain->Resize();
   frmMain->MapWindow();
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakePrint()
{
   // make report and load it to the view
   fViewer->MakeReport( fCurLib.c_str(), fCurFunc.c_str(), 0, "/tmp/memstatprint.txt");
   fText->LoadFile("/tmp/memstatprint.txt");
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeSelection(TGCompositeFrame *frame)
{
   if (!fViewer)
      return;

   TGGroupFrame *grp = new TGGroupFrame(frame, "Selections");
   frame->AddFrame(grp, new TGLayoutHints(kLHintsExpandX));

   // ----- Function
   // text description
   TGLabel *lblFun = new TGLabel(grp, "Function");
   grp->AddFrame(lblFun, new TGLayoutHints(kLHintsExpandX ));
   // a list box of stamps
   TGComboBox *lboxFunctions = new TGComboBox(grp);
   lboxFunctions->Resize(100, 20);
   grp->AddFrame(lboxFunctions, new TGLayoutHints(kLHintsExpandX ));
   lboxFunctions->Connect("Selected(const char*)", "TMemStatViewerGUI", this, "HandleFuncSelect(const char*)");

   // Add default selection - select all
   lboxFunctions->AddEntry("*", 0);
   // Fill values for Functions
   TMemStat::Selection_t container;
   fViewer->GetFillSelection( &container, TMemStat::kFunction );
   for_each(container.begin(), container.end(),
            bind2nd(SStringToListBox(), lboxFunctions));
   lboxFunctions->Select(0);

   // ----- Library
   // text description
   TGLabel *lblLib = new TGLabel(grp, "Libraries");
   grp->AddFrame(lblLib, new TGLayoutHints(kLHintsExpandX));
   // a list box of stamps
   TGComboBox *lboxLibraries = new TGComboBox(grp);
   lboxLibraries->Resize(100, 20);
   grp->AddFrame(lboxLibraries, new TGLayoutHints(kLHintsExpandX ));
   lboxLibraries->Connect("Selected(const char*)", "TMemStatViewerGUI", this, "HandleLibSelect(const char*)");

   // Add default selection - select all
   lboxLibraries->AddEntry("*", 0);
   // Fill values for Functions
   container.clear();
   fViewer->GetFillSelection( &container, TMemStat::kLibrary );
   for_each(container.begin(), container.end(),
            bind2nd(SStringToListBox(), lboxLibraries));
   lboxLibraries->Select(0);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleFuncSelect(const char* _val)
{
   fCurFunc = _val;
   // if _val == "*" then we don't sort
   if ( fCurFunc.find("*") != string::npos )
      fCurFunc.clear();

   MakePrint();
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleLibSelect(const char* _val)
{
   fCurLib = _val;
   // if _val == "*" then we don't sort
   if ( fCurLib.find("*") != string::npos )
      fCurLib.clear();

   MakePrint();
}
