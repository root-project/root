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
//  used for the memory checker                                             //
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
struct SStringToListBox_t : public binary_function<string, TGComboBox*, bool> {
   bool operator()(string str, TGComboBox* box) const {
      if (!box)
         return false;

      box->AddEntry(str.c_str(), box->GetNumberOfEntries());
      return true;
   }
};

//______________________________________________________________________________
struct SFillListBox_t : public binary_function<TObject*, TGComboBox*, bool> {
   bool operator()(TObject *aObj, TGComboBox* box) const {
      if (!aObj || !box)
         return false; //TODO: need an assert "SFillListBox_t: parameter is a NULL pointer"

      if ((aObj->IsA() == TObjString::Class())) {
         TObjString *str(dynamic_cast<TObjString*>(aObj));
         if (!str)
            return false; // TODO: need an assert "SFillListBox_t: Container's element is not a TObjString object."

         SStringToListBox_t()(str->String().Data(), box);
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
   TGCompositeFrame *contMain = new TGCompositeFrame(this, w, h, kVerticalFrame | kFixedWidth | kFixedHeight);
   AddFrame(contMain, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // container for all GUI elements, horizontal divided
   TGCompositeFrame *contLCR = new TGCompositeFrame(contMain, w, h, kHorizontalFrame | kFixedWidth | kFixedHeight);
   contMain->AddFrame(contLCR, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // ************************* content of ContLCR *************************
   // container for GUI elements on left side
   TGCompositeFrame *contLeft = new TGCompositeFrame(contLCR, 160, 200, kVerticalFrame | kFixedWidth | kFitHeight);
   contLCR->AddFrame(contLeft, new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY, 5, 3, 3, 3));

   // left vertical splitter
   TGVSplitter *splitLeft = new TGVSplitter(contLCR);
   splitLeft->SetFrame(contLeft, kTRUE);
   contLCR->AddFrame(splitLeft, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   // container for GUI elements at the center
   TGCompositeFrame *contCenter = new TGCompositeFrame(contLCR, 150, 200, kVerticalFrame | kFixedWidth | kFitHeight);
   contLCR->AddFrame(contCenter, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   TGTab *tab = new TGTab(contCenter, 150, 200);
   contCenter->AddFrame(tab, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   TGCompositeFrame *text = tab->AddTab("Text");
   TGCompositeFrame *graphics = tab->AddTab("Graphics");

   fText = new TGTextView(text);
   text->AddFrame(fText, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // Display everything
   Initialize(option);

   // Graphics View
   // TMemStat must be initialized first
   new TMemStatDrawDlg(graphics, fViewer);

   MakeStampList(contLeft);
   MakeSelection(contLeft);
   MakeContSortStat(contLeft);   // Make content for Sort Statistic
   MakeContSortStamp(contLeft);  // make content for sort Stamps
   MakeContDeep(contLeft);       // make content for sort Stamps

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   // Default View
   fViewer->SetSortStat( TMemStat::kTotalAllocCount );
   fViewer->SetSortStamp( TMemStat::kCurrent );
   MakePrint();
}

//______________________________________________________________________________
TMemStatViewerGUI::~TMemStatViewerGUI()
{
   // a dtor

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
   TGVButtonGroup *sortStatGroup = new TGVButtonGroup(frame, "Statistic type");
   frame->AddFrame(sortStatGroup, new TGLayoutHints(kLHintsExpandX));
   new TGRadioButton(sortStatGroup, "Total Alloc Count", rbtnTotalAllocCount);
   new TGRadioButton(sortStatGroup, "Total Alloc Size", rbtnTotalAllocSize);
   new TGRadioButton(sortStatGroup, "Alloc Count", rbtnAllocCount);
   new TGRadioButton(sortStatGroup, "Alloc Size", rbtnAllocSize);
   sortStatGroup->SetButton(rbtnTotalAllocCount);
   sortStatGroup->Connect("Pressed(Int_t)", "TMemStatViewerGUI", this, "HandleButtonsSortStat(Int_t)");
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeContSortStamp(TGCompositeFrame *frame)
{
   // make windows for Sorting *STAMP* selection

   // sorting stamp option
   TGVButtonGroup *sortStampGroup = new TGVButtonGroup(frame, "Sorting stamp");
   frame->AddFrame(sortStampGroup, new TGLayoutHints(kLHintsExpandX));
   new TGRadioButton(sortStampGroup, "Current", rbtnCurrent);
   new TGRadioButton(sortStampGroup, "Max Size", rbtnMaxSize);
   new TGRadioButton(sortStampGroup, "Max Count", rbtnMaxCount);
   sortStampGroup->SetButton(rbtnCurrent);
   sortStampGroup->Connect("Pressed(Int_t)", "TMemStatViewerGUI", this, "HandleButtonsSortStamp(Int_t)");
}

//______________________________________________________________________________
void TMemStatViewerGUI::MakeStampList(TGCompositeFrame *frame)
{
   // make STAMPs list box

   if (!fViewer)
      return;

   const TObjArray *stampList = fViewer->GetStampList();
   if (!stampList)
      return;

   TGHorizontalFrame *horz = new TGHorizontalFrame(frame);
   frame->AddFrame(horz, new TGLayoutHints(kLHintsExpandX, 2, 2, 10, 2));
   // text description
   TGLabel *lblName = new TGLabel(horz, "Stamp name:");
   horz->AddFrame(lblName,
                  new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 2, 2, 2, 2));

   // a list box of stamps
   TGComboBox *stampListBox = new TGComboBox(horz, lstStamps);
   stampListBox->Resize(100, 20);
   horz->AddFrame(stampListBox, new TGLayoutHints(kLHintsExpandX));
   stampListBox->Connect("Selected(const char*)", "TMemStatViewerGUI", this, "HandleStampSelect(const char*)");

   // filling Combo box of stamps
   TIter iter(stampList);
   for_each(iter.Begin(), TIter::End(),
            bind2nd(SFillListBox_t(), stampListBox));

   const Int_t count(stampListBox->GetNumberOfEntries());
   if (count <= 0)
      return;
   // Selecting the last stamp
   stampListBox->Select(count - 1);

   TObjString *str(dynamic_cast<TObjString*>(stampList->At(stampListBox->GetSelected())));
   if (!str)
      return;

   fViewer->SetCurrentStamp(*str);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleStampSelect(const char* value)
{
   // TODO: Comment me

   fViewer->SetCurrentStamp(value);
   MakePrint();
}

//______________________________________________________________________________
void  TMemStatViewerGUI::MakeContDeep(TGCompositeFrame *frame)
{
   // create and layout "Deep" controls

   TGGroupFrame *contDeep = new TGGroupFrame(frame, "Deepnes");
   contDeep->SetLayoutManager(new TGMatrixLayout(contDeep, 0, 2, 5));
   frame->AddFrame(contDeep, new TGLayoutHints(kLHintsExpandX));

   // ------ Stack Deep
   // text description
   TGLabel *lblStackDeep = new TGLabel(contDeep, "Stack Deep:");
   contDeep->AddFrame(lblStackDeep, 0);
   // number entry box for specifying the stack deepness
   fNmbStackDeep = new TGNumberEntry(contDeep, fViewer->GetStackDeep(), 1, -1, TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 50);
   contDeep->AddFrame(fNmbStackDeep, 0);
   fNmbStackDeep->Connect("ValueSet(Long_t)", "TMemStatViewerGUI", this, "HandleDeep(Long_t)");
   fNmbStackDeep->Resize(60, 20);

   // ------ Sort Deep
   // text description
   TGLabel *lSortDeep = new TGLabel(contDeep, "Sort Deep:");
   contDeep->AddFrame(lSortDeep, 0);
   // number entry box for specifying the number of stamps
   fNmbSortDeep = new TGNumberEntry(contDeep, fViewer->GetSortDeep(), 1, -1, TGNumberFormat::kNESInteger,
                                    TGNumberFormat::kNEANonNegative, TGNumberFormat::kNELLimitMinMax, 1, 50);
   contDeep->AddFrame(fNmbSortDeep, 0);
   fNmbSortDeep->Connect("ValueSet(Long_t)", "TMemStatViewerGUI", this, "HandleDeep(Long_t)");
   fNmbSortDeep->Resize(60, 20);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleButtonsSortStat(Int_t id)
{
   // handles mutual radio button exclusions - set sort stat type
   TMemStat::StatType val;
   HandleRButtons(id, rbtnTotalAllocCount, &val);
   fViewer->SetSortStat(val);
   MakePrint();
}

//___________________________l___________________________________________________
void TMemStatViewerGUI::HandleButtonsSortStamp(Int_t id)
{
   // handles mutual radio button exclusions - set sort stat type
   TMemStat::StampType val;
   HandleRButtons(id, rbtnCurrent, &val);
   fViewer->SetSortStamp(val);
   MakePrint();
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleDeep(Long_t /*id*/)
{
   // handles stack deep

   fViewer->SetStackDeep( fNmbStackDeep->GetIntNumber() );
   fViewer->SetSortDeep( fNmbSortDeep->GetIntNumber() );
   MakePrint();
}

//______________________________________________________________________________
void TMemStatViewerGUI::ShowGUI()
{
   // initialize and show GUI for presentation

   TGMainFrame* frmMain = new TGMainFrame(gClient->GetRoot(), 950, 600);
   frmMain->SetWindowName("TMemStat analysis console");
   frmMain->SetCleanup(kDeepCleanup);

   TMemStatViewerGUI* calibViewer1 = new TMemStatViewerGUI(frmMain, 950, 600, "read");
   frmMain->AddFrame(calibViewer1, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // position relative to the parent's window
   //frmMain->CenterOnParent();
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
   // TODO: Comment me
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
            bind2nd(SStringToListBox_t(), lboxFunctions));
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
            bind2nd(SStringToListBox_t(), lboxLibraries));
   lboxLibraries->Select(0);
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleFuncSelect(const char* _val)
{
   // TODO: Comment me
   fCurFunc = _val;
   // if _val == "*" then we don't sort
   if ( fCurFunc.find("*") != string::npos )
      fCurFunc.clear();

   MakePrint();
}

//______________________________________________________________________________
void TMemStatViewerGUI::HandleLibSelect(const char* _val)
{
   // TODO: Comment me
   fCurLib = _val;
   // if _val == "*" then we don't sort
   if ( fCurLib.find("*") != string::npos )
      fCurLib.clear();

   MakePrint();
}
