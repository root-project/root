// @(#)root/treeviewer:$Name:  $:$Id: TSessionDialogs.cxx
// Author: Marek Biskup, Jakub Madejczyk, Bertrand Bellenot 10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSessionDialogs                                                      //
//                                                                      //
// This file defines several dialogs that are used by TSessionViewer.   //
// The following dialogs are available: TNewChainDlg and TNewQueryDlg.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSessionDialogs.h"
#include "TSessionViewer.h"
#include "TSystem.h"
#include "TGButton.h"
#include "TList.h"
#include "TChain.h"
#include "TDSet.h"
#include "TGTextEntry.h"
#include "TGTextBuffer.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"
#include "TGListView.h"
#include "TGPicture.h"
#include "TGFSContainer.h"
#include "TGFileDialog.h"
#include "TGListTree.h"
#include "TInterpreter.h"
#include "TApplication.h"
#include "TKey.h"
#include "TGTableLayout.h"
#include "TGFileDialog.h"

ClassImp(TNewChainDlg)
ClassImp(TNewQueryDlg)

const char *partypes[] = {
   "Par files",  "*.par",
   "All files",  "*",
    0,            0
};

const char *filetypes[] = {
   "C files",       "*.C",
   "ROOT files",    "*.root",
   "All files",     "*",
   0,               0
};

//////////////////////////////////////////////////////////////////////////
// New Chain Dialog
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TNewChainDlg::TNewChainDlg(const TGWindow *p, const TGWindow *main) :
   TGTransientFrame(p, main, 350, 300, kVerticalFrame)
{
   // Create a new chain dialog box. Used to list chains present in memory
   // and offers the possibility to create new ones by executing macros
   // directly from the associate file container.

   SetCleanup(kDeepCleanup);
   AddFrame(new TGLabel(this, new TGHotString("List of Chains in Memory :")),
            new TGLayoutHints(kLHintsLeft, 5, 5, 5, 5) );

   // Add TGListView used to show objects in memory
   fListView = new TGListView(this, 300, 100);
   fLVContainer = new TGLVContainer(fListView, kSunkenFrame, GetWhitePixel());
   fLVContainer->Associate(fListView);
   fLVContainer->SetViewMode(kLVSmallIcons);
   fLVContainer->SetCleanup(kDeepCleanup);
   AddFrame(fListView, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 4, 4, 4, 4));

   fListView->Connect("DoubleClicked(TGLVEntry*, Int_t)", "TNewChainDlg",
                      this, "OnElementDblClicked(TGLVEntry* ,Int_t)");

   // Add text entry showing type and name of user's selection
   TGCompositeFrame* frmSel = new TGHorizontalFrame(this, 300, 100);
   frmSel->SetCleanup(kDeepCleanup);
   frmSel->AddFrame(new TGLabel(frmSel, new TGHotString("Selected chain :")),
            new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 5, 5, 5, 5) );
   fNameBuf = new TGTextBuffer(100);
   fName = new TGTextEntry(frmSel, fNameBuf);
   fName->Resize(200, fName->GetDefaultHeight());
   fName->Associate(this);
   frmSel->AddFrame(fName, new TGLayoutHints(kLHintsLeft | kLHintsCenterY | kLHintsExpandX, 5, 5, 5, 5));
   AddFrame(frmSel, new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));

   // Add TGListview / TGFileContainer to allow user to execute Macros
   // for the creation of new TChains / TDSets
   TGListView* lv = new TGListView(this, 300, 100);
   AddFrame(lv,new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 5, 5, 5, 5));

   Pixel_t white;
   gClient->GetColorByName("white",white);
   fContents = new TGFileContainer(lv, kSunkenFrame, white);
   fContents->SetCleanup(kDeepCleanup);
   fContents->SetFilter("*.C");
   fContents->SetViewMode(kLVSmallIcons);
   fContents->Associate(this);
   fContents->SetDefaultHeaders();
   fContents->DisplayDirectory();
   fContents->AddFile("..");        // up level directory
   fContents->Resize();
   fContents->StopRefreshTimer();   // stop refreshing

   // position relative to the parent's window
   Window_t wdummy;
   Int_t  ax, ay;
   gVirtualX->TranslateCoordinates( main->GetId(),
                                    fClient->GetDefaultRoot()->GetId(),
                                    0, 0, ax, ay, wdummy);
   Move(ax + 200, ay + 35);

   // Apply and Close buttons
   AddFrame(fApplyButton = new TGTextButton(this, "Apply && Return", 0),
            new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));
   AddFrame(fCloseButton = new TGTextButton(this, "Cancel", 1),
            new TGLayoutHints(kLHintsExpandX, 5, 5, 5, 5));
   fApplyButton->Associate(this);
   fCloseButton->Associate(this);

   SetWindowName("Chains Selection Dialog");
   MapSubwindows();
   Layout();
   Resize(GetDefaultSize());
   MapWindow();
   UpdateList();
}

//______________________________________________________________________________
TNewChainDlg::~TNewChainDlg()
{
   // Delete chain dialog.

   if (IsZombie()) return;
   Cleanup();
}

//______________________________________________________________________________
void TNewChainDlg::OnElementSelected(TObject *obj)
{
   // Emits OnElementSelected signal if dset is not zero
   if (obj) {
      Emit("OnElementSelected(TObject *)", (Long_t)obj);
   }
}

//______________________________________________________________________________
void TNewChainDlg::OnElementDblClicked(TGLVEntry *entry, Int_t)
{
   // Handle double click in the Memory list view and put the type
   // and name of selected object in the text entry
   fChain = (TObject *)entry->GetUserData();
   if (fChain->IsA() == TChain::Class()) {
      TString s = Form("%s : %s" , ((TChain *)fChain)->GetTitle(),
                      ((TChain *)fChain)->GetName());
      fName->SetText(s);
   }
   else if (fChain->IsA() == TDSet::Class()) {
      TString s = Form("%s : %s" , ((TDSet *)fChain)->GetName(),
                      ((TDSet *)fChain)->GetObjName());
      fName->SetText(s);
   }
}

//______________________________________________________________________________
void TNewChainDlg::UpdateList()
{
   // Update Memory list view

   TGLVEntry *item=0;
   TObject *obj = 0;
   fChains = gROOT->GetListOfDataSets();
   fLVContainer->RemoveAll();
   if (!fChains) return;
   TIter next(fChains);
   // loop on the list of chains/datasets in memory,
   // and fill the associated listview
   while ((obj = (TObject *)next())) {
      if (obj->IsA() == TChain::Class()) {
         const char *title = ((TChain *)obj)->GetTitle();
         if (strlen(title) == 0)
            ((TChain *)obj)->SetTitle("TChain");
         item = new TGLVEntry(fLVContainer, ((TChain *)obj)->GetName(),
                              ((TChain *)obj)->GetTitle());
      }
      else if (obj->IsA() == TDSet::Class()) {
         item = new TGLVEntry(fLVContainer, ((TDSet *)obj)->GetObjName(),
                              ((TDSet *)obj)->GetName());
      }
      item->SetUserData(obj);
      fLVContainer->AddItem(item);
   }
   fClient->NeedRedraw(fLVContainer);
   Resize();
}

void TNewChainDlg::DisplayDirectory(const TString &fname)
{
   // Display content of directory

   fContents->SetDefaultHeaders();
   gSystem->ChangeDirectory(fname);
   fContents->ChangeDirectory(fname);
   fContents->DisplayDirectory();
   fContents->AddFile("..");  // up level directory
   Resize();
}

void TNewChainDlg::OnDoubleClick(TGLVEntry* f, Int_t btn)
{
   // Handle double click in the File container

   if (btn!=kButton1) return;
   gVirtualX->SetCursor(fContents->GetId(),gVirtualX->CreateCursor(kWatch));

   TString name(f->GetTitle());

   // Check if the file is a root macro file type
   if (name.Contains(".C")) {
      // form the command
      TString command = Form(".x %s/%s",
            gSystem->UnixPathName(fContents->GetDirectory()),
            name.Data());
      // and process
      gApplication->ProcessLine(command.Data());
      UpdateList();
   } else {
      // if double clicked on a directory, then display it
      DisplayDirectory(name);
   }
   gVirtualX->SetCursor(fContents->GetId(),gVirtualX->CreateCursor(kPointer));
}

//______________________________________________________________________________
Bool_t TNewChainDlg::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process messages for new chain dialog

   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_BUTTON:
               switch (parm1) {

                  case 0:
                     // Apply button
                     OnElementSelected(fChain);
                     DeleteWindow();
                     break;

                  case 1:
                     // Close button
                     fChain = 0;
                     DeleteWindow();
                     break;
               }
               break;
            default:
               break;
         }
         break;

      case kC_CONTAINER:
         switch (GET_SUBMSG(msg)) {
            case kCT_ITEMDBLCLICK:
               if (parm1==kButton1)
                  OnDoubleClick((TGLVEntry*)fContents->GetLastActive(), parm1);
               break;
         }
         break;
      default:
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TNewChainDlg::CloseWindow()
{
   // Close file dialog.
   DeleteWindow();
}


//////////////////////////////////////////////////////////////////////////
// New Query Dialog
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TNewQueryDlg::TNewQueryDlg(TSessionViewer *gui, Int_t Width, Int_t Height,
         TQueryDescription *query, Bool_t editmode) :
         TGTransientFrame(gClient->GetRoot(), gui, Width, Height)
{
   // Create a new Query dialog, used by the Session Viewer, to Edit a Query if
   // the editmode flag is set, or to create a new one if not set

   Window_t wdummy;
   Int_t  ax, ay;
   fEditMode = editmode;
   fChain = 0;
   fQuery = query;
   if (fQuery && fQuery->fChain) {
      fChain = fQuery->fChain;
   }
   Build(gui);
   // if in edit mode, update fields with query description data
   if (editmode && query)
      UpdateFields(query);
   else if (!editmode)
      fTxtQueryName->SetText("Query 1");
   MapSubwindows();
   Resize(Width, Height);
   // hide options frame
   fFrmNewQuery->HideFrame(fFrmMore);
   fBtnMore->SetText(" More >> ");
   SetWMSizeHints(Width+5, Height+25, Width+5, Height+25, 1, 1);
   ChangeOptions(GetOptions() | kFixedSize);
   Layout();
   SetWindowName("Query Dialog");
   // Position relative to parent
   gVirtualX->TranslateCoordinates( fViewer->GetId(),
                                    fClient->GetDefaultRoot()->GetId(),
                                    0, 0, ax, ay, wdummy);
   Move(ax + fViewer->GetWidth()/2, ay + 35);
}

//______________________________________________________________________________
TNewQueryDlg::~TNewQueryDlg()
{
   // Delete query dialog.

   if (IsZombie()) return;
   Cleanup();
}

//______________________________________________________________________________
void TNewQueryDlg::Build(TSessionViewer *gui)
{
   // builds the dialog

   TGButton*   btnTmp;
   fViewer = gui;
   SetLayoutManager(new TGVerticalLayout(this));
   SetCleanup(kDeepCleanup);
   SetMinWidth(500);
   fFrmNewQuery = new TGGroupFrame(this, "New Query");
   fFrmNewQuery->SetCleanup(kDeepCleanup);

   AddFrame(fFrmNewQuery, new TGLayoutHints(kLHintsExpandX |
         kLHintsExpandY, 2, 2, 2, 2));
   fFrmNewQuery->SetLayoutManager(new TGTableLayout(fFrmNewQuery, 6, 5));

   fFrmNewQuery->AddFrame(new TGLabel(fFrmNewQuery, "Query Name :"),
         new TGTableLayoutHints(0, 1, 0, 1, kLHintsCenterY, 0, 5, 4, 0));
   fFrmNewQuery->AddFrame(fTxtQueryName = new TGTextEntry(fFrmNewQuery,
         (const char *)0, 1), new TGTableLayoutHints(1, 2, 0, 1,
         kLHintsCenterY, 5, 5, 4, 0));

   fFrmNewQuery->AddFrame(new TGLabel(fFrmNewQuery, "TChain :"),
         new TGTableLayoutHints(0, 1, 1, 2, kLHintsCenterY, 0, 5, 4, 0));
   fFrmNewQuery->AddFrame(fTxtChain = new TGTextEntry(fFrmNewQuery,
         (const char *)0, 2), new TGTableLayoutHints(1, 2, 1, 2,
         kLHintsCenterY, 5, 5, 4, 0));
   fTxtChain->SetToolTipText("Specify TChain or TDSet from memory or file");
   fFrmNewQuery->AddFrame(btnTmp = new TGTextButton(fFrmNewQuery, "Browse..."),
         new TGTableLayoutHints(2, 3, 1, 2, kLHintsCenterY, 5, 0, 4, 8));
   btnTmp->Connect("Clicked()", "TNewQueryDlg", this, "OnBrowseChain()");

   fFrmNewQuery->AddFrame(new TGLabel(fFrmNewQuery, "Selector :"),
         new TGTableLayoutHints(0, 1, 2, 3, kLHintsCenterY, 0, 5, 0, 0));
   fFrmNewQuery->AddFrame(fTxtSelector = new TGTextEntry(fFrmNewQuery,
         (const char *)0, 3), new TGTableLayoutHints(1, 2, 2, 3,
         kLHintsCenterY, 5, 5, 0, 0));
   fFrmNewQuery->AddFrame(btnTmp = new TGTextButton(fFrmNewQuery, "Browse..."),
         new TGTableLayoutHints(2, 3, 2, 3, kLHintsCenterY, 5, 0, 0, 8));
   btnTmp->Connect("Clicked()", "TNewQueryDlg", this, "OnBrowseSelector()");


   fFrmNewQuery->AddFrame(new TGLabel(fFrmNewQuery, "Options :"),
         new TGTableLayoutHints(0, 1, 3, 4, kLHintsCenterY, 0, 5, 0, 0));
   fFrmNewQuery->AddFrame(fTxtOptions = new TGTextEntry(fFrmNewQuery,
         (const char *)0, 4), new TGTableLayoutHints(1, 2, 3, 4,
         kLHintsCenterY, 5, 0, 0, 8));
   fTxtOptions->SetText("\"ASYN\"");

   fFrmNewQuery->AddFrame(fBtnMore = new TGTextButton(fFrmNewQuery, " Less << "),
         new TGTableLayoutHints(2, 3, 4, 5, kLHintsCenterY, 5, 5, 4, 0));
   fBtnMore->Connect("Clicked()", "TNewQueryDlg", this, "OnNewQueryMore()");

   fFrmMore = new TGCompositeFrame(fFrmNewQuery, 200, 200);
   fFrmMore->SetCleanup(kDeepCleanup);

   fFrmNewQuery->AddFrame(fFrmMore, new TGTableLayoutHints(0, 3, 5, 6,
                          kLHintsExpandX | kLHintsExpandY));
   fFrmMore->SetLayoutManager(new TGTableLayout(fFrmMore, 4, 3));

   fFrmMore->AddFrame(new TGLabel(fFrmMore, "Nb Entries :"),
         new TGTableLayoutHints(0, 1, 0, 1, kLHintsCenterY, 0, 5, 0, 0));
   fFrmMore->AddFrame(fNumEntries = new TGNumberEntry(fFrmMore, 0, 5, -1,
         TGNumberFormat::kNESInteger, TGNumberFormat::kNEAAnyNumber,
         TGNumberFormat::kNELNoLimits), new TGTableLayoutHints(1, 2, 0, 1,
         0, 37, 0, 0, 8));
   fNumEntries->SetIntNumber(-1);
   fFrmMore->AddFrame(new TGLabel(fFrmMore, "First entry :"),
         new TGTableLayoutHints(0, 1, 1, 2, kLHintsCenterY, 0, 5, 0, 0));
   fFrmMore->AddFrame(fNumFirstEntry = new TGNumberEntry(fFrmMore, 0, 5, -1,
         TGNumberFormat::kNESInteger, TGNumberFormat::kNEANonNegative,
         TGNumberFormat::kNELNoLimits), new TGTableLayoutHints(1, 2, 1, 2, 0,
         37, 0, 0, 8));

   fFrmMore->AddFrame(new TGLabel(fFrmMore, "Par file :"),
         new TGTableLayoutHints(0, 1, 2, 3, kLHintsCenterY, 0, 5, 0, 0));
   fFrmMore->AddFrame(fTxtParFile = new TGTextEntry(fFrmMore,
         (const char *)0, 5), new TGTableLayoutHints(1, 2, 2, 3, 0, 37,
         5, 0, 0));
   fFrmMore->AddFrame(btnTmp = new TGTextButton(fFrmMore, "Browse..."),
         new TGTableLayoutHints(2, 3, 2, 3, 0, 6, 0, 0, 8));
   btnTmp->Connect("Clicked()", "TNewQueryDlg", this, "OnBrowseParFile()");

   fFrmMore->AddFrame(new TGLabel(fFrmMore, "Event list :"),
         new TGTableLayoutHints(0, 1, 3, 4, kLHintsCenterY, 0, 5, 0, 0));
   fFrmMore->AddFrame(fTxtEventList = new TGTextEntry(fFrmMore,
         (const char *)0, 6), new TGTableLayoutHints(1, 2, 3, 4, 0, 37,
         5, 0, 0));
   fFrmMore->AddFrame(btnTmp = new TGTextButton(fFrmMore, "Browse..."),
         new TGTableLayoutHints(2, 3, 3, 4, 0, 6, 0, 0, 8));
   btnTmp->Connect("Clicked()", "TNewQueryDlg", this, "OnBrowseEventList()");

   fTxtQueryName->Associate(this);
   fTxtChain->Associate(this);
   fTxtSelector->Associate(this);
   fTxtOptions->Associate(this);
   fNumEntries->Associate(this);
   fNumFirstEntry->Associate(this);
   fTxtParFile->Associate(this);
   fTxtEventList->Associate(this);

   TGCompositeFrame *tmp;
   AddFrame(tmp = new TGCompositeFrame(this, 140, 20, kHorizontalFrame),
                       new TGLayoutHints(kLHintsLeft | kLHintsExpandX));
   tmp->SetCleanup(kDeepCleanup);
   if (fEditMode)
      tmp->AddFrame(fBtnSave = new TGTextButton(tmp, "Save Query"),
         new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 3, 3, 3, 3));
   else
      tmp->AddFrame(fBtnSave = new TGTextButton(tmp, "Add Query"),
         new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 3, 3, 3, 3));
   fBtnSave->Connect("Clicked()", "TNewQueryDlg", this, "OnBtnSaveClicked()");
   tmp->AddFrame(fBtnSubmit = new TGTextButton(tmp, "Save && Submit"),
      new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 3, 3, 3, 3));
   fBtnSubmit->Connect("Clicked()", "TNewQueryDlg", this, "OnBtnSubmitClicked()");
   tmp->AddFrame(fBtnClose = new TGTextButton(tmp, "Close"),
      new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 3, 3, 3, 3));
   fBtnClose->Connect("Clicked()", "TNewQueryDlg", this, "OnBtnCloseClicked()");
}

//______________________________________________________________________________
void TNewQueryDlg::CloseWindow()
{
   // Called when window is closed via the window manager.
   DeleteWindow();
}

//______________________________________________________________________________
void TNewQueryDlg::OnNewQueryMore()
{
   if (fFrmNewQuery->IsVisible(fFrmMore)) {
      fFrmNewQuery->HideFrame(fFrmMore);
      fBtnMore->SetText(" More >> ");
   }
   else {
      fFrmNewQuery->ShowFrame(fFrmMore);
      fBtnMore->SetText(" Less << ");
   }
}

//______________________________________________________________________________
void TNewQueryDlg::OnBrowseChain()
{
   TNewChainDlg *dlg = new TNewChainDlg(fClient->GetRoot(), this);
   dlg->Connect("OnElementSelected(TObject *)", "TNewQueryDlg",
                this, "OnElementSelected(TObject *)");
}

//____________________________________________________________________________
void TNewQueryDlg::OnElementSelected(TObject *obj)
{
   if (obj) {
      fChain = obj;
      if (obj->IsA() == TChain::Class())
         fTxtChain->SetText(((TChain *)fChain)->GetName());
      else if (obj->IsA() == TDSet::Class())
         fTxtChain->SetText(((TDSet *)fChain)->GetObjName());
   }
}

//______________________________________________________________________________
void TNewQueryDlg::OnBrowseSelector()
{
   TGFileInfo fi;
   fi.fFileTypes = filetypes;
   new TGFileDialog(fClient->GetRoot(), this, kFDOpen, &fi);
   if (!fi.fFilename) return;
   fTxtSelector->SetText(gSystem->BaseName(fi.fFilename));
}

//______________________________________________________________________________
void TNewQueryDlg::OnBrowseParFile()
{
   TGFileInfo fi;
   fi.fFileTypes = partypes;
   new TGFileDialog(fClient->GetRoot(), this, kFDOpen, &fi);
   if (!fi.fFilename) return;
   fTxtParFile->SetText(gSystem->BaseName(fi.fFilename));
}

//______________________________________________________________________________
void TNewQueryDlg::OnBrowseEventList()
{

}

//______________________________________________________________________________
void TNewQueryDlg::OnBtnSaveClicked()
{
   Int_t nbElements = 0;
   TQueryDescription *newquery;
   if (fEditMode && fQuery)
      newquery = fQuery;
   else
      newquery = new TQueryDescription();

   newquery->fSelectorString  = fTxtSelector->GetText();
   if (fChain) {
      newquery->fTDSetString  = fChain->GetName();
      newquery->fChain        = fChain;
   }
   else {
      newquery->fTDSetString  = "";
      newquery->fChain         = 0;
   }
   newquery->fQueryName       = fTxtQueryName->GetText();
   newquery->fOptions         = fTxtOptions->GetText();
   newquery->fParFile         = fTxtParFile->GetText();
   newquery->fNoEntries       = fNumEntries->GetIntNumber();
   newquery->fFirstEntry      = fNumFirstEntry->GetIntNumber();

   if (!fEditMode) {
      newquery->fResult = 0;
      newquery->fStatus = TQueryDescription::kSessionQueryCreated;
      fViewer->GetActDesc()->fQueries->Add((TObject *)newquery);
      TGListTreeItem *item = fViewer->GetSessionHierarchy()->FindChildByData(
         fViewer->GetSessionItem(), fViewer->GetActDesc());
      TGListTreeItem *item2 = fViewer->GetSessionHierarchy()->AddItem(item,
         newquery->fQueryName, fViewer->GetQueryConPict(), fViewer->GetQueryConPict());
      item2->SetUserData(newquery);
      fViewer->GetSessionHierarchy()->ClearHighlighted();
      fViewer->GetSessionHierarchy()->HighlightItem(item2);
      fViewer->GetSessionHierarchy()->SetSelected(item2);
      fViewer->OnListTreeClicked(item2, 1, 0, 0);
   }
   else {
      TGListTreeItem *item = fViewer->GetSessionHierarchy()->GetSelected();
      item->SetUserData(newquery);
   }
   if (newquery->fChain) {
      if (newquery->fChain->IsA() == TChain::Class())
         nbElements = ((TChain *)newquery->fChain)->GetEntries();
      else if (newquery->fChain->IsA() == TDSet::Class())
         nbElements = ((TDSet *)newquery->fChain)->GetListOfElements()->GetSize();
   }
   fViewer->GetSessionFrame()->SetNumberOfFiles(nbElements);
   fViewer->GetSessionFrame()->SetEntries(newquery->fNoEntries);
   fViewer->GetSessionFrame()->SetFirstEntry(newquery->fFirstEntry);

   fClient->NeedRedraw(fViewer->GetSessionHierarchy());
   fTxtQueryName->SelectAll();
   fTxtQueryName->SetFocus();
}

//______________________________________________________________________________
void TNewQueryDlg::OnBtnSubmitClicked()
{
   OnBtnSaveClicked();
   fViewer->GetQueryFrame()->OnBtnSubmit();
}

//______________________________________________________________________________
void TNewQueryDlg::OnBtnCloseClicked()
{
   CloseWindow();
}

//______________________________________________________________________________
void TNewQueryDlg::Popup()
{
   MapWindow();
   fTxtQueryName->SetFocus();
}

//______________________________________________________________________________
void TNewQueryDlg::UpdateFields(TQueryDescription *desc)
{
   fQuery = desc;
   fTxtQueryName->SetText(desc->fQueryName);
   fTxtChain->SetText(desc->fTDSetString);
   fTxtSelector->SetText(desc->fSelectorString);
   fTxtOptions->SetText(desc->fOptions);
   fNumEntries->SetIntNumber(desc->fNoEntries);
   fNumFirstEntry->SetIntNumber(desc->fFirstEntry);
   fTxtParFile->SetText(desc->fParFile);
   fTxtEventList->SetText(desc->fEventList);
}
//______________________________________________________________________________
Bool_t TNewQueryDlg::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Process messages for new query dialog
   // essentially used to navigate between text entry fields

   switch (GET_MSG(msg)) {
      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
            case kTE_ENTER:
            case kTE_TAB:
               switch (parm1) {
                  case 1:
                     fTxtChain->SelectAll();
                     fTxtChain->SetFocus();
                     break;
                  case 2:
                     fTxtSelector->SelectAll();
                     fTxtSelector->SetFocus();
                     break;
                  case 3:
                     fTxtOptions->SelectAll();
                     fTxtOptions->SetFocus();
                     break;
                  case 4:
                     fTxtParFile->SelectAll();
                     fTxtParFile->SetFocus();
                     break;
                  case 5:
                     fTxtEventList->SelectAll();
                     fTxtEventList->SetFocus();
                     break;
                  case 6:
                     fTxtQueryName->SelectAll();
                     fTxtQueryName->SetFocus();
                     break;
               }
               break;

            default:
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

