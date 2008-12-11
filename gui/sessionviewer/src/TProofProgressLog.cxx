// @(#)root/sessionviewer:$Id$
// Author: G Ganis, Jul 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TError.h"
#include "TGFrame.h"
#include "TGTextView.h"
#include "TGScrollBar.h"
#include "TGLabel.h"
#include "TProof.h"
#include "TProofProgressDialog.h"
#include "TProofProgressLog.h"
#include "TProofLog.h"
#include "TGNumberEntry.h"
#include "TGListBox.h"
#include "TGMenu.h"
#include "TGButton.h"

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// TProofProgressLog                                                     //
//                                                                       //
// Dialog used to display Proof session logs from the Proof progress     //
// dialog.                                                               //
// It uses TProofMgr::GetSessionLogs() mechanism internally              //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

ClassImp(TProofProgressLog)

//____________________________________________________________________________
TProofProgressLog::TProofProgressLog(TProofProgressDialog *d, Int_t w, Int_t h) :
   TGTransientFrame(gClient->GetRoot(), gClient->GetRoot(), w, h)
{
   // Create a window frame for log messages.

   fDialog = d;
   fProofLog = 0;
   fFullText = kTRUE;
   fTextType = kStd;
   // use hierarchical cleaning
   SetCleanup(kDeepCleanup);

   //The text window
   TGHorizontalFrame *htotal = new TGHorizontalFrame(this, w, h);
   TGVerticalFrame *vtextbox = new TGVerticalFrame(htotal, w, h);
   //fText = new TGTextView(this, w, h);
   fText = new TGTextView(vtextbox, w, h);
   vtextbox->AddFrame(fText, new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY, 3, 3, 3, 3));

   //The frame for choosing workers
   TGVerticalFrame *vworkers = new TGVerticalFrame(htotal);
   TGLabel *label1 = new TGLabel(vworkers,"Choose workers:");

   //The list of workers
   if (!(fLogList = BuildLogList(vworkers))) {
      SetBit(TObject::kInvalidObject);
      return;
   }
   fLogList->Resize(102,52);
   fLogList->SetMultipleSelections(kTRUE); 

   //The SelectAll/ClearAll button
   TGPopupMenu *pm = new TGPopupMenu(gClient->GetRoot());
   pm->AddEntry("Select All", 0);
   pm->AddEntry("Clear All", 1);

   fAllWorkers = new TGSplitButton(vworkers, new TGHotString("Select ...            "), pm);
   fAllWorkers->Connect("ItemClicked(Int_t)", "TProofProgressLog", this, 
                     "Select(Int_t)");
   fAllWorkers->SetSplit(kFALSE);
   //select all for the first display
   Select(0);

   //Display button
   fLogNew = new TGTextButton(vworkers, "&Display");
   fLogNew->Connect("Clicked()", "TProofProgressLog", this, "DoLog(=kFALSE)");
   //fLogNew->Resize(102, 20);
   // fLogNew->SetMargins(1, 1, 0, 1);
   fLogNew->SetTextColor(0xffffff, kFALSE);
   fLogNew->SetBackgroundColor(0x000044);
   vworkers->AddFrame(label1, new TGLayoutHints(kLHintsLeft | kLHintsTop, 7, 2, 5, 2));
   vworkers->AddFrame(fAllWorkers, new TGLayoutHints(kLHintsExpandX | kLHintsTop, 5, 2, 2, 2));
   vworkers->AddFrame(fLogList, new TGLayoutHints(kLHintsExpandX | kLHintsTop | kLHintsExpandY, 2, 2, 5, 2));
   vworkers->AddFrame(fLogNew, new TGLayoutHints(kLHintsExpandX | kLHintsTop , 2, 2, 1, 5));

   htotal->AddFrame(vworkers, new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandY, 2, 2, 2, 2));

   //The lower row of number entries and buttons
   TGHorizontalFrame *hflogbox = new TGHorizontalFrame(vtextbox, 550, 20);
   fClose = new TGTextButton(hflogbox, "  &Close  ");
   fClose->Connect("Clicked()", "TProofProgressLog", this, "CloseWindow()");
   hflogbox->AddFrame(fClose, new TGLayoutHints(kLHintsCenterY |
                                                kLHintsRight, 10, 2, 2, 2));

   //Saving to a file controls
   fSave = new TGTextButton(hflogbox, "&Save");
   fSave->Connect("Clicked()", "TProofProgressLog", this, "SaveToFile()");
   hflogbox->AddFrame(fSave, new TGLayoutHints(kLHintsCenterY | kLHintsRight, 4, 0, 0, 0));
   fFileName = new TGTextEntry(hflogbox);
   fFileName->SetText("<session-tag>.log");
   hflogbox->AddFrame(fFileName, new TGLayoutHints(kLHintsCenterY | kLHintsRight));
   TGLabel *label10 = new TGLabel(hflogbox, "Save to a file:");
   hflogbox->AddFrame(label10, new TGLayoutHints(kLHintsCenterY | kLHintsRight, 50, 2, 2, 2));

   //Choose the number of lines to display
   TGVerticalFrame *vlines = new TGVerticalFrame(hflogbox);
   TGHorizontalFrame *vlines_buttons = new TGHorizontalFrame(vlines);
   TGLabel *label2 = new TGLabel(vlines_buttons, "Lines:");
   vlines_buttons->AddFrame(label2, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));

   fAllLines = new TGCheckButton(vlines_buttons, "all");
   fAllLines->SetToolTipText("Retrieve all lines (service messages excluded)");
   fAllLines->SetState(kButtonUp);
   fAllLines->Connect("Clicked()", "TProofProgressLog", this, "NoLineEntry()");
   vlines_buttons->AddFrame(fAllLines, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));

   fRawLines = new TGCheckButton(vlines_buttons, "svcmsg");
   fRawLines->SetToolTipText("Retrieve all type of lines, service messages included");
   fRawLines->SetState(kButtonUp);
   vlines_buttons->AddFrame(fRawLines, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));

   TGLabel *label11 = new TGLabel(vlines_buttons, "From");
   vlines_buttons->AddFrame(label11, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));

   fLinesFrom = new TGNumberEntry(vlines_buttons, 0, 5, -1, TGNumberFormat::kNESInteger);
   fLinesFrom->SetIntNumber(-100);
   fLinesFrom->GetNumberEntry()->SetToolTipText("Negative values indicate \"tail\" action");


   vlines_buttons->AddFrame(fLinesFrom, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));

   TGLabel *label3 = new TGLabel(vlines_buttons, "to");
   vlines_buttons->AddFrame(label3, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));
   fLinesTo = new TGNumberEntry(vlines_buttons, 0, 5, -1, TGNumberFormat::kNESInteger);
   vlines_buttons->AddFrame(fLinesTo, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));
   vlines->AddFrame(vlines_buttons, new TGLayoutHints(kLHintsCenterY));
   hflogbox->AddFrame(vlines, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2));

   //Grep controls
   TGLabel *label4 = new TGLabel(hflogbox, "Grep for:");
   hflogbox->AddFrame(label4, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 5, 2, 2, 2));
   fGrepText = new TGTextEntry(hflogbox);
   hflogbox->AddFrame(fGrepText, new TGLayoutHints(kLHintsCenterY | kLHintsLeft));

   fGrepButton = new TGTextButton(hflogbox, "Grep");
   fGrepButton->Connect("Clicked()", "TProofProgressLog", this, "DoLog(=kTRUE)");
   hflogbox->AddFrame(fGrepButton, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4, 0, 0, 0));

   vtextbox->AddFrame(hflogbox, new TGLayoutHints(kLHintsBottom | kLHintsLeft | kLHintsExpandX, 2, 2, 2, 2));
   htotal->AddFrame(vtextbox, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY | kLHintsRight, 3, 3, 3, 3));
   AddFrame(htotal, new TGLayoutHints(kLHintsExpandX |
                                        kLHintsExpandY, 3, 3, 3, 3));

   char title[256] = {0};
   strcpy(title,Form("PROOF Processing Logs: %s",
                     (fDialog->fProof ? fDialog->fProof->GetMaster() : "<dummy>")));
   SetWindowName(title);
   SetIconName(title);

   MapSubwindows();

   Resize();

   Window_t wdummy;
   int ax, ay;
   gVirtualX->TranslateCoordinates(GetParent()->GetId(), fDialog->fDialog->GetId(),
       (Int_t)(((TGFrame *)GetParent())->GetWidth() + w),
       (Int_t)(((TGFrame *)GetParent())->GetHeight()- 3*h/2), ax, ay, wdummy);
   Move(ax, ay);

   Popup();
}

//____________________________________________________________________________
TProofProgressLog::~TProofProgressLog()
{
   // Destructor

   // Cleanup the log object
   SafeDelete(fProofLog);

   // Detach from owner dialog
   fDialog->fLogWindow = 0;
   fDialog->fProof->Disconnect("LogMessage(const char*,Bool_t)", this,
                               "LogMessage(const char*,Bool_t)");
}

//____________________________________________________________________________
void TProofProgressLog::Popup()
{
   // Show log window.

   MapWindow();
}

//____________________________________________________________________________
void TProofProgressLog::Clear(Option_t *)
{
   // Clear log window.

   if (fText)
      fText->Clear();
}

//____________________________________________________________________________
void TProofProgressLog::LoadBuffer(const char *buffer)
{
   // Load a text buffer in the window.

   if (fText)
      fText->LoadBuffer(buffer);
}

//____________________________________________________________________________
void TProofProgressLog::LoadFile(const char *file)
{
   // Load a file in the window.

   if (fText)
      fText->LoadFile(file);
}

//____________________________________________________________________________
void TProofProgressLog::AddBuffer(const  char *buffer)
{
   // Add text to the window.

   if (fText) {
      TGText txt;
      txt.LoadBuffer(buffer);
      fText->AddText(&txt);
   }
}

//____________________________________________________________________________
void TProofProgressLog::CloseWindow()
{
   // Handle close button or when closed via window manager action.

   DeleteWindow();
}

//______________________________________________________________________________
TGListBox* TProofProgressLog::BuildLogList(TGFrame *parent)
{
   // Build the list of workers. For this, extract the logs and take the names
   // of TProofLogElements

   TGListBox *c = 0;
   if (!fDialog) {
      Warning("BuildLogList", "dialog instance undefined - do nothing");
      return c;
   }
   TProofMgr *mgr = TProof::Mgr(fDialog->fSessionUrl.Data());
   if (!mgr || !mgr->IsValid()) {
      Warning("BuildLogList", "unable open a manager connection to %s",
                              fDialog->fSessionUrl.Data());
      return c;
   }
   if (!(fProofLog = mgr->GetSessionLogs())) {
      Warning("BuildLogList", "unable to get logs from %s",
                              fDialog->fSessionUrl.Data());
      return c;
   }
   // Create the list-box now
   c = new TGListBox(parent);

   TList *elem = fProofLog->GetListOfLogs();
   TIter next(elem);
   TProofLogElem *pe = 0;

   Int_t is = 0;
   while ((pe=(TProofLogElem*)next())){
      TUrl url(pe->GetTitle());
      TString buf = Form("%s %s", pe->GetName(), url.GetHost());
      c->AddEntry(buf.Data(), is);
      is++;
   }
   return c;

}

//______________________________________________________________________________
void TProofProgressLog::DoLog(Bool_t grep)
{
   // Display the logs

   Clear();

   TString greptext = fGrepText->GetText();
   Int_t from, to;
   if (fAllLines->IsOn()){
      from = 0;
      to = -1;
   } else {
      from = fLinesFrom->GetIntNumber();
      to = fLinesTo->GetIntNumber();
   }

   TProofMgr *mgr = 0;
   if (!grep) {
      if (!fProofLog || !fFullText ||
          ((fTextType != kRaw && fRawLines->IsOn())   ||
           (fTextType != kStd && !fRawLines->IsOn())) ||
          fDialog->fStatus==TProofProgressDialog::kRunning) {
         SafeDelete(fProofLog);
         if ((mgr = TProof::Mgr(fDialog->fSessionUrl.Data()))) {
            if (fRawLines->IsOn()) {
               fProofLog = mgr->GetSessionLogs(0, 0, 0);
               fTextType = kRaw;
            } else {
               fProofLog = mgr->GetSessionLogs();
               fTextType = kStd;
            }
         } else {
            Warning("DoLog", "unable to instantiate a TProofMgr for %s",
                             fDialog->fSessionUrl.Data());
         }
         if (fDialog->fStatus != TProofProgressDialog::kRunning)
            fFullText = kTRUE;
      }
   } else {
      SafeDelete(fProofLog);
      if ((mgr = TProof::Mgr(fDialog->fSessionUrl.Data()))) {
         fProofLog = mgr->GetSessionLogs(0, 0, greptext.Data());
      } else {
         Warning("DoLog", "unable to instantiate a TProofMgr for %s",
                          fDialog->fSessionUrl.Data());
      }
      fTextType = kGrep;
      if (fDialog->fStatus != TProofProgressDialog::kRunning)
         fFullText = kTRUE;
   }
   if (fProofLog) {
      TList *selected = new TList;
      fLogList->GetSelectedEntries(selected);
      TIter next(selected);
      TGTextLBEntry *selentry;
      Bool_t logonly = fProofLog->LogToBox();
      fProofLog->SetLogToBox(kTRUE);

      fProofLog->Connect("Prt(const char*)", "TProofProgressLog",
                           this, "LogMessage(const char*, Bool_t)");
      while ((selentry=(TGTextLBEntry*)next())){
         TString ord = selentry->GetText()->GetString();
         Int_t is = ord.Index(" ");
         if (is != kNPOS) ord.Remove(is);
         fProofLog->Display(ord.Data(), from, to);
      }
      fProofLog->SetLogToBox(logonly);
      fProofLog->Disconnect("Prt(const char*)", this, "LogMessage(const char*, Bool_t)");
      delete selected;
   }
}

//______________________________________________________________________________
void TProofProgressLog::LogMessage(const char *msg, Bool_t all)
{
   // Load/append a log msg in the log frame, if open

   if (all) {
      // load buffer
      LoadBuffer(msg);
   } else {
      // append
      AddBuffer(msg);
   }
}

//______________________________________________________________________________
void TProofProgressLog::SaveToFile()
{
   //Save the logs to a file 
   //Only the name of the file is taken, no expansion

   if (!fProofLog) DoLog();

   // File name: the default is <session-tag>.log
   TString filename = fFileName->GetText();
   if (filename.IsNull() || filename == "<session-tag>.log") {
      filename = (fDialog && fDialog->fProof) ? Form("%s.log", fDialog->fProof->GetName())
                                              : "proof.log";
   }

   TList *selected = new TList;
   fLogList->GetSelectedEntries(selected);
   TIter next(selected);
   TGTextLBEntry *selentry;
   Bool_t writemode=kTRUE;
   const char *option;
   TString ord;
   while ((selentry=(TGTextLBEntry*)next())){
      ord = "";
      const char *name = selentry->GetText()->GetString();
      Int_t i=0;
      while (name[i]!=' ' && i<10) i++;
      ord.Append(name, i);
      //open the file in "w" mode for the first time
      option = writemode ? "w" : "a";
      fProofLog->Save(ord.Data(), filename.Data(), option);
      writemode=kFALSE;
   }

   Info("SaveToFile", "logs saved to file %s", filename.Data());
   return;
}

//______________________________________________________________________________
void TProofProgressLog::NoLineEntry()
{
   //Enable/disable the line number entry

   if (fAllLines->IsOn()){
      //disable the line number entry
      fLinesFrom->SetState(kFALSE);
      fLinesTo->SetState(kFALSE);
   } else {
      fLinesFrom->SetState(kTRUE);
      fLinesTo->SetState(kTRUE);
   }
}

//______________________________________________________________________________
void TProofProgressLog::Select(Int_t id)
{
   //actions of select all/clear all button

   Int_t nen = fLogList->GetNumberOfEntries();
   Bool_t sel = id ? 0 : 1;

   for (Int_t ie=0; ie<nen; ie++) {
      fLogList->Select(ie, sel);
   }
}

