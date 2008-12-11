// @(#)root/sessionviewer:$Id$
// Author: G Ganis, Aug 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofProgressLog
#define ROOT_TProofProgressLog

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// TProofProgressLog                                                     //
//                                                                       //
// Dialog used to display Proof session logs from the Proof progress     //
// dialog.                                                               //
// It uses TProofMgr::GetSessionLogs() mechanism internally              //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGTextView;
class TGTextButton;
class TProofProgressDialog;
class TProofLog;
class TGTextEntry;
class TGNumberEntry;
class TGListBox;
class TGSplitButton;

class TProofProgressLog : public TGTransientFrame {

private:
   enum ETextType { kRaw = 0, kStd = 1, kGrep = 2 };

   TGTextView           *fText;      // text widget
   TGTextButton         *fClose;     // close button
   TGListBox            *fLogList;   // list of workers
   TGTextButton         *fLogNew;    // display logs button
   TProofProgressDialog *fDialog;    // owner dialog
   TProofLog            *fProofLog;  // the log
   TGNumberEntry        *fLinesFrom; // starting line
   TGNumberEntry        *fLinesTo;   // ending line
   TGTextEntry          *fGrepText;  // text to grep for in the logs
   TGTextEntry          *fFileName;  // file to save to
   TGTextButton         *fSave;      // save button
   TGTextButton         *fGrepButton; //grep button
   TGCheckButton        *fAllLines;  // display all lines button
   TGCheckButton        *fRawLines;  // display raw lines button
   TGSplitButton        *fAllWorkers; // display all workers button

   Bool_t                fFullText;    // 0 - when grep was called
   Int_t                 fTextType;   // Type of retrieval

public:
   TProofProgressLog(TProofProgressDialog *d, Int_t w = 700, Int_t h = 300);
   virtual ~TProofProgressLog();

   TGListBox* BuildLogList(TGFrame *parent);
   void       DoLog(Bool_t grep=kFALSE);
   void       LogMessage(const char *msg, Bool_t all);

   void   LoadBuffer(const char *buffer);
   void   AddBuffer(const char *buffer);

   void   LoadFile(const char *file);

   void   Clear(Option_t * = 0);
   void   Popup();
   void   SaveToFile();
   void   NoLineEntry();
   void   Select(Int_t id);
   // slots
   void   CloseWindow();

   ClassDef(TProofProgressLog,0) //Class implementing a log graphic box
};

#endif
