// @(#)root/proof:$Name:  $:$Id: TProofProgressDialog.h,v 1.9 2005/12/12 12:54:27 rdm Exp $
// Author: Fons Rademakers   21/03/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofProgressDialog
#define ROOT_TProofProgressDialog


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressDialog                                                 //
//                                                                      //
// This class provides a query progress bar.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTime
#include "TTime.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TGTransientFrame;
class TGProgressBar;
class TGTextButton;
class TGCheckButton;
class TGLabel;
class TGTextBuffer;
class TGTextEntry;
class TVirtualProof;
class TProofProgressLog;


class TProofProgressDialog {

friend class TProofProgressLog;

private:
   enum EQueryStatus { kRunning = 0, kDone, kStopped, kAborted, kIncomplete };

   TGTransientFrame   *fDialog;  // transient frame, main dialog window
   TGProgressBar      *fBar;     // progress bar
   TGTextButton       *fClose;
   TGTextButton       *fStop;
   TGTextButton       *fAbort;
   TGTextButton       *fLog;
   TGCheckButton      *fKeepToggle;
   TGCheckButton      *fLogQueryToggle;
   TGTextBuffer       *fTextQuery;
   TGTextEntry        *fEntry;
   TGLabel            *fTitleLab;
   TGLabel            *fFilesEvents;
   TGLabel            *fProcessed;
   TGLabel            *fTotal;
   TGLabel            *fRate;
   TGLabel            *fSelector;
   TProofProgressLog  *fLogWindow;       // transient frame for logs
   TVirtualProof      *fProof;
   TTime               fStartTime;
   TTime               fEndTime;
   Long64_t            fPrevProcessed;
   Long64_t            fPrevTotal;
   Long64_t            fFirst;
   Long64_t            fEntries;
   Int_t               fFiles;
   EQueryStatus        fStatus;
   Bool_t              fKeep;
   Bool_t              fLogQuery;

   static Bool_t       fgKeepDefault;
   static Bool_t       fgLogQueryDefault;
   static TString      fgTextQueryDefault;

public:
   TProofProgressDialog(TVirtualProof *proof, const char *selector,
                        Int_t files, Long64_t first, Long64_t entries);
   virtual ~TProofProgressDialog();

   void ResetProgressDialog(const char *sel, Int_t sz, Long64_t fst, Long64_t ent);
   void Progress(Long64_t total, Long64_t processed);
   void IndicateStop(Bool_t aborted);
   void LogMessage(const char *msg, Bool_t all);

   void CloseWindow();
   void DoClose();
   void DoLog();
   void DoKeep(Bool_t on);
   void DoSetLogQuery(Bool_t on);
   void DoStop();
   void DoAbort();

   ClassDef(TProofProgressDialog,0)  //PROOF progress dialog
};

#endif
