// @(#)root/sessionviewer:$Id$
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


#include "TTime.h"
#include "TString.h"

class TGTransientFrame;
class TGProgressBar;
class TGTextButton;
class TGCheckButton;
class TGLabel;
class TGTextBuffer;
class TGTextEntry;
class TProof;
class TProofProgressLog;
class TProofProgressMemoryPlot;
class TNtuple;
class TGraph;
class TGSpeedo;

class TProofProgressDialog {

   friend class TProofProgressLog;
   friend class TProofProgressMemoryPlot;

private:
   enum EQueryStatus { kRunning = 0, kDone, kStopped, kAborted, kIncomplete };

   TGTransientFrame   *fDialog;  // transient frame, main dialog window
   TGProgressBar      *fBar;     // progress bar
   TGTextButton       *fClose;
   TGTextButton       *fStop;
   TGTextButton       *fAbort;
   TGTextButton       *fAsyn;
   TGTextButton       *fLog;
   TGTextButton       *fRatePlot;
   TGTextButton       *fMemPlot;
   TGTextButton       *fUpdtSpeedo;
   TGCheckButton      *fKeepToggle;
   TGCheckButton      *fLogQueryToggle;
   TGTextBuffer       *fTextQuery;
   TGTextEntry        *fEntry;
   TGLabel            *fTitleLab;
   TGLabel            *fFilesEvents;
   TGLabel            *fTimeLab;
   TGLabel            *fProcessed;
   TGLabel            *fEstim;
   TGLabel            *fTotal;
   TGLabel            *fRate;
   TGLabel            *fInit;
   TGLabel            *fSelector;
   Bool_t              fSpeedoEnabled;    // whether to enable the speedometer
   TGSpeedo           *fSpeedo;           // speedometer
   TGCheckButton      *fSmoothSpeedo;     // use smooth speedometer update
   TProofProgressLog  *fLogWindow;        // transient frame for logs
   TProofProgressMemoryPlot *fMemWindow;  // transient frame for memory plots
   TProof             *fProof;
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
   TNtuple            *fRatePoints;
   TGraph             *fRateGraph;
   TGraph             *fMBRtGraph;
   TGraph             *fActWGraph;
   TGraph             *fTotSGraph;
   TGraph             *fEffSGraph;
   Float_t             fInitTime;
   Float_t             fProcTime;
   Double_t            fAvgRate;
   Double_t            fAvgMBRate;
   Int_t               fRightInfo;

   TString             fSessionUrl;

   Float_t             AdjustBytes(Float_t mbs, TString &sf);

   static Bool_t       fgKeepDefault;
   static Bool_t       fgLogQueryDefault;
   static TString      fgTextQueryDefault;

public:
   TProofProgressDialog(TProof *proof, const char *selector,
                        Int_t files, Long64_t first, Long64_t entries);
   virtual ~TProofProgressDialog();

   void ResetProgressDialog(const char *sel, Int_t sz, Long64_t fst, Long64_t ent);
   void Progress(Long64_t total, Long64_t processed);
   void Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                 Float_t initTime, Float_t procTime,
                 Float_t evtrti, Float_t mbrti) {
                 Progress(total, processed, bytesread, initTime, procTime,
                          evtrti, mbrti, -1, -1, -1.); }
   void Progress(Long64_t total, Long64_t processed, Long64_t bytesread,
                 Float_t initTime, Float_t procTime,
                 Float_t evtrti, Float_t mbrti, Int_t actw, Int_t tses, Float_t eses);
   void DisableAsyn();
   void IndicateStop(Bool_t aborted);
   void LogMessage(const char *msg, Bool_t all);

   void CloseWindow();
   void DoClose();
   void DoLog();
   void DoKeep(Bool_t on);
   void DoSetLogQuery(Bool_t on);
   void DoStop();
   void DoAbort();
   void DoAsyn();
   void DoPlotRateGraph();
   void DoMemoryPlot();
   void DoEnableSpeedo();
   void ToggleOdometerInfos();
   void ToggleThreshold();

   ClassDef(TProofProgressDialog,0)  //PROOF progress dialog
};

#endif
