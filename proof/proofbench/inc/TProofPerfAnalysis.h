// @(#)root/proofx:$Id$
// Author: G.Ganis Nov 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofPerfAnalysis
#define ROOT_TProofPerfAnalysis

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPerfAnalysis                                                       //
//                                                                      //
// Set of tools to analyse the performance tree                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TSortedList
#include "TSortedList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TFile;
class TH1F;
class TH2F;
class TList;
class TTree;
class TProofPerfAnalysis : public TNamed {

public:              // public because of Sun CC bug
   class TWrkInfo;

private:
   TFile  *fFile;                // The open performance file
   TString fDirName;             // The name of the subdir with the perfomance tree
   TString fTreeName;            // The name of the performance tree
   TTree  *fTree;                // The performance tree
   TSortedList fWrksInfo;        // Sorted list of workers info
   Float_t fInitTime;            // End of initialization time for this query
   Float_t fMergeTime;           // Begin of merging time for this query
   Float_t fMaxTime;             // Max time for this query (slowest worker)
   TH1F   *fEvents;              // Event distribution per worker
   TH1F   *fPackets;             // Packet distribution per worker
   Double_t fEvtRateMax;         // Max event processing rate per packet
   Double_t fMBRateMax;          // Max MB processing rate per packet
   Double_t fLatencyMax;         // Max retrieval latency per packet

   static Int_t fgDebug;         // Verbosity level

   Int_t CompareOrd(const char *ord1, const char *ord2);
   void  FillFileDist(TH1F *hf, TH1F *hb, TH2F *hx, Bool_t wdet = kFALSE);
   void  FillFileDistOneSrv(TH1F *hx, Bool_t wdet = kFALSE);
   void  FillWrkInfo(Bool_t force = kFALSE);
   TString GetCanvasTitle(const char *t);
   void  GetFileInfo(TList *wl, TList *sl);
   void  LoadTree(TDirectory *dir);

public:

   TProofPerfAnalysis(const char *perffile, const char *title = "",
                  const char *treename = "PROOF_PerfStats");
   virtual ~TProofPerfAnalysis();

   Bool_t IsValid() const { return (fFile && fTree) ? kTRUE : kFALSE; } 
   Bool_t WrkInfoOK() const { return (fWrksInfo.GetSize() > 0) ? kTRUE : kFALSE; } 
   
   void  EventDist();                          // Analyse event and packet distribution
   void  FileDist(Bool_t writedet = kFALSE);   // Analyse the file distribution
   void  LatencyPlot(const char *wrks = 0);    // Packet latency distribution vs time
   void  RatePlot(const char *wrks = 0);       // Rate distribution vs time
   void  WorkerActivity();                     // Analyse the worker activity
   void  PrintWrkInfo(Int_t showlast = 10);    // Print workers info
   void  PrintWrkInfo(const char *wrk);        // Print worker info by name
   
   void  SetDebug(Int_t d = 0);   // Setter for the verbosity level

   ClassDef(TProofPerfAnalysis, 0)   // Set of tools to analyse the performance tree
};

#endif
