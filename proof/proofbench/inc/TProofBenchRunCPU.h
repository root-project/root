// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofBenchRunCPU
#define ROOT_TProofBenchRunCPU

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBenchRunCPU                                                    //
//                                                                      //
// CPU-intensive PROOF benchmark test generates events and fill 1, 2,   //
//or 3-D histograms. No I/O activity is involved.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TProofBenchRun
#include "TProofBenchRun.h"
#endif


class TCanvas;
class TList;
class TProof;
class TProfile;
class TLegend;
class TH2;
class TTree;

class TProofBenchMode;
class TProofNodes;
class TPBHistType;

class TProofBenchRunCPU : public TProofBenchRun {

private:

   TPBHistType *fHistType;              //histogram type
   Int_t        fNHists;                //number of histograms

   Long64_t     fNEvents;               //number of events to generate
   Int_t        fNTries;                //number of tries

   Int_t        fStart;                 //start number of workers to scan
   Int_t        fStop;                  //stop number of workers to scan
   Int_t        fStep;                  //test to be performed every fStep workers

   Int_t        fDraw;                  //draw switch
   Int_t        fDebug;                 //debug switch

   TDirectory*  fDirProofBench;         //directory for proof outputs

   TProofNodes* fNodes;                 //node information

   TList*       fListPerfPlots;         //list of performance plots
   TCanvas*     fCanvas;                //canvas for performance plots

   TProfile    *fProfile_perfstat_event;
   TH2         *fHist_perfstat_event;
   TProfile    *fProfile_perfstat_evtmax;
   TProfile    *fNorm_perfstat_evtmax;
   TProfile    *fProfile_queryresult_event;
   TProfile    *fNorm_queryresult_event;
   TProfile    *fProfile_cpu_eff;

   TLegend     *fProfLegend;            // Legend for profiles
   TLegend     *fNormLegend;            // Legend for norms

   TString      fName;                  //name of CPU run

   void BuildHistos(Int_t start, Int_t stop, Int_t step, Bool_t nx);

protected:

   void FillPerfStatPerfPlots(TTree* t, Int_t nactive);

   Int_t SetParameters();
   Int_t DeleteParameters();

public:

   TProofBenchRunCPU(TPBHistType *histtype = 0,
                     Int_t nhists=16, TDirectory* dirproofbench=0,
                     TProof* proof=0, TProofNodes* nodes=0,
                     Long64_t nevents=1000000, Int_t ntries=2, Int_t start=1,
                     Int_t stop=-1, Int_t step=1, Int_t draw=0, Int_t debug=0);

   virtual ~TProofBenchRunCPU();

   void Run(Long64_t nevents, Int_t start, Int_t stop, Int_t step, Int_t ntries,
            Int_t debug, Int_t draw);
   void Run(const char *, Int_t, Int_t, Int_t, Int_t, Int_t, Int_t) { }

   void DrawPerfPlots();

   void Print(Option_t* option="") const;

   void SetHistType(TPBHistType *histtype);
   void SetNHists(Int_t nhists) { fNHists = nhists; }
   void SetNEvents(Long64_t nevents) { fNEvents = nevents; }
   void SetNTries(Int_t ntries) { fNTries = ntries; }
   void SetStart(Int_t start) { fStart = start; }
   void SetStop(Int_t stop) { fStop = stop; }
   void SetStep(Int_t step) { fStep = step; }
   void SetDraw(Int_t draw) { fDraw = draw; }
   void SetDebug(Int_t debug) { fDebug = debug; }

   void SetDirProofBench(TDirectory* dir) { fDirProofBench = dir; }

   TPBHistType *GetHistType() const { return fHistType; }
   Int_t GetNHists() const { return fNHists; }
   Long64_t GetNEvents() const { return fNEvents; }
   Int_t GetNTries() const { return fNTries; }
   Int_t GetStart() const { return fStart; }
   Int_t GetStop() const { return fStop; }
   Int_t GetStep() const { return fStep; }
   Int_t GetDraw() const { return fDraw; }
   Int_t GetDebug() const { return fDebug; }
   TDirectory* GetDirProofBench() const { return fDirProofBench; }
   TList* GetListPerfPlots() const { return fListPerfPlots; }
   TCanvas* GetCanvas() const { return fCanvas; }
   const char* GetName() const { return fName; }

   TString GetNameStem() const;

   ClassDef(TProofBenchRunCPU,0)     //CPU-intensive PROOF benchmark
};

#endif
