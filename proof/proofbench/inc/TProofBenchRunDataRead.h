// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofBenchRunDataRead
#define ROOT_TProofBenchRunDataRead

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBenchRunDataRead                                               //
//                                                                      //
// I/O-intensive PROOF benchmark test reads in event files distributed  //
// on the cluster. Number of events processed per second and size of    //
// events processed per second are plotted against number of active     //
// workers. Performance rate for unit packets and performance rate      //
// for query are plotted.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TProofBenchRun
#include "TProofBenchRun.h"
#endif

class TProof;
class TCanvas;
class TH2;
class TProfile;
class TTree;
class TFileCollection;

class TProofBenchMode;
class TProofBenchDataSet;
class TProofNodes;
class TPBReadType;

class TProofBenchRunDataRead : public TProofBenchRun {

private:
   TProof* fProof;               //pointer to proof

   TPBReadType *fReadType; //read type
   TProofBenchDataSet *fDS;              //dataset operations handler

   Long64_t fNEvents;            //number of events per file
   Int_t fNTries;                //number of tries
   Int_t fStart;                 //start number of workers
   Int_t fStop;                  //stop number of workers
   Int_t fStep;                  //test to be performed every fStep workers
   Int_t fDebug;                 //debug switch
   Int_t fFilesPerWrk;           //# of files to be processed per worker

   TDirectory  *fDirProofBench;   //directory for proof outputs

   TProofNodes *fNodes;                //list of nodes information

   TList        *fListPerfPlots;            //list of performance plots
   TProfile     *fProfile_perfstat_event;
   TH2          *fHist_perfstat_event;
   TProfile     *fProfile_queryresult_event;
   TProfile     *fNorm_queryresult_event;
   TProfile     *fProfile_perfstat_IO;
   TH2          *fHist_perfstat_IO;
   TProfile     *fProfile_queryresult_IO;
   TProfile     *fNorm_queryresult_IO;

   TCanvas *fCPerfProfiles;      //canvas for performance profile histograms

   TString fName;                //name of this run

   void BuildHistos(Int_t start, Int_t stop, Int_t step, Bool_t nx);

protected:

   void FillPerfStatProfiles(TTree* t, Int_t nactive);

   Int_t SetParameters();
   Int_t DeleteParameters();

public:

   TProofBenchRunDataRead(TProofBenchDataSet *pbds, TPBReadType *readtype = 0,
                          TDirectory* dirproofbench=0, TProof* proof=0, TProofNodes* nodes=0,
                          Long64_t nevents=-1, Int_t ntries=2, Int_t start=1, Int_t stop=-1,
                          Int_t step=1, Int_t debug=0);

   virtual ~TProofBenchRunDataRead();

   void Run(Long64_t, Int_t, Int_t, Int_t, Int_t, Int_t, Int_t) { }
   void Run(const char *dset, Int_t start, Int_t stop, Int_t step, Int_t ntries,
            Int_t debug, Int_t);

   TFileCollection *GetDataSet(const char *dset, Int_t nact, Bool_t nx);

   void DrawPerfProfiles();

   void Print(Option_t* option="") const;

   void SetReadType(TPBReadType *readtype) { fReadType = readtype; }
   void SetNEvents(Long64_t nevents) { fNEvents = nevents; }
   void SetNTries(Int_t ntries) { fNTries = ntries; }
   void SetStart(Int_t start) { fStart = start; }
   void SetStop(Int_t stop) { fStop = stop; }
   void SetStep(Int_t step) { fStep = step; }
   void SetDebug(Int_t debug) { fDebug = debug; }
   void SetDirProofBench(TDirectory* dir) { fDirProofBench = dir; }
   void SetFilesPerWrk(Int_t fpw) { fFilesPerWrk = fpw; }

   TPBReadType *GetReadType() const { return fReadType; }
   Long64_t GetNEvents() const { return fNEvents; }
   Int_t GetNTries() const { return fNTries; }
   Int_t GetStart() const { return fStart; }
   Int_t GetStop() const { return fStop; }
   Int_t GetStep() const { return fStep; }
   Int_t GetDebug() const { return fDebug; }
   TDirectory* GetDirProofBench() const { return fDirProofBench; }
   TCanvas* GetCPerfProfiles() const { return fCPerfProfiles; }
   const char* GetName() const { return fName; }

   TString GetNameStem() const;

   ClassDef(TProofBenchRunDataRead,0)         //IO-intensive PROOF benchmark
};

#endif
