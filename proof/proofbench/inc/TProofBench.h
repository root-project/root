// @(#)root/proofx:$Id$
// Author: G.Ganis, S.Ryu Feb 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofBench
#define ROOT_TProofBench

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBench                                                          //
//                                                                      //
// Steering class for PROOF benchmarks                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TProofBenchTypes
#include "TProofBenchTypes.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TF1;
class TFile;
class TProof;
class TProofBenchRunCPU;
class TProofBenchRunDataRead;
class TProofBenchDataSet;

class TProofBench : public TObject {

private:
   Bool_t  fUnlinkOutfile;       // Whether to remove empty output files

protected:

   TProof* fProof;               // Proof
   TProof* fProofDS;             // Proof to be used for dataset actions
   TFile  *fOutFile;             // Output file
   TString fOutFileName;         // Name of the output file
   Int_t   fNtries;              // Number of times a measurement is repeated
   TPBHistType *fHistType;       // Type of histograms for CPU runs
   Int_t   fNHist;               // Number of histograms to be created in default CPU runs
   TPBReadType *fReadType;       // Type of read (partial, full)
   TString fDataSet;             // Name of the dataset
   Int_t   fNFilesWrk;           // Number of files generated files per worker
   Int_t   fNumWrkMax;           // Max number of workers (required for dynamic setups)

   TString fCPUSel;              // Selector to be used for CPU benchmarks
   TString fCPUPar;              // List of par files to be loaded for CPU benchmarks
   TString fDataSel;             // Selector to be used for data benchmarks
   TString fDataPar;             // List of par files to be loaded for data benchmarks
   TString fDataGenSel;          // Selector to be used for generate data for benchmarks
   TString fDataGenPar;          // List of par files to be loaded to generate data for benchmarks

   TProofBenchRunCPU      *fRunCPU; // Instance to run CPU scans
   TProofBenchRunDataRead *fRunDS;  // Instance to run data-read scans
   TProofBenchDataSet     *fDS;     // Instance to handle datasets operations

   Bool_t fDebug;                // Debug switch

   TNamed *fDescription;         // Strings describing the cluster for this test (saved in the output file)

   static TF1 *fgFp1;                 // Simple 1st degree polynomial
   static TF1 *fgFp1n;                // Normalized 1st degree
   static TF1 *fgFp2;                 // Simple 2nd degree polynomial
   static TF1 *fgFp2n;                // Normalized 2nd degree

   static void AssertFittingFun(Double_t mi, Double_t mx);

public:

   TProofBench(const char *url, const char *outfile = "<default>", const char *proofopt = 0);

   virtual ~TProofBench();

   Int_t RunCPU(Long64_t nevents=-1, Int_t start=-1, Int_t stop=-1, Int_t step=-1);
   Int_t RunCPUx(Long64_t nevents=-1, Int_t start=-1, Int_t stop=-1);
   Int_t RunDataSet(const char *dset = "BenchDataSet",
                    Int_t start = 1, Int_t stop = -1, Int_t step = 1);
   Int_t RunDataSetx(const char *dset = "BenchDataSet", Int_t start = 1, Int_t stop = -1);

   Int_t CopyDataSet(const char *dset, const char *dsetdst, const char *destdir);
   Int_t MakeDataSet(const char *dset = 0, Long64_t nevt = -1, const char *fnroot = "event",
                     Bool_t regenerate = kFALSE);
   Int_t ReleaseCache(const char *dset);
   Int_t RemoveDataSet(const char *dset);
                    
   void  CloseOutFile();
   Int_t OpenOutFile(Bool_t wrt = kFALSE, Bool_t verbose = kTRUE);
   Int_t SetOutFile(const char *outfile, Bool_t verbose = kTRUE);
   const char *GetOutFileName() const { return fOutFileName; }
   void  SetNFilesWrk(Int_t nf = 0) { fNFilesWrk = (nf > 0) ? nf : 4; }
   void  SetNTries(Int_t nt) { if (nt > 0) fNtries = nt; }
   void  SetHistType(TPBHistType *histtype) { fHistType = histtype; }
   void  SetNHist(Int_t nh) { fNHist = nh; }
   void  SetReadType(TPBReadType *readtype) { fReadType = readtype; }

   void  SetCPUSel(const char *sel) { fCPUSel = sel; }
   void  SetCPUPar(const char *par) { fCPUPar = par; }
   void  SetDataSel(const char *sel) { fDataSel = sel; }
   void  SetDataPar(const char *par) { fDataPar = par; }
   void  SetDataGenSel(const char *sel) { fDataGenSel = sel; }
   void  SetDataGenPar(const char *par) { fDataGenPar = par; }

   void  SetProofDS(TProof *p);

   void  SetDebug(Bool_t debug = kTRUE) { fDebug = debug; }

   Bool_t GetDebug() { return fDebug; }

   static void DrawCPU(const char *outfile, const char *opt = "std:", Bool_t verbose = kFALSE, Int_t dofit = 0);
   static void DrawDataSet(const char *outfile, const char *opt = "std:", const char *type = "mbs", Bool_t verbose = kFALSE);
   static void GetPerfSpecs(const char *path = ".", Int_t degfit = 1);

   ClassDef(TProofBench, 0)   // Steering class for PROOF benchmarks
};

#endif
