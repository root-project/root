// @(#)root/proof:$Id$
// Author: G. Ganis Oct 2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofBenchTypes
#define ROOT_TProofBenchTypes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ProofBenchTypes                                                      //
// Const and types used by ProofBench and its selectors                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

const char* const kPROOF_BenchCPUSelPar  = "ProofBenchCPUSel";  // PAR with bench CPU selectors
const char* const kPROOF_BenchDataSelPar = "ProofBenchDataSel";  // PAR with bench data selectors
const char* const kPROOF_BenchSrcDir     = "proof/proofbench/src/";  // Dir with ProofBench Src files
const char* const kPROOF_BenchIncDir     = "proof/proofbench/inc/";  // Dir with ProofBench Inc files
const char* const kPROOF_BenchSelCPUDef  = "TSelHist";   // default CPU selector
const char* const kPROOF_BenchSelDataDef = "TSelEvent";  // default Data Read selector
const char* const kPROOF_BenchSelDataGenDef = "TSelEventGen";  // default Data generator selector

class TPBReadType : public TObject {
public:
   enum EReadType {
      kReadNotSpecified = 0,                    //read type not specified
      kReadFull = 1,                            //read in a full event
      kReadOpt = 2,                             //read in part of an event
      kReadNo = 4                               //do not read in event
   };
private:
   EReadType fType;
   TString fName;
public:
   TPBReadType(EReadType type = kReadOpt) : fType(type), fName("PROOF_Benchmark_ReadType") { }
   virtual ~TPBReadType() { }

   EReadType GetType() const { return fType; }
   Bool_t IsReadFull() const { return (fType == kReadFull) ? kTRUE : kFALSE; }
   Bool_t IsReadOpt() const { return (fType == kReadOpt) ? kTRUE : kFALSE; }
   Bool_t IsReadNo() const { return (fType == kReadNo) ? kTRUE : kFALSE; }
   const char *GetName() const { return fName; }

   ClassDef(TPBReadType, 1)     // Streamable PBReadType
};

class TPBHistType : public TObject {
public:
   enum EHistType {
      kHistNotSpecified = 0,                  //histogram type not specified
      kHist1D = 1,                            //1-D histogram
      kHist2D = 2,                            //2-D histogram
      kHist3D = 4,                            //3-D histogram
      kHistAll = kHist1D | kHist2D | kHist3D  //1-D, 2-D and 3-D histograms
   };
private:
   EHistType fType;
   TString fName;
public:
   TPBHistType(EHistType type = kHist1D) : fType(type), fName("PROOF_Benchmark_HistType") { }
   virtual ~TPBHistType() { }

   EHistType GetType() const { return fType; }
   Bool_t IsHist1D() const { return (fType == kHist1D) ? kTRUE : kFALSE; }
   Bool_t IsHist2D() const { return (fType == kHist2D) ? kTRUE : kFALSE; }
   Bool_t IsHist3D() const { return (fType == kHist3D) ? kTRUE : kFALSE; }
   Bool_t IsHistAll() const { return (fType == kHistAll) ? kTRUE : kFALSE; }
   const char *GetName() const { return fName; }

   ClassDef(TPBHistType, 1)     // Streamable PBHistType
};

class TPBHandleDSType : public TObject {
public:
   enum EHandleDSType {
      kReleaseCache   = 0,               // Release memory cache for the given file
      kCheckCache     = 1,               // Check memory cache for the given files
      kRemoveFiles    = 2,               // Remove (physically) the given files
      kCopyFiles      = 3                // Copy the given files to a destination dir
   };
private:
   EHandleDSType fType;
   TString fName;
public:
   TPBHandleDSType(EHandleDSType type = kReleaseCache) : fType(type), fName("PROOF_Benchmark_HandleDSType") { }
   virtual ~TPBHandleDSType() { }

   EHandleDSType GetType() const { return fType; }
   Bool_t IsReleaseCache() const { return (fType == kReleaseCache) ? kTRUE : kFALSE; }
   Bool_t IsCheckCache() const { return (fType == kCheckCache) ? kTRUE : kFALSE; }
   Bool_t IsRemoveFiles() const { return (fType == kRemoveFiles) ? kTRUE : kFALSE; }
   Bool_t IsCopyFiles() const { return (fType == kCopyFiles) ? kTRUE : kFALSE; }
   const char *GetName() const { return fName; }

   ClassDef(TPBHandleDSType, 1)     // Streamable PBHandleDSType
};

#endif
