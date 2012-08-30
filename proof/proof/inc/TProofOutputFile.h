// @(#)root/proof:$Id$
// Author: Long Tran-Thanh   14/09/07

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofOutputFile
#define ROOT_TProofOutputFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofOutputFile                                                           //
//                                                                      //
// Small class to steer the merging of files produced on workers        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TCollection;
class TString;
class TList;
class TFile;
class TFileCollection;
class TFileMerger;

class TProofOutputFile : public TNamed {

friend class TProof;
friend class TProofPlayer;
friend class TProofPlayerRemote;

public:
   enum ERunType {  kMerge        = 1,      // Type of run: merge or dataset creation
                    kDataset      = 2};
   enum ETypeOpt {  kRemote       = 1,      // Merge from original copies
                    kLocal        = 2,      // Make local copies before merging
                    kCreate       = 4,      // Create dataset
                    kRegister     = 8,      // Register dataset
                    kOverwrite    = 16,     // Force dataset replacement during registration
                    kVerify       = 32};    // Verify the registered dataset

private:
   TProofOutputFile(const TProofOutputFile&); // Not implemented
   TProofOutputFile& operator=(const TProofOutputFile&); // Not implemented

   TString  fDir;            // name of the directory to be exported
   TString  fRawDir;         // name of the local directory where to create the file
   TString  fFileName;
   TString  fOptionsAnchor;  // options and anchor string including delimiters, e.g. "?myopts#myanchor"
   TString  fOutputFileName;
   TString  fWorkerOrdinal;
   TString  fLocalHost;      // Host where the file was created
   Bool_t   fIsLocal;     // kTRUE if the file is in the sandbox
   Bool_t   fMerged;
   ERunType fRunType;     // Type of run (see enum ERunType)
   UInt_t   fTypeOpt;     // Option (see enum ETypeOpt)
   Bool_t   fMergeHistosOneGo;  // If true merge histos in one go (argument to TFileMerger)

   TFileCollection *fDataSet;  // Instance of the file collection in 'dataset' mode
   TFileMerger *fMerger;  // Instance of the file merger in 'merge' mode

   void Init(const char *path, const char *dsname);
   void SetFileName(const char* name) { fFileName = name; }
   void SetDir(const char* dir, Bool_t raw = kFALSE) { if (raw) { fRawDir = dir; } else { fDir = dir; } }
   void SetMerged(Bool_t merged = kTRUE) { fMerged = merged; }
   void SetWorkerOrdinal(const char* ordinal) { fWorkerOrdinal = ordinal; }

   void AddFile(TFileMerger *merger, const char *path);
   void NotifyError(const char *errmsg);
   void Unlink(const char *path);

protected:

public:
   enum EStatusBits {
      kOutputFileNameSet = BIT(16),
      kRetrieve          = BIT(17), // If set, the file is copied to the final destination via the client
      kSwapFile          = BIT(18)  // Set when the represented file is the result of the automatic
                                    // save-to-file functionality 
   };
   TProofOutputFile() : fDir(), fRawDir(), fFileName(), fOptionsAnchor(), fOutputFileName(),
                        fWorkerOrdinal(), fLocalHost(), fIsLocal(kFALSE), fMerged(kFALSE),
                        fRunType(kMerge), fTypeOpt(kRemote), fMergeHistosOneGo(kFALSE),
                        fDataSet(0), fMerger(0) { }
   TProofOutputFile(const char *path, const char *option = "M", const char *dsname = 0);
   TProofOutputFile(const char *path, ERunType type, UInt_t opt = kRemote, const char *dsname = 0);
   virtual ~TProofOutputFile();

   const char *GetDir(Bool_t raw = kFALSE) const { return (raw) ? fRawDir : fDir; }
   TFileCollection *GetFileCollection();
   TFileMerger *GetFileMerger(Bool_t local = kFALSE);
   const char *GetFileName() const { return fFileName; }
   const char *GetLocalHost() const { return fLocalHost; }
   const char *GetOptionsAnchor() const { return fOptionsAnchor; }
   const char *GetOutputFileName() const { return fOutputFileName; }
   const char *GetWorkerOrdinal() const { return fWorkerOrdinal; }

   ERunType    GetRunType() const { return fRunType; }
   UInt_t      GetTypeOpt() const { return fTypeOpt; }
   Bool_t      IsMerge() const { return (fRunType == kMerge) ? kTRUE : kFALSE; }
   Bool_t      IsMerged() const { return fMerged; }
   Bool_t      IsRegister() const { return ((fTypeOpt & kRegister) || (fTypeOpt & kVerify)) ? kTRUE : kFALSE; }

   Bool_t      IsRetrieve() const { return (TestBit(TProofOutputFile::kRetrieve)) ? kTRUE : kFALSE; }
   void        SetRetrieve(Bool_t on = kTRUE) { if (on) { SetBit(TProofOutputFile::kRetrieve);
                                                        } else { ResetBit(TProofOutputFile::kRetrieve); }}

   Int_t AdoptFile(TFile *f);                    // Adopt a TFile already open
   TFile* OpenFile(const char *opt);             // Open a file with the specified name in fFileName1
   Long64_t Merge(TCollection *list);
   void Print(Option_t *option = "") const;
   void SetOutputFileName(const char *name);
   void ResetFileCollection() { fDataSet = 0; }

   static Int_t AssertDir(const char *dirpath);

   ClassDef(TProofOutputFile,5) // Wrapper class to steer the merging of files produced on workers
};

#endif
