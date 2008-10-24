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
class TProofOutputFile;
class TString;
class TList;
class TFile;
class TFileMerger;

class TProofOutputFile : public TNamed {

friend class TProof;
friend class TProofPlayer;

public:

private:
   TProofOutputFile(const TProofOutputFile&); // Not implemented
   TProofOutputFile& operator=(const TProofOutputFile&); // Not implemented

   TString  fDir;         // name of the directory
   TString  fFileName;
   TString  fFileName1;
   TString  fLocation;
   TString  fMode;
   TString  fOutputFileName;
   TString  fWorkerOrdinal;
   Bool_t   fIsLocal;     // kTRUE if the file is in the sandbox
   Bool_t   fMerged;

   TFileMerger *fMerger;  // Instance of the file merger for mode "CENTRAL"

   TString GetTmpName(const char* name);

   void ResolveKeywords(TString &fname);
   void SetFileName(const char* name);
   void SetDir(const char* dir) { fDir = dir; }
   void SetWorkerOrdinal(const char* ordinal) { fWorkerOrdinal = ordinal; }

   void AddFile(TFileMerger *merger, const char *path);
   void NotifyError(const char *errmsg);
   void Unlink(const char *path);

protected:

public:
   TProofOutputFile() : fDir(), fFileName(), fFileName1(), fLocation(),
     fMode(), fOutputFileName(), fWorkerOrdinal(), fIsLocal(kFALSE), fMerged(kFALSE),
     fMerger(0) {}

   TProofOutputFile(const char* path,
                    const char* location = "REMOTE", const char* mode = "CENTRAL");
   virtual ~TProofOutputFile();

   const char* GetDir() const { return fDir; }
   TFileMerger* GetFileMerger(Bool_t local = kFALSE); // Instance of the file merger for mode "CENTRAL"
   const char* GetFileName(Bool_t tmpName = kTRUE) const { return (tmpName) ? fFileName1 : fFileName; }
   const char* GetLocation() const { return fLocation; }
   const char* GetMode() const { return fMode; }
   const char* GetOutputFileName() const { return fOutputFileName; }
   const char* GetWorkerOrdinal() const { return fWorkerOrdinal; }


   Int_t AdoptFile(TFile *f);                    // Adopt a TFile already open
   TFile* OpenFile(const char* opt);             // Open a file with the specified name in fFileName1
   Long64_t Merge(TCollection* list);
   void Print(Option_t *option="") const;
   void SetOutputFileName(const char *name);


   ClassDef(TProofOutputFile,1) // Wrapper class to steer the merging of files produced on workers
};

#endif
