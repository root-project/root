// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelHandleDataSet                                                    //
//                                                                      //
// PROOF selector for file cache release.                               //
// List of files to be cleaned for each node is provided by client.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#define TSelHandleDataSet_cxx

#include "TSelHandleDataSet.h"

#include "TDSet.h"
#include "TEnv.h"
#include "TFile.h"
#include "TMap.h"
#include "TParameter.h"
#include "TProofBenchTypes.h"
#include "TSystem.h"
#include "TUrl.h"
#include "errno.h"
#include <unistd.h>
#include <fcntl.h>


ClassImp(TSelHandleDataSet)

//______________________________________________________________________________
void TSelHandleDataSet::SlaveBegin(TTree *)
{
   // Init the type from the input parameters


   TObject *o = fInput->FindObject("PROOF_Benchmark_HandleDSType");
   if (o) fType = dynamic_cast<TPBHandleDSType *>(o);

   TNamed *n = dynamic_cast<TNamed *>(fInput->FindObject("PROOF_Benchmark_DestDir"));
   if (n) {
      fDestDir = n->GetTitle();
      if (gSystem->AccessPathName(fDestDir)) {
         // Create the directory
         if (gSystem->mkdir(fDestDir, kTRUE) != 0) {
            fDestDir = "";
            Error("SlaveBegin", "could not create dir '%s'", fDestDir.Data());
         } else {
            if (gSystem->AccessPathName(fDestDir, kWritePermission)) {
               fDestDir = "";
               Error("SlaveBegin", "dir '%s' is not writable by this process", fDestDir.Data());
            } else {
               Info("SlaveBegin", "dir '%s' successfully created", fDestDir.Data());
            }
         }
      }
   }

   // Use default if nothing found in the input list
   if (!fType) fType = new TPBHandleDSType(TPBHandleDSType::kReleaseCache);
}

//______________________________________________________________________________
void TSelHandleDataSet::ReleaseCache(const char *fn)
{
   // Release the memory cache associated with file 'fn'.

#if defined(R__LINUX)
   TString filename(fn);
   Int_t fd;
   fd = open(filename.Data(), O_RDONLY);
   if (fd > -1) {
      fdatasync(fd);
      posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      close(fd);
      Info("ReleaseCache", "file cache for file '%s' cleaned ...", filename.Data());
   } else {
      Error("ReleaseCache", "cannot open file '%s' for cache clean up; errno=%d",
                            filename.Data(), errno);
   }
#else
   Info("ReleaseCache", "dummy function: file '%s' untouched ...", fn);
#endif
   // Done
   return;
}

//______________________________________________________________________________
void TSelHandleDataSet::CheckCache(const char * /*fn*/)
{
   // Check the memory cache associated with file 'fn'.

   Warning("CheckCache", "cache checker not implemented yet");
   // Done
   return;
}

//______________________________________________________________________________
void TSelHandleDataSet::RemoveFile(const char *fn)
{
   // Physically remove the file 'fn'.

   if (!gSystem->AccessPathName(fn, kWritePermission)) {
      if (gSystem->Unlink(fn) != 0) {
         Error("RemoveFile", "problems removing file '%s' ...", fn);
      } else {
         if (!gSystem->AccessPathName(fn))
            Warning("RemoveFile", "'unlink' returned success but the file still exists ('%s')", fn);
      }
   } else {
      if (!gSystem->AccessPathName(fn)) {
         Error("RemoveFile", "file '%s' cannot removed by this process", fn);
      } else {
         Error("RemoveFile", "file '%s' does not exists", fn);
      }
   }
   // Done
   return;
}

//______________________________________________________________________________
void TSelHandleDataSet::CopyFile(const char *fn)
{
   // Copy file 'fn' to fDestDir

   // Check if we have a destination dir
   if (fDestDir.IsNull()) {
      Error("CopyFile", "destination dir undefined: file '%s' not copied", fn);
      return;
   }

   TString basefn = gSystem->BaseName(TUrl(fn, kTRUE).GetFile());
   TString dst = TString::Format("%s/%s", fDestDir.Data(), basefn.Data());
   if (!TFile::Cp(fn, dst.Data())) {
      Error("CopyFile", "problems copying file '%s' to '%s'", fn, dst.Data());
      return;
   }
   Info("CopyFile", "file '%s' created ...", dst.Data());

   // Done
   return;
}

//______________________________________________________________________________
Bool_t TSelHandleDataSet::Process(Long64_t entry)
{
   // The Process() function is called for each entry in the tree (or possibly
   // keyed object in the case of PROOF) to be processed. The entry argument
   // specifies which entry in the currently loaded tree is to be processed.
   // It can be passed to either TTree::GetEntry() or TBranch::GetEntry()
   // to read either all or the required parts of the data. When processing
   // keyed objects with PROOF, the object is already loaded and is available
   // via the fObject pointer.
   //
   // This function should contain the "body" of the analysis. It can contain
   // simple or elaborate selection criteria, run algorithms on the data
   // of the event and typically fill histograms.

   // WARNING when a selector is used with a TChain, you must use
   //  the pointer to the current TTree to call GetEntry(entry).
   //  The entry is always the local entry number in the current tree.
   //  Assuming that fChain is the pointer to the TChain being processed,
   //  use fChain->GetTree()->GetEntry(entry).

   TDSetElement *fCurrent = 0;
   TPair *elemPair = 0;
   if (fInput && (elemPair = dynamic_cast<TPair *>
                      (fInput->FindObject("PROOF_CurrentElement")))) {
      if ((fCurrent = dynamic_cast<TDSetElement *>(elemPair->Value()))) {
         Info("Process", "entry %lld: file: '%s'", entry, fCurrent->GetName());
      } else {
         Error("Process", "entry %lld: no file specified!", entry);
         return kFALSE;
      }
   }

   // Resolve the file type; this also adjusts names for Xrd based systems
   TUrl url(fCurrent->GetName());
   url.SetAnchor(0);
   TString lfname = gEnv->GetValue("Path.Localroot", "");
   TFile::EFileType type = TFile::GetType(url.GetUrl(), "", &lfname);
   if (type == TFile::kLocal &&
       strcmp(url.GetProtocol(),"root") && strcmp(url.GetProtocol(),"xrd"))
      lfname = url.GetFileAndOptions();

   if (fType->GetType() == TPBHandleDSType::kReleaseCache) {
      // Release the file cache
      if (type == TFile::kLocal) {
         ReleaseCache(lfname);
      } else if (type == TFile::kFile) {
         ReleaseCache(url.GetFile());
      } else {
         Error("Process",
               "attempt to call ReleaseCache for a non-local file: '%s'", url.GetUrl());
      }
   } else if (fType->GetType() == TPBHandleDSType::kCheckCache) {
      // Check the file cache
      if (type == TFile::kLocal) {
         CheckCache(lfname);
      } else if (type == TFile::kFile) {
         CheckCache(url.GetFile());
      } else {
         Error("Process",
               "attempt to call CheckCache for a non-local file: '%s'", url.GetUrl());
      }
   } else if (fType->GetType() == TPBHandleDSType::kRemoveFiles) {
      // Remove the file
      RemoveFile(url.GetFileAndOptions());
   } else if (fType->GetType() == TPBHandleDSType::kCopyFiles) {
      // Copy file
      CopyFile(url.GetFileAndOptions());
   } else {
      // Type unknown
      Warning("Process", "type: %d is unknown", fType->GetType());
   }

   return kTRUE;
}

