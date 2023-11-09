// @(#)root/io:$Id$
// Author: Philippe Canal, Witold Pokorski, and Guilherme Amadio

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TBufferMerger.hxx"

#include "TBufferFile.h"
#include "TError.h"
#include "TROOT.h"
#include "TVirtualMutex.h"

#include <utility>

namespace ROOT {

TBufferMerger::TBufferMerger(const char *name, Option_t *option, Int_t compress)
{
   // We cannot chain constructors or use in-place initialization here because
   // instantiating a TBufferMerger should not alter gDirectory's state.
   TDirectory::TContext ctxt;
   Init(std::unique_ptr<TFile>(TFile::Open(name, option, /* title */ name, compress)));
}

TBufferMerger::TBufferMerger(std::unique_ptr<TFile> output)
{
   Init(std::move(output));
}

void TBufferMerger::Init(std::unique_ptr<TFile> output)
{
   if (!output || !output->IsWritable() || output->IsZombie())
      Error("TBufferMerger", "cannot write to output file");

   fMerger.OutputFile(std::move(output));
}

TBufferMerger::~TBufferMerger()
{
   for (const auto &f : fAttachedFiles)
      if (!f.expired()) Fatal("TBufferMerger", " TBufferMergerFiles must be destroyed before the server");

   // Since we support purely incremental merging, Merge does not write the target objects
   // that are attached to the file (TTree and histograms) and thus we need to write them
   // now.
   if (TFile *out = fMerger.GetOutputFile())
      out->Write("",TObject::kOverwrite);
}

std::shared_ptr<TBufferMergerFile> TBufferMerger::GetFile()
{
   R__LOCKGUARD(gROOTMutex);
   std::shared_ptr<TBufferMergerFile> f(new TBufferMergerFile(*this));
   gROOT->GetListOfFiles()->Remove(f.get());
   fAttachedFiles.push_back(f);
   return f;
}

const char *TBufferMerger::GetMergeOptions()
{
   return fMerger.GetMergeOptions();
}


void TBufferMerger::SetMergeOptions(const TString& options)
{
   fMerger.SetMergeOptions(options);
}

void TBufferMerger::Merge(ROOT::TBufferMergerFile *memfile)
{
   std::lock_guard q(fMergeMutex);
   memfile->WriteStreamerInfo();
   fMerger.AddFile(memfile);
   fMerger.PartialMerge(TFileMerger::kAll | TFileMerger::kIncremental | TFileMerger::kDelayWrite |
                        TFileMerger::kKeepCompression);
   fMerger.Reset();
}

} // namespace ROOT
