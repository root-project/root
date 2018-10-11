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
namespace Experimental {

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

   if (!fQueue.empty())
      Merge();
}

std::shared_ptr<TBufferMergerFile> TBufferMerger::GetFile()
{
   R__LOCKGUARD(gROOTMutex);
   std::shared_ptr<TBufferMergerFile> f(new TBufferMergerFile(*this));
   gROOT->GetListOfFiles()->Remove(f.get());
   fAttachedFiles.push_back(f);
   return f;
}

size_t TBufferMerger::GetQueueSize() const
{
   return fQueue.size();
}

void TBufferMerger::Push(TBufferFile *buffer)
{
   {
      std::lock_guard<std::mutex> lock(fQueueMutex);
      fBuffered += buffer->BufferSize();
      fQueue.push(buffer);
   }

   if (fBuffered > fAutoSave)
      Merge();
}

size_t TBufferMerger::GetAutoSave() const
{
   return fAutoSave;
}

void TBufferMerger::SetAutoSave(size_t size)
{
   fAutoSave = size;
}

void TBufferMerger::Merge()
{
   if (fMergeMutex.try_lock()) {
      std::queue<TBufferFile *> queue;
      {
         std::lock_guard<std::mutex> q(fQueueMutex);
         std::swap(queue, fQueue);
         fBuffered = 0;
      }

      while (!queue.empty()) {
         std::unique_ptr<TBufferFile> buffer{queue.front()};
         fMerger.AddAdoptFile(
            new TMemFile(fMerger.GetOutputFileName(), buffer->Buffer(), buffer->BufferSize(), "READ"));
         queue.pop();
      }

      fMerger.PartialMerge();
      fMerger.Reset();
      fMergeMutex.unlock();
   }
}

} // namespace Experimental
} // namespace ROOT
