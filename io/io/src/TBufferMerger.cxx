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
   fMergingThread.reset(new std::thread([&]() { this->WriteOutputFile(); }));
}

TBufferMerger::~TBufferMerger()
{
   for (const auto &f : fAttachedFiles)
      if (!f.expired()) Fatal("TBufferMerger", " TBufferMergerFiles must be destroyed before the server");

   this->Push(nullptr);
   fMergingThread->join();
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

void TBufferMerger::RegisterCallback(const std::function<void(void)> &f)
{
   fCallback = f;
}

void TBufferMerger::Push(TBufferFile *buffer)
{
   {
      std::lock_guard<std::mutex> lock(fQueueMutex);
      fQueue.push(buffer);
   }
   fDataAvailable.notify_one();
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
   fBuffered = 0;
   fMerger.PartialMerge();
   fMerger.Reset();

   if (fCallback)
      fCallback();
}

void TBufferMerger::WriteOutputFile()
{
   std::unique_ptr<TBufferFile> buffer;

   while (true) {
      std::unique_lock<std::mutex> lock(fQueueMutex);
      fDataAvailable.wait(lock, [this]() { return !this->fQueue.empty(); });

      buffer.reset(fQueue.front());
      fQueue.pop();
      lock.unlock();

      if (!buffer)
         break;

      fBuffered += buffer->BufferSize();
      fMerger.AddAdoptFile(new TMemFile(fMerger.GetOutputFileName(), buffer->Buffer(), buffer->BufferSize(), "read"));

      if (fBuffered > fAutoSave)
         Merge();
   }

   Merge();
}

} // namespace Experimental
} // namespace ROOT
