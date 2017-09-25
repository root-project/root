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
#include "TFileMerger.h"
#include "TROOT.h"
#include "TVirtualMutex.h"

namespace ROOT {
namespace Experimental {

TBufferMerger::TBufferMerger(const char *name, Option_t *option, Int_t compress)
   : fName(name), fOption(option), fCompress(compress), fAutoSave(0),
     fMergingThread(new std::thread([&]() { this->WriteOutputFile(); }))
{
}

TBufferMerger::~TBufferMerger()
{
   for (auto f : fAttachedFiles)
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

void TBufferMerger::SetAutoSave(size_t size)
{
   fAutoSave = size;
}

void TBufferMerger::WriteOutputFile()
{
   size_t buffered = 0;
   std::vector<TMemFile *> memfiles;
   std::unique_ptr<TBufferFile> buffer;
   TFileMerger merger;

   merger.ResetBit(kMustCleanup);

   {
      R__LOCKGUARD(gROOTMutex);
      merger.OutputFile(fName.c_str(), fOption.c_str(), fCompress);
   }

   while (true) {
      std::unique_lock<std::mutex> lock(fQueueMutex);
      fDataAvailable.wait(lock, [this]() { return !this->fQueue.empty(); });

      buffer.reset(fQueue.front());
      fQueue.pop();
      lock.unlock();

      if (!buffer)
         break;

      Long64_t length;
      buffer->SetReadMode();
      buffer->SetBufferOffset();
      buffer->ReadLong64(length);
      buffered += length;

      {
         R__LOCKGUARD(gROOTMutex);
         memfiles.push_back(new TMemFile(fName.c_str(), buffer->Buffer() + buffer->Length(), length, "read"));
         buffer->SetBufferOffset(buffer->Length() + length);
         merger.AddFile(memfiles.back(), false);

         if (buffered > fAutoSave) {
            buffered = 0;
            merger.PartialMerge();
            merger.Reset();
            memfiles.clear();
         }
      }

      if (fCallback)
         fCallback();
   }

   R__LOCKGUARD(gROOTMutex);
   merger.PartialMerge();
   merger.Reset();
}

} // namespace Experimental
} // namespace ROOT
