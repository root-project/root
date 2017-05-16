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

namespace ROOT {
namespace Experimental {

TBufferMerger::TBufferMerger(const char *name, Option_t *option, Int_t compress)
   : fName(name), fOption(option), fCompress(compress),
     fMergingThread(new std::thread([&]() { this->WriteOutputFile(); }))
{
}

TBufferMerger::~TBufferMerger()
{
   for (auto f : fAttachedFiles)
      if (!f.expired()) Fatal("TBufferMerger", " TBufferMergerFiles must be destroyed before the server");

   this->Push(nullptr);
   fCV.notify_one();

   fMergingThread->join();
}

std::shared_ptr<TBufferMergerFile> TBufferMerger::GetFile()
{
   std::lock_guard<std::mutex> lk(fFilesMutex);
   std::shared_ptr<TBufferMergerFile> f(new TBufferMergerFile(*this));
   fAttachedFiles.push_back(f);
   return f;
}

void TBufferMerger::Push(TBufferFile *buffer)
{
   {
      std::lock_guard<std::mutex> lock(fQueueMutex);
      fQueue.push(buffer);
   }
   fCV.notify_one();
}

void TBufferMerger::WriteOutputFile()
{
   bool done = false;
   std::unique_lock<std::mutex> wlock(fWriteMutex);
   TDirectoryFile::TContext context;
   TFileMerger merger;

   merger.OutputFile(fName, fOption, fCompress);

   while (!done) {
      fCV.wait(wlock, [this]() { return !this->fQueue.empty(); });

      while (!fQueue.empty()) {
         std::unique_ptr<TBufferFile> buffer;

         {
            std::lock_guard<std::mutex> qlock(fQueueMutex);
            buffer.reset(fQueue.front());
            fQueue.pop();
         }

         if (!buffer) {
            done = true;
            break;
         }

         Long64_t length;
         buffer->SetReadMode();
         buffer->SetBufferOffset();
         buffer->ReadLong64(length);

         {
            TDirectory::TContext ctxt;
            auto tmp = new TMemFile(fName, buffer->Buffer() + buffer->Length(), length, "READ");
            buffer->SetBufferOffset(buffer->Length() + length);
            merger.AddAdoptFile(tmp);
            merger.PartialMerge();
            merger.Reset();
         }
      }
   }
}

} // namespace Experimental
} // namespace ROOT
