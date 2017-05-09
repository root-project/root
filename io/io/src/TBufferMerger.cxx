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

#include "TBits.h"
#include "TBufferFile.h"
#include "TClass.h"
#include "TError.h"
#include "TFileCacheWrite.h"
#include "TFileMerger.h"
#include "TKey.h"
#include "TMath.h"
#include "TMemFile.h"
#include "TSystem.h"
#include "TTimeStamp.h"

namespace {

Bool_t R__NeedInitialMerge(TDirectory *dir)
{
   if (dir == 0) return false;

   TIter nextkey(dir->GetListOfKeys());
   for (TKey *key = (TKey *)nextkey(); key != nullptr; key = (TKey *)nextkey()) {
      TClass *cl = TClass::GetClass(key->GetClassName());
      if (cl->InheritsFrom(TDirectory::Class())) {
         TDirectory *subdir = (TDirectory *)dir->GetList()->FindObject(key->GetName());
         if (!subdir) {
            subdir = (TDirectory *)key->ReadObj();
         }
         if (R__NeedInitialMerge(subdir)) {
            return true;
         }
      } else {
         if (0 != cl->GetResetAfterMerge()) {
            return true;
         }
      }
   }
   return false;
}

void R__DeleteObject(TDirectory *dir, Bool_t withReset)
{
   if (dir == 0) return;

   TIter nextkey(dir->GetListOfKeys());
   TKey *key;
   while ((key = (TKey *)nextkey())) {
      TClass *cl = TClass::GetClass(key->GetClassName());
      if (cl->InheritsFrom(TDirectory::Class())) {
         TDirectory *subdir = (TDirectory *)dir->GetList()->FindObject(key->GetName());
         if (!subdir) {
            subdir = (TDirectory *)key->ReadObj();
         }
         R__DeleteObject(subdir, withReset);
      } else {
         Bool_t todelete = false;
         if (withReset) {
            todelete = (0 != cl->GetResetAfterMerge());
         } else {
            todelete = (0 == cl->GetResetAfterMerge());
         }
         if (todelete) {
            key->Delete();
            dir->GetListOfKeys()->Remove(key);
            delete key;
         }
      }
   }
}

void R__MigrateKey(TDirectory *destination, TDirectory *source)
{
   if (destination == 0 || source == 0) return;
   TIter nextkey(source->GetListOfKeys());

   TKey *key;

   while ((key = (TKey *)nextkey())) {
      TClass *cl = TClass::GetClass(key->GetClassName());
      if (cl->InheritsFrom(TDirectory::Class())) {
         TDirectory *source_subdir = (TDirectory *)source->GetList()->FindObject(key->GetName());
         if (!source_subdir) {
            source_subdir = (TDirectory *)key->ReadObj();
         }
         TDirectory *destination_subdir = destination->GetDirectory(key->GetName());
         if (!destination_subdir) {
            destination_subdir = destination->mkdir(key->GetName());
         }
         R__MigrateKey(destination_subdir, source_subdir);
      } else {
         TKey *oldkey = destination->GetKey(key->GetName());
         if (oldkey) {
            oldkey->Delete();
            delete oldkey;
         }
         // a priori the files are from the same client
         TKey *newkey = new TKey(destination, *key, 0 /* pidoffset */);
         destination->GetFile()->SumBuffer(newkey->GetObjlen());
         newkey->WriteFile(0);
         if (destination->GetFile()->TestBit(TFile::kWriteError)) {
            return;
         }
      }
   }
   destination->SaveSelf();
}

struct ClientInfo {
   TFile *fFile; // This object does *not* own the file, it's owned by the owner of ClientInfo
   TString fLocalName;
   UInt_t fContactsCount;
   TTimeStamp fLastContact;
   Double_t fTimeSincePrevContact;

   ClientInfo() : fFile(0), fLocalName(), fContactsCount(0), fTimeSincePrevContact(0) {}

   ClientInfo(const char *filename, UInt_t clientId) : fFile(0), fContactsCount(0), fTimeSincePrevContact(0)
   {
      fLocalName.Form("%s-%d-%d", filename, clientId, gSystem->GetPid());
   }

   void Set(TFile *file)
   {
      // Register the new file as coming from this client.
      if (file != fFile) {
         // We need to keep any of the keys from the previous file that
         // are not in the new file.
         if (fFile) {
            R__MigrateKey(fFile, file);
            // delete the previous memory file (if any)
            delete file;
         } else {
            fFile = file;
         }
      }
      TTimeStamp now;
      fTimeSincePrevContact = now.AsDouble() - fLastContact.AsDouble();
      fLastContact = now;
      ++fContactsCount;
   }
};

struct ThreadFileMerger : public TObject {
   TString fFilename;
   TBits fClientsContact;
   UInt_t fNClientsContact;
   std::vector<ClientInfo> fClients;
   TTimeStamp fLastMerge;
   TFileMerger fMerger;

   ThreadFileMerger(const char *filename, Bool_t writeCache = false)
      : fFilename(filename), fNClientsContact(0), fMerger(false, true)
   {
      fMerger.SetPrintLevel(0);
      fMerger.OutputFile(filename, "RECREATE");
      if (writeCache) new TFileCacheWrite(fMerger.GetOutputFile(), 32 * 1024 * 1024);
   }

   ~ThreadFileMerger()
   {
      for (auto &&client : fClients) delete client.fFile;
   }

   const char *GetName() const { return fFilename; }

   ULong_t Hash() const { return fFilename.Hash(); }

   Bool_t InitialMerge(TFile *input)
   {
      fMerger.AddFile(input);
      Bool_t result = fMerger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kResetable);
      R__DeleteObject(input, true);
      return result;
   }

   Bool_t NeedFinalMerge() { return fClientsContact.CountBits() > 0; }

   Bool_t NeedMerge(Float_t clientThreshold)
   {
      if (fClients.size() == 0) {
         return false;
      }

      // Calculate average and rms of the time between the last 2 contacts.
      Double_t sum = 0;
      Double_t sum2 = 0;
      for (unsigned int c = 0; c < fClients.size(); ++c) {
         sum += fClients[c].fTimeSincePrevContact;
         sum2 += fClients[c].fTimeSincePrevContact * fClients[c].fTimeSincePrevContact;
      }
      Double_t avg = sum / fClients.size();
      Double_t sigma = sum2 ? TMath::Sqrt(sum2 / fClients.size() - avg * avg) : 0;
      Double_t target = avg + 2 * sigma;
      TTimeStamp now;
      if ((now.AsDouble() - fLastMerge.AsDouble()) > target) {
         return true;
      }
      Float_t cut = clientThreshold * fClients.size();
      return fClientsContact.CountBits() > cut || fNClientsContact > 2 * cut;
   }

   Bool_t Merge()
   {
      // Merge the current inputs into the output file.

      // Remove object that can *not* be incrementally merged
      // and will *not* be reset by the client code.
      R__DeleteObject(fMerger.GetOutputFile(), false);

      for (unsigned int f = 0; f < fClients.size(); ++f) {
         fMerger.AddFile(fClients[f].fFile);
      }

      Bool_t result = fMerger.PartialMerge(TFileMerger::kAllIncremental);

      // Remove any 'resetable' object (like TTree) from the input file
      // so that they will not be re-merged. Keep only the object that
      // always need to be re-merged (Histograms).

      for (unsigned int f = 0; f < fClients.size(); ++f) {
         if (fClients[f].fFile) {
            R__DeleteObject(fClients[f].fFile, true);
         } else {
            // We back up the file (probably due to memory constraint)
            TFile *file = TFile::Open(fClients[f].fLocalName, "UPDATE");
            // Remove object that can be incrementally merged and
            // will be reset by the client code.
            R__DeleteObject(file, true);
            file->Write();
            delete file;
         }
      }
      fLastMerge = TTimeStamp();
      fNClientsContact = 0;
      fClientsContact.Clear();

      return result;
   }

   void RegisterClient(UInt_t clientId, TFile *file)
   {
      ++fNClientsContact;

      fClientsContact.SetBitNumber(clientId);

      if (fClients.size() < clientId + 1) {
         fClients.push_back(ClientInfo(fFilename, clientId));
      }
      fClients[clientId].Set(file);
   }
};

} // unnamed namespace

namespace ROOT {
namespace Experimental {

TBufferMerger::TBufferMerger(const char *name, Option_t *option, const char *ftitle, Int_t compress)
   : fFile(TFile::Open(name, option, ftitle, compress)), fMergingThread(new std::thread([this]() { this->Listen(); }))
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
   std::shared_ptr<TBufferMergerFile> f;
   {
      std::lock_guard<std::mutex> lk(fFilesMutex);
      f.reset(new TBufferMergerFile(*this));
      fAttachedFiles.push_back(f);
   }
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

void TBufferMerger::Listen()
{
   std::unique_lock<std::mutex> wlock(fWriteMutex);

   bool done = false;
   THashTable mergers;

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

         buffer->SetReadMode();
         buffer->SetBufferOffset();

         Long64_t length;
         TString filename;

         buffer->ReadTString(filename);
         buffer->ReadLong64(length);

         // UPDATE because we need to remove the TTree after merging them.
         TMemFile *transient = new TMemFile(filename, buffer->Buffer() + buffer->Length(), length, "UPDATE");

         buffer->SetBufferOffset(buffer->Length() + length);

         // control how often the histogram are merged.  Here as soon as half the clients have reported.
         const Float_t clientThreshold = 0.75;

         ThreadFileMerger *info = (ThreadFileMerger *)mergers.FindObject(filename);

         if (!info) {
            info = new ThreadFileMerger(filename, false);
            mergers.Add(info);
         }

         if (R__NeedInitialMerge(transient)) {
            info->InitialMerge(transient);
         }

         info->RegisterClient(0, transient);

         if (info->NeedMerge(clientThreshold)) info->Merge();

         transient = nullptr;
      }
   }

   TIter next(&mergers);
   ThreadFileMerger *info;
   while ((info = (ThreadFileMerger *)next())) {
      if (info->NeedFinalMerge()) {
         info->Merge();
      }
   }

   mergers.Delete();
}

} // namespace Experimental
} // namespace ROOT
