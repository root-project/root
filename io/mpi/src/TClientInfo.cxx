// @(#)root/io:$Id$
// Author: Amit Bashyal, August 2018

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClientInfo.h"
#include "TSystem.h"
#include "TClass.h"
#include "TKey.h"

ClassImp(TClientInfo);

TClientInfo::TClientInfo() : fFile(0), fLocalName(), fContactsCount(0), fTimeSincePrevContact(0) {}

TClientInfo::~TClientInfo() {}
TClientInfo::TClientInfo(const char *filename, UInt_t clientId) : fFile(0), fContactsCount(0), fTimeSincePrevContact(0)
{
   fLocalName.Form("%s-%d-%d", filename, clientId, gSystem->GetPid());
}

void TClientInfo::SetFile(TFile *file)
{
   {
      // Register the new file as coming from this client.
      if (file != fFile) {
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
}

void TClientInfo::R__DeleteObject(TDirectory *dir, Bool_t withReset)
{
   if (dir == 0)
      return;

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
         Bool_t todelete = kFALSE;
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

void TClientInfo::R__MigrateKey(TDirectory *destination, TDirectory *source)
{
   if (destination == 0 || source == 0)
      return;
   TIter nextkey(source->GetListOfKeys());
   // std::cout<<"TClientInfo::Trying to migrate the keys here"<<std::endl;
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
         R__MigrateKey(destination, source);
      } else {
         TKey *oldkey = destination->GetKey(key->GetName());
         if (oldkey) {
            oldkey->Delete();
            delete oldkey;
         }
         TKey *newkey = new TKey(destination, *key, 0 /* pidoffset */); // a priori the file are from the same client ..
         destination->GetFile()->SumBuffer(newkey->GetObjlen());
         newkey->WriteFile(0);
         if (destination->GetFile()->TestBit(TFile::kWriteError)) {
            return;
         }
      }
   }
   destination->SaveSelf();
}
