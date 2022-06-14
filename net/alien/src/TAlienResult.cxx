// @(#)root/alien:$Id: 11fde82f21e66ae11add660ef69f33597f089efb $
// Author: Fons Rademakers   23/5/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienResult                                                         //
//                                                                      //
// Class defining interface to a Alien result set.                      //
// Objects of this class are created by TGrid methods.                  //
//                                                                      //
// Related classes are TAlien.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienResult.h"
#include "TObjString.h"
#include "TMap.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TUrl.h"
#include "TFileInfo.h"
#include "TEntryList.h"
#include <cstdlib>


ClassImp(TAlienResult);

////////////////////////////////////////////////////////////////////////////////
/// Cleanup object.

TAlienResult::~TAlienResult()
{
   TIter next(this);
   while (TMap * obj = (TMap *) next()) {
      obj->DeleteAll();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Dump result set.

void TAlienResult::DumpResult()
{
   std::cout << "BEGIN DUMP" << std::endl;
   TIter next(this);
   TMap *map;
   while ((map = (TMap *) next())) {
      TIter next2(map->GetTable());
      TPair *pair;
      while ((pair = (TPair *) next2())) {
         TObjString *keyStr = dynamic_cast < TObjString * >(pair->Key());
         TObjString *valueStr =
             dynamic_cast < TObjString * >(pair->Value());

         if (keyStr) {
            std::cout << "Key: " << keyStr->GetString() << "   ";
         }
         if (valueStr) {
            std::cout << "Value: " << valueStr->GetString();
         }
         std::cout << std::endl;
      }
   }

   std::cout << "END DUMP" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
///Return a file name.

const char *TAlienResult::GetFileName(UInt_t i) const
{
   if (At(i)) {
      TObjString *entry;
      if ((entry = (TObjString *) ((TMap *) At(i))->GetValue("name"))) {
         return entry->GetName();
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the entry list, if evtlist was defined as a tag.

const TEntryList *TAlienResult::GetEntryList(UInt_t i) const
{
   if (At(i)) {
      TEntryList *entry;
      if ((entry = (TEntryList *) ((TMap *) At(i))->GetValue("evlist"))) {
         return entry;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return file name path.

const char *TAlienResult::GetFileNamePath(UInt_t i) const
{
   if (At(i)) {
      TObjString *entry;
      if ((entry = (TObjString *) ((TMap *) At(i))->GetValue("name"))) {
         TObjString *path;
         if ((path = (TObjString *) ((TMap *) At(i))->GetValue("path"))) {
            fFilePath =
                TString(path->GetName()) + TString(entry->GetName());
            return fFilePath;
         }
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return path.

const char *TAlienResult::GetPath(UInt_t i) const
{
   if (At(i)) {
      TObjString *entry;
      if ((entry = (TObjString *) ((TMap *) At(i))->GetValue("path"))) {
         return entry->GetName();
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the key.

const char *TAlienResult::GetKey(UInt_t i, const char *key) const
{
   if (At(i)) {
      TObjString *entry;
      if ((entry = (TObjString *) ((TMap *) At(i))->GetValue(key))) {
         return entry->GetName();
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the key.

Bool_t TAlienResult::SetKey(UInt_t i, const char *key, const char *value)
{
   if (At(i)) {
      TPair *entry;
      if ((entry = (TPair *) ((TMap *) At(i))->FindObject(key))) {
         TObject *val = ((TMap *) At(i))->Remove((TObject *) entry->Key());
         if (val) {
            delete val;
         }
      }
      ((TMap *) At(i))->Add(new TObjString(key), new TObjString(value));
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a file info list.

TList *TAlienResult::GetFileInfoList() const
{
   TList *newfileinfolist = new TList();

   newfileinfolist->SetOwner(kTRUE);

   for (Int_t i = 0; i < GetSize(); i++) {

      Long64_t size = -1;
      if (GetKey(i, "size"))
         size = atol(GetKey(i, "size"));

      const char *md5 = GetKey(i, "md5");
      const char *uuid = GetKey(i, "guid");
      const char *msd = GetKey(i, "msd");

      if (md5 && !md5[0])
         md5 = 0;
      if (uuid && !uuid[0])
         uuid = 0;
      if (msd && !msd[0])
         msd = 0;

      TString turl = GetKey(i, "turl");

      if (msd) {
         TUrl urlturl(turl);
         TString options = urlturl.GetOptions();
         options += "&msd=";
         options += msd;
         urlturl.SetOptions(options);
         turl = urlturl.GetUrl();
      }
      Info("GetFileInfoList", "Adding Url %s with Msd %s\n", turl.Data(),
           msd);
      newfileinfolist->Add(new TFileInfo(turl, size, uuid, md5));
   }
   return newfileinfolist;
}

////////////////////////////////////////////////////////////////////////////////
/// Print the AlienResult info.

void TAlienResult::Print(Option_t * option) const
{
   Long64_t totaldata = 0;
   Int_t totalfiles = 0;

   if (TString(option) != TString("all")) {
      // the default print out format is for a query
      for (Int_t i = 0; i < GetSize(); i++) {
         if (TString(option) == TString("l")) {
            printf("( %06d ) LFN: %-80s   Size[Bytes]: %10s   GUID: %s\n",
                   i, GetKey(i, "lfn"), GetKey(i, "size"), GetKey(i,
                                                                  "guid"));
         } else {
            printf("( %06d ) LFN: .../%-48s   Size[Bytes]: %10s   GUID: %s\n",
                   i, gSystem->BaseName(GetKey(i, "lfn")), GetKey(i, "size"),
                   GetKey(i, "guid"));
         }
         if (GetKey(i, "size")) {
            totaldata += atol(GetKey(i, "size"));
            totalfiles++;
         }
      }
      printf("------------------------------------------------------------\n");
      printf("-> Result contains %.02f MB in %d Files.\n",
             totaldata / 1024. / 1024., totalfiles);
   } else {
      TIter next(this);
      TMap *map;
      Int_t i = 1;
      while ((map = (TMap *) next())) {
         TIter next2(map->GetTable());
         TPair *pair;
         printf("------------------------------------------------------------\n");
         while ((pair = (TPair *) next2())) {
            TObjString *keyStr =
                dynamic_cast < TObjString * >(pair->Key());
            TObjString *valueStr =
                dynamic_cast < TObjString * >(pair->Value());
            if (keyStr && valueStr)
               printf("( %06d ) [ -%16s ]  = %s\n", i, keyStr->GetName(),
                      valueStr->GetName());
         }
         i++;
      }
   }
}
