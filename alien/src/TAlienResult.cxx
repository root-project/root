// @(#)root/alien:$Name:  $:$Id: TAlienResult.cxx,v 1.5 2005/05/20 11:13:30 rdm Exp $
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


ClassImp(TAlienResult)

//______________________________________________________________________________
void TAlienResult::DumpResult()
{
   // Dump result set.

   cout << "BEGIN DUMP" << endl;
   TIter next(this);
   TMap *map;
   while ((map = (TMap*) next())) {
      TIter next2(map->GetTable());
      TPair *pair;
      while ((pair = (TPair*) next2())) {
         TObjString *keyStr = dynamic_cast<TObjString*>(pair->Key());
         TObjString* valueStr = dynamic_cast<TObjString*>(pair->Value());

         if (keyStr) {
            cout << "Key: " << keyStr->GetString() << "   ";
         }
         if (valueStr) {
            cout << "Value: " << valueStr->GetString();
         }
         cout << endl;
      }
   }

   cout << "END DUMP" << endl;
}

//______________________________________________________________________________
const char *TAlienResult::GetFileName(UInt_t i) const
{
   if (At(i)) {
     TObjString* entry;
     if ((entry=(TObjString*)((TMap*)At(i))->GetValue("name"))) {
        return entry->GetName();
     }
   }
   return 0;
}

//______________________________________________________________________________
const char *TAlienResult::GetFileNamePath(UInt_t i) const
{
   if (At(i)) {
      TObjString* entry;
      if ((entry=(TObjString*)((TMap*)At(i))->GetValue("name"))) {
         TObjString* path;
         if ((path=(TObjString*)((TMap*)At(i))->GetValue("path"))) {
            fFilePath = TString(path->GetName()) + TString(entry->GetName());
            return fFilePath;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
const char *TAlienResult::GetPath(UInt_t i) const
{
   if (At(i)) {
      TObjString* entry;
      if ((entry=(TObjString*)((TMap*)At(i))->GetValue("path"))) {
         return entry->GetName();
      }
   }
   return 0;
}

//______________________________________________________________________________
const char *TAlienResult::GetKey(UInt_t i, const char* key) const
{
   if (At(i)) {
      TObjString* entry;
      if ((entry=(TObjString*)((TMap*)At(i))->GetValue(key))) {
         return entry->GetName();
      }
   }
   return 0;
}
