// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfTypes                                                         //
//                                                                      //
// A collection of TDataType designed to hold the typedef information   //
// and numerical type information.  The collection is populated on      //
// demand.                                                              //
//                                                                      //
// Besides the built-in types (int, float) a typedef is explicitly      //
// added to the collection (and thus visible via ls or Print) only if   //
// it is requested explicitly.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TListOfTypes.h"

#include "TInterpreter.h"
#include "TDataType.h"
#include "TVirtualMutex.h"

TListOfTypes::TListOfTypes() : THashTable(100, 3)
{
   // Constructor
   TDataType::AddBuiltins(this);
}

TObject *TListOfTypes::FindObject(const char *name) const
{
   // Specialize FindObject to do search for the
   // typedef if its not already in the list

   return FindType(name);
}

TDataType *TListOfTypes::FindType(const char *name) const
{
   // Look for a type, first in the hast table
   // then in the interpreter.

   TDataType *result = static_cast<TDataType*>(THashTable::FindObject(name));
   if (!result) {
      R__LOCKGUARD2(gInterpreterMutex);

      TypedefInfo_t  *info = gInterpreter->TypedefInfo_Factory(name);
      if (gInterpreter->TypedefInfo_IsValid(info)) {
         result = new TDataType(info);
         // Double check we did not get a different spelling of an
         // already existing typedef.
         if (strcmp(name,result->GetName()) != 0) {
            TDataType *alt = static_cast<TDataType*>(THashTable::FindObject(result->GetName()));
            if (!alt)
               const_cast<TListOfTypes*>(this)->Add(result);
            else {
               delete result;
               result = alt;
            }
         } else {
            const_cast<TListOfTypes*>(this)->Add(result);
         }
      } else {
         gInterpreter->TypedefInfo_Delete(info);
      }
   }
   return result;
}
