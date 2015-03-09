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

#include "TEnum.h"
#include "TClassTable.h"
#include "TROOT.h"
#include "TClass.h"
#include "TProtoClass.h"
#include "TListOfEnums.h"

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

static bool NameExistsElsewhere(const char* name){

   // Is this a scope?
   // We look into the list of classes available,
   // the ones in the dictionaries and the protoclasses.
   if (gROOT->GetListOfClasses()->FindObject(name) ||
       TClassTable::GetDictNorm(name) ||
       TClassTable::GetProtoNorm(name)) return true;

   // Is this an enum?
   TObject* theEnum = nullptr;
   const auto lastPos = strrchr(name, ':');
   if (lastPos != nullptr) {
      // We have a scope
      const auto enName = lastPos + 1;
      const auto scopeNameSize = ((Long64_t)lastPos - (Long64_t)name) / sizeof(decltype(*lastPos)) - 1;
#ifdef R__WIN32
      char *scopeName = new char[scopeNameSize + 1];
#else
      char scopeName[scopeNameSize + 1]; // on the stack, +1 for the terminating character '\0'
#endif
      strncpy(scopeName, name, scopeNameSize);
      scopeName[scopeNameSize] = '\0';
      // We have now an enum name and a scope name
      // We look first in the classes
      if(auto scope = dynamic_cast<TClass*>(gROOT->GetListOfClasses()->FindObject(scopeName))){
         theEnum = ((TListOfEnums*)scope->GetListOfEnums(false))->THashList::FindObject(enName);
      }
      // And then if not found in the protoclasses
      if (!theEnum){
         if (auto scope = TClassTable::GetProtoNorm(scopeName)){
            if (auto listOfEnums = (TListOfEnums*)scope->GetListOfEnums())
               theEnum = listOfEnums->THashList::FindObject(enName);
         }
      }
#ifdef R__WIN32
      delete [] scopeName;
#endif
   } else { // Here we look in the global scope
      theEnum = ((TListOfEnums*)gROOT->GetListOfEnums())->THashList::FindObject(name);
   }

  return nullptr != theEnum;

}

TDataType *TListOfTypes::FindType(const char *name) const
{
   // Look for a type, first in the hast table
   // then in the interpreter.

   TDataType *result = static_cast<TDataType*>(THashTable::FindObject(name));
   if (!result) {

      if (NameExistsElsewhere(name)) {
         return nullptr;
      }

      // We perform now a lookup

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
