// @(#)root/meta:$Id$
// Author: Fons Rademakers   20/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TDictionary

This class defines an abstract interface that must be implemented
by all classes that contain dictionary information.

The dictionary is defined by the following classes:
~~~ {.cpp}
TDataType                              (typedef definitions)
TGlobal                                (global variables)
TGlobalFunc                            (global functions)
TClass                                 (classes)
   TBaseClass                          (base classes)
   TDataMember                         (class datamembers)
   TMethod                             (class methods)
      TMethodArg                       (method arguments)
~~~
All the above classes implement the TDictionary abstract interface.
Note: the indentation shows aggregation not inheritance.
~~~ {.cpp}
TMethodCall                            (method call environment)
~~~
\image html base_tdictionary.png
*/

#include "TDictionary.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TDataType.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"


ClassImp(TDictionary);

TDictionary::TDictionary(const TDictionary& dict):
   TNamed(dict),
   fAttributeMap(dict.fAttributeMap ?
                 ((TDictAttributeMap*)dict.fAttributeMap->Clone()) : nullptr ),
   fUpdatingTransactionCount(0)
{
   // Copy constructor, cloning fAttributeMap.
}

TDictionary::~TDictionary()
{
   // Destruct a TDictionary, delete the attribute map.
   delete fAttributeMap;
}

TDictionary &TDictionary::operator=(const TDictionary& dict)
{
  // Assignment op, cloning fAttributeMap.
  TNamed::operator=(dict);

  delete fAttributeMap;
  fAttributeMap = nullptr;
  if (dict.fAttributeMap)
    fAttributeMap = ((TDictAttributeMap*)dict.fAttributeMap->Clone());

  return *this;
}

void TDictionary::CreateAttributeMap()
{
   //Create a TDictAttributeMap for a TClass to be able to add attribute pairs
   //key-value to the TClass.

   if (!fAttributeMap)
      fAttributeMap = new TDictAttributeMap;
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the type (class, fundamental type, typedef etc)
/// named "name". Returned object is either a TClass or TDataType.
/// Returns `nullptr` if the type is unknown.

TDictionary* TDictionary::GetDictionary(const char* name)
{
   // Start with typedef, the query is way faster than TClass::GetClass().
   if (auto* ret = (TDictionary*)gROOT->GetListOfTypes()->FindObject(name)) {
      if (auto *dtRet = dynamic_cast<TDataType*>(ret)) {
         if (dtRet->GetType() <= 0) {
            // Not a numeric type. Is it a known class?
            if (auto *clRet = TClass::GetClass(name, true))
               return clRet;
         }
      }
      return ret;
   }

   return TClass::GetClass(name, true);
}

TDictionary* TDictionary::GetDictionary(const std::type_info &typeinfo)
{
   // Retrieve the type (class, fundamental type, typedef etc)
   // with typeid typeinfo. Returned object is either a TClass or TDataType.
   // Returns 0 if the type is unknown.

   EDataType datatype = TDataType::GetType(typeinfo);
   TDictionary* ret = TDataType::GetDataType(datatype);
   if (ret) return ret;

   return TClass::GetClass(typeinfo, true);
}

Bool_t TDictionary::UpdateInterpreterStateMarker()
{
   // Return true if there were any transactions that could have changed the
   // state of the object.
   ULong64_t currentTransaction = gInterpreter->GetInterpreterStateMarker();
   if (currentTransaction == fUpdatingTransactionCount) {
      return false;
   }
   fUpdatingTransactionCount = currentTransaction;
   return true;
}
