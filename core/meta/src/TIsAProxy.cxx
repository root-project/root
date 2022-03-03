// @(#)root/meta:$Id$
// Author: Markus Frank 20/05/2005

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TClass.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TIsAProxy.h"

#include <map>
#include <type_traits>

/** \class TIsAProxy
TIsAProxy implementation class.
*/

namespace {
   struct DynamicType {
      // Helper class to enable typeid on any address
      // Used in code similar to:
      //    typeid( * (DynamicType*) void_ptr );
      virtual ~DynamicType() {}
   };

   typedef std::map<const void*, TClass*> ClassMap_t; // Internal type map
   inline ClassMap_t *GetMap(const void* p)
   {
      return (ClassMap_t*)p;
   }

   inline ClassMap_t::value_type* ToPair(void*p)
   {
      return (ClassMap_t::value_type*)p;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Standard initializing constructor

TIsAProxy::TIsAProxy(const std::type_info& typ)
   : fType(&typ), fClass(nullptr),
     fSubTypesReaders(0), fSubTypesWriteLockTaken(kFALSE), fNextLastSlot(0),
     fInit(kFALSE), fVirtual(kFALSE)
{
   static_assert(sizeof(ClassMap_t)<=sizeof(fSubTypes), "ClassMap size is to large for array");

   ::new(fSubTypes) ClassMap_t();
   for(auto& slot : fLasts)
      slot = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Standard destructor

TIsAProxy::~TIsAProxy()
{
   ClassMap_t* m = GetMap(fSubTypes);
   m->clear();
   m->~ClassMap_t();
}

////////////////////////////////////////////////////////////////////////////////
/// Set class pointer
///   This method is not thread safe

void TIsAProxy::SetClass(TClass *cl)
{
   GetMap(fSubTypes)->clear();
   fClass = cl;
   for(auto& slot : fLasts)
      slot = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// IsA callback

TClass* TIsAProxy::operator()(const void *obj)
{
   if ( !fInit )  {
      R__WRITE_LOCKGUARD(ROOT::gCoreMutex);
      if ( !fClass && fType ) {
         fClass = TClass::GetClass(*fType);
      }
      if ( !fClass ) return nullptr;
      fVirtual = (*fClass).ClassProperty() & kClassHasVirtual;
      fInit = kTRUE;
   }
   if ( !obj || !fVirtual )  {
      return fClass;
   }
   // Avoid the case that the first word is a virtual_base_offset_table instead of
   // a virtual_function_table
   Long_t offset = **(Long_t**)obj;
   if ( offset == 0 ) {
      return fClass;
   }

   DynamicType* ptr = (DynamicType*)obj;
   const std::type_info* typ = &typeid(*ptr);

   if ( typ == fType )  {
     return fClass;
   }
   for(auto& slot : fLasts) {
      auto last = ToPair(slot);
      if ( last && typ == last->first )  {
         return last->second;
      }
   }

   // Check if type is already in sub-class cache
   auto last = ToPair(FindSubType(typ));
   if ( last == nullptr )  {
      // Last resort: lookup root class
      auto cls = TClass::GetClass(*typ);
      if (cls)
         last = ToPair(CacheSubType(typ,cls));
      else
         return nullptr; // Don't record failed searches (a library might be loaded between now and the next search).
   }

   UChar_t next = fNextLastSlot++;
   if (next >= fgMaxLastSlot) {
      UChar_t expected_value = next + 1;
      next = next % fgMaxLastSlot;
      fNextLastSlot.compare_exchange_strong(expected_value, next + 1);
   }
   fLasts[next].store(last);

   return last ? last->second : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// See if we have already cached the TClass that correspond to this std::type_info.

inline void* TIsAProxy::FindSubType(const std::type_info* type) const
{
   bool needToWait = kTRUE;
   do {
     ++fSubTypesReaders;

     //See if there is a writer, if there is we need to release
     // our reader count so that the writer can proceed
     if(fSubTypesWriteLockTaken) {
       --fSubTypesReaders;
       while(fSubTypesWriteLockTaken) {}
     } else {
       needToWait = kFALSE;
     }
   } while(needToWait);

   void* returnValue = nullptr;
   auto const map = GetMap(fSubTypes);

   auto found = map->find(type);
   if(found != map->end()) {
      returnValue = &(*found);
   }
   --fSubTypesReaders;
   return returnValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Record the TClass found for a std::type_info, so that we can retrieved it faster.

void* TIsAProxy::CacheSubType(const std::type_info* type, TClass* cls)
{
   //See if another thread has the write lock, wait if it does
   Bool_t expected = kFALSE;
   while(! fSubTypesWriteLockTaken.compare_exchange_strong(expected,kTRUE) ) {
      expected = kFALSE;
   };

   //See if there are any readers
   while(fSubTypesReaders > 0);

   auto map = GetMap(fSubTypes);
   auto ret = map->emplace(type,cls);
   if (!ret.second) {
      // type is already in the map, let's update it.
      (*ret.first).second = cls;
   }

   fSubTypesWriteLockTaken = kFALSE;
   return &(*(ret.first));
}
