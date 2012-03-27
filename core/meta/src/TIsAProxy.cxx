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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClass                                                               //
//                                                                      //
// TIsAProxy implementation class.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace {
   struct DynamicType {
      // Helper class to enable typeid on any address
      // Used in code similar to:
      //    typeid( * (DynamicType*) void_ptr );
      virtual ~DynamicType() {}
   };
}

typedef std::map<Long_t, TClass*> ClassMap_t; // Internal type map
inline ClassMap_t *GetMap(void* p)
{
   return (ClassMap_t*)p;
}

//______________________________________________________________________________
TIsAProxy::TIsAProxy(const std::type_info& typ, void* ctxt)
   : fType(&typ), fLastType(&typ), fClass(0), fLastClass(0),
     fVirtual(false), fContext(ctxt), fInit(false)
{
   // Standard initializing constructor

   ::new(fSubTypes) ClassMap_t();
   if ( sizeof(ClassMap_t) > sizeof(fSubTypes) ) {
      Fatal("TIsAProxy::TIsAProxy",
         "Classmap size is badly adjusted: it needs %u instead of %u bytes.",
         (UInt_t)sizeof(ClassMap_t), (UInt_t)sizeof(fSubTypes));
   }
}

//______________________________________________________________________________
TIsAProxy::TIsAProxy(const TIsAProxy& iap) :
  TVirtualIsAProxy(iap),
  fType(iap.fType),
  fLastType(iap.fLastType),
  fClass(iap.fClass),
  fLastClass(iap.fLastClass),
  fVirtual(iap.fVirtual),
  fContext(iap.fContext),
  fInit(iap.fInit)
{
   //copy constructor
   for(Int_t i=0; i<72; i++) fSubTypes[i]=iap.fSubTypes[i];
}

//______________________________________________________________________________
TIsAProxy& TIsAProxy::operator=(const TIsAProxy& iap)
{
   //assignement operator
   if(this!=&iap) {
      TVirtualIsAProxy::operator=(iap);
      fType=iap.fType;
      fLastType=iap.fLastType;
      fClass=iap.fClass;
      fLastClass=iap.fLastClass;
      for(Int_t i=0; i<72; i++) fSubTypes[i]=iap.fSubTypes[i];
      fVirtual=iap.fVirtual;
      fContext=iap.fContext;
      fInit=iap.fInit;
   }
   return *this;
}

//______________________________________________________________________________
TIsAProxy::~TIsAProxy()
{
   // Standard destructor

   ClassMap_t* m = GetMap(fSubTypes);
   m->clear();
   m->~ClassMap_t();
}

//______________________________________________________________________________
void TIsAProxy::SetClass(TClass *cl)
{
   // Set class pointer
   GetMap(fSubTypes)->clear();
   fClass = fLastClass = cl;
}

//______________________________________________________________________________
TClass* TIsAProxy::operator()(const void *obj)
{
   // IsA callback

   if ( !fInit )  {
      fInit = kTRUE;
      if ( !fClass && fType ) fClass = TClass::GetClass(*fType);
      if ( !fClass) return 0;
      fClass->Property();
      if ( fClass->GetClassInfo() )  {
         fVirtual = (gCint->ClassInfo_ClassProperty(fClass->GetClassInfo())&G__CLS_HASVIRTUAL) == G__CLS_HASVIRTUAL;
      }
   }
   if ( !obj || !fVirtual )  {
      return fClass;
   } else  {
      // Avoid the case that the first word is a virtual_base_offset_table instead of
      // a virtual_function_table
      Long_t offset = **(Long_t**)obj;
      if ( offset == 0 ) return fClass;

      DynamicType* ptr = (DynamicType*)obj;
      const std::type_info* typ = &typeid(*ptr);

      if ( typ == fType )  {
         return fClass;
      }
      else if ( typ == fLastType )  {
         return fLastClass;
      }
      // Check if type is already in sub-class cache
      else if ( 0 != (fLastClass=(*GetMap(fSubTypes))[long(typ)]) )  {
         fLastType = typ;
      }
      // Last resort: lookup root class
      else   {
         fLastClass = TClass::GetClass(*typ);
         fLastType = typ;
         (*GetMap(fSubTypes))[long(fLastType)] = fLastClass;
      }
   }
   return fLastClass;
}
