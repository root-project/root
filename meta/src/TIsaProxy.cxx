// @(#)root/meta:$Name:  $:$Id: TClass.cxx,v 1.166 2005/05/23 17:00:27 pcanal Exp $
// Author: Rene Brun   07/01/95

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TClass
#include "TClass.h"
#endif

#ifndef ROOT_TError
#include "TError.h"
#endif

#include "Api.h"
#include "TIsaProxy.h"

#include <map>


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClass                                                               //
//                                                                      //
// TIsaProxy implementation class.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

namespace {  
   struct DynamicType {    
      virtual ~DynamicType() {}  
   };  
}

typedef std::map<long, TClass*> ClassMap; // Internal type map
inline ClassMap& i_map(void* p)  { return *(ClassMap*)p; }

//______________________________________________________________________________
TIsaProxy::TIsaProxy(const std::type_info& typ, void* ctxt)
   : fType(&typ), fLastType(&typ), fClass(0), fLastClass(0), 
     fVirtual(false), fContext(ctxt), fInit(false)
{
   // Standard initializing constructor
   ::new(fSubTypes) ClassMap();
   if ( sizeof(ClassMap) > sizeof(fSubTypes) ) {
      Fatal("TIsaProxy",
         "Classmap size is badly adjusted: it needs %d instead of %d bytes.",
         sizeof(ClassMap), sizeof(fSubTypes));
   }
}

//______________________________________________________________________________
TIsaProxy::~TIsaProxy()
{
   // Standard destructor
   ClassMap* m = &i_map(fSubTypes);
   m->clear();
   m->~ClassMap();
}

//______________________________________________________________________________
void TIsaProxy::SetClass(TClass* cl)  {
   // Set class pointer
   i_map(fSubTypes).clear();
   fClass = fLastClass = cl;
}

//______________________________________________________________________________
TClass* TIsaProxy::operator()(const void* obj)  {
   /// IsA callback
   if ( !fInit )  {
      fInit = true;
      if ( !fClass && fType ) fClass = gROOT->GetClass(*fType);
      fClass->Property();
      if ( fClass->GetClassInfo() )  {
         fVirtual = (fClass->GetClassInfo()->ClassProperty()&G__CLS_HASVIRTUAL) == G__CLS_HASVIRTUAL;
      }
   }
   if ( !obj || !fVirtual )  {
      return fClass;
   }
   else  {
      // Avoid the case that the first word is a virtual_base_offset_table instead of
      // a virtual_function_table
      long offset = **(long**)obj;
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
      else if ( 0 != (fLastClass=(*(ClassMap*)fSubTypes)[long(typ)]) )  {
         fLastType = typ;
      }
      // Last resort: lookup root class
      else   {
         fLastClass = ROOT::GetROOT()->GetClass(*typ);
         (*(ClassMap*)fSubTypes)[long(fLastType=typ)] = fLastClass;
      }
   }
   return fLastClass;
}
