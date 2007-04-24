// @(#)root/cont:$Name:  $:$Id: TGenCollectionProxy.cxx,v 1.28 2006/05/19 07:30:04 brun Exp $
// Author: Markus Frank 28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGenCollectionProxy
//
// Proxy around an arbitrary container, which implements basic
// functionality and iteration.
//
// In particular this is used to implement splitting and abstract
// element access of any container. Access to compiled code is necessary
// to implement the abstract iteration sequence and functionality like
// size(), clear(), resize(). resize() may be a void operation.
//
//////////////////////////////////////////////////////////////////////////

#include "TGenCollectionProxy.h"
#include "TStreamerElement.h"
#include "TClassEdit.h"
#include "Property.h"
#include "TClass.h"
#include "TError.h"
#include "TROOT.h"
#include "Api.h"
#include "Riostream.h"
#include "TVirtualMutex.h"

#define MESSAGE(which,text)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  class TGenVectorProxy
//
//   Local optimization class.
//
//   Collection proxies get copied. On copy we switch the type of the
//   proxy to the concrete STL type. The concrete types are optimized
//   for element access.
//
//////////////////////////////////////////////////////////////////////////
class TGenVectorProxy : public TGenCollectionProxy {
public:
   // Standard Destructor
   TGenVectorProxy(const TGenCollectionProxy& c) : TGenCollectionProxy(c)
{
   }
   // Standard Destructor
   virtual ~TGenVectorProxy()
{
   }
   // Return the address of the value at index 'idx'
   virtual void* At(UInt_t idx)
{
      if ( fEnv && fEnv->object ) {
         fEnv->idx = idx;
         switch( idx ) {
         case 0:
            return fEnv->start = fFirst.invoke(fEnv);
         default:
            if (! fEnv->start ) fEnv->start = fFirst.invoke(fEnv);
            return ((char*)fEnv->start) + fValDiff*idx;
         }
      }
      Fatal("TGenVectorProxy","At> Logic error - no proxy object set.");
      return 0;
   }
   // Call to delete/destruct individual item
   virtual void DeleteItem(bool force, void* ptr) const
   {
      if ( force && ptr ) {
         fVal->DeleteItem(ptr);
      }
   }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  class TGenListProxy
//
//   Localoptimization class.
//
//   Collection proxies get copied. On copy we switch the type of the
//   proxy to the concrete STL type. The concrete types are optimized
//   for element access.
//
//////////////////////////////////////////////////////////////////////////
class TGenListProxy : public TGenVectorProxy {
public:
   // Standard Destructor
   TGenListProxy(const TGenCollectionProxy& c) : TGenVectorProxy(c)
{
   }
   // Standard Destructor
   virtual ~TGenListProxy()
{
   }
   // Return the address of the value at index 'idx'
   void* At(UInt_t idx)
{
      if ( fEnv && fEnv->object ) {
         switch( idx ) {
         case 0:
            fEnv->idx = idx;
            return fEnv->start = fFirst.invoke(fEnv);
         default:  {
            fEnv->idx = idx - fEnv->idx;
            if (! fEnv->start ) fEnv->start = fFirst.invoke(fEnv);
            void* result = fNext.invoke(fEnv);
            fEnv->idx = idx;
            return result;
         }
         }
      }
      Fatal("TGenListProxy","At> Logic error - no proxy object set.");
      return 0;
   }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// class TGenSetProxy
//
//   Localoptimization class.
//
//   Collection proxies get copied. On copy we switch the type of the
//   proxy to the concrete STL type. The concrete types are optimized
//   for element access.
//
//////////////////////////////////////////////////////////////////////////
class TGenSetProxy : public TGenVectorProxy {
public:
   // Standard Destructor
   TGenSetProxy(const TGenCollectionProxy& c) : TGenVectorProxy(c)
{
   }
   // Standard Destructor
   virtual ~TGenSetProxy()
{
   }
   // Return the address of the value at index 'idx'
   void* At(UInt_t idx)
{
      if ( fEnv && fEnv->object ) {
         if ( fEnv->temp ) {
            return (((char*)fEnv->temp)+idx*fValDiff);
         }
         switch( idx ) {
         case 0:
            fEnv->idx = idx;
            return fEnv->start = fFirst.invoke(fEnv);
         default:  {
            fEnv->idx = idx - fEnv->idx;
            if (! fEnv->start ) fEnv->start = fFirst.invoke(fEnv);
            void* result = fNext.invoke(fEnv);
            fEnv->idx = idx;
            return result;
         }
         }
      }
      Fatal("TGenSetProxy","At> Logic error - no proxy object set.");
      return 0;
   }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  class TGenMapProxy
//
//   Localoptimization class.
//
//   Collection proxies get copied. On copy we switch the type of the
//   proxy to the concrete STL type. The concrete types are optimized
//   for element access.
//
//////////////////////////////////////////////////////////////////////////
class TGenMapProxy : public TGenSetProxy {
public:
   // Standard Destructor
   TGenMapProxy(const TGenCollectionProxy& c) : TGenSetProxy(c)
{
   }
   // Standard Destructor
   virtual ~TGenMapProxy()
{
   }
   // Call to delete/destruct individual item
   virtual void DeleteItem(Bool_t /* force */, void* ptr) const
   {
      if ( fKey->fCase&G__BIT_ISPOINTER ) {
         fKey->DeleteItem(*(void**)ptr);
      }
      if ( fVal->fCase&G__BIT_ISPOINTER ) {
         char *addr = ((char*)ptr)+fValOffset;
         fVal->DeleteItem(*(void**)addr);
      }
   }
};


//______________________________________________________________________________
TGenCollectionProxy::Value::Value(const Value& copy)
{
   // Constructor.

   fType   = copy.fType;
   fCase   = copy.fCase;
   fKind   = copy.fKind;
   fSize   = copy.fSize;
   fCtor   = copy.fCtor;
   fDtor   = copy.fDtor;
   fDelete = copy.fDelete;
}

//______________________________________________________________________________
TGenCollectionProxy::Value::Value(const std::string& inside_type)
{
   // Constructor.

   std::string inside = (inside_type.find("const ")==0) ? inside_type.substr(6) : inside_type;
   fCase = 0;
   fCtor = 0;
   fDtor = 0;
   fDelete = 0;
   fSize = std::string::npos;
   fKind = kNoType_t;
   std::string intype = TClassEdit::ShortType(inside.c_str(),TClassEdit::kDropTrailStar );
   if ( inside.substr(0,6) == "string" || inside.substr(0,11) == "std::string" ) {
      fCase = kBIT_ISSTRING;
      fType = gROOT->GetClass("string");
      fCtor = fType->GetNew();
      fDtor = fType->GetDestructor();
      fDelete = fType->GetDelete();
      switch(inside[inside.length()-1]) {
      case '*':
         fCase |= G__BIT_ISPOINTER;
         fSize = sizeof(void*);
         break;
      default:
         fSize = sizeof(std::string);
         break;
      }
   }
   else {
      // In the case where we have an emulated class,
      // if the class is nested (in a class or a namespace),
      // calling G__TypeInfo ti(inside.c_str());
      // might fail because CINT does not known the nesting
      // scope, so let's first look for an emulated class:
      fType = gROOT->GetClass(intype.c_str());
      if (fType && !fType->IsLoaded()) {
         if (intype != inside) {
            fCase |= G__BIT_ISPOINTER;
            fSize = sizeof(void*);
         }
         fCase  |= G__BIT_ISCLASS;
         fCtor   = fType->GetNew();
         fDtor   = fType->GetDestructor();
         fDelete = fType->GetDelete();
      } else {
         G__TypeInfo ti(inside.c_str());
         if ( !ti.IsValid() ) {
            if (intype != inside) {
               fCase |= G__BIT_ISPOINTER;
               fSize = sizeof(void*);
            }
            fType = gROOT->GetClass(intype.c_str());
            if (fType) {
               fCase  |= G__BIT_ISCLASS;
               fCtor   = fType->GetNew();
               fDtor   = fType->GetDestructor();
               fDelete = fType->GetDelete();
            }
            else {
               // either we have an Emulated enum or a really unknown class!
               // let's just claim its an enum :(
               fCase = G__BIT_ISENUM;
               fSize = sizeof(Int_t);
               fKind = kInt_t;
            }
         }
         else {
            Long_t prop = ti.Property();
            if ( prop&G__BIT_ISPOINTER ) {
               fSize = sizeof(void*);
            }
            if ( prop&G__BIT_ISSTRUCT ) {
               prop |= G__BIT_ISCLASS;
            }
            if ( prop&G__BIT_ISCLASS ) {
               fType = gROOT->GetClass(intype.c_str());
               R__ASSERT(fType);
               fCtor   = fType->GetNew();
               fDtor   = fType->GetDestructor();
               fDelete = fType->GetDelete();
            }
            else if ( prop&G__BIT_ISFUNDAMENTAL ) {
               TDataType *fundType = gROOT->GetType( intype.c_str() );
               fKind = (EDataType)fundType->GetType();
               if ( 0 == strcmp("bool",fundType->GetFullTypeName()) ) {
                  fKind = (EDataType)kBOOL_t;
               }
               fSize = ti.Size();
               R__ASSERT(fKind>0 && fKind<0x16 || (fKind==-1&&(prop&G__BIT_ISPOINTER)) );
            }
            else if ( prop&G__BIT_ISENUM ) {
               fSize = sizeof(int);
               fKind = kInt_t;
            }
            fCase = prop & (G__BIT_ISPOINTER|G__BIT_ISFUNDAMENTAL|G__BIT_ISENUM|G__BIT_ISCLASS);
            if (fType == TString::Class() && (fCase&G__BIT_ISPOINTER)) {
               fCase |= kBIT_ISTSTRING;
            }
         }
      }
   }
   if ( fSize == std::string::npos ) {
      if ( fType == 0 ) {
         Fatal("TGenCollectionProxy","Could not find %s!",inside.c_str());
      }
      fSize = fType->Size();
   }
}

void TGenCollectionProxy::Value::DeleteItem(void* ptr)
{
   // Delete an item.

   if ( ptr && fCase&G__BIT_ISPOINTER ) {
      if ( fDelete ) {
         (*fDelete)(ptr);
      }
      else if ( fType ) {
         fType->Destructor(ptr);
      }
      else {
         ::operator delete(ptr);
      }
   }
}

//______________________________________________________________________________
TGenCollectionProxy::TGenCollectionProxy(const TGenCollectionProxy& copy)
   : TVirtualCollectionProxy(copy.fClass),
     fTypeinfo(copy.fTypeinfo)
{
   // Build a proxy for an emulated container.
   fEnv            = 0;
   fName           = copy.fName;
   fPointers       = copy.fPointers;
   fSTL_type       = copy.fSTL_type;
   fSize.call      = copy.fSize.call;
   fNext.call      = copy.fNext.call;
   fFirst.call     = copy.fFirst.call;
   fClear.call     = copy.fClear.call;
   fResize.call    = copy.fResize.call;
   fDestruct.call  = copy.fDestruct.call;
   fConstruct.call = copy.fConstruct.call;
   fFeed.call      = copy.fFeed.call;
   fCollect.call   = copy.fCollect.call;
   fValOffset      = copy.fValOffset;
   fValDiff        = copy.fValDiff;
   fValue          = copy.fValue ? new Value(*copy.fValue) : 0;
   fVal            = copy.fVal   ? new Value(*copy.fVal)   : 0;
   fKey            = copy.fKey   ? new Value(*copy.fKey)   : 0;
}

//______________________________________________________________________________
TGenCollectionProxy::TGenCollectionProxy(Info_t info, size_t iter_size)
   : TVirtualCollectionProxy(0),
     fTypeinfo(info)
{
   // Build a proxy for a collection whose type is described by 'collectionClass'.
   fEnv             = 0;
   fSize.call       = 0;
   fFirst.call      = 0;
   fNext.call       = 0;
   fClear.call      = 0;
   fResize.call     = 0;
   fDestruct.call   = 0;
   fConstruct.call  = 0;
   fCollect.call    = 0;
   fFeed.call       = 0;
   fValue           = 0;
   fKey             = 0;
   fVal             = 0;
   fValOffset       = 0;
   fValDiff         = 0;
   fPointers        = false;
   Env_t e;
   if ( iter_size > sizeof(e.buff) ) {
      Fatal("TGenCollectionProxy",
            "%s %s are too large:%d bytes. Maximum is:%d bytes",
            "Iterators for collection",
            fClass->GetName(),
            iter_size,
            sizeof(e.buff));
   }
}

namespace {
   typedef std::vector<ROOT::Environ<char[64]>* > Proxies_t;
   void clearProxies(Proxies_t& v)
   {
      // Clear out the proxies.

      for(Proxies_t::iterator i=v.begin(); i != v.end(); ++i) {
         ROOT::Environ<char[64]> *e = *i;
         if ( e ) {
            if ( e->temp ) ::free(e->temp);
            delete e;
         }
      }
      v.clear();
   }
}
//______________________________________________________________________________
TGenCollectionProxy::~TGenCollectionProxy()
{
   // Standard destructor
   clearProxies(fProxyList);
   clearProxies(fProxyKept);

   if ( fValue ) delete fValue;
   if ( fVal   ) delete fVal;
   if ( fKey   ) delete fKey;
}

//______________________________________________________________________________
TVirtualCollectionProxy* TGenCollectionProxy::Generate() const
{
   // Virtual copy constructor
   if ( !fClass ) Initialize();
   switch(fSTL_type) {
   case TClassEdit::kVector:
      return new TGenVectorProxy(*this);
   case TClassEdit::kList:
      return new TGenListProxy(*this);
   case TClassEdit::kMap:
   case TClassEdit::kMultiMap:
      return new TGenMapProxy(*this);
   case TClassEdit::kSet:
   case TClassEdit::kMultiSet:
      return new TGenSetProxy(*this);
   default:
      return new TGenCollectionProxy(*this);
   }
}

//______________________________________________________________________________
TGenCollectionProxy *TGenCollectionProxy::Initialize() const
{
   // Proxy initializer
   TGenCollectionProxy* p = const_cast<TGenCollectionProxy*>(this);
   if ( fClass ) return p;
   return p->InitializeEx();
}

//______________________________________________________________________________
void TGenCollectionProxy::CheckFunctions() const
{
   // Check existence of function pointers
   if ( 0 == fSize.call ) {
      Fatal("TGenCollectionProxy","No 'size' function pointer for class %s present.",fName.c_str());
   }
   if ( 0 == fResize.call ) {
      Fatal("TGenCollectionProxy","No 'resize' function for class %s present.",fName.c_str());
   }
   if ( 0 == fNext.call  ) {
      Fatal("TGenCollectionProxy","No 'next' function for class %s present.",fName.c_str());
   }
   if ( 0 == fFirst.call ) {
      Fatal("TGenCollectionProxy","No 'begin' function for class %s present.",fName.c_str());
   }
   if ( 0 == fClear.call ) {
      Fatal("TGenCollectionProxy","No 'clear' function for class %s present.",fName.c_str());
   }
   if ( 0 == fConstruct.call ) {
      Fatal("TGenCollectionProxy","No 'block constructor' function for class %s present.",fName.c_str());
   }
   if ( 0 == fDestruct.call ) {
      Fatal("TGenCollectionProxy","No 'block destructor' function for class %s present.",fName.c_str());
   }
   if ( 0 == fFeed.call ) {
      Fatal("TGenCollectionProxy","No 'data feed' function for class %s present.",fName.c_str());
   }
   if ( 0 == fCollect.call ) {
      Fatal("TGenCollectionProxy","No 'data collect' function for class %s present.",fName.c_str());
   }
}

//______________________________________________________________________________
TGenCollectionProxy *TGenCollectionProxy::InitializeEx()
{
   // Proxy initializer
   R__LOCKGUARD2(gCollectionMutex);
   if (fClass) return this;

   TClass *cl = gROOT->GetClass(fTypeinfo);
   if ( cl ) {
      fEnv    = 0;
      fName   = cl->GetName();
      fPointers  = false;
      int nested = 0;
      std::vector<std::string> inside;
      int num = TClassEdit::GetSplit(cl->GetName(),inside,nested);
      if ( num > 1 ) {
         std::string nam;
         if ( inside[0].find("stdext::hash_") != std::string::npos )
            inside[0].replace(3,10,"::");
         if ( inside[0].find("__gnu_cxx::hash_") != std::string::npos )
            inside[0].replace(0,16,"std::");
         fSTL_type = TClassEdit::STLKind(inside[0].c_str());
         int slong = sizeof(void*);
         switch ( fSTL_type ) {
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            nam = "pair<"+inside[1]+","+inside[2];
            nam += (nam[nam.length()-1]=='>') ? " >" : ">";
            fValue = new Value(nam);
            fVal   = new Value(inside[2]);
            fKey   = new Value(inside[1]);
            fPointers = fPointers || (0 != (fKey->fCase&G__BIT_ISPOINTER));
            if ( 0 == fValDiff ) {
               fValDiff = fKey->fSize + fVal->fSize;
               fValDiff += (slong - fKey->fSize%slong)%slong;
               fValDiff += (slong - fValDiff%slong)%slong;
            }
            if ( 0 == fValOffset ) {
               fValOffset = fKey->fSize;
               fValOffset += (slong - fKey->fSize%slong)%slong;
            }
            break;
         default:
            fValue = new Value(inside[1]);
            fVal   = new Value(*fValue);
            if ( 0 == fValDiff ) {
               fValDiff = fVal->fSize;
               fValDiff += (slong - fValDiff%slong)%slong;
            }
            break;
         }
         fPointers = fPointers || (0 != (fVal->fCase&G__BIT_ISPOINTER));
         fClass = cl;
         return this;
      }
      Fatal("TGenCollectionProxy","Components of %s not analysed!",cl->GetName());
   }
   Fatal("TGenCollectionProxy","Collection class %s not found!",fTypeinfo.name());
   return 0;
}

//______________________________________________________________________________
TClass *TGenCollectionProxy::GetCollectionClass()
{
   // Return a pointer to the TClass representing the container
   return fClass ? fClass : Initialize()->fClass;
}

//______________________________________________________________________________
UInt_t TGenCollectionProxy::Sizeof() const
{
   // Return the sizeof the collection object.
   return fClass->Size();
}

//______________________________________________________________________________
Bool_t TGenCollectionProxy::HasPointers() const
{
   // Return true if the content is of type 'pointer to'

   // The content of a map and multimap is always a 'pair' and hence
   // fPointers means "Flag to indicate if containee has pointers (key or value)"
   // so we need to ignore its value for map and multimap;
   return fPointers && !(fSTL_type == TClassEdit::kMap || fSTL_type == TClassEdit::kMultiMap);
}

//______________________________________________________________________________
TClass *TGenCollectionProxy::GetValueClass()
{
   // Return a pointer to the TClass representing the content.
   return fValue ? fValue->fType.GetClass() : 0;
}

//______________________________________________________________________________
void TGenCollectionProxy::SetValueClass(TClass *new_Value_type)
{
   // Set pointer to the TClass representing the content.
   fValue->fType = new_Value_type;
}

//______________________________________________________________________________
EDataType TGenCollectionProxy::GetType()
{
   // If the content is a simple numerical value, return its type (see TDataType)
   return fValue->fKind;
}

//______________________________________________________________________________
void* TGenCollectionProxy::At(UInt_t idx)
{
   // Return the address of the value at index 'idx'
   if ( fEnv && fEnv->object ) {
      switch (fSTL_type) {
      case TClassEdit::kVector:
         fEnv->idx = idx;
         switch( idx ) {
         case 0:
            return fEnv->start = fFirst.invoke(fEnv);
         default:
            if (! fEnv->start ) fEnv->start = fFirst.invoke(fEnv);
            return ((char*)fEnv->start) + fValDiff*idx;
         }
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
         if ( fEnv->temp ) {
            return (((char*)fEnv->temp)+idx*fValDiff);
         }
      default:
         switch( idx ) {
         case 0:
            fEnv->idx = idx;
            return fEnv->start = fFirst.invoke(fEnv);
         default:  {
            fEnv->idx = idx - fEnv->idx;
            if (! fEnv->start ) fEnv->start = fFirst.invoke(fEnv);
            void* result = fNext.invoke(fEnv);
            fEnv->idx = idx;
            return result;
         }
         }
      }
   }
   Fatal("TGenCollectionProxy","At> Logic error - no proxy object set.");
   return 0;
}

//______________________________________________________________________________
void TGenCollectionProxy::Clear(const char* opt)
{
   // Clear the emulated collection.
   if ( fEnv && fEnv->object ) {
      if ( fPointers && opt && *opt=='f' ) {
         size_t i, n = *(size_t*)fSize.invoke(fEnv);
         if ( n > 0 ) {
            for (i=0; i<n; ++i)
               DeleteItem(true, TGenCollectionProxy::At(i));
         }
      }
      fClear.invoke(fEnv);
   }
}

//______________________________________________________________________________
UInt_t TGenCollectionProxy::Size() const
{
   // Return the current size of the container
   if ( fEnv && fEnv->object ) {
      return *(size_t*)fSize.invoke(fEnv);
   }
   Fatal("TGenCollectionProxy","Size> Logic error - no proxy object set.");
   return 0;
}

//______________________________________________________________________________
void TGenCollectionProxy::Resize(UInt_t n, Bool_t force)
{
   // Resize the container
   if ( fEnv && fEnv->object ) {
      if ( force && fPointers ) {
         size_t i, nold = *(size_t*)fSize.invoke(fEnv);
         if ( n != nold ) {
            for (i=n; i<nold; ++i)
               DeleteItem(true, *(void**)TGenCollectionProxy::At(i));
         }
      }
      MESSAGE(3, "Resize(n)" );
      fEnv->size = n;
      fResize.invoke(fEnv);
      return;
   }
   Fatal("TGenCollectionProxy","Resize> Logic error - no proxy object set.");
}

//______________________________________________________________________________
void* TGenCollectionProxy::Allocate(UInt_t n, Bool_t /* forceDelete */ )
{
   // Allocate the needed space.

   if ( fEnv && fEnv->object ) {
      switch ( fSTL_type ) {
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
         if ( fPointers )
            Clear("force");
         else
            fClear.invoke(fEnv);
         ++fEnv->refCount;
         fEnv->size  = n;
         if ( fEnv->space < fValDiff*n ) {
            fEnv->temp = fEnv->temp ? ::realloc(fEnv->temp,fValDiff*n) : ::malloc(fValDiff*n);
            fEnv->space = fValDiff*n;
         }
         fEnv->start = fEnv->temp;
         fConstruct.invoke(fEnv);
         return fEnv;
      case TClassEdit::kVector:
      case TClassEdit::kList:
      case TClassEdit::kDeque:
         if( fPointers ) {
            Clear("force");
         }
         fEnv->size = n;
         fResize.invoke(fEnv);
         return fEnv;
      }
   }
   return 0;
}

//______________________________________________________________________________
void TGenCollectionProxy::Commit(void* env)
{
   // Commit the change.

   switch (fSTL_type) {
   case TClassEdit::kVector:
   case TClassEdit::kList:
   case TClassEdit::kDeque:
      return;
   case TClassEdit::kMap:
   case TClassEdit::kMultiMap:
   case TClassEdit::kSet:
   case TClassEdit::kMultiSet:
      if ( env ) {
         Env_t* e = (Env_t*)env;
         if ( e->object ) {
            e->start = e->temp;
            fFeed.invoke(e);
         }
         fDestruct.invoke(e);
         e->start = 0;
         --e->refCount;
      }
      return;
   default:
      return;
   }
}

//______________________________________________________________________________
void TGenCollectionProxy::PushProxy(void *objstart)
{
   // Add an object.

   if ( !fClass ) Initialize();
   if ( !fProxyList.empty() ) {
      Env_t* back = fProxyList.back();
      if ( back->object == objstart ) {
         back->refCount++;
         fProxyList.push_back(back);
         fEnv = back;
         return;
      }
   }
   Env_t* e    = 0;
   if ( fProxyKept.empty() ) {
      e = new Env_t();
      e->space = 0;
      e->temp  = 0;
   }
   else {
      e = fProxyKept.back();
      fProxyKept.pop_back();
   }
   e->size     = 0;
   e->refCount = 1;
   e->object   = objstart;
   e->start    = 0;
   e->idx      = 0;
   ::memset(e->buff,0,sizeof(e->buff));
   fProxyList.push_back(e);
   fEnv = e;
}

//______________________________________________________________________________
void TGenCollectionProxy::PopProxy()
{
   // Remove the last object.

   if ( !fProxyList.empty() ) {
      Env_t* e = fProxyList.back();
      if ( --e->refCount <= 0 ) {
         fProxyKept.push_back(e);
      }
      fProxyList.pop_back();
   }
   fEnv = fProxyList.empty() ? 0 : fProxyList.back();
}

//______________________________________________________________________________
void TGenCollectionProxy::DeleteItem(Bool_t force, void* ptr) const
{
   // Call to delete/destruct individual item.
   if ( force && ptr ) {
      switch (fSTL_type) {
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
         if ( fKey->fCase&G__BIT_ISPOINTER ) {
            fKey->DeleteItem(*(void**)ptr);
         }
         if ( fVal->fCase&G__BIT_ISPOINTER ) {
            char *addr = ((char*)ptr)+fValOffset;
            fVal->DeleteItem(*(void**)addr);
         }
         break;
      default:
         if ( fVal->fCase&G__BIT_ISPOINTER ) {
            fVal->DeleteItem(*(void**)ptr);
         }
         break;
      }
   }
}

//______________________________________________________________________________
void TGenCollectionProxy::Streamer(TBuffer &buff)
{
   // Streamer Function.
   if ( fEnv ) {
      GetCollectionClass()->Streamer( fEnv->object, buff );
      return;
   }
   Fatal("TGenCollectionProxy","Streamer> Logic error - no proxy object set.");
}

//______________________________________________________________________________
void TGenCollectionProxy::Streamer(TBuffer &buff, void *objp, int /* siz */ )
{
   // Streamer I/O overload
   TPushPop env(this, objp);
   Streamer(buff);
}

//______________________________________________________________________________
void TGenCollectionProxy::operator()(TBuffer &b, void *objp)
{
   // TClassStreamer IO overload
   Streamer(b, objp, 0);
}

