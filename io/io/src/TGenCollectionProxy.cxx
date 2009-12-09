// @(#)root/io:$Id$
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
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"
#include "TClassEdit.h"
#include "TClass.h"
#include "TError.h"
#include "TROOT.h"
#include "TInterpreter.h"
#include "Riostream.h"
#include "TVirtualMutex.h"
#include <stdlib.h>

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
      if ( fEnv && fEnv->fObject ) {
         fEnv->fIdx = idx;
         switch( idx ) {
         case 0:
            return fEnv->fStart = fFirst.invoke(fEnv);
         default:
            if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
            return ((char*)fEnv->fStart) + fValDiff*idx;
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
//  class TGenVectorBoolProxy
//
//   Local optimization class.
//
//   Collection proxies get copied. On copy we switch the type of the
//   proxy to the concrete STL type. The concrete types are optimized
//   for element access.
//
//////////////////////////////////////////////////////////////////////////
class TGenVectorBoolProxy : public TGenCollectionProxy {
   bool fLastValue;
   
public:
   TGenVectorBoolProxy(const TGenCollectionProxy& c) : TGenCollectionProxy(c), fLastValue(false)
   {
      // Standard Constructor.
   }
   virtual ~TGenVectorBoolProxy()
   {
      // Standard Destructor.
   }
   virtual void* At(UInt_t idx)
   {
      // Return the address of the value at index 'idx'
      
      // However we can 'take' the address of the content of vector<bool>.
      if ( fEnv && fEnv->fObject ) {
         switch( idx ) {
            case 0:
               fEnv->fStart = fFirst.invoke(fEnv);
               fEnv->fIdx = idx;
               break;
            default:
               fEnv->fIdx = idx - fEnv->fIdx;
               if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
               fNext.invoke(fEnv);
               fEnv->fIdx = idx;
               break;
         }
         typedef ROOT::TCollectionProxyInfo::Type<std::vector<bool> >::Env_t EnvType_t;
         EnvType_t *e = (EnvType_t*)fEnv;
         fLastValue = *(e->iter());
         return &fLastValue;
      }
      Fatal("TGenVectorProxy","At> Logic error - no proxy object set.");
      return 0;
   }
   
   virtual void DeleteItem(bool force, void* ptr) const
   {
      // Call to delete/destruct individual item
      if ( force && ptr ) {
         fVal->DeleteItem(ptr);
      }
   }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  class TGenBitsetProxy
//
//   Local optimization class.
//
//   Collection proxies get copied. On copy we switch the type of the
//   proxy to the concrete STL type. The concrete types are optimized
//   for element access.
//
//////////////////////////////////////////////////////////////////////////
class TGenBitsetProxy : public TGenCollectionProxy {
   
public:
   TGenBitsetProxy(const TGenCollectionProxy& c) : TGenCollectionProxy(c)
   {
      // Standard Constructor.
   }
   virtual ~TGenBitsetProxy()
   {
      // Standard Destructor.
   }
   virtual void* At(UInt_t idx)
   {
      // Return the address of the value at index 'idx'
      
      // However we can 'take' the address of the content of vector<bool>.
      if ( fEnv && fEnv->fObject ) {
         switch( idx ) {
            case 0:
               fEnv->fStart = fFirst.invoke(fEnv);
               fEnv->fIdx = idx;
               break;
            default:
               fEnv->fIdx = idx - fEnv->fIdx;
               if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
               fNext.invoke(fEnv);
               fEnv->fIdx = idx;
               break;
         }
         typedef ROOT::TCollectionProxyInfo::Environ<std::pair<size_t,bool> > EnvType_t;
         EnvType_t *e = (EnvType_t*)fEnv;
         return &(e->fIterator.second);
      }
      Fatal("TGenVectorProxy","At> Logic error - no proxy object set.");
      return 0;
   }
   
   virtual void DeleteItem(bool force, void* ptr) const
   {
      // Call to delete/destruct individual item
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
      if ( fEnv && fEnv->fObject ) {
         switch( idx ) {
         case 0:
            fEnv->fIdx = idx;
            return fEnv->fStart = fFirst.invoke(fEnv);
         default:  {
            fEnv->fIdx = idx - fEnv->fIdx;
            if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
            void* result = fNext.invoke(fEnv);
            fEnv->fIdx = idx;
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
      if ( fEnv && fEnv->fObject ) {
         if ( fEnv->fUseTemp ) {
            return (((char*)fEnv->fTemp)+idx*fValDiff);
         }
         switch( idx ) {
         case 0:
            fEnv->fIdx = idx;
            return fEnv->fStart = fFirst.invoke(fEnv);
         default:  {
            fEnv->fIdx = idx - fEnv->fIdx;
            if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
            void* result = fNext.invoke(fEnv);
            fEnv->fIdx = idx;
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
      fType = TClass::GetClass("string");
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
      fType = TClass::GetClass(intype.c_str());
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
#if defined(NOT_YET)
         // Because the TStreamerInfo of the contained classes
         // is stored only when tbere at least one element in
         // the collection, we might not even have an emulated
         // class.  So go the long route to avoid errors
         // issued by CINT ....
         G__value gval = G__string2type_body(inside.c_str(),2);
         G__TypeInfo ti(gval);
#else
         //G__TypeInfo ti(inside.c_str());
         TypeInfo_t *ti = gCint->TypeInfo_Factory();
         gCint->TypeInfo_Init(ti,inside.c_str());
#endif   
         if ( !gCint->TypeInfo_IsValid(ti) ) {
            if (intype != inside) {
               fCase |= G__BIT_ISPOINTER;
               fSize = sizeof(void*);
            }
            fType = TClass::GetClass(intype.c_str());
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
            Long_t prop = gCint->TypeInfo_Property(ti);
            if ( prop&G__BIT_ISPOINTER ) {
               fSize = sizeof(void*);
            }
            if ( prop&G__BIT_ISSTRUCT ) {
               prop |= G__BIT_ISCLASS;
            }
            if ( prop&G__BIT_ISCLASS ) {
               fType = TClass::GetClass(intype.c_str());
               R__ASSERT(fType);
               fCtor   = fType->GetNew();
               fDtor   = fType->GetDestructor();
               fDelete = fType->GetDelete();
            }
            else if ( prop&G__BIT_ISFUNDAMENTAL ) {
               TDataType *fundType = gROOT->GetType( intype.c_str() );
               if (fundType==0) {
                  if (intype != "long double") {
                     Error("TGenCollectionProxy","Unknown fundamental type %s",intype.c_str());
                  }
                  fSize = sizeof(int);
                  fKind = kInt_t;
               } else {
                  fKind = (EDataType)fundType->GetType();
                  if ( 0 == strcmp("bool",fundType->GetFullTypeName()) ) {
                     fKind = (EDataType)kBOOL_t;
                  }
                  fSize = gCint->TypeInfo_Size(ti);
                  R__ASSERT((fKind>0 && fKind<0x16) || (fKind==-1&&(prop&G__BIT_ISPOINTER)) );
               }
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
         gCint->TypeInfo_Delete(ti);
      }
   }
   if ( fSize == std::string::npos ) {
      if ( fType == 0 ) {
         // The caller should check the validity by calling IsValid()
      } else {
         fSize = fType->Size();
      }
   }
}

Bool_t TGenCollectionProxy::Value::IsValid()
{
   // Return true if the Value has been properly initialized.
   
   return fSize != std::string::npos;
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
   fCreateEnv.call = copy.fCreateEnv.call;
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
   fCreateEnv.call  = 0;
   fFeed.call       = 0;
   fValue           = 0;
   fKey             = 0;
   fVal             = 0;
   fValOffset       = 0;
   fValDiff         = 0;
   fPointers        = false;
   Env_t e;
   if ( iter_size > sizeof(e.fIterator) ) {
      Fatal("TGenCollectionProxy",
            "%s %s are too large:%d bytes. Maximum is:%d bytes",
            "Iterators for collection",
            fClass->GetName(),
            iter_size,
            sizeof(e.fIterator));
   }
}

//______________________________________________________________________________
TGenCollectionProxy::TGenCollectionProxy(const ROOT::TCollectionProxyInfo &info, TClass *cl)
   : TVirtualCollectionProxy(cl),
     fTypeinfo(info.fInfo)
{
   // Build a proxy for a collection whose type is described by 'collectionClass'.
   fEnv            = 0;
   fValDiff        = info.fValueDiff;
   fValOffset      = info.fValueOffset;
   fSize.call      = info.fSizeFunc;
   fResize.call    = info.fResizeFunc;
   fNext.call      = info.fNextFunc;
   fFirst.call     = info.fFirstFunc;
   fClear.call     = info.fClearFunc;
   fConstruct.call = info.fConstructFunc;
   fDestruct.call  = info.fDestructFunc;
   fFeed.call      = info.fFeedFunc;
   fCollect.call   = info.fCollectFunc;
   fCreateEnv.call = info.fCreateEnv;
   
   if (cl) {
      fName = cl->GetName();
   }
   CheckFunctions();

   fValue           = 0;
   fKey             = 0;
   fVal             = 0;
   fPointers        = false;

   Env_t e;
   if ( info.fIterSize > sizeof(e.fIterator) ) {
      Fatal("TGenCollectionProxy",
            "%s %s are too large:%d bytes. Maximum is:%d bytes",
            "Iterators for collection",
            fClass->GetName(),
            info.fIterSize,
            sizeof(e.fIterator));
   }
}

namespace {
   typedef std::vector<ROOT::TCollectionProxyInfo::EnvironBase* > Proxies_t;
   void clearProxies(Proxies_t& v)
   {
      // Clear out the proxies.

      for(Proxies_t::iterator i=v.begin(); i != v.end(); ++i) {
         ROOT::TCollectionProxyInfo::EnvironBase *e = *i;
         if ( e ) {
            if ( e->fTemp ) ::free(e->fTemp);
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
   if ( !fValue ) Initialize();

   if( fPointers )
      return new TGenCollectionProxy(*this);

   switch(fSTL_type) {
   case TClassEdit::kBitSet: {
      return new TGenBitsetProxy(*this);
   }
   case TClassEdit::kVector: {
      if (fValue->fKind == (EDataType)kBOOL_t) {
         return new TGenVectorBoolProxy(*this);
      } else {
         return new TGenVectorProxy(*this);
      }         
   }
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
   if ( fValue ) return p;
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
   if (0 == fCreateEnv.call ) {
      Fatal("TGenCollectionProxy","No 'environment creation' function for class %s present.",fName.c_str());
   }
}

//______________________________________________________________________________
static TGenCollectionProxy::Value *R__CreateValue(const std::string &name)
{
   // Utility routine to issue a Fatal error is the Value object is not valid
   TGenCollectionProxy::Value *val = new TGenCollectionProxy::Value( name );
   if ( !val->IsValid() ) {
      Fatal("TGenCollectionProxy","Could not find %s!",name.c_str());
   }
   return val;
}
      
//______________________________________________________________________________
TGenCollectionProxy *TGenCollectionProxy::InitializeEx()
{
   // Proxy initializer
   R__LOCKGUARD2(gCollectionMutex);
   if (fValue) return this;

   TClass *cl = fClass ? fClass.GetClass() : TClass::GetClass(fTypeinfo);
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
            fValue = R__CreateValue(nam);

            fVal   = R__CreateValue(inside[2]);
            fKey   = R__CreateValue(inside[1]);
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
         case TClassEdit::kBitSet:
            inside[1] = "bool";
            // Intentional fall through
         default:
            fValue = R__CreateValue(inside[1]);

            fVal   = new Value(*fValue);
            if ( 0 == fValDiff ) {
               fValDiff = fVal->fSize;
               fValDiff += (slong - fValDiff%slong)%slong;
            }
            break;
         }

         // Optimizing does not work with member wise streaming
         if (TVirtualStreamerInfo::GetStreamMemberWise() && fValue->fType.GetClass()) {
            Bool_t optim = TVirtualStreamerInfo::CanOptimize();
            TVirtualStreamerInfo::Optimize(kFALSE);
            fValue->fType.GetClass()->GetStreamerInfo()->Compile();
            TVirtualStreamerInfo::Optimize(optim);
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

   // Initialize proxy in case it hasn't been initialized yet
   if( !fValue )
      Initialize();

   // The content of a map and multimap is always a 'pair' and hence
   // fPointers means "Flag to indicate if containee has pointers (key or value)"
   // so we need to ignore its value for map and multimap;
   return fPointers && !(fSTL_type == TClassEdit::kMap || fSTL_type == TClassEdit::kMultiMap);
}

//______________________________________________________________________________
TClass *TGenCollectionProxy::GetValueClass()
{
   // Return a pointer to the TClass representing the content.

   if (!fValue) Initialize();
   return fValue ? fValue->fType.GetClass() : 0;
}

//______________________________________________________________________________
void TGenCollectionProxy::SetValueClass(TClass *new_Value_type)
{
   // Set pointer to the TClass representing the content.

   if (!fValue) Initialize();
   fValue->fType = new_Value_type;
}

//______________________________________________________________________________
EDataType TGenCollectionProxy::GetType()
{
   // If the content is a simple numerical value, return its type (see TDataType)

   if ( !fValue ) Initialize();
   return fValue->fKind;
}

//______________________________________________________________________________
void* TGenCollectionProxy::At(UInt_t idx)
{
   // Return the address of the value at index 'idx'
   if ( fEnv && fEnv->fObject ) {
      switch (fSTL_type) {
      case TClassEdit::kVector:
         fEnv->fIdx = idx;
         switch( idx ) {
         case 0:
            return fEnv->fStart = fFirst.invoke(fEnv);
         default:
            if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
            return ((char*)fEnv->fStart) + fValDiff*idx;
         }
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
         if ( fEnv->fUseTemp ) {
            return (((char*)fEnv->fTemp)+idx*fValDiff);
         }
         // Intentional fall through.
      default:
         switch( idx ) {
         case 0:
            fEnv->fIdx = idx;
            return fEnv->fStart = fFirst.invoke(fEnv);
         default:  {
            fEnv->fIdx = idx - fEnv->fIdx;
            if (! fEnv->fStart ) fEnv->fStart = fFirst.invoke(fEnv);
            void* result = fNext.invoke(fEnv);
            fEnv->fIdx = idx;
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
   if ( fEnv && fEnv->fObject ) {
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
   if ( fEnv && fEnv->fObject ) {
      return *(size_t*)fSize.invoke(fEnv);
   }
   Fatal("TGenCollectionProxy","Size> Logic error - no proxy object set.");
   return 0;
}

//______________________________________________________________________________
void TGenCollectionProxy::Resize(UInt_t n, Bool_t force)
{
   // Resize the container
   if ( fEnv && fEnv->fObject ) {
      if ( force && fPointers ) {
         size_t i, nold = *(size_t*)fSize.invoke(fEnv);
         if ( n != nold ) {
            for (i=n; i<nold; ++i)
               DeleteItem(true, *(void**)TGenCollectionProxy::At(i));
         }
      }
      MESSAGE(3, "Resize(n)" );
      fEnv->fSize = n;
      fResize.invoke(fEnv);
      return;
   }
   Fatal("TGenCollectionProxy","Resize> Logic error - no proxy object set.");
}

//______________________________________________________________________________
void* TGenCollectionProxy::Allocate(UInt_t n, Bool_t /* forceDelete */ )
{
   // Allocate the needed space.

   if ( fEnv && fEnv->fObject ) {
      switch ( fSTL_type ) {
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
         if ( fPointers )
            Clear("force");
         else
            fClear.invoke(fEnv);
         ++fEnv->fRefCount;
         fEnv->fSize  = n;
         if ( fEnv->fSpace < fValDiff*n ) {
            fEnv->fTemp = fEnv->fTemp ? ::realloc(fEnv->fTemp,fValDiff*n) : ::malloc(fValDiff*n);
            fEnv->fSpace = fValDiff*n;
         }
         fEnv->fUseTemp = kTRUE;
         fEnv->fStart = fEnv->fTemp;
         fConstruct.invoke(fEnv);
         return fEnv;
      case TClassEdit::kVector:
      case TClassEdit::kList:
      case TClassEdit::kDeque:
         if( fPointers ) {
            Clear("force");
         }
         fEnv->fSize = n;
         fResize.invoke(fEnv);
         return fEnv;

      case TClassEdit::kBitSet:
         // Nothing to do.
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
   case TClassEdit::kBitSet:
      return;
   case TClassEdit::kMap:
   case TClassEdit::kMultiMap:
   case TClassEdit::kSet:
   case TClassEdit::kMultiSet:
      if ( env ) {
         EnvironBase_t* e = (EnvironBase_t*)env;
         if ( e->fObject ) {
            e->fStart = e->fTemp;
            fFeed.invoke(e);
         }
         fDestruct.invoke(e);
         e->fStart = 0;
         --e->fRefCount;
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

   if ( !fValue ) Initialize();
   if ( !fProxyList.empty() ) {
      EnvironBase_t* back = fProxyList.back();
      if ( back->fObject == objstart ) {
         ++back->fRefCount;
         fProxyList.push_back(back);
         fEnv = back;
         return;
      }
   }
   EnvironBase_t* e    = 0;
   if ( fProxyKept.empty() ) {
      e = (EnvironBase_t*)fCreateEnv.invoke();
      e->fSpace = 0;
      e->fTemp  = 0;
      e->fUseTemp = kFALSE;
   }
   else {
      e = fProxyKept.back();
      fProxyKept.pop_back();
   }
   e->fSize     = 0;
   e->fRefCount = 1;
   e->fObject   = objstart;
   e->fStart    = 0;
   e->fIdx      = 0;
   // ::memset(e->buff,0,sizeof(e->buff));
   fProxyList.push_back(e);
   fEnv = e;
}

//______________________________________________________________________________
void TGenCollectionProxy::PopProxy()
{
   // Remove the last object.

   if ( !fProxyList.empty() ) {
      EnvironBase_t* e = fProxyList.back();
      if ( --e->fRefCount <= 0 ) {
         fProxyKept.push_back(e);
         e->fUseTemp = kFALSE;
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
      GetCollectionClass()->Streamer( fEnv->fObject, buff );
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

