// @(#)root/cont:$Name:  $:$Id: TGenVectorProxy.cxx,v 1.26 2004/10/13 15:30:22 rdm Exp $
// Author: Markus Frank 28/10/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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
#include <iostream>

#define MESSAGE(which,text)  

/** @class TGenVectorProxy TGenCollectionProxy.cxx cont/TGenCollectionProxy.cxx
  *
  * Localoptimization class.
  *
  * Collection proxies get copied. On copy we switch the type of the 
  * proxy to the concrete STL type. The concrete types are optimized 
  * for element access.
  *
  * @author  M.Frank
  * @version 1.0
  * @date    10/10/2004
  */
class TGenVectorProxy : public TGenCollectionProxy {
public:
  /// Standard Destructor
  TGenVectorProxy(const TGenCollectionProxy& c) : TGenCollectionProxy(c)  {  
  }
  /// Standard Destructor
  virtual ~TGenVectorProxy() {
  }
  /// Return the address of the value at index 'idx'
  virtual void* At(UInt_t idx)   {
    if ( fEnv && fEnv->object )   {
      fEnv->idx = idx;
      switch( idx ) {
        case 0:
          return fEnv->start = fFirst.invoke(fEnv);
        default:
          return ((char*)fEnv->start) + fValDiff*idx;
      }
    }
    Fatal("TGenVectorProxy","At> Logic error - no proxy object set.");
    return 0;
  }
  /// Call to delete/destruct individual item
  virtual void DeleteItem(bool force, void* ptr)  const  {
    if ( force && ptr )  {
      if ( fValue->fDelete )  {
        (*fValue->fDelete)(ptr);
      }
      else if ( fVal->fType )  {
        fVal->fType->Destructor(ptr);
      }
      else {
        ::operator delete(ptr);
      }
    }
  }
};

/** @class TGenListProxy TGenCollectionProxy.cxx cont/TGenCollectionProxy.cxx
  *
  * Localoptimization class.
  *
  * Collection proxies get copied. On copy we switch the type of the 
  * proxy to the concrete STL type. The concrete types are optimized 
  * for element access.
  *
  * @author  M.Frank
  * @version 1.0
  * @date    10/10/2004
  */
class TGenListProxy : public TGenVectorProxy {
public:
  /// Standard Destructor
  TGenListProxy(const TGenCollectionProxy& c) : TGenVectorProxy(c)  {  
  }
  /// Standard Destructor
  virtual ~TGenListProxy() {
  }
  /// Return the address of the value at index 'idx'
  void* At(UInt_t idx)   {
    if ( fEnv && fEnv->object )   {
      switch( idx ) {
        case 0:
          fEnv->idx = idx;
          return fEnv->start = fFirst.invoke(fEnv);
        default:  {
          fEnv->idx = idx - fEnv->idx;
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

/** @class TGenSetProxy TGenCollectionProxy.cxx cont/TGenCollectionProxy.cxx
  *
  * Localoptimization class.
  *
  * Collection proxies get copied. On copy we switch the type of the 
  * proxy to the concrete STL type. The concrete types are optimized 
  * for element access.
  *
  * @author  M.Frank
  * @version 1.0
  * @date    10/10/2004
  */
class TGenSetProxy : public TGenVectorProxy {
public:
  /// Standard Destructor
  TGenSetProxy(const TGenCollectionProxy& c) : TGenVectorProxy(c)  {  
  }
  /// Standard Destructor
  virtual ~TGenSetProxy() {
  }
  /// Return the address of the value at index 'idx'
  void* At(UInt_t idx)   {
    if ( fEnv && fEnv->object )   {
      if ( fEnv->temp )  {
        return (((char*)fEnv->temp)+idx*fValDiff);
      }
      switch( idx ) {
        case 0:
          fEnv->idx = idx;
          return fEnv->start = fFirst.invoke(fEnv);
        default:  {
          fEnv->idx = idx - fEnv->idx;
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

/** @class TGenMapProxy TGenCollectionProxy.cxx cont/TGenCollectionProxy.cxx
  *
  * Localoptimization class.
  *
  * Collection proxies get copied. On copy we switch the type of the 
  * proxy to the concrete STL type. The concrete types are optimized 
  * for element access.
  *
  * @author  M.Frank
  * @version 1.0
  * @date    10/10/2004
  */
class TGenMapProxy : public TGenSetProxy {
public:
  /// Standard Destructor
  TGenMapProxy(const TGenCollectionProxy& c) : TGenSetProxy(c)  {  
  }
  /// Standard Destructor
  virtual ~TGenMapProxy() {
  }
  /// Call to delete/destruct individual item
  virtual void DeleteItem(bool /* force */, void* ptr)  const  {
    if ( fKey->fCase&G__BIT_ISPOINTER )  {
      if ( *(void**)ptr )  {
        (*fKey->fDelete)(*(void**)ptr);
      }
    }
    if ( fVal->fCase&G__BIT_ISPOINTER )  {
      char *addr = ((char*)ptr)+fKey->fSize;
      if ( *(void**)addr ) (*fVal->fDelete)(*(void**)addr);
    }
  }
};

TGenCollectionProxy::Value::Value(const Value& copy)  {
  fType   = copy.fType;
  fCase   = copy.fCase;
  fKind   = copy.fKind;
  fSize   = copy.fSize;
  fCtor   = copy.fCtor;
  fDtor   = copy.fDtor;
  fDelete = copy.fDelete;
}

TGenCollectionProxy::Value::Value(const std::string& inside)  {
  fType = 0;
  fCase = 0;
  fCtor = 0;
  fDtor = 0;
  fDelete = 0;
  fSize = std::string::npos;
  fKind = kNoType_t;
  std::string intype = TClassEdit::ShortType(inside.c_str(),TClassEdit::kDropTrailStar );
  if ( inside.substr(0,6) == "string" || inside.substr(0,11) == "std::string" ) {
    fCase = R__BIT_ISSTRING;
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
    G__TypeInfo ti(inside.c_str());
    if ( !ti.IsValid() ) {
      if (intype != inside) {
        fCase |= G__BIT_ISPOINTER;
        fSize = sizeof(void*);
      }
      fType = gROOT->GetClass(intype.c_str());
      if (fType)  {
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
      long P = ti.Property();      
      if ( P&G__BIT_ISPOINTER ) {
        fSize = sizeof(void*);
      }
      if ( P&G__BIT_ISSTRUCT ) {
        P |= G__BIT_ISCLASS;
      }
      if ( P&G__BIT_ISCLASS ) {
        fType = gROOT->GetClass(intype.c_str());
        Assert(fType);
        fCtor   = fType->GetNew();
        fDtor   = fType->GetDestructor();
        fDelete = fType->GetDelete();
      }
      else if ( P&G__BIT_ISFUNDAMENTAL ) {
        TDataType *fundType = gROOT->GetType( intype.c_str() );
        fKind = (EDataType)fundType->GetType();
        fSize = ti.Size();
        Assert(fKind>0 && fKind<20);
      }
      else if ( P&G__BIT_ISENUM ) {
        fSize = sizeof(int);
        fKind = kInt_t;
      }
      fCase = P & (G__BIT_ISPOINTER|G__BIT_ISFUNDAMENTAL|G__BIT_ISENUM|G__BIT_ISCLASS);
      if (fType == TString::Class() && (fCase&G__BIT_ISPOINTER)) {
        fCase |= R__BIT_ISTSTRING;
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

/// Build a proxy for an emulated container.
TGenCollectionProxy::TGenCollectionProxy(const TGenCollectionProxy& copy)
: TVirtualCollectionProxy(copy.fClass),
  fTypeinfo(copy.fTypeinfo)
{
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

/// Build a proxy for a collection whose type is described by 'collectionClass'.
TGenCollectionProxy::TGenCollectionProxy(Info_t info, size_t iter_size)
: TVirtualCollectionProxy(0),
  fTypeinfo(info)
{
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

/// Standard destructor
TGenCollectionProxy::~TGenCollectionProxy()   {
  for(Proxies_t::iterator i=fProxyList.begin(); i != fProxyList.end(); ++i)  {
    if ( (*i) ) delete (*i);
  }
  fProxyList.clear();
  if ( fValue ) delete fValue;
  if ( fVal   ) delete fVal;
  if ( fKey   ) delete fKey;
}

/// Virtual copy constructor
TVirtualCollectionProxy* TGenCollectionProxy::Generate() const  { 
  if ( !fClass ) Initialize();
  switch(fSTL_type)  {
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

/// Proxy initializer
TGenCollectionProxy *TGenCollectionProxy::Initialize()  const { 
  TGenCollectionProxy* p = const_cast<TGenCollectionProxy*>(this);
  if ( fClass ) return p;
  return p->InitializeEx();
}

/// Check existence of function pointers
void TGenCollectionProxy::CheckFunctions()  const   {
  if ( 0 == fSize.call )   {
    Fatal("TGenCollectionProxy","No 'size' function pointer for class %s present.",fName.c_str());
  }
  if ( 0 == fResize.call )   {
    Fatal("TGenCollectionProxy","No 'resize' function for class %s present.",fName.c_str());
  }
  if ( 0 == fNext.call  )   {
    Fatal("TGenCollectionProxy","No 'next' function for class %s present.",fName.c_str());
  }
  if ( 0 == fFirst.call )   {
    Fatal("TGenCollectionProxy","No 'begin' function for class %s present.",fName.c_str());
  }
  if ( 0 == fClear.call )   {
    Fatal("TGenCollectionProxy","No 'clear' function for class %s present.",fName.c_str());
  }
  if ( 0 == fConstruct.call )   {
    Fatal("TGenCollectionProxy","No 'block constructor' function for class %s present.",fName.c_str());
  }
  if ( 0 == fDestruct.call )   {
    Fatal("TGenCollectionProxy","No 'block destructor' function for class %s present.",fName.c_str());
  }
  if ( 0 == fFeed.call )   {
    Fatal("TGenCollectionProxy","No 'data feed' function for class %s present.",fName.c_str());
  }
  if ( 0 == fCollect.call )   {
    Fatal("TGenCollectionProxy","No 'data collect' function for class %s present.",fName.c_str());
  }
}

/// Proxy initializer
TGenCollectionProxy *TGenCollectionProxy::InitializeEx() { 
  fClass = gROOT->GetClass(fTypeinfo);
  if ( fClass )  {
    fEnv    = 0;
    fName   = fClass->GetName();
    fPointers  = false;
    int nested = 0;
    std::vector<std::string> inside;
    int num = TClassEdit::GetSplit(fClass->GetName(),inside,nested);
    if ( num > 1 )  {
      std::string nam;
      fSTL_type = TClassEdit::STLKind(inside[0].c_str());
      switch ( fSTL_type )  {
        case TClassEdit::kMap:
        case TClassEdit::kMultiMap:
          nam = "pair<"+inside[1]+","+inside[2];
          nam += (nam[nam.length()-1]=='>') ? " >" : ">";
          fValue = new Value(nam);
          fVal   = new Value(inside[2]);
          fKey   = new Value(inside[1]);
          fPointers = fPointers || (0 != (fKey->fCase&G__BIT_ISPOINTER));
          if ( 0 == fValDiff )  {
            fValDiff = fKey->fSize + fVal->fSize;
          }
          if ( 0 == fValOffset )  {
            fValOffset = fKey->fSize;
          }
          break;
        default:
          fValue = new Value(inside[1]);
          fVal   = new Value(*fValue);
          if ( 0 == fValDiff )  {
            fValDiff = fVal->fSize;
          }
          break;
      }
      fPointers = fPointers || (0 != (fVal->fCase&G__BIT_ISPOINTER));
      return this;
    }
    Fatal("TGenCollectionProxy","Components of %s not analysed!",fClass->GetName());
  }
  Fatal("TGenCollectionProxy","Collection class %s not found!",fTypeinfo.name());
  return 0;
}

/// Return a pointer to the TClass representing the container
TClass *TGenCollectionProxy::GetCollectionClass()  {
  return fClass ? fClass : Initialize()->fClass;
}

/// Return the sizeof the collection object. 
UInt_t TGenCollectionProxy::Sizeof() const  {
  return fClass->Size();
}

/// Return true if the content is of type 'pointer to'
Bool_t TGenCollectionProxy::HasPointers() const   {
  return fPointers;
}

/// Return a pointer to the TClass representing the content.
TClass *TGenCollectionProxy::GetValueClass()  {
  return fValue->fType;
}

/// Set pointer to the TClass representing the content.
void TGenCollectionProxy::SetValueClass(TClass *new_Value_type)  {
  fValue->fType = new_Value_type;
}

/// If the content is a simple numerical value, return its type (see TDataType)
EDataType TGenCollectionProxy::GetType()   {
  return fValue->fKind;
}

/// Return the address of the value at index 'idx'
void* TGenCollectionProxy::At(UInt_t idx)   {
  if ( fEnv && fEnv->object )   {
    switch (fSTL_type)  {
      case TClassEdit::kVector:
        fEnv->idx = idx;
        switch( idx ) {
          case 0:
            return fEnv->start = fFirst.invoke(fEnv);
          default:
            return ((char*)fEnv->start) + fValDiff*idx;
        }
      case TClassEdit::kSet:
      case TClassEdit::kMultiSet:
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
        if ( fEnv->temp )  {
          return (((char*)fEnv->temp)+idx*fValDiff);
        }
      default:
        switch( idx ) {
          case 0:
            fEnv->idx = idx;
            return fEnv->start = fFirst.invoke(fEnv);
          default:  {
            fEnv->idx = idx - fEnv->idx;
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

/// Clear the emulated collection.  
void TGenCollectionProxy::Clear(const char* opt)  {
  if ( fEnv && fEnv->object )   {
    if ( fPointers && opt && *opt=='f' )  {
      size_t i, n = *(size_t*)fSize.invoke(fEnv);
      if ( n > 0 )  {
        for (i=0; i<n; ++i)
          DeleteItem(true, TGenCollectionProxy::At(i));
      }
    }
    fClear.invoke(fEnv);
  }
}

/// Return the current size of the container
UInt_t TGenCollectionProxy::Size() const   {
  if ( fEnv && fEnv->object )   {
    return *(size_t*)fSize.invoke(fEnv);
  }
  Fatal("TGenCollectionProxy","Size> Logic error - no proxy object set.");
  return 0;
}

/// Resize the container
void TGenCollectionProxy::Resize(UInt_t n, Bool_t force)  {
  if ( fEnv && fEnv->object )   {
    if ( force && fPointers )  {
      size_t i, nold = *(size_t*)fSize.invoke(fEnv);
      if ( n != nold )  {
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

void* TGenCollectionProxy::Allocate(UInt_t n, Bool_t /* forceDelete */ )  {
  if ( fEnv && fEnv->object )   {
    switch ( fSTL_type )  {
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
        fEnv->start = fEnv->temp = ::operator new(fValDiff*n);
        fConstruct.invoke(fEnv);
        return fEnv;
      case TClassEdit::kVector:
      case TClassEdit::kList:
      case TClassEdit::kDeque:
        if( fPointers )  {
          Clear("force");
        }
        fEnv->size = n;
        fResize.invoke(fEnv);
        return fEnv;
    }
  }
  return 0;
}

void TGenCollectionProxy::Commit(void* env)  {
  switch (fSTL_type)  {
    case TClassEdit::kVector:
    case TClassEdit::kList:
    case TClassEdit::kDeque:
      return;
    case TClassEdit::kMap:
    case TClassEdit::kMultiMap: 
    case TClassEdit::kSet:
    case TClassEdit::kMultiSet: 
      if ( env )  {
        Env_t* e = (Env_t*)env;
        if ( e->object )   {
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

void TGenCollectionProxy::PushProxy(void *objstart) {
  if ( !fClass ) Initialize();
  if ( !fProxyList.empty() )  {
    Env_t* back = fProxyList.back();
    if ( back->object == objstart )  {
      back->refCount++;
      fProxyList.push_back(back);
      fEnv = back;
      return;
    }
  }
  Env_t* e    = new Env_t();
  e->size     = 0;
  e->refCount = 1;
  e->object   = objstart;
  e->temp     = e->start = 0;
  ::memset(e->buff,0,sizeof(e->buff));
  fProxyList.push_back(e);
  fEnv = e;
}

void TGenCollectionProxy::PopProxy() {
  if ( !fProxyList.empty() )  {
    Env_t* e = fProxyList.back();
    if ( --e->refCount <= 0 )  {
      if ( e->temp ) 
        ::operator delete(e->temp);
      delete e;
    }
    fProxyList.pop_back();
  }
  fEnv = 0;
}

/// Call to delete/destruct individual item
void TGenCollectionProxy::DeleteItem(bool force, void* ptr)  const  {
  // This needs correction in the LCG dict...
  // Pointers are not initialized to NULL!
  if ( force && ptr )  {
    switch (fSTL_type)  {
      case TClassEdit::kMap:
      case TClassEdit::kMultiMap:
        if ( fKey->fCase&G__BIT_ISPOINTER )  {
          if (*(void**)ptr) (*fKey->fDelete)(*(void**)ptr);
        }
        if ( fVal->fCase&G__BIT_ISPOINTER )  {
          char *addr = ((char*)ptr)+fKey->fSize;
          if (*(void**)addr) (*fVal->fDelete)(*(void**)addr);
        }
        // (*fValue->fDtor)(ptr); No: pair must stay intact !
        break;
      default:
        if ( fVal->fCase&G__BIT_ISPOINTER )  {
          (*fVal->fDelete)(*(void**)ptr);
        }
        break;
    }
  }
}

void TGenCollectionProxy::Streamer(TBuffer &buff)  {
  if ( fEnv )   {
    GetCollectionClass()->Streamer( fEnv->object, buff );
    return;
  }
  Fatal("TGenCollectionProxy","Streamer> Logic error - no proxy object set.");
}

/// Streamer I/O overload
void TGenCollectionProxy::Streamer(TBuffer &buff, void *objp, int /* siz */ ) {
  TPushPop env(this, objp);
  Streamer(buff);
}

/// TClassStreamer IO overload
void TGenCollectionProxy::operator()(TBuffer &b, void *objp) {
  Streamer(b, objp, 0); 
}

