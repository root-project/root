// @(#)root/cont:$Name:  $:$Id: TEmulatedCollectionProxy.cxx,v 1.6 2004/11/11 06:06:41 brun Exp $
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
// TEmulatedCollectionProxy
//
// Streamer around an arbitrary container, which implements basic 
// functionality and iteration.
//
// In particular this is used to implement splitting and abstract
// element access of any container. Access to compiled code is necessary
// to implement the abstract iteration sequence and functionality like
// size(), clear(), resize(). resize() may be a void operation.
//
//////////////////////////////////////////////////////////////////////////

#include "TEmulatedCollectionProxy.h"
#include "TStreamerInfo.h"
#include "TClassEdit.h"
#include "TError.h"
#include "TROOT.h"
#include <iostream>

/// Build a Streamer for an emulated vector whose type is 'name'.
TEmulatedCollectionProxy::TEmulatedCollectionProxy(const TEmulatedCollectionProxy& copy)
: TGenCollectionProxy(copy)
{
}

/// Build a Streamer for a collection whose type is described by 'collectionClass'.
TEmulatedCollectionProxy::TEmulatedCollectionProxy(const char* cl_name)
: TGenCollectionProxy(typeid(std::vector<char>), sizeof(std::vector<char>::iterator))
{
  fName = cl_name;
  Initialize();
}

/// Standard destructor
TEmulatedCollectionProxy::~TEmulatedCollectionProxy()   {
}

/// Virtual copy constructor
TVirtualCollectionProxy* TEmulatedCollectionProxy::Generate() const  { 
  if ( !fClass ) Initialize();
  return new TEmulatedCollectionProxy(*this);
}

/// Proxy initializer
TGenCollectionProxy *TEmulatedCollectionProxy::InitializeEx() { 
  fClass = gROOT->GetClass(fName.c_str());
  fEnv = 0;
  fKey = 0;
  if ( fClass )  {
    int nested = 0;
    std::vector<std::string> inside;
    fPointers  = false;
    int num = TClassEdit::GetSplit(fName.c_str(),inside,nested);
    if ( num > 1 )  {
      std::string nam;
      fSTL_type = TClassEdit::STLKind(inside[0].c_str());
      // std::cout << "Initialized " << typeid(*this).name() << ":" << fName << std::endl;
      int slong = sizeof(void*);
      switch ( fSTL_type )  {
        case TClassEdit::kMap:
        case TClassEdit::kMultiMap:
          nam = "pair<"+inside[1]+","+inside[2];
          nam += (nam[nam.length()-1]=='>') ? " >" : ">";
          fValue = new Value(nam);
          fKey   = new Value(inside[1]);
          fVal   = new Value(inside[2]);
          fPointers |= 0 != (fKey->fCase&G__BIT_ISPOINTER);
          if ( 0 == fValDiff )  {
            fValDiff = fKey->fSize + fVal->fSize;
            fValDiff += (slong - fKey->fSize%slong)%slong;
            fValDiff += (slong - fValDiff%slong)%slong;
            
          }
          if ( 0 == fValOffset )  {
            fValOffset  = fKey->fSize;
            fValOffset += (slong - fKey->fSize%slong)%slong;
          }
          break;
        default:
          fValue = new Value(inside[1]);
          fVal   = new Value(*fValue);
          if ( 0 == fValDiff )  {
            fValDiff  = fVal->fSize;
            fValDiff += (slong - fValDiff%slong)%slong;
          }
          break;
      }
      fPointers |= 0 != (fVal->fCase&G__BIT_ISPOINTER);
      return this;
    }
    Fatal("TEmulatedCollectionProxy","Components of %s not analysed!",fClass->GetName());
  }
  Fatal("TEmulatedCollectionProxy","Collection class %s not found!",fTypeinfo.name());
  return 0;
}

/// Return the current size of the container
UInt_t TEmulatedCollectionProxy::Size() const   {
  if ( fEnv && fEnv->object )   {
    return fEnv->size = PCont_t(fEnv->object)->size()/fValDiff;
  }
  Fatal("TEmulatedCollectionProxy","Size> Logic error - no proxy object set.");
  return 0;
}

/// Clear the emulated collection.  
void TEmulatedCollectionProxy::Clear(const char* opt)  {
  Resize(0, opt && *opt=='f');
}

/// Shrink the container
void TEmulatedCollectionProxy::Shrink(UInt_t nCurr, UInt_t left, Bool_t /* force */ )  {
  typedef std::string  String_t;
  PCont_t c   = PCont_t(fEnv->object);
  char* addr  = ((char*)fEnv->start) + fValDiff*left;
  size_t i;

  switch ( fSTL_type )  {
    case TClassEdit::kMap:
    case TClassEdit::kMultiMap:
      addr = ((char*)fEnv->start) + fValDiff*left;
      switch(fKey->fCase)  {
        case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
        case G__BIT_ISENUM:
          break;
        case G__BIT_ISCLASS:
          for( i= fKey->fType ? left : nCurr; i<nCurr; ++i, addr += fValDiff )  {
            // Call emulation in case non-compiled content
            fKey->fType->Destructor(addr, kTRUE);
          }
          break;
        case R__BIT_ISSTRING:
          for( i=left; i<nCurr; ++i, addr += fValDiff )
            ((std::string*)addr)->~String_t();
          break;
        case G__BIT_ISPOINTER|G__BIT_ISCLASS:
          for( i=left; i<nCurr; ++i, addr += fValDiff )  {
            StreamHelper* h = (StreamHelper*)addr;
            void* ptr = h->ptr();
            fKey->fType->Destructor(ptr);
            h->set(0);
          }
        case G__BIT_ISPOINTER|R__BIT_ISSTRING:
          for( i=nCurr; i<left; ++i, addr += fValDiff )   {
            StreamHelper* h = (StreamHelper*)addr;
            delete (std::string*)h->ptr();
            h->set(0);
          }
          break;
        case G__BIT_ISPOINTER|R__BIT_ISTSTRING|G__BIT_ISCLASS:
          for( i=nCurr; i<left; ++i, addr += fValDiff )   {
            StreamHelper* h = (StreamHelper*)addr;
            delete (TString*)h->ptr();
            h->set(0);
          }
          break;
      }
      addr = ((char*)fEnv->start)+fValOffset+fValDiff*left;
      // DO NOT break; just continue

    // General case for all values
    default:
      switch( fVal->fCase )  {
        case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
        case G__BIT_ISENUM:
          break;
        case G__BIT_ISCLASS:
          for( i=left; i<nCurr; ++i, addr += fValDiff )  {
            // Call emulation in case non-compiled content
            fVal->fType->Destructor(addr,kTRUE);
          }
          break;
        case R__BIT_ISSTRING:
          for( i=left; i<nCurr; ++i, addr += fValDiff )
            ((std::string*)addr)->~String_t();
          break;
        case G__BIT_ISPOINTER|G__BIT_ISCLASS:
          for( i=left; i<nCurr; ++i, addr += fValDiff )  {
            StreamHelper* h = (StreamHelper*)addr;
            void* p = h->ptr();
            if ( p )  {
              fVal->fType->Destructor(p);
            }
            h->set(0);
          }
        case G__BIT_ISPOINTER|R__BIT_ISSTRING:
          for( i=nCurr; i<left; ++i, addr += fValDiff )   {
            StreamHelper* h = (StreamHelper*)addr;
            delete (std::string*)h->ptr();
            h->set(0);
          }
          break;
        case G__BIT_ISPOINTER|R__BIT_ISTSTRING|G__BIT_ISCLASS:
          for( i=nCurr; i<left; ++i, addr += fValDiff )   {
            StreamHelper* h = (StreamHelper*)addr;
            delete (TString*)h->ptr();
            h->set(0);
          }
          break;
      }
  }
  c->resize(left*fValDiff,0);
  fEnv->start = left>0 ? &(*c->begin()) : 0;
  return;
}

/// Expand the container
void TEmulatedCollectionProxy::Expand(UInt_t nCurr, UInt_t left)  {
  size_t i;
  PCont_t c   = PCont_t(fEnv->object);
  c->resize(left*fValDiff,0);
  fEnv->start = left>0 ? &(*c->begin()) : 0;
  char* addr = ((char*)fEnv->start) + fValDiff*nCurr;
  switch ( fSTL_type )  {
    case TClassEdit::kMap:
    case TClassEdit::kMultiMap:
      switch(fKey->fCase)  {
        case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
        case G__BIT_ISENUM:
          break;
        case G__BIT_ISCLASS:
          for( i=nCurr; i<left; ++i, addr += fValDiff )
            fKey->fType->New(addr);
          break;
        case R__BIT_ISSTRING:
          for( i=nCurr; i<left; ++i, addr += fValDiff )
            ::new(addr) std::string();
          break;
        case G__BIT_ISPOINTER|G__BIT_ISCLASS:
        case G__BIT_ISPOINTER|R__BIT_ISSTRING:
        case G__BIT_ISPOINTER|R__BIT_ISTSTRING|G__BIT_ISCLASS:
          for( i=nCurr; i<left; ++i, addr += fValDiff )
            *(void**)addr = 0;
          break;
      }
      addr = ((char*)fEnv->start)+fValOffset+fValDiff*nCurr;
      // DO NOT break; just continue

    // General case for all values
    default:
      switch(fVal->fCase)  {
        case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
        case G__BIT_ISENUM:
          break;
        case G__BIT_ISCLASS:
          for( i=nCurr; i<left; ++i, addr += fValDiff ) {
            fVal->fType->New(addr);
          }
         break;
        case R__BIT_ISSTRING:
          for( i=nCurr; i<left; ++i, addr += fValDiff )
            ::new(addr) std::string();
          break;
        case G__BIT_ISPOINTER|G__BIT_ISCLASS:
        case G__BIT_ISPOINTER|R__BIT_ISSTRING:
        case G__BIT_ISPOINTER|R__BIT_ISTSTRING|G__BIT_ISCLASS:
          for( i=nCurr; i<left; ++i, addr += fValDiff )
            *(void**)addr = 0;
          break;
      }
      break;
  }
}
/// Resize the container
void TEmulatedCollectionProxy::Resize(UInt_t left, Bool_t force)  {
  if ( fEnv && fEnv->object )   {
    size_t nCurr = Size();
    PCont_t c = PCont_t(fEnv->object);
    fEnv->start = nCurr>0 ? &(*c->begin()) : 0;
    if ( left == nCurr )  {
      return;
    }
    else if ( left < nCurr )  {
      Shrink(nCurr, left, force);
      return;
    }
    Expand(nCurr, left);
    return;
  }
  Fatal("TEmulatedCollectionProxy","Resize> Logic error - no proxy object set.");
}

/// Return the address of the value at index 'idx'
void* TEmulatedCollectionProxy::At(UInt_t idx)   {
  if ( fEnv && fEnv->object )   {
    PCont_t c = PCont_t(fEnv->object);
    size_t  s = c->size();
    if ( idx >= (s/fValDiff) )  {
      return 0;
    }
    return idx<(c->size()/fValDiff) ? ((char*)&(*c->begin()))+idx*fValDiff : 0;
  }
  Fatal("TEmulatedCollectionProxy","At> Logic error - no proxy object set.");
  return 0;
}

void* TEmulatedCollectionProxy::Allocate(UInt_t n, Bool_t forceDelete)  {
  Resize(n, forceDelete);
  return fEnv;
}

void TEmulatedCollectionProxy::Commit(void* /* env */ )  {
}

/// Object input streamer
void TEmulatedCollectionProxy::ReadItems(int nElements, TBuffer &b)  {
  bool vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion()<=3;
  StreamHelper* itm = (StreamHelper*)At(0);
  switch (fVal->fCase) {
    case G__BIT_ISFUNDAMENTAL:  //  Only handle primitives this way
    case G__BIT_ISENUM:
      switch( int(fVal->fKind) )   {
        case kBool_t:    b.ReadFastArray(&itm->boolean   , nElements); break;
        case kChar_t:    b.ReadFastArray(&itm->s_char    , nElements); break;
        case kShort_t:   b.ReadFastArray(&itm->s_short   , nElements); break;
        case kInt_t:     b.ReadFastArray(&itm->s_int     , nElements); break;
        case kLong_t:    b.ReadFastArray(&itm->s_long    , nElements); break;
        case kLong64_t:  b.ReadFastArray(&itm->s_longlong, nElements); break;
        case kFloat_t:   b.ReadFastArray(&itm->flt       , nElements); break;
        case kDouble_t:  b.ReadFastArray(&itm->dbl       , nElements); break;
        case kBOOL_t:    b.ReadFastArray(&itm->boolean   , nElements); break;
        case kUChar_t:   b.ReadFastArray(&itm->u_char    , nElements); break;
        case kUShort_t:  b.ReadFastArray(&itm->u_short   , nElements); break;
        case kUInt_t:    b.ReadFastArray(&itm->u_int     , nElements); break;
        case kULong_t:   b.ReadFastArray(&itm->u_long    , nElements); break;
        case kULong64_t: b.ReadFastArray(&itm->u_longlong, nElements); break;
        case kDouble32_t:b.ReadFastArrayDouble32(&itm->dbl,nElements); break;
        case kchar:
        case kNoType_t:
        case kOther_t:
          Error("TEmulatedCollectionProxy","fType %d is not supported yet!\n",fVal->fKind);
      }
      break;
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
    case G__BIT_ISCLASS: 
      DOLOOP( b.StreamObject(i,fVal->fType) );
    case R__BIT_ISSTRING:
      DOLOOP( i->read_std_string(b) );
    case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( i->read_any_object(fVal,b) );
    case G__BIT_ISPOINTER|R__BIT_ISSTRING:
      DOLOOP( i->read_std_string_pointer(b) );
    case G__BIT_ISPOINTER|R__BIT_ISTSTRING|G__BIT_ISCLASS:  
      DOLOOP( i->read_tstring_pointer(vsn3,b) );
  }
#undef DOLOOP
}

/// Object output streamer
void TEmulatedCollectionProxy::WriteItems(int nElements, TBuffer &b)  {
  StreamHelper* itm = (StreamHelper*)At(0);
  switch (fVal->fCase) {
    case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
    case G__BIT_ISENUM:
      itm = (StreamHelper*)At(0);
      switch( int(fVal->fKind) )   {
        case kBool_t:    b.WriteFastArray(&itm->boolean   , nElements); break;
        case kChar_t:    b.WriteFastArray(&itm->s_char    , nElements); break;
        case kShort_t:   b.WriteFastArray(&itm->s_short   , nElements); break;
        case kInt_t:     b.WriteFastArray(&itm->s_int     , nElements); break;
        case kLong_t:    b.WriteFastArray(&itm->s_long    , nElements); break;
        case kLong64_t:  b.WriteFastArray(&itm->s_longlong, nElements); break;
        case kFloat_t:   b.WriteFastArray(&itm->flt       , nElements); break;
        case kDouble_t:  b.WriteFastArray(&itm->dbl       , nElements); break;
        case kBOOL_t:    b.WriteFastArray(&itm->boolean   , nElements); break;
        case kUChar_t:   b.WriteFastArray(&itm->u_char    , nElements); break;
        case kUShort_t:  b.WriteFastArray(&itm->u_short   , nElements); break;
        case kUInt_t:    b.WriteFastArray(&itm->u_int     , nElements); break;
        case kULong_t:   b.WriteFastArray(&itm->u_long    , nElements); break;
        case kULong64_t: b.WriteFastArray(&itm->u_longlong, nElements); break;
        case kDouble32_t:b.WriteFastArrayDouble32(&itm->dbl,nElements); break;
        case kchar:
        case kNoType_t:
        case kOther_t:
          Error("TEmulatedCollectionProxy","fType %d is not supported yet!\n",fVal->fKind);
      }
      break;
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
    case G__BIT_ISCLASS: 
      DOLOOP( b.StreamObject(i,fVal->fType) );
    case R__BIT_ISSTRING:
      DOLOOP( TString(i->c_str()).Streamer(b) );
    case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( b.WriteObjectAny(i->ptr(),fVal->fType) );
    case R__BIT_ISSTRING|G__BIT_ISPOINTER:
      DOLOOP( i->write_std_string_pointer(b) );
    case R__BIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:
      DOLOOP( i->write_tstring_pointer(b) );
  }
#undef DOLOOP
}

/// TClassStreamer IO overload
void TEmulatedCollectionProxy::Streamer(TBuffer &b) {
  if ( b.IsReading() ) {  //Read mode 
    int nElements = 0;
    b >> nElements;
    if ( fEnv->object )  {
      Resize(nElements,true);
    }
    if ( nElements > 0 )  {
      ReadItems(nElements, b);
    }
  }
  else {     // Write case
    int nElements = fEnv->object ? *(size_t*)fSize.invoke(fEnv) : 0;
    b << nElements;
    if ( nElements > 0 )  {
      WriteItems(nElements, b);
    }
  }
}
