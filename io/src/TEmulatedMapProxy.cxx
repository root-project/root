// @(#)root/cont:$Name:  $:$Id: TEmulatedMapProxy.cxx,v 1.1 2004/10/29 18:03:10 brun Exp $
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
// TEmulatedMapProxy
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

#include "TEmulatedMapProxy.h"
#include "TStreamerInfo.h"
#include "TClassEdit.h"
#include "TError.h"

/// Build a Streamer for an emulated vector whose type is 'name'.
TEmulatedMapProxy::TEmulatedMapProxy(const TEmulatedMapProxy& copy)
: TEmulatedCollectionProxy(copy)
{
  if ( !(fSTL_type == TClassEdit::kMap || fSTL_type == TClassEdit::kMultiMap) )  {
    Fatal("TEmulatedMapProxy","Class %s is not a map-type!",fName.c_str());
  }
}

/// Build a Streamer for a collection whose type is described by 'collectionClass'.
TEmulatedMapProxy::TEmulatedMapProxy(const char* cl_name)
: TEmulatedCollectionProxy(cl_name)
{
  fName = cl_name;
  Initialize();
  if ( !(fSTL_type == TClassEdit::kMap || fSTL_type == TClassEdit::kMultiMap) )  {
    Fatal("TEmulatedMapProxy","Class %s is not a map-type!",fName.c_str());
  }
}

/// Standard destructor
TEmulatedMapProxy::~TEmulatedMapProxy()   {
}

/// Virtual copy constructor
TVirtualCollectionProxy* TEmulatedMapProxy::Generate() const  { 
  if ( !fClass ) Initialize();
  return new TEmulatedMapProxy(*this);
}

/// Return the address of the value at index 'idx'
void* TEmulatedMapProxy::At(UInt_t idx)   {
  if ( fEnv && fEnv->object )   {
    PCont_t c = PCont_t(fEnv->object);
    return idx<(c->size()/fValDiff) ? ((char*)&(*c->begin())) + idx*fValDiff : 0;
  }
  Fatal("TEmulatedMapProxy","At> Logic error - no proxy object set.");
  return 0;
}

/// Return the current size of the container
UInt_t TEmulatedMapProxy::Size() const   {
  if ( fEnv && fEnv->object )   {
    PCont_t c = PCont_t(fEnv->object);
    return fEnv->size = (c->size()/fValDiff);
  }
  Fatal("TEmulatedMapProxy","Size> Logic error - no proxy object set.");
  return 0;
}

/// Map input streamer
void TEmulatedMapProxy::ReadMap(int nElements, TBuffer &b)  {
  bool   vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion()<=3;
  int    idx, loop, off[2] = {0, fValOffset };
  Value  *v, *val[2] = { fKey, fVal };
  StreamHelper* helper;
  float f;
  char* addr = 0; 
  char* temp = (char*)At(0);
  for ( idx = 0; idx < nElements; ++idx )  {
    addr = temp + idx*fValDiff;
    for ( loop=0; loop<2; loop++)  {
      addr += off[loop];
      helper = (StreamHelper*)addr;
      v = val[loop];
      switch (v->fCase) {
        case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
        case G__BIT_ISENUM:
          switch( int(v->fKind) )   {
            case kChar_t:    b >> helper->s_char;      break;
            case kShort_t:   b >> helper->s_short;     break;
            case kInt_t:     b >> helper->s_int;       break;
            case kLong_t:    b >> helper->s_long;      break;
            case kLong64_t:  b >> helper->s_longlong;  break;
            case kFloat_t:   b >> helper->flt;         break;
            case kDouble_t:  b >> helper->dbl;         break;
            case kBOOL_t:    b >> helper->boolean;     break;
            case kUChar_t:   b >> helper->u_char;      break;
            case kUShort_t:  b >> helper->u_short;     break;
            case kUInt_t:    b >> helper->u_int;       break;
            case kULong_t:   b >> helper->u_long;      break;
            case kULong64_t: b >> helper->u_longlong;  break;
            case kDouble32_t:b >> f; 
                             helper->dbl = double(f);  break;
            case kchar:
            case kNoType_t:
            case kOther_t:
              Error("TEmulatedMapProxy","fType %d is not supported yet!\n",v->fKind);
          }
          break;
        case G__BIT_ISCLASS:
          b.StreamObject(helper,v->fType);
          break;
        case R__BIT_ISSTRING:
          helper->read_std_string(b);
          break;
        case G__BIT_ISPOINTER|G__BIT_ISCLASS:
          helper->set(b.ReadObjectAny(v->fType));
          break;
        case G__BIT_ISPOINTER|R__BIT_ISSTRING:
#ifndef R__AIX
          helper->read_std_string_pointer(b);
#endif
          break;
        case G__BIT_ISPOINTER|R__BIT_ISTSTRING|G__BIT_ISCLASS:
          helper->read_tstring_pointer(vsn3,b);
          break;
      }
    }
  }
}

/// Map output streamer
void TEmulatedMapProxy::WriteMap(int nElements, TBuffer &b)  {
  Value  *v, *val[2] = { fKey, fVal };
  int    off[2]      = { 0, fValOffset };
  StreamHelper* i;
  char* addr = 0; 
  char* temp = (char*)At(0);
  for (int loop, idx = 0; idx < nElements; ++idx )  {
    addr = temp + idx*fValDiff;
    for ( loop = 0; loop<2; ++loop )  {
      addr += off[loop];
      i = (StreamHelper*)addr;
      v = val[loop];
      switch (v->fCase) {
        case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
        case G__BIT_ISENUM:
          switch( int(v->fKind) )   {
            case kChar_t:    b << i->s_char;      break;
            case kShort_t:   b << i->s_short;     break;
            case kInt_t:     b << i->s_int;       break;
            case kLong_t:    b << i->s_long;      break;
            case kLong64_t:  b << i->s_longlong;  break;
            case kFloat_t:   b << i->flt;         break;
            case kDouble_t:  b << i->dbl;         break;
            case kBOOL_t:    b << i->boolean;     break;
            case kUChar_t:   b << i->u_char;      break;
            case kUShort_t:  b << i->u_short;     break;
            case kUInt_t:    b << i->u_int;       break;
            case kULong_t:   b << i->u_long;      break;
            case kULong64_t: b << i->u_longlong;  break;
            case kDouble32_t:b << float(i->dbl);  break;
            case kchar:
            case kNoType_t:
            case kOther_t:
              Error("TEmulatedMapProxy","fType %d is not supported yet!\n",v->fKind);
          }
          break;
        case G__BIT_ISCLASS: 
          b.StreamObject(i,v->fType);
          break;
        case R__BIT_ISSTRING:
          TString(i->c_str()).Streamer(b);
          break;
        case G__BIT_ISPOINTER|G__BIT_ISCLASS:
          b.WriteObjectAny(i->ptr(),v->fType);
          break;
        case R__BIT_ISSTRING|G__BIT_ISPOINTER:
#ifndef R__AIX
          i->write_std_string_pointer(b);
#endif
          break;
        case R__BIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:
          i->write_tstring_pointer(b);
          break;
      }
    }
  }
}

/// TClassStreamer IO overload
void TEmulatedMapProxy::Streamer(TBuffer &b) {
  if ( b.IsReading() ) {  //Read mode 
    int nElements = 0;
    b >> nElements;
    if ( fEnv->object )  {
      Resize(nElements,true);
    }
    if ( nElements > 0 )  {
      ReadMap(nElements, b);
    }
  }
  else {     // Write case
    int nElements = fEnv->object ? Size() : 0;
    b << nElements;
    if ( nElements > 0 )  {
      WriteMap(nElements, b);
    }
  }
}
