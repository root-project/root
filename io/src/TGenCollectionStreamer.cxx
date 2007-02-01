// @(#)root/cont:$Name:  $:$Id: TGenCollectionStreamer.cxx,v 1.12 2007/01/29 10:53:16 brun Exp $
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
// TGenCollectionStreamer
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

#include "TGenCollectionStreamer.h"
#include "TClassEdit.h"
#include "TError.h"
#include "TROOT.h"
#include "TStreamerInfo.h"
#include "Riostream.h"

TGenCollectionStreamer::TGenCollectionStreamer(const TGenCollectionStreamer& copy)
   : TGenCollectionProxy(copy)
{
   // Build a Streamer for an emulated vector whose type is 'name'.
}

TGenCollectionStreamer::TGenCollectionStreamer(Info_t info, size_t iter_size)
   : TGenCollectionProxy(info, iter_size)
{
   // Build a Streamer for a collection whose type is described by 'collectionClass'.
}

TGenCollectionStreamer::TGenCollectionStreamer(const ::ROOT::TCollectionProxyInfo &info) 
   : TGenCollectionProxy(info)
{
   // Build a Streamer for a collection whose type is described by 'collectionClass'.
}

TGenCollectionStreamer::~TGenCollectionStreamer()
{
   // Standard destructor.
}

TVirtualCollectionProxy* TGenCollectionStreamer::Generate() const
{
   // Virtual copy constructor.
   if ( !fClass ) Initialize();
   return new TGenCollectionStreamer(*this);
}

void TGenCollectionStreamer::ReadPrimitives(int nElements, TBuffer &b)
{ 
   // Primitive input streamer.
   size_t len = fValDiff*nElements;
   char   buffer[8096];
   Bool_t   feed = false;
   void*  memory = 0;
   StreamHelper* itm = 0;
   fEnv->size = nElements;
   switch ( fSTL_type )  {
   case TClassEdit::kVector:
      if ( fVal->fKind != EDataType(kBOOL_t) )  {
         itm = (StreamHelper*)fResize.invoke(fEnv);
         break;
      }
   default:
      feed = true;
      itm = (StreamHelper*)(len<sizeof(buffer) ? buffer : memory=::operator new(len));
      break;
   }
   fEnv->start = itm;
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
      Error("TGenCollectionStreamer","fType %d is not supported yet!\n",fVal->fKind);
   }
   if ( feed )  {    // need to feed in data...
      fEnv->start = fFeed.invoke(fEnv);
      if ( memory )  {
         ::operator delete(memory);
      }
   }
}

void TGenCollectionStreamer::ReadObjects(int nElements, TBuffer &b)
{
   // Object input streamer.
   Bool_t vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion()<=3;
   size_t len = fValDiff*nElements;
   StreamHelper* itm = 0;
   char   buffer[8096];
   void*  memory = 0;

   fEnv->size = nElements;
   switch ( fSTL_type )  {
      // Simple case: contiguous memory. get address of first, then jump.
   case TClassEdit::kVector:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
   itm = (StreamHelper*)fResize.invoke(fEnv);
   switch (fVal->fCase) {
   case G__BIT_ISCLASS:
      DOLOOP( b.StreamObject(i,fVal->fType) );
   case kBIT_ISSTRING:
      DOLOOP( i->read_std_string(b) );
   case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( i->set(b.ReadObjectAny(fVal->fType)) );
   case G__BIT_ISPOINTER|kBIT_ISSTRING:
      DOLOOP( i->read_std_string_pointer(b) );
   case G__BIT_ISPOINTER|kBIT_ISTSTRING|G__BIT_ISCLASS:
      DOLOOP( i->read_tstring_pointer(vsn3,b) );
   }
#undef DOLOOP
   break;

   // No contiguous memory, but resize is possible
   // Hence accessing objects using At(i) should be not too much an overhead
   case TClassEdit::kList:
   case TClassEdit::kDeque:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)TGenCollectionProxy::At(idx); { x ;} ++idx;} break;}
   fResize.invoke(fEnv);
   switch (fVal->fCase) {
   case G__BIT_ISCLASS:
      DOLOOP( b.StreamObject(i,fVal->fType) );
   case kBIT_ISSTRING:
      DOLOOP( i->read_std_string(b) );
   case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( i->set( b.ReadObjectAny(fVal->fType) ) );
   case G__BIT_ISPOINTER|kBIT_ISSTRING:
      DOLOOP( i->read_std_string_pointer(b) );
   case G__BIT_ISPOINTER|kBIT_ISTSTRING|G__BIT_ISCLASS:
      DOLOOP( i->read_tstring_pointer(vsn3,b) );
   }
#undef DOLOOP
   break;

   // Rather troublesome case: Objects can only be fed into the container
   // Once they are created. Need to take memory from stack or heap.
   case TClassEdit::kMultiSet:
   case TClassEdit::kSet:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;}}
   fEnv->start = itm = (StreamHelper*)(len<sizeof(buffer) ? buffer : memory=::operator new(len));
   fConstruct.invoke(fEnv);
   switch ( fVal->fCase ) {
   case G__BIT_ISCLASS:
      DOLOOP( b.StreamObject(i,fVal->fType) )
         fFeed.invoke(fEnv);
      fDestruct.invoke(fEnv);
      break;
   case kBIT_ISSTRING:
      DOLOOP( i->read_std_string(b) )
         fFeed.invoke(fEnv);
      fDestruct.invoke(fEnv);
      break;
   case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( i->set(b.ReadObjectAny(fVal->fType)) );
      fFeed.invoke(fEnv);
      break;
   case G__BIT_ISPOINTER|kBIT_ISSTRING:
      DOLOOP( i->read_std_string_pointer(b) )
         fFeed.invoke(fEnv);
      break;
   case G__BIT_ISPOINTER|kBIT_ISTSTRING|G__BIT_ISCLASS:
      DOLOOP( i->read_tstring_pointer(vsn3,b) )
         fFeed.invoke(fEnv);
      break;
   }
#undef DOLOOP
   break;
   default:
      break;
   }
   if ( memory ) {
      ::operator delete(memory);
   }
}

void TGenCollectionStreamer::ReadMap(int nElements, TBuffer &b)
{
   // Map input streamer.
   Bool_t vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion()<=3;
   size_t len = fValDiff*nElements;
   Value  *v;
   char buffer[8096], *addr, *temp;
   void* memory = 0;
   StreamHelper* i;
   float f;
   fEnv->size  = nElements;
   fEnv->start = (len<sizeof(buffer) ? buffer : memory=::operator new(len));
   addr = temp = (char*)fEnv->start;
   fConstruct.invoke(fEnv);
   for ( int loop, idx = 0; idx < nElements; ++idx )  {
      addr = temp + fValDiff*idx;
      v = fKey;
      for ( loop=0; loop<2; loop++ )  {
         i = (StreamHelper*)addr;
         switch (v->fCase) {
         case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
         case G__BIT_ISENUM:
            switch( int(v->fKind) )   {
            case kBool_t:    b >> i->boolean;      break;
            case kChar_t:    b >> i->s_char;      break;
            case kShort_t:   b >> i->s_short;     break;
            case kInt_t:     b >> i->s_int;       break;
            case kLong_t:    b >> i->s_long;      break;
            case kLong64_t:  b >> i->s_longlong;  break;
            case kFloat_t:   b >> i->flt;         break;
            case kDouble_t:  b >> i->dbl;         break;
            case kBOOL_t:    b >> i->boolean;     break;
            case kUChar_t:   b >> i->u_char;      break;
            case kUShort_t:  b >> i->u_short;     break;
            case kUInt_t:    b >> i->u_int;       break;
            case kULong_t:   b >> i->u_long;      break;
            case kULong64_t: b >> i->u_longlong;  break;
            case kDouble32_t:b >> f;
               i->dbl = double(f);  break;
            case kchar:
            case kNoType_t:
            case kOther_t:
               Error("TGenCollectionStreamer","fType %d is not supported yet!\n",v->fKind);
            }
            break;
         case G__BIT_ISCLASS:
            b.StreamObject(i,v->fType);
            break;
         case kBIT_ISSTRING:
            i->read_std_string(b);
            break;
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:
            i->set(b.ReadObjectAny(v->fType));
            break;
         case G__BIT_ISPOINTER|kBIT_ISSTRING:
            i->read_std_string_pointer(b);
            break;
         case G__BIT_ISPOINTER|kBIT_ISTSTRING|G__BIT_ISCLASS:
            i->read_tstring_pointer(vsn3,b);
            break;
         }
         v = fVal;
         addr += fValOffset;
      }
   }
   fFeed.invoke(fEnv);
   fDestruct.invoke(fEnv);
   if ( memory ) {
      ::operator delete(memory);
   }
}

void TGenCollectionStreamer::WritePrimitives(int nElements, TBuffer &b)
{
   // Primitive output streamer.
   size_t len = fValDiff*nElements;
   char   buffer[8192];
   void*  memory  = 0;
   StreamHelper* itm = 0;
   switch ( fSTL_type )  {
   case TClassEdit::kVector:
      if ( fVal->fKind != EDataType(kBOOL_t) )  {
         itm = (StreamHelper*)(fEnv->start = fFirst.invoke(fEnv));
         break;
      }
   default:
      fEnv->start = itm = (StreamHelper*) (len<sizeof(buffer) ? buffer : memory=::operator new(len));
      fCollect.invoke(fEnv);
      break;
   }
   switch( int(fVal->fKind) )   {
   case kBool_t:    b.WriteFastArray(&itm->boolean    , nElements); break;
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
      Error("TGenCollectionStreamer","fType %d is not supported yet!\n",fVal->fKind);
   }
   if ( memory )  {
      ::operator delete(memory);
   }
}

void TGenCollectionStreamer::WriteObjects(int nElements, TBuffer &b)
{
   // Object output streamer.
   StreamHelper* itm = 0;
   switch(fSTL_type)  {
      // Simple case: contiguous memory. get address of first, then jump.
   case TClassEdit::kVector:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
   itm = (StreamHelper*)fFirst.invoke(fEnv);
   switch (fVal->fCase) {
   case G__BIT_ISCLASS:
      DOLOOP( b.StreamObject(i,fVal->fType));
      break;
   case kBIT_ISSTRING:
      DOLOOP( TString(i->c_str()).Streamer(b));
      break;
   case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( b.WriteObjectAny(i->ptr(),fVal->fType) );
      break;
   case kBIT_ISSTRING|G__BIT_ISPOINTER:
      DOLOOP( i->write_std_string_pointer(b));
      break;
   case kBIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:
      DOLOOP( i->write_tstring_pointer(b));
      break;
   }
#undef DOLOOP
   break;

   // No contiguous memory, but resize is possible
   // Hence accessing objects using At(i) should be not too much an overhead
   case TClassEdit::kList:
   case TClassEdit::kDeque:
   case TClassEdit::kMultiSet:
   case TClassEdit::kSet:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)TGenCollectionProxy::At(idx); { x ;} ++idx;} break;}
   switch (fVal->fCase) {
   case G__BIT_ISCLASS:
      DOLOOP( b.StreamObject(i,fVal->fType));
   case kBIT_ISSTRING:
      DOLOOP( TString(i->c_str()).Streamer(b));
   case G__BIT_ISPOINTER|G__BIT_ISCLASS:
      DOLOOP( b.WriteObjectAny(i->ptr(),fVal->fType) );
   case kBIT_ISSTRING|G__BIT_ISPOINTER:
      DOLOOP( i->write_std_string_pointer(b));
   case kBIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:
      DOLOOP( i->write_tstring_pointer(b));
   }
#undef DOLOOP
   break;
   default:
      break;
   }
}

void TGenCollectionStreamer::WriteMap(int nElements, TBuffer &b)
{
   // Map output streamer
   StreamHelper* i;
   Value  *v;

   for (int loop, idx = 0; idx < nElements; ++idx )  {
      char* addr = (char*)TGenCollectionProxy::At(idx);
      v = fKey;
      for ( loop = 0; loop<2; ++loop )  {
         i = (StreamHelper*)addr;
         switch (v->fCase) {
         case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
         case G__BIT_ISENUM:
            switch( int(v->fKind) )   {
            case kBool_t:    b << i->boolean;     break;
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
               Error("TGenCollectionStreamer","fType %d is not supported yet!\n",v->fKind);
            }
            break;
         case G__BIT_ISCLASS:
            b.StreamObject(i,v->fType);
            break;
         case kBIT_ISSTRING:
            TString(i->c_str()).Streamer(b);
            break;
         case G__BIT_ISPOINTER|G__BIT_ISCLASS:
            b.WriteObjectAny(i->ptr(),v->fType);
            break;
         case kBIT_ISSTRING|G__BIT_ISPOINTER:
            i->write_std_string_pointer(b);
            break;
         case kBIT_ISTSTRING|G__BIT_ISCLASS|G__BIT_ISPOINTER:
            i->write_tstring_pointer(b);
            break;
         }
         addr += fValOffset;
         v = fVal;
      }
   }
}

void TGenCollectionStreamer::Streamer(TBuffer &b)
{
   // TClassStreamer IO overload.
   if ( b.IsReading() ) {  //Read mode
      int nElements = 0;
      b >> nElements;
      if ( fEnv->object )   {
         TGenCollectionProxy::Clear("force");
      }
      if ( nElements > 0 )  {
         switch(fSTL_type)  {
         case TClassEdit::kVector:
         case TClassEdit::kList:
         case TClassEdit::kDeque:
         case TClassEdit::kMultiSet:
         case TClassEdit::kSet:
            switch (fVal->fCase) {
            case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
            case G__BIT_ISENUM:
               ReadPrimitives(nElements, b);
               return;
            default:
               ReadObjects(nElements, b);
               return;
            }
            break;
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            ReadMap(nElements, b);
            break;
         }
      }
   }
   else {     // Write case
      int nElements = fEnv->object ? *(size_t*)fSize.invoke(fEnv) : 0;
      b << nElements;
      if ( nElements > 0 )  {
         switch(fSTL_type)  {
         case TClassEdit::kVector:
         case TClassEdit::kList:
         case TClassEdit::kDeque:
         case TClassEdit::kMultiSet:
         case TClassEdit::kSet:
            switch (fVal->fCase) {
            case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
            case G__BIT_ISENUM:
               WritePrimitives(nElements, b);
               return;
            default:
               WriteObjects(nElements, b);
               return;
            }
            break;
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            WriteMap(nElements, b);
            break;
         }
      }
   }
}
