// @(#)root/io:$Id$
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
#include "TStreamerElement.h"
#include "Riostream.h"
#include "TVirtualCollectionIterators.h"

TGenCollectionStreamer::TGenCollectionStreamer(const TGenCollectionStreamer& copy)
      : TGenCollectionProxy(copy), fReadBufferFunc(&TGenCollectionStreamer::ReadBufferDefault)
{
   // Build a Streamer for an emulated vector whose type is 'name'.
}

TGenCollectionStreamer::TGenCollectionStreamer(Info_t info, size_t iter_size)
      : TGenCollectionProxy(info, iter_size), fReadBufferFunc(&TGenCollectionStreamer::ReadBufferDefault)
{
   // Build a Streamer for a collection whose type is described by 'collectionClass'.
}

TGenCollectionStreamer::TGenCollectionStreamer(const ::ROOT::TCollectionProxyInfo &info, TClass *cl)
      : TGenCollectionProxy(info, cl), fReadBufferFunc(&TGenCollectionStreamer::ReadBufferDefault)
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
   if (!fClass) Initialize();
   return new TGenCollectionStreamer(*this);
}


void TGenCollectionStreamer::ReadPrimitives(int nElements, TBuffer &b)
{
   // Primitive input streamer.
   size_t len = fValDiff * nElements;
   char   buffer[8096];
   Bool_t   feed = false;
   void*  memory = 0;
   StreamHelper* itm = 0;
   fEnv->fSize = nElements;
   switch (fSTL_type)  {
      case TClassEdit::kVector:
         if (fVal->fKind != EDataType(kBOOL_t))  {
            fResize(fEnv->fObject,fEnv->fSize);
            fEnv->fIdx = 0;
            
            TVirtualVectorIterators iterators(fFunctionCreateIterators);
            iterators.CreateIterators(fEnv->fObject);
            itm = (StreamHelper*)iterators.fBegin;
            fEnv->fStart = itm;
            break;
         }
      default:
         feed = true;
         itm = (StreamHelper*)(len < sizeof(buffer) ? buffer : memory =::operator new(len));
         break;
   }
   fEnv->fStart = itm;
   switch (int(fVal->fKind))   {
      case kBool_t:
         b.ReadFastArray(&itm->boolean   , nElements);
         break;
      case kChar_t:
         b.ReadFastArray(&itm->s_char    , nElements);
         break;
      case kShort_t:
         b.ReadFastArray(&itm->s_short   , nElements);
         break;
      case kInt_t:
         b.ReadFastArray(&itm->s_int     , nElements);
         break;
      case kLong_t:
         b.ReadFastArray(&itm->s_long    , nElements);
         break;
      case kLong64_t:
         b.ReadFastArray(&itm->s_longlong, nElements);
         break;
      case kFloat_t:
         b.ReadFastArray(&itm->flt       , nElements);
         break;
      case kFloat16_t:
         b.ReadFastArrayFloat16(&itm->flt, nElements);
         break;
      case kDouble_t:
         b.ReadFastArray(&itm->dbl       , nElements);
         break;
      case kBOOL_t:
         b.ReadFastArray(&itm->boolean   , nElements);
         break;
      case kUChar_t:
         b.ReadFastArray(&itm->u_char    , nElements);
         break;
      case kUShort_t:
         b.ReadFastArray(&itm->u_short   , nElements);
         break;
      case kUInt_t:
         b.ReadFastArray(&itm->u_int     , nElements);
         break;
      case kULong_t:
         b.ReadFastArray(&itm->u_long    , nElements);
         break;
      case kULong64_t:
         b.ReadFastArray(&itm->u_longlong, nElements);
         break;
      case kDouble32_t:
         b.ReadFastArrayDouble32(&itm->dbl, nElements);
         break;
      case kchar:
      case kNoType_t:
      case kOther_t:
         Error("TGenCollectionStreamer", "fType %d is not supported yet!\n", fVal->fKind);
   }
   if (feed)  {      // need to feed in data...
      fEnv->fStart = fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
      if (memory)  {
         ::operator delete(memory);
      }
   }
}

void TGenCollectionStreamer::ReadObjects(int nElements, TBuffer &b)
{
   // Object input streamer.
   Bool_t vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion() <= 3;
   size_t len = fValDiff * nElements;
   StreamHelper* itm = 0;
   char   buffer[8096];
   void*  memory = 0;
   
   TClass* onFileValClass = (fOnFileClass ? fOnFileClass->GetCollectionProxy()->GetValueClass() : 0);

   fEnv->fSize = nElements;
   switch (fSTL_type)  {
         // Simple case: contiguous memory. get address of first, then jump.
      case TClassEdit::kVector:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
         fResize(fEnv->fObject,fEnv->fSize);
         fEnv->fIdx = 0;
         
         {
            TVirtualVectorIterators iterators(fFunctionCreateIterators);
            iterators.CreateIterators(fEnv->fObject);
            itm = (StreamHelper*)iterators.fBegin;
         }
         fEnv->fStart = itm;
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(b.StreamObject(i, fVal->fType, onFileValClass ));
            case kBIT_ISSTRING:
               DOLOOP(i->read_std_string(b));
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               DOLOOP(i->set(b.ReadObjectAny(fVal->fType)));
            case G__BIT_ISPOINTER | kBIT_ISSTRING:
               DOLOOP(i->read_std_string_pointer(b));
            case G__BIT_ISPOINTER | kBIT_ISTSTRING | G__BIT_ISCLASS:
               DOLOOP(i->read_tstring_pointer(vsn3, b));
         }
#undef DOLOOP
         break;

         // No contiguous memory, but resize is possible
         // Hence accessing objects using At(i) should be not too much an overhead
      case TClassEdit::kList:
      case TClassEdit::kDeque:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)TGenCollectionProxy::At(idx); { x ;} ++idx;} break;}
         fResize(fEnv->fObject,fEnv->fSize);
         fEnv->fIdx = 0;
         fEnv->fStart = 0;
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(b.StreamObject(i, fVal->fType, onFileValClass));
            case kBIT_ISSTRING:
               DOLOOP(i->read_std_string(b));
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               DOLOOP(i->set(b.ReadObjectAny(fVal->fType)));
            case G__BIT_ISPOINTER | kBIT_ISSTRING:
               DOLOOP(i->read_std_string_pointer(b));
            case G__BIT_ISPOINTER | kBIT_ISTSTRING | G__BIT_ISCLASS:
               DOLOOP(i->read_tstring_pointer(vsn3, b));
         }
#undef DOLOOP
         break;

         // Rather troublesome case: Objects can only be fed into the container
         // Once they are created. Need to take memory from stack or heap.
      case TClassEdit::kMultiSet:
      case TClassEdit::kSet:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;}}
         fEnv->fStart = itm = (StreamHelper*)(len < sizeof(buffer) ? buffer : memory =::operator new(len));
         fConstruct(itm,nElements);
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(b.StreamObject(i, fVal->fType, onFileValClass));
               fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
               fDestruct(fEnv->fStart,fEnv->fSize);
               break;
            case kBIT_ISSTRING:
               DOLOOP(i->read_std_string(b))
               fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
               fDestruct(fEnv->fStart,fEnv->fSize);
               break;
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               DOLOOP(i->set(b.ReadObjectAny(fVal->fType)));
               fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
               break;
            case G__BIT_ISPOINTER | kBIT_ISSTRING:
               DOLOOP(i->read_std_string_pointer(b))
               fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
               break;
            case G__BIT_ISPOINTER | kBIT_ISTSTRING | G__BIT_ISCLASS:
               DOLOOP(i->read_tstring_pointer(vsn3, b));
               fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
               break;
         }
#undef DOLOOP
         break;
      default:
         break;
   }
   if (memory) {
      ::operator delete(memory);
   }
}

void TGenCollectionStreamer::ReadPairFromMap(int nElements, TBuffer &b)
{
   // Input streamer to convert a map into another collection

   Bool_t vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion() <= 3;
   size_t len = fValDiff * nElements;
   StreamHelper* itm = 0;
   char   buffer[8096];
   void*  memory = 0;

   TStreamerInfo *pinfo = (TStreamerInfo*)fVal->fType->GetStreamerInfo();
   R__ASSERT(pinfo);
   R__ASSERT(fVal->fCase == G__BIT_ISCLASS);

   int nested = 0;
   std::vector<std::string> inside;
   TClassEdit::GetSplit(pinfo->GetName(), inside, nested);
   Value first(inside[1]);
   Value second(inside[2]);
   fValOffset = ((TStreamerElement*)pinfo->GetElements()->At(1))->GetOffset();

   fEnv->fSize = nElements;
   switch (fSTL_type)  {
         // Simple case: contiguous memory. get address of first, then jump.
      case TClassEdit::kVector:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
         fResize(fEnv->fObject,fEnv->fSize);
         fEnv->fIdx = 0;
         
         {
            TVirtualVectorIterators iterators(fFunctionCreateIterators);
            iterators.CreateIterators(fEnv->fObject);
            itm = (StreamHelper*)iterators.fBegin;
         }
         fEnv->fStart = itm;
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(
                  ReadMapHelper(i, &first, vsn3, b);
                  ReadMapHelper((StreamHelper*)(((char*)i) + fValOffset), &second, vsn3, b)
               );
         }
#undef DOLOOP
         break;

         // No contiguous memory, but resize is possible
         // Hence accessing objects using At(i) should be not too much an overhead
      case TClassEdit::kList:
      case TClassEdit::kDeque:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)TGenCollectionProxy::At(idx); { x ;} ++idx;} break;}
         fResize(fEnv->fObject,fEnv->fSize);
         fEnv->fIdx = 0;
         {
            TVirtualVectorIterators iterators(fFunctionCreateIterators);
            iterators.CreateIterators(fEnv->fObject);
            fEnv->fStart = iterators.fBegin;
         }
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(
                  char **where = (char**)(void*) & i;
                  pinfo->ReadBuffer(b, where, -1);
               );
         }
#undef DOLOOP
         break;

         // Rather troublesome case: Objects can only be fed into the container
         // Once they are created. Need to take memory from stack or heap.
      case TClassEdit::kMultiSet:
      case TClassEdit::kSet:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;}}
         fEnv->fStart = itm = (StreamHelper*)(len < sizeof(buffer) ? buffer : memory =::operator new(len));
         fConstruct(itm,nElements);
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(
                  char **where = (char**)(void*) & i;
                  pinfo->ReadBuffer(b, where, -1);
               );
               fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
               fDestruct(fEnv->fStart,fEnv->fSize);
               break;
         }
#undef DOLOOP
         break;
      default:
         break;
   }
   if (memory) {
      ::operator delete(memory);
   }
}


void TGenCollectionStreamer::ReadMapHelper(StreamHelper *i, Value *v, Bool_t vsn3,  TBuffer &b)
{
   // helper class to read std::map
   
   float f;

   switch (v->fCase) {
      case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
      case G__BIT_ISENUM:
         switch (int(v->fKind))   {
            case kBool_t:
               b >> i->boolean;
               break;
            case kChar_t:
               b >> i->s_char;
               break;
            case kShort_t:
               b >> i->s_short;
               break;
            case kInt_t:
               b >> i->s_int;
               break;
            case kLong_t:
               b >> i->s_long;
               break;
            case kLong64_t:
               b >> i->s_longlong;
               break;
            case kFloat_t:
               b >> i->flt;
               break;
            case kFloat16_t:
               b >> f;
               i->flt = float(f);
               break;
            case kDouble_t:
               b >> i->dbl;
               break;
            case kBOOL_t:
               b >> i->boolean;
               break;
            case kUChar_t:
               b >> i->u_char;
               break;
            case kUShort_t:
               b >> i->u_short;
               break;
            case kUInt_t:
               b >> i->u_int;
               break;
            case kULong_t:
               b >> i->u_long;
               break;
            case kULong64_t:
               b >> i->u_longlong;
               break;
            case kDouble32_t:
               b >> f;
               i->dbl = double(f);
               break;
            case kchar:
            case kNoType_t:
            case kOther_t:
               Error("TGenCollectionStreamer", "fType %d is not supported yet!\n", v->fKind);
         }
         break;
      case G__BIT_ISCLASS:
         b.StreamObject(i, v->fType);
         break;
      case kBIT_ISSTRING:
         i->read_std_string(b);
         break;
      case G__BIT_ISPOINTER | G__BIT_ISCLASS:
         i->set(b.ReadObjectAny(v->fType));
         break;
      case G__BIT_ISPOINTER | kBIT_ISSTRING:
         i->read_std_string_pointer(b);
         break;
      case G__BIT_ISPOINTER | kBIT_ISTSTRING | G__BIT_ISCLASS:
         i->read_tstring_pointer(vsn3, b);
         break;
   }
}

void TGenCollectionStreamer::ReadMap(int nElements, TBuffer &b)
{
   // Map input streamer.
   Bool_t vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion() <= 3;
   size_t len = fValDiff * nElements;
   Value  *v;
   char buffer[8096], *addr, *temp;
   void* memory = 0;
   StreamHelper* i;
   float f;
   fEnv->fSize  = nElements;
   fEnv->fStart = (len < sizeof(buffer) ? buffer : memory =::operator new(len));
   addr = temp = (char*)fEnv->fStart;
   fConstruct(addr,nElements);
   for (int loop, idx = 0; idx < nElements; ++idx)  {
      addr = temp + fValDiff * idx;
      v = fKey;
      for (loop = 0; loop < 2; loop++)  {
         i = (StreamHelper*)addr;
         switch (v->fCase) {
            case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
            case G__BIT_ISENUM:
               switch (int(v->fKind))   {
                  case kBool_t:
                     b >> i->boolean;
                     break;
                  case kChar_t:
                     b >> i->s_char;
                     break;
                  case kShort_t:
                     b >> i->s_short;
                     break;
                  case kInt_t:
                     b >> i->s_int;
                     break;
                  case kLong_t:
                     b >> i->s_long;
                     break;
                  case kLong64_t:
                     b >> i->s_longlong;
                     break;
                  case kFloat_t:
                     b >> i->flt;
                     break;
                  case kFloat16_t:
                     b >> f;
                     i->flt = float(f);
                     break;
                  case kDouble_t:
                     b >> i->dbl;
                     break;
                  case kBOOL_t:
                     b >> i->boolean;
                     break;
                  case kUChar_t:
                     b >> i->u_char;
                     break;
                  case kUShort_t:
                     b >> i->u_short;
                     break;
                  case kUInt_t:
                     b >> i->u_int;
                     break;
                  case kULong_t:
                     b >> i->u_long;
                     break;
                  case kULong64_t:
                     b >> i->u_longlong;
                     break;
                  case kDouble32_t:
                     b >> f;
                     i->dbl = double(f);
                     break;
                  case kchar:
                  case kNoType_t:
                  case kOther_t:
                     Error("TGenCollectionStreamer", "fType %d is not supported yet!\n", v->fKind);
               }
               break;
            case G__BIT_ISCLASS:
               b.StreamObject(i, v->fType);
               break;
            case kBIT_ISSTRING:
               i->read_std_string(b);
               break;
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               i->set(b.ReadObjectAny(v->fType));
               break;
            case G__BIT_ISPOINTER | kBIT_ISSTRING:
               i->read_std_string_pointer(b);
               break;
            case G__BIT_ISPOINTER | kBIT_ISTSTRING | G__BIT_ISCLASS:
               i->read_tstring_pointer(vsn3, b);
               break;
         }
         v = fVal;
         addr += fValOffset;
      }
   }
   fFeed(fEnv->fStart,fEnv->fObject,fEnv->fSize);
   fDestruct(fEnv->fStart,fEnv->fSize);
   if (memory) {
      ::operator delete(memory);
   }
}

void TGenCollectionStreamer::WritePrimitives(int nElements, TBuffer &b)
{
   // Primitive output streamer.
   size_t len = fValDiff * nElements;
   char   buffer[8192];
   void*  memory  = 0;
   StreamHelper* itm = 0;
   switch (fSTL_type)  {
      case TClassEdit::kVector:
         if (fVal->fKind != EDataType(kBOOL_t))  {
            itm = (StreamHelper*)(fEnv->fStart = fFirst.invoke(fEnv));
            break;
         }
      default:
         fEnv->fStart = itm = (StreamHelper*)(len < sizeof(buffer) ? buffer : memory =::operator new(len));
         fCollect.invoke(fEnv);
         break;
   }
   switch (int(fVal->fKind))   {
      case kBool_t:
         b.WriteFastArray(&itm->boolean    , nElements);
         break;
      case kChar_t:
         b.WriteFastArray(&itm->s_char    , nElements);
         break;
      case kShort_t:
         b.WriteFastArray(&itm->s_short   , nElements);
         break;
      case kInt_t:
         b.WriteFastArray(&itm->s_int     , nElements);
         break;
      case kLong_t:
         b.WriteFastArray(&itm->s_long    , nElements);
         break;
      case kLong64_t:
         b.WriteFastArray(&itm->s_longlong, nElements);
         break;
      case kFloat_t:
         b.WriteFastArray(&itm->flt       , nElements);
         break;
      case kFloat16_t:
         b.WriteFastArrayFloat16(&itm->flt, nElements);
         break;
      case kDouble_t:
         b.WriteFastArray(&itm->dbl       , nElements);
         break;
      case kBOOL_t:
         b.WriteFastArray(&itm->boolean   , nElements);
         break;
      case kUChar_t:
         b.WriteFastArray(&itm->u_char    , nElements);
         break;
      case kUShort_t:
         b.WriteFastArray(&itm->u_short   , nElements);
         break;
      case kUInt_t:
         b.WriteFastArray(&itm->u_int     , nElements);
         break;
      case kULong_t:
         b.WriteFastArray(&itm->u_long    , nElements);
         break;
      case kULong64_t:
         b.WriteFastArray(&itm->u_longlong, nElements);
         break;
      case kDouble32_t:
         b.WriteFastArrayDouble32(&itm->dbl, nElements);
         break;
      case kchar:
      case kNoType_t:
      case kOther_t:
         Error("TGenCollectionStreamer", "fType %d is not supported yet!\n", fVal->fKind);
   }
   if (memory)  {
      ::operator delete(memory);
   }
}

void TGenCollectionStreamer::WriteObjects(int nElements, TBuffer &b)
{
   // Object output streamer.
   StreamHelper* itm = 0;
   switch (fSTL_type)  {
         // Simple case: contiguous memory. get address of first, then jump.
      case TClassEdit::kVector:
#define DOLOOP(x) {int idx=0; while(idx<nElements) {StreamHelper* i=(StreamHelper*)(((char*)itm) + fValDiff*idx); { x ;} ++idx;} break;}
         itm = (StreamHelper*)fFirst.invoke(fEnv);
         switch (fVal->fCase) {
            case G__BIT_ISCLASS:
               DOLOOP(b.StreamObject(i, fVal->fType));
               break;
            case kBIT_ISSTRING:
               DOLOOP(TString(i->c_str()).Streamer(b));
               break;
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               DOLOOP(b.WriteObjectAny(i->ptr(), fVal->fType));
               break;
            case kBIT_ISSTRING | G__BIT_ISPOINTER:
               DOLOOP(i->write_std_string_pointer(b));
               break;
            case kBIT_ISTSTRING | G__BIT_ISCLASS | G__BIT_ISPOINTER:
               DOLOOP(i->write_tstring_pointer(b));
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
               DOLOOP(b.StreamObject(i, fVal->fType));
            case kBIT_ISSTRING:
               DOLOOP(TString(i->c_str()).Streamer(b));
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               DOLOOP(b.WriteObjectAny(i->ptr(), fVal->fType));
            case kBIT_ISSTRING | G__BIT_ISPOINTER:
               DOLOOP(i->write_std_string_pointer(b));
            case kBIT_ISTSTRING | G__BIT_ISCLASS | G__BIT_ISPOINTER:
               DOLOOP(i->write_tstring_pointer(b));
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

   for (int loop, idx = 0; idx < nElements; ++idx)  {
      char* addr = (char*)TGenCollectionProxy::At(idx);
      v = fKey;
      for (loop = 0; loop < 2; ++loop)  {
         i = (StreamHelper*)addr;
         switch (v->fCase) {
            case G__BIT_ISFUNDAMENTAL:  // Only handle primitives this way
            case G__BIT_ISENUM:
               switch (int(v->fKind))   {
                  case kBool_t:
                     b << i->boolean;
                     break;
                  case kChar_t:
                     b << i->s_char;
                     break;
                  case kShort_t:
                     b << i->s_short;
                     break;
                  case kInt_t:
                     b << i->s_int;
                     break;
                  case kLong_t:
                     b << i->s_long;
                     break;
                  case kLong64_t:
                     b << i->s_longlong;
                     break;
                  case kFloat_t:
                     b << i->flt;
                     break;
                  case kFloat16_t:
                     b << float(i->flt);
                     break;
                  case kDouble_t:
                     b << i->dbl;
                     break;
                  case kBOOL_t:
                     b << i->boolean;
                     break;
                  case kUChar_t:
                     b << i->u_char;
                     break;
                  case kUShort_t:
                     b << i->u_short;
                     break;
                  case kUInt_t:
                     b << i->u_int;
                     break;
                  case kULong_t:
                     b << i->u_long;
                     break;
                  case kULong64_t:
                     b << i->u_longlong;
                     break;
                  case kDouble32_t:
                     b << float(i->dbl);
                     break;
                  case kchar:
                  case kNoType_t:
                  case kOther_t:
                     Error("TGenCollectionStreamer", "fType %d is not supported yet!\n", v->fKind);
               }
               break;
            case G__BIT_ISCLASS:
               b.StreamObject(i, v->fType);
               break;
            case kBIT_ISSTRING:
               TString(i->c_str()).Streamer(b);
               break;
            case G__BIT_ISPOINTER | G__BIT_ISCLASS:
               b.WriteObjectAny(i->ptr(), v->fType);
               break;
            case kBIT_ISSTRING | G__BIT_ISPOINTER:
               i->write_std_string_pointer(b);
               break;
            case kBIT_ISTSTRING | G__BIT_ISCLASS | G__BIT_ISPOINTER:
               i->write_tstring_pointer(b);
               break;
         }
         addr += fValOffset;
         v = fVal;
      }
   }
}

template <typename basictype>
void TGenCollectionStreamer::ReadBufferVectorPrimitives(TBuffer &b, void *obj)
{
   int nElements = 0;
   b >> nElements;
   fResize(obj,nElements);
   
   TVirtualVectorIterators iterators(fFunctionCreateIterators);
   iterators.CreateIterators(obj);
   b.ReadFastArray((basictype*)iterators.fBegin, nElements);
}

void TGenCollectionStreamer::ReadBufferVectorPrimitivesFloat16(TBuffer &b, void *obj)
{
   int nElements = 0;
   b >> nElements;
   fResize(obj,nElements);
   
   TVirtualVectorIterators iterators(fFunctionCreateIterators);
   iterators.CreateIterators(obj);
   b.ReadFastArrayFloat16((Float16_t*)iterators.fBegin, nElements);
}

void TGenCollectionStreamer::ReadBufferVectorPrimitivesDouble32(TBuffer &b, void *obj)
{
   int nElements = 0;
   b >> nElements;
   fResize(obj,nElements);
   
   TVirtualVectorIterators iterators(fFunctionCreateIterators);
   iterators.CreateIterators(obj);
   b.ReadFastArrayDouble32((Double32_t*)iterators.fBegin, nElements);
}



void TGenCollectionStreamer::ReadBuffer(TBuffer &b, void *obj, const TClass *onFileClass)
{
   // Call the specialized function.  The first time this call ReadBufferDefault which
   // actually set to fReadBufferFunc to the 'right' specialized version.
   
   SetOnFileClass((TClass*)onFileClass);
   (this->*fReadBufferFunc)(b,obj);
}

void TGenCollectionStreamer::ReadBuffer(TBuffer &b, void *obj)
{
   // Call the specialized function.  The first time this call ReadBufferDefault which
   // actually set to fReadBufferFunc to the 'right' specialized version.
   
   (this->*fReadBufferFunc)(b,obj);
}

void TGenCollectionStreamer::ReadBufferDefault(TBuffer &b, void *obj)
{
 
   fReadBufferFunc = &TGenCollectionStreamer::ReadBufferGeneric;
   // We will need this later, so let's make sure it is initialized.
   if (!GetFunctionCreateIterators()) {
      Fatal("TGenCollectionStreamer::ReadBufferDefault","No CreateIterators function for %s",fName.c_str());
   }
   if (fSTL_type == TClassEdit::kVector && ( fVal->fCase == G__BIT_ISFUNDAMENTAL || fVal->fCase == G__BIT_ISENUM ) )
   {
      // Only handle primitives this way
      switch (int(fVal->fKind))   {
         case kBool_t:
            // Nothing use generic for now
            break;
         case kChar_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Char_t>;
            break;
         case kShort_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Short_t>;
            break;
         case kInt_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Int_t>;
            break;
         case kLong_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Long_t>;
            break;
         case kLong64_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Long64_t>;
            break;
         case kFloat_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Float_t>;
            break;
         case kFloat16_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitivesFloat16;
            break;
         case kDouble_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<Double_t>;
            break;
//         case kBOOL_t:
//            fReadBufferFunc = &ReadBufferVectorPrimitives<>;
//            break;
         case kUChar_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<UChar_t>;
            break;
         case kUShort_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<UShort_t>;
            break;
         case kUInt_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<UInt_t>;
            break;
         case kULong_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<ULong_t>;
            break;
         case kULong64_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitives<ULong64_t>;
            break;
         case kDouble32_t:
            fReadBufferFunc = &TGenCollectionStreamer::ReadBufferVectorPrimitivesDouble32;
            break;
         case kchar:
         case kNoType_t:
         case kOther_t:
            // Nothing use the generic for now
            break;
      }
   }
   (this->*fReadBufferFunc)(b,obj);
}

void TGenCollectionStreamer::ReadBufferGeneric(TBuffer &b, void *obj)
{
   TVirtualCollectionProxy::TPushPop env(this, obj);

   int nElements = 0;
   b >> nElements;

   if (nElements == 0) {
      if (obj) {
         TGenCollectionProxy::Clear("force");
      }
   } else if (nElements > 0)  {
      switch (fSTL_type)  {
         case TClassEdit::kBitSet:
            if (obj) {
               if (fProperties & kNeedDelete)   {
                  TGenCollectionProxy::Clear("force");
               }  else {
                  fClear.invoke(fEnv);
               }
            }
            ReadPrimitives(nElements, b);
            return;
         case TClassEdit::kVector:
            if (obj) {
               if (fProperties & kNeedDelete)   {
                  TGenCollectionProxy::Clear("force");
               } // a resize will be called in ReadPrimitives/ReadObjects.
               else if (fVal->fKind == EDataType(kBOOL_t)) {
                  fClear.invoke(fEnv);                  
               }
            }
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
         case TClassEdit::kList:
         case TClassEdit::kDeque:
         case TClassEdit::kMultiSet:
         case TClassEdit::kSet:
            if (obj) {
               if (fProperties & kNeedDelete)   {
                  TGenCollectionProxy::Clear("force");
               }  else {
                  fClear.invoke(fEnv);
               }
            }
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
            if (obj) {
               if (fProperties & kNeedDelete)   {
                  TGenCollectionProxy::Clear("force");
               }  else {
                  fClear.invoke(fEnv);
               }
            }
            ReadMap(nElements, b);
            break;
      }
   }
}

void TGenCollectionStreamer::Streamer(TBuffer &b)
{
   // TClassStreamer IO overload.
   if (b.IsReading()) {    //Read mode
      int nElements = 0;
      b >> nElements;
      if (fEnv->fObject)   {
         TGenCollectionProxy::Clear("force");
      }
      if (nElements > 0)  {
         switch (fSTL_type)  {
            case TClassEdit::kBitSet:
               ReadPrimitives(nElements, b);
               return;
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
   } else {    // Write case
      int nElements = fEnv->fObject ? *(size_t*)fSize.invoke(fEnv) : 0;
      b << nElements;
      if (nElements > 0)  {
         switch (fSTL_type)  {
            case TClassEdit::kBitSet:
               WritePrimitives(nElements, b);
               return;
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

void TGenCollectionStreamer::StreamerAsMap(TBuffer &b)
{
   // TClassStreamer IO overload.
   if (b.IsReading()) {    //Read mode
      int nElements = 0;
      b >> nElements;
      if (fEnv->fObject)   {
         TGenCollectionProxy::Clear("force");
      }
      if (nElements > 0)  {
         switch (fSTL_type)  {
            case TClassEdit::kMap:
            case TClassEdit::kMultiMap:
               ReadMap(nElements, b);
               break;
            case TClassEdit::kVector:
            case TClassEdit::kList:
            case TClassEdit::kDeque:
            case TClassEdit::kMultiSet:
            case TClassEdit::kSet: {
                  ReadPairFromMap(nElements, b);
                  break;
               }
            default:
               break;
         }
      }
   } else {    // Write case
      Streamer(b);
   }
}
