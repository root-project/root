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
#include "TStreamerElement.h"
#include "TStreamerInfo.h"
#include "TClassEdit.h"
#include "TError.h"
#include "TROOT.h"
#include "Riostream.h"

#include "TVirtualMutex.h" // For R__LOCKGUARD
#include "TInterpreter.h"  // For gInterpreterMutex

//
// Utility function to allow the creation of a TClass for a std::pair without
// a dictionary (See end of file for implementation
//

static TStreamerElement* R__CreateEmulatedElement(const char *dmName, const char *dmFull, Int_t offset);
static TStreamerInfo *R__GenerateTClassForPair(const std::string &f, const std::string &s);

TEmulatedCollectionProxy::TEmulatedCollectionProxy(const TEmulatedCollectionProxy& copy)
   : TGenCollectionProxy(copy)
{
   // Build a Streamer for an emulated vector whose type is 'name'.
   fProperties |= kIsEmulated;
}

TEmulatedCollectionProxy::TEmulatedCollectionProxy(const char* cl_name, Bool_t silent)
   : TGenCollectionProxy(typeid(std::vector<char>), sizeof(std::vector<char>::iterator))
{
   // Build a Streamer for a collection whose type is described by 'collectionClass'.

   fName = cl_name;
   if ( this->TEmulatedCollectionProxy::InitializeEx(silent) ) {
      fCreateEnv = TGenCollectionProxy::Env_t::Create;
   }
   fProperties |= kIsEmulated;
}

TEmulatedCollectionProxy::~TEmulatedCollectionProxy()
{
   // Standard destructor
   if ( fEnv && fEnv->fObject ) {
      Clear();
   }
}

TVirtualCollectionProxy* TEmulatedCollectionProxy::Generate() const
{
   // Virtual copy constructor

   if ( !fClass ) Initialize(kFALSE);
   return new TEmulatedCollectionProxy(*this);
}

void TEmulatedCollectionProxy::Destructor(void* p, Bool_t dtorOnly) const
{
   // Virtual destructor

   if (!p) return;
   if (!fEnv || fEnv->fObject != p) { // Envoid the cost of TPushPop if we don't need it
      // FIXME: This is not thread safe.
      TVirtualCollectionProxy::TPushPop env(const_cast<TEmulatedCollectionProxy*>(this), p);
      const_cast<TEmulatedCollectionProxy*>(this)->Clear("force");
   } else {
      const_cast<TEmulatedCollectionProxy*>(this)->Clear("force");
   }
   if (dtorOnly) {
      ((Cont_t*)p)->~Cont_t();
   } else {
      delete (Cont_t*) p;
   }
}

void TEmulatedCollectionProxy::DeleteArray(void* p, Bool_t dtorOnly) const
{
   // Virtual array destructor

   // Cannot implement this properly, we do not know
   // how many elements are in the array.
   Warning("DeleteArray", "Cannot properly delete emulated array of %s at %p, I don't know how many elements it has!", fClass->GetName(), p);
   if (!dtorOnly) {
      delete[] (Cont_t*) p;
   }
}

TGenCollectionProxy *TEmulatedCollectionProxy::InitializeEx(Bool_t silent)
{
   // Proxy initializer
   R__LOCKGUARD2(gInterpreterMutex);
   if (fClass) return this;


   TClass *cl = TClass::GetClass(fName.c_str());
   fEnv = 0;
   fKey = 0;
   if ( cl )  {
      int nested = 0;
      std::vector<std::string> inside;
      fPointers  = false;
      int num = TClassEdit::GetSplit(fName.c_str(),inside,nested);
      if ( num > 1 )  {
         std::string nam;
         if ( inside[0].find("stdext::hash_") != std::string::npos ) {
            inside[0].replace(3,10,"::");
         }
         if ( inside[0].find("__gnu_cxx::hash_") != std::string::npos ) {
            inside[0].replace(0,16,"std::");
         }
         fSTL_type = TClassEdit::STLKind(inside[0].c_str());
         // Note: an emulated collection proxy is never really associative
         // since under-neath is actually an array.

         // std::cout << "Initialized " << typeid(*this).name() << ":" << fName << std::endl;
         int slong = sizeof(void*);
         switch ( fSTL_type )  {
            case ROOT::kSTLmap:
            case ROOT::kSTLmultimap:
               nam = "pair<"+inside[1]+","+inside[2];
               nam += (nam[nam.length()-1]=='>') ? " >" : ">";
               if (0==TClass::GetClass(nam.c_str())) {
                  // We need to emulate the pair
                  R__GenerateTClassForPair(inside[1],inside[2]);
               }
               fValue = new Value(nam,silent);
               fKey   = new Value(inside[1],silent);
               fVal   = new Value(inside[2],silent);
               if ( !(*fValue).IsValid() || !fKey->IsValid() || !fVal->IsValid() ) {
                  return 0;
               }
               fPointers |= 0 != (fKey->fCase&kIsPointer);
               if (fPointers || (0 != (fKey->fProperties&kNeedDelete))) {
                  fProperties |= kNeedDelete;
               }
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
            case ROOT::kSTLbitset:
               inside[1] = "bool";
               // Intentional fall through
            default:
               fValue = new Value(inside[1],silent);
               fVal   = new Value(*fValue);
               if ( !(*fValue).IsValid() || !fVal->IsValid() ) {
                  return 0;
               }
               if ( 0 == fValDiff )  {
                  fValDiff  = fVal->fSize;
                  if (fVal->fCase != kIsFundamental) {
                     fValDiff += (slong - fValDiff%slong)%slong;
                  }
               }
               break;
         }
         fPointers |= 0 != (fVal->fCase&kIsPointer);
         if (fPointers || (0 != (fVal->fProperties&kNeedDelete))) {
            fProperties |= kNeedDelete;
         }
         fClass = cl;
         return this;
      }
      Fatal("TEmulatedCollectionProxy","Components of %s not analysed!",cl->GetName());
   }
   Fatal("TEmulatedCollectionProxy","Collection class %s not found!",fTypeinfo.name());
   return 0;
}

Bool_t TEmulatedCollectionProxy::IsValid() const
{
   // Return true if the collection proxy was well initialized.
   return  (0 != fCreateEnv.call);
}

UInt_t TEmulatedCollectionProxy::Size() const
{
   // Return the current size of the container

   if ( fEnv && fEnv->fObject )   {
      return fEnv->fSize = PCont_t(fEnv->fObject)->size()/fValDiff;
   }
   Fatal("TEmulatedCollectionProxy","Size> Logic error - no proxy object set.");
   return 0;
}

void TEmulatedCollectionProxy::Clear(const char* opt)
{
   // Clear the emulated collection.
   Resize(0, opt && *opt=='f');
}

void TEmulatedCollectionProxy::Shrink(UInt_t nCurr, UInt_t left, Bool_t force )
{
   // Shrink the container

   typedef std::string  String_t;
   PCont_t c   = PCont_t(fEnv->fObject);
   char* addr  = ((char*)fEnv->fStart) + fValDiff*left;
   size_t i;

   switch ( fSTL_type )  {
      case ROOT::kSTLmap:
      case ROOT::kSTLmultimap:
         addr = ((char*)fEnv->fStart) + fValDiff*left;
         switch(fKey->fCase)  {
            case kIsFundamental:  // Only handle primitives this way
            case kIsEnum:
               break;
            case kIsClass:
               for( i= fKey->fType ? left : nCurr; i<nCurr; ++i, addr += fValDiff ) {
                  // Call emulation in case non-compiled content
                  fKey->fType->Destructor(addr, kTRUE);
               }
               break;
            case kBIT_ISSTRING:
               for( i=left; i<nCurr; ++i, addr += fValDiff ) {
                  ((std::string*)addr)->~String_t();
               }
               break;
            case kIsPointer|kIsClass:
               for( i=left; i<nCurr; ++i, addr += fValDiff )  {
                  StreamHelper* h = (StreamHelper*)addr;
                  //Eventually we'll need to delete this
                  //(but only when needed).
                  void* ptr = h->ptr();
                  if (force) fKey->fType->Destructor(ptr);
                  h->set(0);
               }
               break;
            case kIsPointer|kBIT_ISSTRING:
               for( i=nCurr; i<left; ++i, addr += fValDiff )   {
                  StreamHelper* h = (StreamHelper*)addr;
                  //Eventually we'll need to delete this
                  //(but only when needed).
                  if (force) delete (std::string*)h->ptr();
                  h->set(0);
               }
               break;
            case kIsPointer|kBIT_ISTSTRING|kIsClass:
               for( i=nCurr; i<left; ++i, addr += fValDiff )   {
                  StreamHelper* h = (StreamHelper*)addr;
                  if (force) delete (TString*)h->ptr();
                  h->set(0);
               }
               break;
         }
         addr = ((char*)fEnv->fStart)+fValOffset+fValDiff*left;
         // DO NOT break; just continue

         // General case for all values
      default:
         switch( fVal->fCase )  {
            case kIsFundamental:  // Only handle primitives this way
            case kIsEnum:
               break;
            case kIsClass:
               for( i=left; i<nCurr; ++i, addr += fValDiff )  {
                  // Call emulation in case non-compiled content
                  fVal->fType->Destructor(addr,kTRUE);
               }
               break;
            case kBIT_ISSTRING:
               for( i=left; i<nCurr; ++i, addr += fValDiff )
                  ((std::string*)addr)->~String_t();
               break;
            case kIsPointer|kIsClass:
               for( i=left; i<nCurr; ++i, addr += fValDiff )  {
                  StreamHelper* h = (StreamHelper*)addr;
                  void* p = h->ptr();
                  if ( p && force )  {
                     fVal->fType->Destructor(p);
                  }
                  h->set(0);
               }
               break;
            case kIsPointer|kBIT_ISSTRING:
               for( i=nCurr; i<left; ++i, addr += fValDiff )   {
                  StreamHelper* h = (StreamHelper*)addr;
                  if (force) delete (std::string*)h->ptr();
                  h->set(0);
               }
               break;
            case kIsPointer|kBIT_ISTSTRING|kIsClass:
               for( i=nCurr; i<left; ++i, addr += fValDiff )   {
                  StreamHelper* h = (StreamHelper*)addr;
                  if (force) delete (TString*)h->ptr();
                  h->set(0);
               }
               break;
         }
   }
   c->resize(left*fValDiff,0);
   fEnv->fStart = left>0 ? &(*c->begin()) : 0;
   return;
}

void TEmulatedCollectionProxy::Expand(UInt_t nCurr, UInt_t left)
{
   // Expand the container
   size_t i;
   PCont_t c   = PCont_t(fEnv->fObject);
   c->resize(left*fValDiff,0);
   void *oldstart = fEnv->fStart;
   fEnv->fStart = left>0 ? &(*c->begin()) : 0;

   char* addr = ((char*)fEnv->fStart) + fValDiff*nCurr;
   switch ( fSTL_type )  {
      case ROOT::kSTLmap:
      case ROOT::kSTLmultimap:
         switch(fKey->fCase)  {
            case kIsFundamental:  // Only handle primitives this way
            case kIsEnum:
               break;
            case kIsClass:
               if (oldstart && oldstart != fEnv->fStart) {
                  Long_t offset = 0;
                  for( i=0; i<=nCurr; ++i, offset += fValDiff ) {
                     // For now 'Move' only register the change of location
                     // so per se this is wrong since the object are copied via memcpy
                     // rather than a copy (or move) constructor.
                     fKey->fType->Move(((char*)oldstart)+offset,((char*)fEnv->fStart)+offset);
                  }
               }
               for( i=nCurr; i<left; ++i, addr += fValDiff )
                  fKey->fType->New(addr);
               break;
            case kBIT_ISSTRING:
               for( i=nCurr; i<left; ++i, addr += fValDiff )
                  ::new(addr) std::string();
               break;
            case kIsPointer|kIsClass:
            case kIsPointer|kBIT_ISSTRING:
            case kIsPointer|kBIT_ISTSTRING|kIsClass:
               for( i=nCurr; i<left; ++i, addr += fValDiff )
                  *(void**)addr = 0;
               break;
         }
         addr = ((char*)fEnv->fStart)+fValOffset+fValDiff*nCurr;
         // DO NOT break; just continue

         // General case for all values
      default:
         switch(fVal->fCase)  {
            case kIsFundamental:  // Only handle primitives this way
            case kIsEnum:
               break;
            case kIsClass:
               if (oldstart && oldstart != fEnv->fStart) {
                  Long_t offset = 0;
                  for( i=0; i<=nCurr; ++i, offset += fValDiff ) {
                     // For now 'Move' only register the change of location
                     // so per se this is wrong since the object are copied via memcpy
                     // rather than a copy (or move) constructor.
                     fVal->fType->Move(((char*)oldstart)+offset,((char*)fEnv->fStart)+offset);
                  }
               }
               for( i=nCurr; i<left; ++i, addr += fValDiff ) {
                  fVal->fType->New(addr);
               }
               break;
            case kBIT_ISSTRING:
               for( i=nCurr; i<left; ++i, addr += fValDiff )
                  ::new(addr) std::string();
               break;
            case kIsPointer|kIsClass:
            case kIsPointer|kBIT_ISSTRING:
            case kIsPointer|kBIT_ISTSTRING|kIsClass:
               for( i=nCurr; i<left; ++i, addr += fValDiff )
                  *(void**)addr = 0;
               break;
         }
         break;
   }
}

void TEmulatedCollectionProxy::Resize(UInt_t left, Bool_t force)
{
   // Resize the container

   if ( fEnv && fEnv->fObject )   {
      size_t nCurr = Size();
      PCont_t c = PCont_t(fEnv->fObject);
      fEnv->fStart = nCurr>0 ? &(*c->begin()) : 0;
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

void* TEmulatedCollectionProxy::At(UInt_t idx)
{
   // Return the address of the value at index 'idx'
   if ( fEnv && fEnv->fObject )   {
      PCont_t c = PCont_t(fEnv->fObject);
      size_t  s = c->size();
      if ( idx >= (s/fValDiff) )  {
         return 0;
      }
      return idx<(s/fValDiff) ? ((char*)&(*c->begin()))+idx*fValDiff : 0;
   }
   Fatal("TEmulatedCollectionProxy","At> Logic error - no proxy object set.");
   return 0;
}

void* TEmulatedCollectionProxy::Allocate(UInt_t n, Bool_t forceDelete)
{
   // Allocate the necessary space.

   Resize(n, forceDelete);
   return fEnv->fObject;
}

//______________________________________________________________________________
void TEmulatedCollectionProxy::Insert(const void * /* data */, void * /*container*/, size_t /*size*/)
{
   // Insert data into the container where data is a C-style array of the actual type contained in the collection
   // of the given size.   For associative container (map, etc.), the data type is the pair<key,value>.

   Fatal("Insert","Not yet implemented, require copy of objects.");
}

void TEmulatedCollectionProxy::Commit(void* /* env */ )
{
}

void TEmulatedCollectionProxy::ReadItems(int nElements, TBuffer &b)
{
   // Object input streamer
   Bool_t vsn3 = b.GetInfo() && b.GetInfo()->GetOldVersion()<=3;
   StreamHelper* itm = (StreamHelper*)At(0);
   switch (fVal->fCase) {
      case kIsFundamental:  //  Only handle primitives this way
      case kIsEnum:
         switch( int(fVal->fKind) )   {
            case kBool_t:    b.ReadFastArray(&itm->boolean   , nElements); break;
            case kChar_t:    b.ReadFastArray(&itm->s_char    , nElements); break;
            case kShort_t:   b.ReadFastArray(&itm->s_short   , nElements); break;
            case kInt_t:     b.ReadFastArray(&itm->s_int     , nElements); break;
            case kLong_t:    b.ReadFastArray(&itm->s_long    , nElements); break;
            case kLong64_t:  b.ReadFastArray(&itm->s_longlong, nElements); break;
            case kFloat_t:   b.ReadFastArray(&itm->flt       , nElements); break;
            case kFloat16_t: b.ReadFastArrayFloat16(&itm->flt, nElements); break;
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

      case kIsClass:
         DOLOOP( b.StreamObject(i,fVal->fType) );
      case kBIT_ISSTRING:
         DOLOOP( i->read_std_string(b) );
      case kIsPointer|kIsClass:
         DOLOOP( i->read_any_object(fVal,b) );
      case kIsPointer|kBIT_ISSTRING:
         DOLOOP( i->read_std_string_pointer(b) );
      case kIsPointer|kBIT_ISTSTRING|kIsClass:
         DOLOOP( i->read_tstring_pointer(vsn3,b) );
   }

#undef DOLOOP

}

void TEmulatedCollectionProxy::WriteItems(int nElements, TBuffer &b)
{
   // Object output streamer
   StreamHelper* itm = (StreamHelper*)At(0);
   switch (fVal->fCase) {
      case kIsFundamental:  // Only handle primitives this way
      case kIsEnum:
         itm = (StreamHelper*)At(0);
            switch( int(fVal->fKind) )   {
            case kBool_t:    b.WriteFastArray(&itm->boolean   , nElements); break;
            case kChar_t:    b.WriteFastArray(&itm->s_char    , nElements); break;
            case kShort_t:   b.WriteFastArray(&itm->s_short   , nElements); break;
            case kInt_t:     b.WriteFastArray(&itm->s_int     , nElements); break;
            case kLong_t:    b.WriteFastArray(&itm->s_long    , nElements); break;
            case kLong64_t:  b.WriteFastArray(&itm->s_longlong, nElements); break;
            case kFloat_t:   b.WriteFastArray(&itm->flt       , nElements); break;
            case kFloat16_t: b.WriteFastArrayFloat16(&itm->flt, nElements); break;
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
      case kIsClass:
         DOLOOP( b.StreamObject(i,fVal->fType) );
      case kBIT_ISSTRING:
         DOLOOP( TString(i->c_str()).Streamer(b) );
      case kIsPointer|kIsClass:
         DOLOOP( b.WriteObjectAny(i->ptr(),fVal->fType) );
      case kBIT_ISSTRING|kIsPointer:
         DOLOOP( i->write_std_string_pointer(b) );
      case kBIT_ISTSTRING|kIsClass|kIsPointer:
         DOLOOP( i->write_tstring_pointer(b) );
   }
#undef DOLOOP
}

void TEmulatedCollectionProxy::ReadBuffer(TBuffer &b, void *obj, const TClass *onfileClass)
{
   // Read portion of the streamer.

   SetOnFileClass((TClass*)onfileClass);
   ReadBuffer(b,obj);
}

void TEmulatedCollectionProxy::ReadBuffer(TBuffer &b, void *obj)
{
   // Read portion of the streamer.

   TPushPop env(this,obj);
   int nElements = 0;
   b >> nElements;
   if ( fEnv->fObject )  {
      Resize(nElements,true);
   }
   if ( nElements > 0 )  {
      ReadItems(nElements, b);
   }
}

void TEmulatedCollectionProxy::Streamer(TBuffer &b)
{
   // TClassStreamer IO overload
   if ( b.IsReading() ) {  //Read mode
      int nElements = 0;
      b >> nElements;
      if ( fEnv->fObject )  {
         Resize(nElements,true);
      }
      if ( nElements > 0 )  {
         ReadItems(nElements, b);
      }
   }
   else {     // Write case
      int nElements = fEnv->fObject ? Size() : 0;
      b << nElements;
      if ( nElements > 0 )  {
         WriteItems(nElements, b);
      }
   }
}

//
// Utility functions
//
static TStreamerElement* R__CreateEmulatedElement(const char *dmName, const char *dmFull, Int_t offset)
{
   // Create a TStreamerElement for the type 'dmFull' and whose data member name is 'dmName'.

   TString s1( TClassEdit::ShortType(dmFull,0) );
   TString dmType( TClassEdit::ShortType(dmFull,1) );
   Bool_t dmIsPtr = (s1 != dmType);
   const char *dmTitle = "Emulation";

   TDataType *dt = gROOT->GetType(dmType);
   if (dt && dt->GetType() > 0 ) {  // found a basic type
      Int_t dsize,dtype;
      dtype = dt->GetType();
      dsize = dt->Size();
      if (dmIsPtr && dtype != kCharStar) {
         Error("Pair Emulation Building","%s is not yet supported in pair emulation",
               dmFull);
         return 0;
      } else {
         TStreamerElement *el = new TStreamerBasicType(dmName,dmTitle,offset,dtype,dmFull);
         el->SetSize(dsize);
         return el;
      }
   } else {

      static const char *full_string_name = "basic_string<char,char_traits<char>,allocator<char> >";
      if (strcmp(dmType,"string") == 0 || strcmp(dmType,"std::string") == 0 || strcmp(dmType,full_string_name)==0 ) {
         return new TStreamerSTLstring(dmName,dmTitle,offset,dmFull,dmIsPtr);
      }
      if (TClassEdit::IsSTLCont(dmType)) {
         return new TStreamerSTL(dmName,dmTitle,offset,dmFull,dmFull,dmIsPtr);
      }
      TClass *clm = TClass::GetClass(dmType);
      if (!clm) {
         // either we have an Emulated enum or a really unknown class!
         // let's just claim its an enum :(
         Int_t dtype = kInt_t;
         return new TStreamerBasicType(dmName,dmTitle,offset,dtype,dmFull);
      }
      // a pointer to a class
      if ( dmIsPtr ) {
         if (clm->IsTObject()) {
            return new TStreamerObjectPointer(dmName,dmTitle,offset,dmFull);
         } else {
            return new TStreamerObjectAnyPointer(dmName,dmTitle,offset,dmFull);
         }
      }
      // a class
      if (clm->IsTObject()) {
         return new TStreamerObject(dmName,dmTitle,offset,dmFull);
      } else if(clm == TString::Class() && !dmIsPtr) {
         return new TStreamerString(dmName,dmTitle,offset);
      } else {
         return new TStreamerObjectAny(dmName,dmTitle,offset,dmFull);
      }
   }
}


static TStreamerInfo *R__GenerateTClassForPair(const std::string &fname, const std::string &sname)
{
   // Generate a TStreamerInfo for a std::pair<fname,sname>
   // This TStreamerInfo is then used as if it was read from a file to generate
   // and emulated TClass.

   TStreamerInfo *i = (TStreamerInfo*)TClass::GetClass("pair<const int,int>")->GetStreamerInfo()->Clone();
   std::string pname = "pair<"+fname+","+sname;
   pname += (pname[pname.length()-1]=='>') ? " >" : ">";
   i->SetName(pname.c_str());
   i->SetClass(0);
   i->GetElements()->Delete();
   TStreamerElement *fel = R__CreateEmulatedElement("first", fname.c_str(), 0);
   Int_t size = 0;
   if (fel) {
      i->GetElements()->Add( fel );

      size = fel->GetSize();
      Int_t sp = sizeof(void *);
      //align the non-basic data types (required on alpha and IRIX!!)
      if (size%sp != 0) size = size - size%sp + sp;
   } else {
      delete i;
      return 0;
   }
   TStreamerElement *second = R__CreateEmulatedElement("second", sname.c_str(), size);
   if (second) {
      i->GetElements()->Add( second );
   } else {
      delete i;
      return 0;
   }
   Int_t oldlevel = gErrorIgnoreLevel;
   // Hide the warning about the missing pair dictionary.
   gErrorIgnoreLevel = kError;
   i->BuildCheck();
   gErrorIgnoreLevel = oldlevel;
   i->BuildOld();
   return i;
}
