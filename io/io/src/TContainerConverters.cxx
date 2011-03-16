// @(#)root/io:$Id$
// Author: Philippe Canal  11/11/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Helper classes to convert collection from ROOT collection to STL     //
// collections                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TContainerConverters.h"
#include "TClonesArray.h"
#include "TStreamerInfo.h"
#include "TVirtualCollectionProxy.h"
#include "TError.h"
#include "TGenCollectionStreamer.h"
#include "TClassStreamer.h"
#include <stdlib.h>

namespace {
   const Int_t  kMapOffset = 2;
}

//________________________________________________________________________
TConvertClonesArrayToProxy::TConvertClonesArrayToProxy(
   TVirtualCollectionProxy *proxy,
   Bool_t isPointer, Bool_t isPrealloc) :
      fIsPointer(isPointer),
      fIsPrealloc(isPrealloc),
      fProxy(proxy?proxy->Generate():0)
{
   // Constructor.

   if (isPointer) fOffset = sizeof(TClonesArray*);
   else fOffset = sizeof(TClonesArray*);
}

//________________________________________________________________________
TConvertClonesArrayToProxy::~TConvertClonesArrayToProxy()
{
   // Destructor.

   delete fProxy;
}

//________________________________________________________________________
void TConvertClonesArrayToProxy::operator()(TBuffer &b, void *pmember, Int_t size)
{
   // Read a TClonesArray from the TBuffer b and load it into a (stl) collection

   TStreamerInfo *subinfo = (TStreamerInfo*)fProxy->GetValueClass()->GetStreamerInfo();
   R__ASSERT(subinfo);

   Int_t   nobjects, dummy;
   char    nch;
   TString s;
   char classv[256];
   void *env;
   UInt_t start, bytecount;

   R__ASSERT(b.IsReading());

   Bool_t needAlloc = fIsPointer && !fIsPrealloc;

   if (needAlloc) {
      char *addr = (char*)pmember;
      for(Int_t k=0; k<size; ++k, addr += fOffset ) {
         if (*(void**)addr && TStreamerInfo::CanDelete()) {
            fProxy->GetValueClass()->Destructor(*(void**)addr,kFALSE); // call delete and desctructor
         }
         //*(void**)addr = fProxy->New();
         //TClonesArray *clones = (TClonesArray*)ReadObjectAny(TClonesArray::Class());
      }
   }

   char *addr = (char*)pmember;
   if (size==0) size=1;
   for(Int_t k=0; k<size; ++k, addr += fOffset ) {

      if (needAlloc) {
         // Read the class name.

         // make sure fMap is initialized
         b.InitMap();

         // before reading object save start position
         UInt_t startpos = b.Length();

         // attempt to load next object as TClass clCast
         UInt_t tag;       // either tag or byte count
         TClass *clRef = b.ReadClass(TClonesArray::Class(), &tag);

         if (clRef==0) {
            // Got a reference to an already read object.
            if (b.GetBufferVersion() > 0) {
               tag += b.GetBufferDisplacement();
            } else {
               if (tag > (UInt_t)b.GetMapCount()) {
                  Error("TConvertClonesArrayToProxy", "object tag too large, I/O buffer corrupted");
                  return;
               }
            }
            void *objptr;
            b.GetMappedObject( tag, objptr, clRef);
            if ( objptr == (void*)-1 ) {
               Error("TConvertClonesArrayToProxy",
                  "Object can not be found in the buffer's map (at %d)",tag);
               continue;
            }
            if ( objptr == 0 ) {
               if (b.GetBufferVersion()==0) continue;

               // No object found at this location in map. It might have been skipped
               // as part of a skipped object. Try to explicitely read the object.
               b.MapObject(*(void**)addr, fProxy->GetCollectionClass(), 0);
               Int_t currentpos = b.Length();
               b.SetBufferOffset( tag - kMapOffset );

               (*this)(b,&objptr,1);
               b.SetBufferOffset( currentpos);

               if (objptr==0) continue;

               clRef = fProxy->GetCollectionClass();

            }
            R__ASSERT(clRef);
            if (clRef==TClonesArray::Class()) {
               Error("TConvertClonesArrayToProxy",
                  "Object refered to has not been converted from TClonesArray to %s",
                  fProxy->GetCollectionClass()->GetName());
               continue;
            } else if (clRef!=fProxy->GetCollectionClass()) {
               Error("TConvertClonesArrayToProxy",
                  "Object refered to is of type %s instead of %s",
                  clRef->GetName(),fProxy->GetCollectionClass()->GetName());
               continue;
            }
            *(void**)addr = objptr;
            continue;

         } else if (clRef != TClonesArray::Class()) {
            Warning("TConvertClonesArrayToProxy",
                    "Only the TClonesArray part of %s will be read into %s!\n",
                    (clRef!=((TClass*)-1)&&clRef) ? clRef->GetName() : "N/A",
                    fProxy->GetCollectionClass()->GetName());
         } else {
            *(void**)addr = fProxy->New();
            if (b.GetBufferVersion()>0) {
               b.MapObject(*(void**)addr, fProxy->GetCollectionClass(), startpos+kMapOffset);
            } else {
               b.MapObject(*(void**)addr, fProxy->GetCollectionClass(), b.GetMapCount() );
            }
         }
      }
      void *obj;
      if (fIsPointer) obj = *(void**)addr;
      else obj = addr;

      TObject objdummy;
      Version_t v = b.ReadVersion(&start, &bytecount);

      //if (v == 3) {
      //   const int_t koldbypassstreamer = bit(14);
      //   if (testbit(koldbypassstreamer)) bypassstreamer();
      //}
      if (v > 2) objdummy.Streamer(b);
      TString fName;
      if (v > 1) fName.Streamer(b);
      s.Streamer(b);
      strncpy(classv,s.Data(),255);
      Int_t clv = 0;
      char *semicolon = strchr(classv,';');
      if (semicolon) {
         *semicolon = 0;
         clv = atoi(semicolon+1);
      }
      TClass *cl = TClass::GetClass(classv);
      if (!cl) {
         printf("TClonesArray::Streamer expecting class %s\n", classv);
         b.CheckByteCount(start, bytecount, TClonesArray::Class());
         return;
      }

      b >> nobjects;
      if (nobjects < 0) nobjects = -nobjects;  // still there for backward compatibility
      b >> dummy; // fLowerBound is ignored
      if (cl != subinfo->GetClass()) {
         Error("TClonesArray::Conversion to vector","Bad class");
      }
      TVirtualCollectionProxy::TPushPop helper( fProxy, obj );
      env = fProxy->Allocate(nobjects,true);

      if (objdummy.TestBit(TClonesArray::kBypassStreamer)) {

         subinfo->ReadBufferSTL(b,fProxy,nobjects,-1,0);

      } else {
         for (Int_t i = 0; i < nobjects; i++) {
            b >> nch;
            if (nch) {
               void* elem = fProxy->At(i);
               b.StreamObject(elem,subinfo->GetClass());
            }
         }
      }
      fProxy->Commit(env);
      b.CheckByteCount(start, bytecount,TClonesArray::Class());
   }
}

//________________________________________________________________________
TConvertMapToProxy::TConvertMapToProxy(TClassStreamer *streamer,
                                       Bool_t isPointer, Bool_t isPrealloc) :
   fIsPointer(isPointer),
   fIsPrealloc(isPrealloc),
   fSizeOf(0),
   fProxy(0),
   fCollectionStreamer(0)
{
   // Constructor.

   TCollectionClassStreamer *middleman = dynamic_cast<TCollectionClassStreamer*>(streamer);
   if (middleman) {
      fProxy = middleman->GetXYZ();
      fCollectionStreamer = dynamic_cast<TGenCollectionStreamer*>(middleman->GetXYZ());

      if (isPointer) fSizeOf = sizeof(void*);
      else fSizeOf = fProxy->GetCollectionClass()->Size();

      if (fProxy->GetValueClass()->GetStreamerInfo() == 0
          || fProxy->GetValueClass()->GetStreamerInfo()->GetElements()->At(1) == 0 ) {
         // We do not have enough information on the pair (or its not a pair).
         fCollectionStreamer = 0;
      }
   }
}



//________________________________________________________________________
void TConvertMapToProxy::operator()(TBuffer &b, void *pmember, Int_t size)
{
   // Read a std::map or std::multimap from the TBuffer b and load it into a (stl) collection

   R__ASSERT(b.IsReading());
   R__ASSERT(fCollectionStreamer);

   Bool_t needAlloc = fIsPointer && !fIsPrealloc;

   R__ASSERT(!needAlloc); // not yet implemented

   if (needAlloc) {
      char *addr = (char*)pmember;
      for(Int_t k=0; k<size; ++k, addr += fSizeOf ) {
         if (*(void**)addr && TStreamerInfo::CanDelete()) {
            fProxy->GetValueClass()->Destructor(*(void**)addr,kFALSE); // call delete and desctructor
         }
         //*(void**)addr = fProxy->New();
         //TClonesArray *clones = (TClonesArray*)ReadObjectAny(TClonesArray::Class());
      }
   }

   
   char *addr = (char*)pmember;
   if (size==0) size=1;
   for(Int_t k=0; k<size; ++k, addr += fSizeOf) {

      if (needAlloc) {
         
         // Read the class name.

      }

      void *obj;
      if (fIsPointer) obj = *(void**)addr;
      else obj = addr;

      TVirtualCollectionProxy::TPushPop env(fProxy, obj);
      fCollectionStreamer->StreamerAsMap(b);

   }
}
