// @(#)root/cont:$Id$
// Author: Fons Rademakers   21/10/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TArray
\ingroup Containers
Abstract array base class. Used by TArrayC, TArrayS, TArrayI,
TArrayL, TArrayF and TArrayD.
Data member is public for historical reasons.
*/

#include "TArray.h"
#include "TError.h"
#include "TClass.h"
#include "TBuffer.h"


ClassImp(TArray);

////////////////////////////////////////////////////////////////////////////////
/// Generate an out-of-bounds error. Always returns false.

Bool_t TArray::OutOfBoundsError(const char *where, Int_t i) const
{
   ::Error(where, "index %d out of bounds (size: %d, this: 0x%zx)", i, fN, (size_t)this);
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Read TArray object from buffer. Simplified version of
/// TBuffer::ReadObject (does not keep track of multiple
/// references to same array).

TArray *TArray::ReadArray(TBuffer &b, const TClass *clReq)
{
   R__ASSERT(b.IsReading());

   // Make sure ReadArray is initialized
   b.InitMap();

   // Before reading object save start position
   UInt_t startpos = UInt_t(b.Length());

   UInt_t tag;
   TClass *clRef = b.ReadClass(clReq, &tag);

   TArray *a;
   if (!clRef) {

      a = nullptr;

   } else {

      a = (TArray *) clRef->New();
      if (!a) {
         ::Error("TArray::ReadArray", "could not create object of class %s",
                 clRef->GetName());
         // Exception
         return nullptr;
      }

      a->Streamer(b);

      b.CheckByteCount(startpos, tag, clRef);
   }

   return a;
}

////////////////////////////////////////////////////////////////////////////////
/// Write TArray object to buffer. Simplified version of
/// TBuffer::WriteObject (does not keep track of multiple
/// references to the same array).

void TArray::WriteArray(TBuffer &b, const TArray *a)
{
   R__ASSERT(b.IsWriting());

   // Make sure WriteMap is initialized
   b.InitMap();

   if (!a) {

      b << (UInt_t) 0;

   } else {

      // Reserve space for leading byte count
      UInt_t cntpos = UInt_t(b.Length());
      b.SetBufferOffset(Int_t(cntpos+sizeof(UInt_t)));

      TClass *cl = a->IsA();
      b.WriteClass(cl);

      ((TArray *)a)->Streamer(b);

      // Write byte count
      b.SetByteCount(cntpos);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Write TArray or derived object to buffer.

TBuffer &operator<<(TBuffer &buf, const TArray *obj)
{
   TArray::WriteArray(buf, obj);
   return buf;
}
