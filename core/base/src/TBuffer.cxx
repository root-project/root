// @(#)root/base:$Id: 6da0b5b613bbcfaa3a5cd4074e7b2be2448dfb31 $
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TBuffer
\ingroup Base

Buffer base class used for serializing objects.
*/

#include "TBuffer.h"
#include "TClass.h"
#include "TProcessID.h"

const Int_t  kExtraSpace        = 8;   // extra space at end of buffer (used for free block count)

ClassImp(TBuffer);

/// Default streamer implementation used by ClassDefInline to avoid
/// requirement to include TBuffer.h
void ROOT::Internal::DefaultStreamer(TBuffer &R__b, const TClass *cl, void *objpointer)
{
   if (R__b.IsReading())
      R__b.ReadClassBuffer(cl, objpointer);
   else
      R__b.WriteClassBuffer(cl, objpointer);
}

////////////////////////////////////////////////////////////////////////////////
/// The user has provided memory than we don't own, thus we can not extent it
/// either.

static char *R__NoReAllocState(void *, char *, size_t, size_t)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create an I/O buffer object. Mode should be either TBuffer::kRead or
/// TBuffer::kWrite. By default the I/O buffer has a size of
/// TBuffer::kInitialSize (1024) bytes.

TBuffer::TBuffer(EMode mode)
{
   fBufSize      = kInitialSize;
   fMode         = mode;
   fVersion      = 0;
   fParent       = 0;

   SetBit(kIsOwner);

   fBuffer = new char[fBufSize+kExtraSpace];

   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( 0 );
}

////////////////////////////////////////////////////////////////////////////////
/// Create an I/O buffer object. Mode should be either TBuffer::kRead or
/// TBuffer::kWrite.

TBuffer::TBuffer(EMode mode, Int_t bufsiz)
{
   if (bufsiz < kMinimalSize) bufsiz = kMinimalSize;
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fParent   = 0;

   SetBit(kIsOwner);

   fBuffer = new char[fBufSize+kExtraSpace];

   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( nullptr, nullptr );
}

////////////////////////////////////////////////////////////////////////////////
/// Create an I/O buffer object. Mode should be either TBuffer::kRead or
/// TBuffer::kWrite. By default the I/O buffer has a size of
/// TBuffer::kInitialSize (1024) bytes. An external buffer can be passed
/// to TBuffer via the buf argument. By default this buffer will be adopted
/// unless adopt is false.
///
/// If the new buffer is _not_ adopted and no memory allocation routine
/// is provided, a Fatal error will be issued if the Buffer attempts to
/// expand.

TBuffer::TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt, ReAllocStateFun_t reallocfunc, void *reallocData)
  : fReAllocData(reallocData)
{
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fParent   = 0;

   SetBit(kIsOwner);

   if (buf) {
      fBuffer = (char *)buf;
      if ( (fMode&kWrite)!=0 ) {
         fBufSize -= kExtraSpace;
      }
      if (!adopt) ResetBit(kIsOwner);
   } else {
      if (fBufSize < kMinimalSize) {
         fBufSize = kMinimalSize;
      }
      fBuffer = new char[fBufSize+kExtraSpace];
   }
   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( reallocfunc );

   if (buf && ( (fMode&kWrite)!=0 ) && fBufSize < 0) {
      Expand( kMinimalSize );
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete an I/O buffer object.

TBuffer::~TBuffer()
{
   if (TestBit(kIsOwner)) {
      //printf("Deleting fBuffer=%lx\n", fBuffer);
      delete [] fBuffer;
   }
   fBuffer = 0;
   fParent = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Automatically calculate a new size and expand the buffer to fit at least size_needed.
/// The goals is to minimize the number of memory allocation and the memory allocation
/// which avoiding too much memory wastage.
///
/// If the size_needed is larger than the current size, the policy
/// is to expand to double the current size or the size_needed which ever is largest.

void TBuffer::AutoExpand(Int_t size_needed)
{
   if (size_needed > fBufSize) {
      if (size_needed > 2*fBufSize) {
         Expand(size_needed);
      } else {
         Expand(2*fBufSize);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Sets a new buffer in an existing TBuffer object. If newsiz=0 then the
/// new buffer is expected to have the same size as the previous buffer.
/// The current buffer position is reset to the start of the buffer.
/// If the TBuffer owned the previous buffer, it will be deleted prior
/// to accepting the new buffer. By default the new buffer will be
/// adopted unless adopt is false.
///
/// If the new buffer is _not_ adopted and no memory allocation routine
/// is provided, a Fatal error will be issued if the Buffer attempts to
/// expand.

void TBuffer::SetBuffer(void *buf, UInt_t newsiz, Bool_t adopt, ReAllocStateFun_t reallocfunc, void *reallocData)
{
   if (fBuffer && TestBit(kIsOwner))
      delete [] fBuffer;

   if (adopt)
      SetBit(kIsOwner);
   else
      ResetBit(kIsOwner);

   fBuffer = (char *)buf;
   fBufCur = fBuffer;
   if (newsiz > 0) {
      if ( (fMode&kWrite)!=0 ) {
         fBufSize = newsiz - kExtraSpace;
      } else {
         fBufSize = newsiz;
      }
   }
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( reallocfunc, reallocData );

   if (buf && ( (fMode&kWrite)!=0 ) && fBufSize < 0) {
      Expand( kMinimalSize );
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Expand (or shrink) the I/O buffer to newsize bytes.
/// If copy is true (the default), the existing content of the
/// buffer is preserved, otherwise the buffer is returned zero-ed out.
///
/// In order to avoid losing data, if the current length is greater than
/// the requested size, we only shrink down to the current length.

void TBuffer::Expand(Int_t newsize, Bool_t copy)
{
   Int_t l  = Length();
   if ( (l > newsize) && copy ) {
      newsize = l;
   }
   if ( (fMode&kWrite)!=0 ) {
      fBuffer  = fReAllocFunc(fReAllocData, fBuffer, newsize+kExtraSpace,
                              copy ? fBufSize+kExtraSpace : 0);
   } else {
      fBuffer  = fReAllocFunc(fReAllocData, fBuffer, newsize,
                              copy ? fBufSize : 0);
   }
   if (fBuffer == 0) {
      if (fReAllocFunc == TStorage::ReAllocState) {
         Fatal("Expand","Failed to expand the data buffer using TStorage::ReAllocState.");
      } else if (fReAllocFunc == R__NoReAllocState) {
         Fatal("Expand","Failed to expand the data buffer because TBuffer does not own it and no custom memory reallocator was provided.");
      } else {
         Fatal("Expand","Failed to expand the data buffer using custom memory reallocator 0x%lx.", (Long_t)fReAllocFunc);
      }
   }
   fBufSize = newsize;
   fBufCur  = fBuffer + l;
   fBufMax  = fBuffer + fBufSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Return pointer to parent of this buffer.

TObject *TBuffer::GetParent() const
{
   return fParent;
}

////////////////////////////////////////////////////////////////////////////////
/// Set parent owning this buffer.

void TBuffer::SetParent(TObject *parent)
{
   fParent = parent;
}
////////////////////////////////////////////////////////////////////////////////
/// Return the reallocation method currently used.

ReAllocStateFun_t TBuffer::GetReAllocFunc() const
{
   return fReAllocFunc;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the reallocation state currently used.
void *TBuffer::GetReAllocData() const
{
   return fReAllocData;
}

////////////////////////////////////////////////////////////////////////////////
/// Set which memory reallocation method to use.  If reallocafunc is null,
/// reset it to the default value (TStorage::ReAlloc).
//
//  Also allows the user to provide a void* as state for the reallocation.

void  TBuffer::SetReAllocFunc(ReAllocStateFun_t reallocfunc, void *reallocData)
{
   if (reallocfunc) {
      fReAllocFunc = reallocfunc;
   } else {
      if (TestBit(kIsOwner)) {
         fReAllocFunc = TStorage::ReAllocState;
      } else {
         fReAllocFunc = R__NoReAllocState;
      }
   }
   fReAllocData = reallocData;
}

char *R__ReAllocShared(void *obj_void, char *current, size_t new_size, size_t old_size)
{
   TBuffer *owner_buffer = static_cast<TBuffer*>(obj_void);
   if (current != owner_buffer->Buffer()) {
       owner_buffer->Fatal("ReAllocShared", "Re-allocating a non-shared buffer as shared.");
   }
   owner_buffer->Expand(new_size, old_size);  // If old_size is non-zero, then we are copying over the old memory.
   return owner_buffer->Buffer();
}

////////////////////////////////////////////////////////////////////////////////
/// Share the underlying memory allocation with another buffer.
//
// This causes the passed TBuffer object to share our memory buffer.  This is
// useful if two objects want to have their own view of the TBuffer state but
// see identical data.
//
// Internally, not only do both buffers get the same memory location but a
// resize done by the "slave" buffer updates the "owner" buffer (the opposite
// is not true!).
//
// This is most useful if the "slave" does all the writings and the "owner"
// only does reads.
//
/*
Bool_t TBuffer::SetSlaveBuffer(TBuffer &other)
{
   this->fBuffer = other.SetReAllocFunc(R__ReAllocShared, this);
   return true;
}
*/

////////////////////////////////////////////////////////////////////////////////
/// Set buffer in read mode.

void TBuffer::SetReadMode()
{
   if ( (fMode&kWrite)!=0 ) {
      // We had reserved space for the free block count,
      // release it,
      fBufSize += kExtraSpace;
   }
   fMode = kRead;
}

////////////////////////////////////////////////////////////////////////////////
/// Set buffer in write mode.

void TBuffer::SetWriteMode()
{
   if ( (fMode&kWrite)==0 ) {
      // We had not yet reserved space for the free block count,
      // reserve it now.
      fBufSize -= kExtraSpace;
   }
   fMode = kWrite;
}

////////////////////////////////////////////////////////////////////////////////
/// Forward to TROOT::GetClass().

TClass *TBuffer::GetClass(const std::type_info &typeinfo)
{
   return TClass::GetClass(typeinfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Forward to TROOT::GetClass().

TClass *TBuffer::GetClass(const char *className)
{
   return TClass::GetClass(className);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the current Process-ID.

TProcessID *TBuffer::ReadProcessID(UShort_t pidf)
{
   if (!pidf) return TProcessID::GetPID(); //may happen when cloning an object
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Always return 0 (current processID).

UShort_t TBuffer::WriteProcessID(TProcessID *)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Push a new data cache area onto the list of area to be used for
/// temporarily store 'missing' data members.

void TBuffer::PushDataCache(TVirtualArray *obj)
{
   fCacheStack.push_back(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Return the 'current' data cache area from the list of area to be used for
/// temporarily store 'missing' data members.

TVirtualArray *TBuffer::PeekDataCache() const
{
   if (fCacheStack.empty()) return 0;
   return fCacheStack.back();
}

////////////////////////////////////////////////////////////////////////////////
/// Pop and Return the 'current' data cache area from the list of area to be used for
/// temporarily store 'missing' data members.

TVirtualArray *TBuffer::PopDataCache()
{
   TVirtualArray *val = PeekDataCache();
   fCacheStack.pop_back();
   return val;
}

