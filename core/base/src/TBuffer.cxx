// @(#)root/base:$Id$
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer                                                              //
//                                                                      //
// Buffer base class used for serializing objects.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBuffer.h"
#include "TClass.h"
#include "TProcessID.h"

const Int_t  kExtraSpace        = 8;   // extra space at end of buffer (used for free block count)

ClassImp(TBuffer)

//______________________________________________________________________________
static char *R__NoReAllocChar(char *, size_t, size_t)
{
   // The user has provided memory than we don't own, thus we can not extent it
   // either.
   return 0;
}

//______________________________________________________________________________
static inline ULong_t Void_Hash(const void *ptr)
{
   // Return hash value for this object.

   return TString::Hash(&ptr, sizeof(void*));
}


//______________________________________________________________________________
TBuffer::TBuffer(EMode mode)
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite. By default the I/O buffer has a size of
   // TBuffer::kInitialSize (1024) bytes.

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

//______________________________________________________________________________
TBuffer::TBuffer(EMode mode, Int_t bufsiz)
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite.

   if (bufsiz < kMinimalSize) bufsiz = kMinimalSize;
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fParent   = 0;

   SetBit(kIsOwner);

   fBuffer = new char[fBufSize+kExtraSpace];

   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( 0 );
}

//______________________________________________________________________________
TBuffer::TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt, ReAllocCharFun_t reallocfunc)
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite. By default the I/O buffer has a size of
   // TBuffer::kInitialSize (1024) bytes. An external buffer can be passed
   // to TBuffer via the buf argument. By default this buffer will be adopted
   // unless adopt is false.
   // If the new buffer is _not_ adopted and no memory allocation routine
   // is provided, a Fatal error will be issued if the Buffer attempts to
   // expand.
   
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

//______________________________________________________________________________
TBuffer::~TBuffer()
{
   // Delete an I/O buffer object.

   if (TestBit(kIsOwner)) {
      //printf("Deleting fBuffer=%lx\n", fBuffer);
      delete [] fBuffer;
   }
   fBuffer = 0;
   fParent = 0;
}


//______________________________________________________________________________
void TBuffer::AutoExpand(Int_t size_needed)
{
   // Automatically calculate a new size and expand the buffer to fit at least size_needed.
   // The goals is to minimize the number of memory allocation and the memory allocation
   // which avoiding too much memory wastage.
   // If the size_needed is larger than the current size, the policy
   // is to expand to double the current size or the size_needed which ever is largest.
   
   if (size_needed > fBufSize) {
      if (size_needed > 2*fBufSize) {
         Expand(size_needed);
      } else {
         Expand(2*fBufSize);
      }
   }
}

//______________________________________________________________________________
void TBuffer::SetBuffer(void *buf, UInt_t newsiz, Bool_t adopt, ReAllocCharFun_t reallocfunc)
{
   // Sets a new buffer in an existing TBuffer object. If newsiz=0 then the
   // new buffer is expected to have the same size as the previous buffer.
   // The current buffer position is reset to the start of the buffer.
   // If the TBuffer owned the previous buffer, it will be deleted prior
   // to accepting the new buffer. By default the new buffer will be
   // adopted unless adopt is false.
   // If the new buffer is _not_ adopted and no memory allocation routine
   // is provided, a Fatal error will be issued if the Buffer attempts to
   // expand.

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
   
   SetReAllocFunc( reallocfunc );

   if (buf && ( (fMode&kWrite)!=0 ) && fBufSize < 0) {
      Expand( kMinimalSize );
   }
}

//______________________________________________________________________________
void TBuffer::Expand(Int_t newsize)
{
   // Expand (or shrink) the I/O buffer to newsize bytes.

   Int_t l  = Length();
   if ( (fMode&kWrite)!=0 ) {
      fBuffer  = fReAllocFunc(fBuffer, newsize+kExtraSpace,
                              fBufSize+kExtraSpace);
   } else {
      fBuffer  = fReAllocFunc(fBuffer, newsize,
                              fBufSize);
   }
   if (fBuffer == 0) {
      if (fReAllocFunc == TStorage::ReAllocChar) {
         Fatal("Expand","Failed to expand the data buffer using TStorage::ReAllocChar.");
      } if (fReAllocFunc == R__NoReAllocChar) {
         Fatal("Expand","Failed to expand the data buffer because TBuffer does not own it and no custom memory reallocator was provided.");         
      } else {
         Fatal("Expand","Failed to expand the data buffer using custom memory reallocator 0x%lx.", (Long_t)fReAllocFunc);
      }
   }
   fBufSize = newsize;
   fBufCur  = fBuffer + l;
   fBufMax  = fBuffer + fBufSize;
}

//______________________________________________________________________________
TObject *TBuffer::GetParent() const
{
   // Return pointer to parent of this buffer.

   return fParent;
}

//______________________________________________________________________________
void TBuffer::SetParent(TObject *parent)
{
   // Set parent owning this buffer.

   fParent = parent;
}
//______________________________________________________________________________
ReAllocCharFun_t TBuffer::GetReAllocFunc() const
{
   // Return the reallocation method currently used.
   return fReAllocFunc;
}

//______________________________________________________________________________
void  TBuffer::SetReAllocFunc(ReAllocCharFun_t reallocfunc )
{
   // Set which memory reallocation method to use.  If reallocafunc is null,
   // reset it to the defaul value (TStorage::ReAlloc)
   
   if (reallocfunc) {
      fReAllocFunc = reallocfunc;
   } else {
      if (TestBit(kIsOwner)) {
         fReAllocFunc = TStorage::ReAllocChar;
      } else {
         fReAllocFunc = R__NoReAllocChar;
      }
   }
}

//______________________________________________________________________________
void TBuffer::SetReadMode()
{
   // Set buffer in read mode.

   fMode = kRead;
}

//______________________________________________________________________________
void TBuffer::SetWriteMode()
{
   // Set buffer in write mode.

   fMode = kWrite;
}

//______________________________________________________________________________
TClass *TBuffer::GetClass(const type_info &typeinfo)
{
   // Forward to TROOT::GetClass().

   return TClass::GetClass(typeinfo);
}

//______________________________________________________________________________
TClass *TBuffer::GetClass(const char *className)
{
   // Forward to TROOT::GetClass().

   return TClass::GetClass(className);
}

//______________________________________________________________________________
TProcessID *TBuffer::ReadProcessID(UShort_t pidf)
{
   // Return the current PRocessID.

   if (!pidf) return TProcessID::GetPID(); //may happen when cloning an object
   return 0;
}

//______________________________________________________________________________
UShort_t TBuffer::WriteProcessID(TProcessID *)
{
   // Always return 0 (current processID).

   return 0;
}

//______________________________________________________________________________
void TBuffer::PushDataCache(TVirtualArray *obj)
{
   // Push a new data cache area onto the list of area to be used for
   // temporarily store 'missing' data members.
   
   fCacheStack.push_back(obj);
}

//______________________________________________________________________________
TVirtualArray *TBuffer::PeekDataCache() const
{
   // Return the 'current' data cache area from the list of area to be used for
   // temporarily store 'missing' data members.
   
   if (fCacheStack.empty()) return 0;
   return fCacheStack.back();
}

//______________________________________________________________________________
TVirtualArray *TBuffer::PopDataCache()
{
   // Pop and Return the 'current' data cache area from the list of area to be used for
   // temporarily store 'missing' data members.
   
   TVirtualArray *val = PeekDataCache();
   fCacheStack.pop_back();
   return val;
}

