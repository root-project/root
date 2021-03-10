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

constexpr Int_t kExtraSpace    = 8;   // extra space at end of buffer (used for free block count)
constexpr Int_t kMaxBufferSize  = 0x7FFFFFFE;  // largest possible size.


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

static char *R__NoReAllocChar(char *, size_t, size_t)
{
   return nullptr;
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
   fParent       = nullptr;

   SetBit(kIsOwner);

   fBuffer = new char[fBufSize+kExtraSpace];

   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( nullptr );
}

////////////////////////////////////////////////////////////////////////////////
/// Create an I/O buffer object. Mode should be either TBuffer::kRead or
/// TBuffer::kWrite.

TBuffer::TBuffer(EMode mode, Int_t bufsiz)
{
   if (bufsiz < 0)
      Fatal("TBuffer","Request to create a buffer with a negative size, likely due to an integer overflow: 0x%x for a max of 0x%x.", bufsiz, kMaxBufferSize);
   if (bufsiz < kMinimalSize) bufsiz = kMinimalSize;
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fParent   = nullptr;

   SetBit(kIsOwner);

   fBuffer = new char[fBufSize+kExtraSpace];

   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;

   SetReAllocFunc( nullptr );
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

TBuffer::TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt, ReAllocCharFun_t reallocfunc)
{
   if (bufsiz < 0)
      Fatal("TBuffer","Request to create a buffer with a negative size, likely due to an integer overflow: 0x%x for a max of 0x%x.", bufsiz, kMaxBufferSize);
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fParent   = nullptr;

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
      fBuffer = new char[(Long64_t)fBufSize+kExtraSpace];
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
   fBuffer = nullptr;
   fParent = nullptr;
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
   if (size_needed < 0) {
      Fatal("AutoExpand","Request to expand to a negative size, likely due to an integer overflow: 0x%x for a max of 0x%x.", size_needed, kMaxBufferSize);
   }
   if (size_needed > fBufSize) {
      Long64_t doubling = 2LLU * fBufSize;
      if (doubling > kMaxBufferSize)
         doubling = kMaxBufferSize;
      if (size_needed > doubling) {
         Expand(size_needed);
      } else {
         Expand(doubling);
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

void TBuffer::SetBuffer(void *buf, UInt_t newsiz, Bool_t adopt, ReAllocCharFun_t reallocfunc)
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

   SetReAllocFunc( reallocfunc );

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
   const Int_t extraspace = (fMode&kWrite)!=0 ? kExtraSpace : 0;

   if ( ((Long64_t)newsize+extraspace) > kMaxBufferSize) {
      if (l < kMaxBufferSize) {
         newsize = kMaxBufferSize - extraspace;
      } else {
         Fatal("Expand","Requested size (%d) is too large (max is %d).", newsize, kMaxBufferSize);
      }
   }
   if ( (fMode&kWrite)!=0 ) {
      fBuffer  = fReAllocFunc(fBuffer, newsize+kExtraSpace,
                              copy ? fBufSize+kExtraSpace : 0);
   } else {
      fBuffer  = fReAllocFunc(fBuffer, newsize,
                              copy ? fBufSize : 0);
   }
   if (fBuffer == nullptr) {
      if (fReAllocFunc == TStorage::ReAllocChar) {
         Fatal("Expand","Failed to expand the data buffer using TStorage::ReAllocChar.");
      } else if (fReAllocFunc == R__NoReAllocChar) {
         Fatal("Expand","Failed to expand the data buffer because TBuffer does not own it and no custom memory reallocator was provided.");
      } else {
         Fatal("Expand","Failed to expand the data buffer using custom memory reallocator 0x%zx.", (size_t)fReAllocFunc);
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

ReAllocCharFun_t TBuffer::GetReAllocFunc() const
{
   return fReAllocFunc;
}

////////////////////////////////////////////////////////////////////////////////
/// Set which memory reallocation method to use.  If reallocafunc is null,
/// reset it to the default value (TStorage::ReAlloc)

void  TBuffer::SetReAllocFunc(ReAllocCharFun_t reallocfunc )
{
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
   return nullptr;
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
   if (fCacheStack.empty()) return nullptr;
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

////////////////////////////////////////////////////////////////////////////////
/// Byte-swap N primitive-elements in the buffer.
/// Bulk API relies on this function.

Bool_t TBuffer::ByteSwapBuffer(Long64_t n, EDataType type)
{
   char *input_buf = GetCurrent();
   if ((type == EDataType::kShort_t) || (type == EDataType::kUShort_t)) {
#ifdef R__BYTESWAP
      Short_t *buf __attribute__((aligned(8))) = reinterpret_cast<Short_t*>(input_buf);
      for (int idx=0; idx<n; idx++) {
         Short_t tmp = *reinterpret_cast<Short_t*>(buf + idx); // Makes a copy of the values; frombuf can't handle aliasing.
         char *tmp_ptr = reinterpret_cast<char *>(&tmp);
         frombuf(tmp_ptr, buf + idx);
      }
#endif
   } else if ((type == EDataType::kFloat_t) || (type == EDataType::kInt_t) || (type == EDataType::kUInt_t)) {
#ifdef R__BYTESWAP
      Float_t *buf __attribute__((aligned(8))) = reinterpret_cast<Float_t*>(input_buf);
      for (int idx=0; idx<n; idx++) {
         Float_t tmp = *reinterpret_cast<Float_t*>(buf + idx); // Makes a copy of the values; frombuf can't handle aliasing.
         char *tmp_ptr = reinterpret_cast<char *>(&tmp);
         frombuf(tmp_ptr, buf + idx);
      }
#endif
   } else if ((type == EDataType::kDouble_t) || (type == EDataType::kLong64_t) || (type == EDataType::kULong64_t)) {
#ifdef R__BYTESWAP
      Double_t *buf __attribute__((aligned(8))) = reinterpret_cast<Double_t*>(input_buf);
      for (int idx=0; idx<n; idx++) {
         Double_t tmp = *reinterpret_cast<Double_t*>(buf + idx); // Makes a copy of the values; frombuf can't handle aliasing.
         char *tmp_ptr = reinterpret_cast<char*>(&tmp);
         frombuf(tmp_ptr, buf + idx);
      }
#endif
   } else {
      return false;
   }

   return true;
}
