// @(#)root/base:$Name:  $:$Id: TBuffer.cxx,v 1.100 2007/01/19 16:47:59 brun Exp $
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

#include "TROOT.h"
#include "TBuffer.h"

const Int_t  kExtraSpace        = 8;   // extra space at end of buffer (used for free block count)

ClassImp(TBuffer)

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
}

//______________________________________________________________________________
TBuffer::TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt) 
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite. By default the I/O buffer has a size of
   // TBuffer::kInitialSize (1024) bytes. An external buffer can be passed
   // to TBuffer via the buf argument. By default this buffer will be adopted
   // unless adopt is false.

   if (!buf && bufsiz < kMinimalSize) bufsiz = kMinimalSize;
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fParent   = 0;

   SetBit(kIsOwner);

   if (buf) {
      fBuffer = (char *)buf;
      if (!adopt) ResetBit(kIsOwner);
   } else
      fBuffer = new char[fBufSize+kExtraSpace];
   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;
}

//______________________________________________________________________________
TBuffer::TBuffer(const TBuffer &b) : TObject(b)
{
   // TBuffer copy ctor.
}

//______________________________________________________________________________
void TBuffer::operator=(const TBuffer&)
{
   // TBuffer assignment operator.
   //return *this;
}

//______________________________________________________________________________
TBuffer::~TBuffer()
{
   // Delete an I/O buffer object.

   if (TestBit(kIsOwner)) {
      //printf("Deleting fBuffer=%x\n",fBuffer);
      delete [] fBuffer;
   }
   fBuffer = 0;
   fParent = 0;

   //delete fMap;
   //delete fClassMap;
}

//______________________________________________________________________________
void TBuffer::SetBuffer(void *buf, UInt_t newsiz, Bool_t adopt)
{
   // Sets a new buffer in an existing TBuffer object. If newsiz=0 then the
   // new buffer is expected to have the same size as the previous buffer.
   // The current buffer position is reset to the start of the buffer.
   // If the TBuffer owned the previous buffer, it will be deleted prior
   // to accepting the new buffer. By default the new buffer will be
   // adopted unless adopt is false.

   if (fBuffer && TestBit(kIsOwner))
      delete [] fBuffer;

   if (adopt)
      SetBit(kIsOwner);
   else
      ResetBit(kIsOwner);

   fBuffer = (char *)buf;
   fBufCur = fBuffer;
   if (newsiz > 0) fBufSize = newsiz;
   fBufMax = fBuffer + fBufSize;
}

//______________________________________________________________________________
void TBuffer::Expand(Int_t newsize)
{
   // Expand the I/O buffer to newsize bytes.

   Int_t l  = Length();
   fBuffer  = TStorage::ReAllocChar(fBuffer, newsize+kExtraSpace,
                                    fBufSize+kExtraSpace);
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
   // Forward to TROOT::GetClass

   return gROOT->GetClass(typeinfo);
}

//______________________________________________________________________________
TClass *TBuffer::GetClass(const char *className)
{
   // Forward to TROOT::GetClass

   return gROOT->GetClass(className);
}

