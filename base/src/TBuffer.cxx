// @(#)root/base:$Name:  $:$Id: TBuffer.cxx,v 1.16 2001/06/01 11:01:53 brun Exp $
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

#include <string.h>

#include "TFile.h"
#include "TBuffer.h"
#include "TExMap.h"
#include "TObjPtr.h"
#include "TClass.h"
#include "TStorage.h"
#include "TMath.h"
#include "TError.h"

#if defined(__linux) && defined(__i386__)
//#define USE_BSWAPCPY
#endif

#ifdef USE_BSWAPCPY
#include "Bswapcpy.h"
#endif


const UInt_t kNullTag           = 0;
const UInt_t kNewClassTag       = 0xFFFFFFFF;
const UInt_t kClassMask         = 0x80000000;  // OR the class index with this
const UInt_t kByteCountMask     = 0x40000000;  // OR the byte count with this
const UInt_t kMaxMapCount       = 0x3FFFFFFE;  // last valid fMapCount and byte count
const Version_t kByteCountVMask = 0x4000;      // OR the version byte count with this
const Version_t kMaxVersion     = 0x3FFF;      // highest possible version number
const Int_t  kExtraSpace        = 8;   // extra space at end of buffer (used for free block count)
const Int_t  kMapOffset         = 2;   // first 2 map entries are taken by null obj and self obj

Int_t TBuffer::fgMapSize   = kMapSize;


ClassImp(TBuffer)
//______________________________________________________________________________
TBuffer::TBuffer(EMode mode, Int_t bufsiz, void *buf)
{
   // Create an I/O buffer object. Mode should be either TBuffer::kRead or
   // TBuffer::kWrite. By default the I/O buffer has a size of
   // TBuffer::kInitialSize (1024) bytes.

   // Before using the buffer make sure some assumptions are true
   Assert(sizeof(Short_t) == 2);
   Assert(sizeof(Int_t) == 4);
// commented lines below in view of support for longlong on 32 bits machines
//#ifdef R__B64
//   Assert(sizeof(Long_t) == 8);
//#else
//   Assert(sizeof(Long_t) == 4);
//#endif
   Assert(sizeof(Float_t) == 4);
   Assert(sizeof(Double_t) == 8);

   if (!buf && bufsiz < kMinimalSize) bufsiz = kMinimalSize;
   fBufSize  = bufsiz;
   fMode     = mode;
   fVersion  = 0;
   fMapCount = 0;
   fMapSize  = fgMapSize;
   fReadMap  = 0;
   fDisplacement = 0;

   if (buf)
      fBuffer = (char *)buf;
   else
      fBuffer = new char[fBufSize+kExtraSpace];
   fBufCur = fBuffer;
   fBufMax = fBuffer + fBufSize;
}

//______________________________________________________________________________
TBuffer::~TBuffer()
{
   // Delete an I/O buffer object.

   delete [] fBuffer;
   fBuffer = 0;

   if (IsReading())
      delete fReadMap;
   else
      delete fWriteMap;

   fReadMap = 0;
}

//______________________________________________________________________________
void frombufOld(char *&buf, Long_t *x)
{
// files written with versions older than 3.00/06 had a non-portable
// implementation of Long_t/ULong_t. These types should not have been used at all.
// However, because some users had already written many files with these types
// we provide this dirty patch for "backward compatibility"
   
#ifdef R__BYTESWAP
#ifdef R__B64
   char *sw = (char *)x;
   sw[0] = buf[7];
   sw[1] = buf[6];
   sw[2] = buf[5];
   sw[3] = buf[4];
   sw[4] = buf[3];
   sw[5] = buf[2];
   sw[6] = buf[1];
   sw[7] = buf[0];
#else
   char *sw = (char *)x;
   sw[0] = buf[3];
   sw[1] = buf[2];
   sw[2] = buf[1];
   sw[3] = buf[0];
#endif
#else
   memcpy(x, buf, sizeof(Long_t));
#endif
   buf += sizeof(Long_t);
}

//______________________________________________________________________________
TBuffer &TBuffer::operator>>(Long_t &l)
{
   if (gFile && gFile->GetVersion() < 30006) {
      frombufOld(fBufCur, &l);
   } else {
      frombuf(fBufCur, &l);
   }   
   return *this;
}

//______________________________________________________________________________
void TBuffer::SetBuffer(void *buf, UInt_t newsiz)
{
   // Set buffer address

   fBuffer = (char *)buf;
   fBufCur = fBuffer;
   if (newsiz > 0) fBufSize = newsiz;
   fBufMax = fBuffer + fBufSize;
}

//______________________________________________________________________________
void TBuffer::CheckCount(UInt_t offset)
{
   // Check if offset is not too large (< kMaxMapCount) when writing.

   if (IsWriting()) {
      if (offset >= kMaxMapCount) {
         Error("CheckCount", "buffer offset too large (larger than %d)", kMaxMapCount);
         // exception
      }
   }
}

//______________________________________________________________________________
UInt_t TBuffer::CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass)
{
   // Check for object in the read map. If the object is 0 it still has to be
   // read. Try to read it from the buffer starting at location offset. If the
   // object is -1 then it really does not exist and we return 0. If the object
   // exists just return the offset.

   // in position 0 we always have the reference to the null object
   if (!offset) return offset;

   Long_t cli;

   if (readClass) {
      if ((cli = fReadMap->GetValue(offset)) == 0) {
         // No class found at this location in map. It might have been skipped
         // as part of a skipped object. Try to explicitely read the class.

         // save fBufCur and set to place specified by offset (-kMapOffset-sizeof(bytecount))
         char *bufsav = fBufCur;
         fBufCur = (char *)(fBuffer + offset-kMapOffset-sizeof(UInt_t));

         TClass *c = ReadClass(cl);
         if (c == (TClass*) -1) {
            // mark class as really not available
            fReadMap->Remove(offset);
            fReadMap->Add(offset, -1);
            offset = 0;
            Warning("CheckObject", "reference to unavailable class %s,"
                    " pointers of this type will be 0", cl ? cl->GetName() : "TObject");
         }

         fBufCur = bufsav;

      } else if (cli == -1) {

         // class really does not exist
         return 0;
      }

   } else {

      if ((cli = fReadMap->GetValue(offset)) == 0) {
         // No object found at this location in map. It might have been skipped
         // as part of a skipped object. Try to explicitely read the object.

         // save fBufCur and set to place specified by offset (-kMapOffset)
         char *bufsav = fBufCur;
         fBufCur = (char *)(fBuffer + offset-kMapOffset);

         TObject *obj = ReadObject(cl);
         if (!obj) {
            // mark object as really not available
            fReadMap->Remove(offset);
            fReadMap->Add(offset, -1);
            offset = 0;
            Warning("CheckObject", "reference to object of unavailable class %s,"
                    " pointer will be 0", cl ? cl->GetName() : "TObject");
         }

         fBufCur = bufsav;

      } else if (cli == -1) {

         // object really does not exist
         return 0;
      }

   }

   return offset;
}

//______________________________________________________________________________
void TBuffer::Expand(Int_t newsize)
{
   // Expand the I/O buffer to newsize bytes.

   Int_t l = Length();

   fBuffer = (char *) TStorage::ReAlloc(fBuffer,
                                        (newsize+kExtraSpace) * sizeof(char),
                                        (fBufSize+kExtraSpace) * sizeof(char));
   fBufSize = newsize;
   fBufCur  = fBuffer + l;
   fBufMax  = fBuffer + fBufSize;
}

//______________________________________________________________________________
void TBuffer::MapObject(const TObject *obj, UInt_t offset)
{
   // Add object to the fWriteMap or fReadMap containers (depending on the mode).
   // If obj is not 0 add object to the map (in read mode also add 0 objects to
   // the map). This method may only be called outside this class just before
   // calling obj->Streamer() to prevent self reference of obj, in case obj
   // contains (via via) a pointer to itself. In that case offset must be 1
   // (default value for offset).

   if (IsWriting()) {
      if (!fWriteMap)
         InitMap();

      if (obj) {
         CheckCount(offset);
         fWriteMap->Add(((TObject*)obj)->TObject::Hash(), (Long_t)obj, offset);
         fMapCount++;
      }
   } else {
      if (!fReadMap)
         InitMap();

      fReadMap->Add(offset, (Long_t)obj);
      fMapCount++;
   }
}

//______________________________________________________________________________
void TBuffer::SetReadParam(Int_t mapsize)
{
   // Set the initial size of the map used to store object and class
   // references during reading. The default size is kMapSize=503.
   // Increasing the default has the benefit that when reading many
   // small objects the map does not need to be resized too often
   // (the system is always dynamic, even with the default everything
   // will work, only the initial resizing will cost some time).
   // This method can only be called directly after the creation of
   // the TBuffer, before any reading is done. Globally this option
   // can be changed using SetGlobalReadParam().

   Assert(IsReading());
   Assert(fReadMap == 0);

   fMapSize = mapsize;
}

//______________________________________________________________________________
void TBuffer::SetWriteParam(Int_t mapsize)
{
   // Set the initial size of the hashtable used to store object and class
   // references during writing. The default size is kMapSize=503.
   // Increasing the default has the benefit that when writing many
   // small objects the hashtable does not get too many collisions
   // (the system is always dynamic, even with the default everything
   // will work, only a large number of collisions will cost performance).
   // For optimal performance hashsize should always be a prime.
   // This method can only be called directly after the creation of
   // the TBuffer, before any writing is done. Globally this option
   // can be changed using SetGlobalWriteParam().

   Assert(IsWriting());
   Assert(fWriteMap == 0);

   fMapSize = mapsize;
}

//______________________________________________________________________________
void TBuffer::InitMap()
{
   // Create the fWriteMap or fReadMap containers and initialize them
   // with the null object.

   if (IsWriting()) {
      if (!fWriteMap) {
         fWriteMap = new TExMap(fMapSize);
         fMapCount = 0;
      }
   } else {
      if (!fReadMap) {
         fReadMap = new TExMap(fMapSize);
         fReadMap->Add(0, kNullTag);      // put kNullTag in slot 0
         fMapCount = 1;
      }
   }
}

//______________________________________________________________________________
void TBuffer::ResetMap()
{
   // Delete existing fWriteMap or fReadMap and reset map counter.

   if (IsWriting()) {
      delete fWriteMap;
      fWriteMap = 0;
   } else {
      delete fReadMap;
      fReadMap = 0;
   }
   fMapCount = 0;
   fDisplacement = 0;
}

//______________________________________________________________________________
void TBuffer::SetByteCount(UInt_t cntpos, Bool_t packInVersion)
{
   // Set byte count at position cntpos in the buffer. Generate warning if
   // count larger than kMaxMapCount. The count is excluded its own size.

   UInt_t cnt = UInt_t(fBufCur - fBuffer) - cntpos - sizeof(UInt_t);
   char  *buf = (char *)(fBuffer + cntpos);

   // if true, pack byte count in two consecutive shorts, so it can
   // be read by ReadVersion()
   if (packInVersion) {
      union {
         UInt_t    cnt;
         Version_t vers[2];
      } v;
      v.cnt = cnt;
#ifdef R__BYTESWAP
      tobuf(buf, Version_t(v.vers[1] | kByteCountVMask));
      tobuf(buf, v.vers[0]);
#else
      tobuf(buf, Version_t(v.vers[0] | kByteCountVMask));
      tobuf(buf, v.vers[1]);
#endif
   } else
      tobuf(buf, cnt | kByteCountMask);

   if (cnt >= kMaxMapCount) {
      Error("WriteByteCount", "bytecount too large (more than %d)", kMaxMapCount);
      // exception
   }
}

//______________________________________________________________________________
Int_t TBuffer::CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss)
{
   // Check byte count with current buffer position. They should
   // match. If not print warning and position buffer in correct
   // place determined by the byte count. Startpos is position of
   // first byte where the byte count is written in buffer.
   // Returns 0 if everything is ok, otherwise the bytecount offset
   // (< 0 when read too little, >0 when read too much).

   if (!bcnt) return 0;

   Int_t  offset = 0;

   Long_t endpos = Long_t(fBuffer) + startpos + bcnt + sizeof(UInt_t);

   if (Long_t(fBufCur) != endpos) {
      if (clss) {
         if (Long_t(fBufCur) < endpos)
            Error("CheckByteCount", "object of class %s read too few bytes",
                  clss->GetName());
         if (Long_t(fBufCur) > endpos)
            Error("CheckByteCount", "object of class %s read too many bytes",
                  clss->GetName());
            Warning("CheckByteCount","%s::Streamer() not in sync with data on file, fix Streamer()",
                    clss->GetName());
      }
      offset = Int_t(Long_t(fBufCur) - endpos);
      //gROOT->Message(1005, this);

      fBufCur = (char *) endpos;
   }
   return offset;
}

//______________________________________________________________________________
Int_t TBuffer::ReadBuf(void *buf, Int_t max)
{
   // Read max bytes from the I/O buffer into buf. The function returns
   // the actual number of bytes read.

   Assert(IsReading());

   if (max == 0) return 0;

   Int_t n = TMath::Min(max, (Int_t)(fBufMax - fBufCur));

   memcpy(buf, fBufCur, n);
   fBufCur += n;

   return n;
}

//______________________________________________________________________________
void TBuffer::WriteBuf(const void *buf, Int_t max)
{
   // Write max bytes from buf into the I/O buffer.

   Assert(IsWriting());

   if (max == 0) return;

   if (fBufCur + max > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+max));

   memcpy(fBufCur, buf, max);
   fBufCur += max;
}

//______________________________________________________________________________
Text_t *TBuffer::ReadString(Text_t *s, Int_t max)
{
   // Read string from I/O buffer. String is read till 0 character is
   // found or till max-1 characters are read (i.e. string s has max
   // bytes allocated). If max = -1 no check on number of character is
   // made, reading continues till 0 character is found.

   Assert(IsReading());

   char  ch;
   Int_t nr = 0;

   if (max == -1) max = kMaxInt;

   while (nr < max-1) {

      *this >> ch;

      // stop when 0 read
      if (ch == 0) break;

      s[nr++] = ch;
   }

   s[nr] = 0;
   return s;
}

//______________________________________________________________________________
void TBuffer::WriteString(const Text_t *s)
{
   // Write string to I/O buffer. Writes string upto and including the
   // terminating 0.

   WriteBuf(s, (strlen(s)+1)*sizeof(Text_t));
}

//______________________________________________________________________________
Int_t TBuffer::ReadArray(Char_t *&c)
{
   // Read array of characters from the I/O buffer. Returns the number of
   // characters read. If argument is a 0 pointer then space will be
   // allocated for the array.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!c) c = new Char_t[n];

   Int_t l = sizeof(Char_t)*n;
   memcpy(c, fBufCur, l);
   fBufCur += l;
   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadArray(Short_t *&h)
{
   // Read array of shorts from the I/O buffer. Returns the number of shorts
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!h) h = new Short_t[n];

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(h, fBufCur, n);
   fBufCur += sizeof(Short_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &h[i]);
# endif
#else
   Int_t l = sizeof(Short_t)*n;
   memcpy(h, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadArray(Int_t *&ii)
{
   // Read array of ints from the I/O buffer. Returns the number of ints
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!ii) ii = new Int_t[n];

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(ii, fBufCur, n);
   fBufCur += sizeof(Int_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ii[i]);
# endif
#else
   Int_t l = sizeof(Int_t)*n;
   memcpy(ii, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadArray(Long_t *&ll)
{
   // Read array of longs from the I/O buffer. Returns the number of longs
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!ll) ll = new Long_t[n];

   if (gFile && gFile->GetVersion() < 30006) {
      for (int i = 0; i < n; i++) frombufOld(fBufCur, &ll[i]);
   } else {
      for (int i = 0; i < n; i++) frombuf(fBufCur, &ll[i]);
   }
   return n;
}
//______________________________________________________________________________
Int_t TBuffer::ReadArray(Float_t *&f)
{
   // Read array of floats from the I/O buffer. Returns the number of floats
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!f) f = new Float_t[n];

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(f, fBufCur, n);
   fBufCur += sizeof(Float_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &f[i]);
# endif
#else
   Int_t l = sizeof(Float_t)*n;
   memcpy(f, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadArray(Double_t *&d)
{
   // Read array of doubles from the I/O buffer. Returns the number of doubles
   // read. If argument is a 0 pointer then space will be allocated for the
   // array.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!d) d = new Double_t[n];

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &d[i]);
#else
   Int_t l = sizeof(Double_t)*n;
   memcpy(d, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadStaticArray(Char_t *c)
{
   // Read array of characters from the I/O buffer. Returns the number of
   // characters read.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!c) return 0;

   Int_t l = sizeof(Char_t)*n;
   memcpy(c, fBufCur, l);
   fBufCur += l;
   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadStaticArray(Short_t *h)
{
   // Read array of shorts from the I/O buffer. Returns the number of shorts
   // read.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!h) return 0;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(h, fBufCur, n);
   fBufCur += sizeof(Short_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &h[i]);
# endif
#else
   Int_t l = sizeof(Short_t)*n;
   memcpy(h, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadStaticArray(Int_t *ii)
{
   // Read array of ints from the I/O buffer. Returns the number of ints
   // read.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!ii) return 0;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(ii, fBufCur, n);
   fBufCur += sizeof(Int_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &ii[i]);
# endif
#else
   Int_t l = sizeof(Int_t)*n;
   memcpy(ii, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadStaticArray(Long_t *ll)
{
   // Read array of longs from the I/O buffer. Returns the number of longs
   // read.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!ll) return 0;

   if (gFile && gFile->GetVersion() < 30006) {
      for (int i = 0; i < n; i++) frombufOld(fBufCur, &ll[i]);
   } else {
      for (int i = 0; i < n; i++) frombuf(fBufCur, &ll[i]);
   }
   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadStaticArray(Float_t *f)
{
   // Read array of floats from the I/O buffer. Returns the number of floats
   // read.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!f) return 0;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(f, fBufCur, n);
   fBufCur += sizeof(Float_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &f[i]);
# endif
#else
   Int_t l = sizeof(Float_t)*n;
   memcpy(f, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
Int_t TBuffer::ReadStaticArray(Double_t *d)
{
   // Read array of doubles from the I/O buffer. Returns the number of doubles
   // read.

   Assert(IsReading());

   Int_t n;
   *this >> n;

   if (!n) return n;

   if (!d) return 0;

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &d[i]);
#else
   Int_t l = sizeof(Double_t)*n;
   memcpy(d, fBufCur, l);
   fBufCur += l;
#endif

   return n;
}

//______________________________________________________________________________
void TBuffer::ReadFastArray(Char_t *c, Int_t n)
{
   // Read array of n characters from the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Char_t)*n;
   memcpy(c, fBufCur, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBuffer::ReadFastArray(Short_t *h, Int_t n)
{
   // Read array of n shorts from the I/O buffer.

   if (n <= 0) return;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(h, fBufCur, n);
   fBufCur += sizeof(Short_t)*n;
# else
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &h[i]);
# endif
#else
   Int_t l = sizeof(Short_t)*n;
   memcpy(h, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::ReadFastArray(Int_t *ii, Int_t n)
{
   // Read array of n ints from the I/O buffer.

   if (n <= 0) return;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(ii, fBufCur, n);
   fBufCur += sizeof(Int_t)*n;
# else
   //char *sw = (char*)ii;
   for (int i = 0; i < n; i++) {
      frombuf(fBufCur, &ii[i]);
      //sw[0] = fBufCur[3];
      //sw[1] = fBufCur[2];
      //sw[2] = fBufCur[1];
      //sw[3] = fBufCur[0];
      //fBufCur += 4;
      //sw += 4;
   }
# endif
#else
   Int_t l = sizeof(Int_t)*n;
   memcpy(ii, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::ReadFastArray(Long_t *ll, Int_t n)
{
   // Read array of n longs from the I/O buffer.

   if (n <= 0) return;

   if (gFile && gFile->GetVersion() < 30006) {
      for (int i = 0; i < n; i++) frombufOld(fBufCur, &ll[i]);
   } else {
      for (int i = 0; i < n; i++) frombuf(fBufCur, &ll[i]);
   }
}

//______________________________________________________________________________
void TBuffer::ReadFastArray(Float_t *f, Int_t n)
{
   // Read array of n floats from the I/O buffer.

   if (n <= 0) return;

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(f, fBufCur, n);
   fBufCur += sizeof(Float_t)*n;
# else
//   char *sw = (char*)f;
   for (int i = 0; i < n; i++) {
      frombuf(fBufCur, &f[i]);
      //sw[0] = fBufCur[3];
      //sw[1] = fBufCur[2];
      //sw[2] = fBufCur[1];
      //sw[3] = fBufCur[0];
      //fBufCur += 4;
      //sw += 4;
   }
# endif
#else
   Int_t l = sizeof(Float_t)*n;
   memcpy(f, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::ReadFastArray(Double_t *d, Int_t n)
{
   // Read array of n doubles from the I/O buffer.

   if (n <= 0) return;

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      frombuf(fBufCur, &d[i]);
#else
   Int_t l = sizeof(Double_t)*n;
   memcpy(d, fBufCur, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteArray(const Char_t *c, Int_t n)
{
   // Write array of n characters into the I/O buffer.

   Assert(IsWriting());

   *this << n;

   if (!n) return;

   Assert(c);

   Int_t l = sizeof(Char_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

   memcpy(fBufCur, c, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBuffer::WriteArray(const Short_t *h, Int_t n)
{
   // Write array of n shorts into the I/O buffer.

   Assert(IsWriting());

   *this << n;

   if (!n) return;

   Assert(h);

   Int_t l = sizeof(Short_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(fBufCur, h, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, h[i]);
# endif
#else
   memcpy(fBufCur, h, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteArray(const Int_t *ii, Int_t n)
{
   // Write array of n ints into the I/O buffer.

   Assert(IsWriting());

   *this << n;

   if (!n) return;

   Assert(ii);

   Int_t l = sizeof(Int_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, ii, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, ii[i]);
# endif
#else
   memcpy(fBufCur, ii, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteArray(const Long_t *ll, Int_t n)
{
   // Write array of n longs into the I/O buffer.

   Assert(IsWriting());

   *this << n;

   if (!n) return;

   Assert(ll);

   Int_t l = 8*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));
   for (int i = 0; i < n; i++) tobuf(fBufCur, ll[i]);
}

//______________________________________________________________________________
void TBuffer::WriteArray(const Float_t *f, Int_t n)
{
   // Write array of n floats into the I/O buffer.

   Assert(IsWriting());

   *this << n;

   if (!n) return;

   Assert(f);

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, f, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, f[i]);
# endif
#else
   memcpy(fBufCur, f, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteArray(const Double_t *d, Int_t n)
{
   // Write array of n doubles into the I/O buffer.

   Assert(IsWriting());

   *this << n;

   if (!n) return;

   Assert(d);

   Int_t l = sizeof(Double_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, d[i]);
#else
   memcpy(fBufCur, d, l);
   fBufCur += l;
#endif
}


//______________________________________________________________________________
void TBuffer::WriteFastArray(const Char_t *c, Int_t n)
{
   // Write array of n characters into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Char_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

   memcpy(fBufCur, c, l);
   fBufCur += l;
}

//______________________________________________________________________________
void TBuffer::WriteFastArray(const Short_t *h, Int_t n)
{
   // Write array of n shorts into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Short_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy16(fBufCur, h, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, h[i]);
# endif
#else
   memcpy(fBufCur, h, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteFastArray(const Int_t *ii, Int_t n)
{
   // Write array of n ints into the I/O buffer.
   if (n <= 0) return;

   Int_t l = sizeof(Int_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, ii, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, ii[i]);
# endif
#else
   memcpy(fBufCur, ii, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteFastArray(const Long_t *ll, Int_t n)
{
   // Write array of n longs into the I/O buffer.

   if (n <= 0) return;

   Int_t l = 8*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

   for (int i = 0; i < n; i++) tobuf(fBufCur, ll[i]);
}

//______________________________________________________________________________
void TBuffer::WriteFastArray(const Float_t *f, Int_t n)
{
   // Write array of n floats into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Float_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
# ifdef USE_BSWAPCPY
   bswapcpy32(fBufCur, f, n);
   fBufCur += l;
# else
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, f[i]);
# endif
#else
   memcpy(fBufCur, f, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
void TBuffer::WriteFastArray(const Double_t *d, Int_t n)
{
   // Write array of n doubles into the I/O buffer.

   if (n <= 0) return;

   Int_t l = sizeof(Double_t)*n;
   if (fBufCur + l > fBufMax) Expand(TMath::Max(2*fBufSize, fBufSize+l));

#ifdef R__BYTESWAP
   for (int i = 0; i < n; i++)
      tobuf(fBufCur, d[i]);
#else
   memcpy(fBufCur, d, l);
   fBufCur += l;
#endif
}

//______________________________________________________________________________
TObject *TBuffer::ReadObject(const TClass *clReq)
{
   // Read object from I/O buffer. clReq can be used to cross check
   // if the actually read object is of the requested class.

   Assert(IsReading());

   // make sure fReadMap is initialized
   InitMap();

   // before reading object save start position
   UInt_t startpos = UInt_t(fBufCur-fBuffer);

   // attempt to load next object as TClass clReq
   UInt_t tag;       // either tag or byte count
   TClass *clRef = ReadClass(clReq, &tag);

   // check if object has not already been read
   // (this can only happen when called via CheckObject())
   TObject *obj;
   if (fVersion > 0) {
      obj = (TObject *) fReadMap->GetValue(startpos+kMapOffset);
      if (obj) {
         if (obj == (TObject*) -1)
            obj = 0;
         else if (clReq && !obj->IsA()->InheritsFrom(clReq)) {
            Error("ReadObject", "got object of wrong class");
            // exception
         }
         CheckByteCount(startpos, tag, 0);
         return obj;
      }
   }

   // unknown class, skip to next object and return 0 obj
   if (clRef == (TClass*) -1) {
      if (fVersion > 0)
         MapObject((TObject*) -1, startpos+kMapOffset);
      else
         MapObject(0, fMapCount);
      CheckByteCount(startpos, tag, 0);
      return 0;
   }

   if (!clRef) {

      // got a reference to an already read object
      if (fVersion > 0) {
         tag += fDisplacement;
         tag = CheckObject(tag, clReq);
      } else {
         if (tag > (UInt_t)fReadMap->GetSize()) {
            Error("ReadObject", "object tag too large, I/O buffer corrupted");
            return 0;
            // exception
         }
      } 

      obj = (TObject *) fReadMap->GetValue(tag);
      if (obj && clReq && !obj->IsA()->InheritsFrom(clReq)) {
         Error("ReadObject", "got object of wrong class");
         // exception
      }

   } else {

      // allocate a new object based on the class found
      obj = (TObject *)clRef->New();
      if (!obj) {
         Error("ReadObject", "could not create object of class %s", clRef->GetName());
         // exception
         return 0;
      }

      // add to fReadMap before reading rest of object
      if (fVersion > 0)
         MapObject(obj, startpos+kMapOffset);
      else
         MapObject(obj, fMapCount);

      // let the object read itself
      if (clRef->GetClassInfo()) {
         obj->Streamer(*this);
      } else {
         //fake class has no Streamer
         if (gDebug > 0) Warning("ReadObject","%s::Streamer not available, using TClass::ReadBuffer instead",clRef->GetName());
         clRef->ReadBuffer(*this,obj);
      }

      CheckByteCount(startpos, tag, clRef);
   }

   return obj;
}

//______________________________________________________________________________
void TBuffer::WriteObject(const TObject *obj)
{
   // Write object to I/O buffer.

   Assert(IsWriting());

   // make sure fWriteMap is initialized
   InitMap();

   ULong_t idx;

   if (!obj) {

      // save kNullTag to represent NULL pointer
      *this << kNullTag;

   } else if ((idx = (ULong_t)fWriteMap->GetValue(((TObject*)obj)->TObject::Hash(), (Long_t)obj)) != 0) {

      // truncation is OK the value we did put in the map is an 30-bit offset
      // and not a pointer
      UInt_t objIdx = UInt_t(idx);

      // save index of already stored object
      *this << objIdx;

   } else {

      // reserve space for leading byte count
      UInt_t cntpos = UInt_t(fBufCur-fBuffer);
      fBufCur += sizeof(UInt_t);

      // write class of object first
      TClass *cl = obj->IsA();
      WriteClass(cl);

      // add to map before writing rest of object (to handle self reference)
      // (+kMapOffset so it's != kNullTag)
      MapObject(obj, cntpos+kMapOffset);

      // let the object write itself (cast const away)
      ((TObject *)obj)->Streamer(*this);

      // write byte count
      SetByteCount(cntpos);
   }
}

//______________________________________________________________________________
TClass *TBuffer::ReadClass(const TClass *clReq, UInt_t *objTag)
{
   // Read class definition from I/O buffer. clReq can be used to cross check
   // if the actually read object is of the requested class. objTag is
   // set in case the object is a reference to an already read object.

   Assert(IsReading());

   // read byte count and/or tag (older files don't have byte count)
   UInt_t bcnt, tag, startpos = 0;
   *this >> bcnt;
   if (!(bcnt & kByteCountMask) || bcnt == kNewClassTag) {
      tag  = bcnt;
      bcnt = 0;
   } else {
      fVersion = 1;
      startpos = UInt_t(fBufCur-fBuffer);
      *this >> tag;
   }

   // in case tag is object tag return tag
   if (!(tag & kClassMask)) {
      if (objTag) *objTag = tag;
      return 0;
   }

   TClass *cl;
   if (tag == kNewClassTag) {

      // got a new class description followed by a new object
      // (class can be 0 if class dictionary is not found, in that
      // case object of this class must be skipped)
      cl = TClass::Load(*this);

      // add class to fReadMap for later reference
      if (fVersion > 0) {
         // check if class was already read
         TClass *cl1 = (TClass *)fReadMap->GetValue(startpos+kMapOffset);
         if (cl1 != cl)
            MapObject(cl ? cl : (TObject*) -1, startpos+kMapOffset);
      } else
         MapObject(cl, fMapCount);

   } else {

      // got a tag to an already seen class
      UInt_t clTag = (tag & ~kClassMask);

      if (fVersion > 0) {
         clTag += fDisplacement;
         clTag = CheckObject(clTag, clReq, kTRUE);
      } else {
         if (clTag == 0 || clTag > (UInt_t)fReadMap->GetSize()) {
            Error("ReadClass", "illegal class tag=%d (0<tag<=%d), I/O buffer corrupted",
                  clTag, fReadMap->GetSize());
            // exception
         }
      }

      // class can be 0 if dictionary was not found
      cl = (TClass *)fReadMap->GetValue(clTag);
   }

   if (cl && clReq && !cl->InheritsFrom(clReq)) {
      Error("ReadClass", "got wrong class: %s",cl->GetName());
      // exception
   }

   // return bytecount in objTag
   if (objTag) *objTag = (bcnt & ~kByteCountMask);

   // case of unknown class
   if (!cl) cl = (TClass*)-1;

   return cl;
}

//______________________________________________________________________________
void TBuffer::WriteClass(const TClass *cl)
{
   // Write class description to I/O buffer.

   Assert(IsWriting());

   ULong_t idx;

   if ((idx = (ULong_t)fWriteMap->GetValue(((TObject *)cl)->TObject::Hash(), (Long_t)cl)) != 0) {

      // truncation is OK the value we did put in the map is an 30-bit offset
      // and not a pointer
      UInt_t clIdx = UInt_t(idx);

      // save index of already stored class
      *this << (clIdx | kClassMask);

   } else {

      // offset in buffer where class info is written
      UInt_t offset = UInt_t(fBufCur-fBuffer);

      // save new class tag
      *this << kNewClassTag;

      // write class name
      cl->Store(*this);

      // store new class reference in fWriteMap (+kMapOffset so it's != kNullTag)
      MapObject(cl, offset+kMapOffset);
   }
}

//______________________________________________________________________________
Version_t TBuffer::ReadVersion(UInt_t *startpos, UInt_t *bcnt)
{
   // Read class version from I/O buffer.

   Version_t version;

   if (startpos && bcnt) {
      // before reading object save start position
      *startpos = UInt_t(fBufCur-fBuffer);

      // read byte count (older files don't have byte count)
      // byte count is packed in two individual shorts, this to be
      // backward compatible with old files that have at this location
      // only a single short (i.e. the version)
      union {
         UInt_t     cnt;
         Version_t  vers[2];
      } v;
#ifdef R__BYTESWAP
      *this >> v.vers[1];
      *this >> v.vers[0];
#else
      *this >> v.vers[0];
      *this >> v.vers[1];
#endif

      // no bytecount, backup and read version
      if (!(v.cnt & kByteCountMask)) {
         fBufCur -= sizeof(UInt_t);
         v.cnt = 0;
      }
      *bcnt = (v.cnt & ~kByteCountMask);
      *this >> version;
//printf("Reading version=%d at pos=%d, bytecount=%d\n",version,*startpos,*bcnt);

   } else {

      // not interested in byte count
      *this >> version;

      // if this is a byte count, then skip next short and read version
      if (version & kByteCountVMask) {
         *this >> version;
         *this >> version;
      }
//printf("Reading version=%d at pos=%d\n",version,startpos);
   }

   return version;
}

//______________________________________________________________________________
UInt_t TBuffer::WriteVersion(const TClass *cl, Bool_t useBcnt)
{
   // Write class version to I/O buffer.

   UInt_t cntpos = 0;
   if (useBcnt) {
      // reserve space for leading byte count
      cntpos   = UInt_t(fBufCur-fBuffer);
      fBufCur += sizeof(UInt_t);
   }

   Version_t version = cl->GetClassVersion();
//printf("Writing version=%d at pos=%d for class:%s\n",version,cntpos,cl->GetName());
   if (version > kMaxVersion) {
      Error("WriteVersion", "version number cannot be larger than %hd)",
            kMaxVersion);
      version = kMaxVersion;
   }

   *this << version;

   // return position where to store possible byte count
   return cntpos;
}

//---- Static functions --------------------------------------------------------

//______________________________________________________________________________
void TBuffer::SetGlobalReadParam(Int_t mapsize)
{
   // Set the initial size of the map used to store object and class
   // references during reading. The default size is kMapSize=503.
   // Increasing the default has the benefit that when reading many
   // small objects the array does not need to be resized too often
   // (the system is always dynamic, even with the default everything
   // will work, only the initial resizing will cost some time).
   // Per TBuffer object this option can be changed using SetReadParam().

   fgMapSize = mapsize;
}

//______________________________________________________________________________
void TBuffer::SetGlobalWriteParam(Int_t mapsize)
{
   // Set the initial size of the hashtable used to store object and class
   // references during writing. The default size is kMapSize=503.
   // Increasing the default has the benefit that when writing many
   // small objects the hashtable does not get too many collisions
   // (the system is always dynamic, even with the default everything
   // will work, only a large number of collisions will cost performance).
   // For optimal performance hashsize should always be a prime.
   // Per TBuffer object this option can be changed using SetWriteParam().

   fgMapSize = mapsize;
}

//______________________________________________________________________________
Int_t TBuffer::GetGlobalReadParam()
{
   // Get default read map size.

   return fgMapSize;
}

//______________________________________________________________________________
Int_t TBuffer::GetGlobalWriteParam()
{
   // Get default write map size.

   return fgMapSize;
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
