// @(#)root/base:$Name:  $:$Id: TBuffer.h,v 1.2 2000/05/24 10:31:47 brun Exp $
// Author: Fons Rademakers   04/05/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBuffer
#define ROOT_TBuffer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer                                                              //
//                                                                      //
// Buffer base class used for serializing objects.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_Bytes
#include "Bytes.h"
#endif

class TClass;
class TExMap;


class TBuffer : public TObject {

protected:
   Bool_t  fMode;          //Read or write mode
   Int_t   fVersion;       //Buffer format version
   Int_t   fBufSize;       //Size of buffer
   char   *fBuffer;        //Buffer used to store objects
   char   *fBufCur;        //Current position in buffer
   char   *fBufMax;        //End of buffer
   Int_t   fMapCount;      //Number of objects or classes in map
   Int_t   fMapSize;       //Default size of map
   Int_t   fDisplacement;  //Value to be added to the map offsets

   union {
      TExMap *fReadMap;    //Map containing id,object references during reading
      TExMap *fWriteMap;   //Map containing object,id pairs during writing
   };

   static Int_t fgMapSize; //Default map size for all TBuffer objects

   // Default ctor
   TBuffer() : fMode(0), fBuffer(0) { fReadMap = 0; }

   // TBuffer objects cannot be copied or assigned
   TBuffer(const TBuffer &) { }
   void operator=(const TBuffer &) { }

   void   CheckCount(UInt_t offset);
   UInt_t CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass = kFALSE);

   void Expand(Int_t newsize);  //Expand buffer to newsize

   Int_t Read(const char *name) { return TObject::Read(name); }
   Int_t Write(const char *name, Int_t opt, Int_t bufs)
                                 { return TObject::Write(name, opt, bufs); }

public:
   enum EMode { kRead = 0, kWrite = 1 };
   enum { kInitialSize = 1024, kMinimalSize = 128 };
   enum { kMapSize = 503 };

   TBuffer(EMode mode, Int_t bufsiz = kInitialSize, void *buf = 0);
   virtual ~TBuffer();

   void     MapObject(const TObject *obj, UInt_t offset = 1);
   virtual void Reset() { SetBufferOffset(); ResetMap(); }
   void     InitMap();
   void     ResetMap();
   void     SetReadMode();
   void     SetReadParam(Int_t mapsize);
   void     SetWriteMode();
   void     SetWriteParam(Int_t mapsize);
   void     SetBuffer(void *buf, UInt_t bufsiz = 0);
   void     SetBufferOffset(Int_t offset = 0) { fBufCur = fBuffer+offset; }

   char    *Buffer() const { return fBuffer; }
   Int_t    BufferSize() const { return fBufSize; }
   void     DetachBuffer() { fBuffer = 0; }
   Int_t    Length() const { return (Int_t)(fBufCur - fBuffer); }

   Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss);
   void     SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);

   Bool_t   IsReading() const { return (fMode & kWrite) == 0; }
   Bool_t   IsWriting() const { return (fMode & kWrite) != 0; }

   Int_t    ReadBuf(void *buf, Int_t max);
   void     WriteBuf(const void *buf, Int_t max);

   char    *ReadString(char *s, Int_t max);
   void     WriteString(const char *s);

   Version_t ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0);
   UInt_t    WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);

   TClass  *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   void     WriteClass(const TClass *cl);

   TObject *ReadObject(const TClass *cl);
   void     WriteObject(const TObject *obj);

   void     SetBufferDisplacement(Int_t skipped)
            { fDisplacement =  (Int_t)(Length() - skipped); }
   void     SetBufferDisplacement()
            { fDisplacement = 0; }
   Int_t    GetBufferDisplacement() { return fDisplacement; }

   Int_t    ReadArray(Char_t   *&c);
   Int_t    ReadArray(UChar_t  *&c);
   Int_t    ReadArray(Short_t  *&h);
   Int_t    ReadArray(UShort_t *&h);
   Int_t    ReadArray(Int_t    *&i);
   Int_t    ReadArray(UInt_t   *&i);
   Int_t    ReadArray(Long_t   *&l);
   Int_t    ReadArray(ULong_t  *&l);
   Int_t    ReadArray(Float_t  *&f);
   Int_t    ReadArray(Double_t *&d);

   Int_t    ReadStaticArray(Char_t   *c);
   Int_t    ReadStaticArray(UChar_t  *c);
   Int_t    ReadStaticArray(Short_t  *h);
   Int_t    ReadStaticArray(UShort_t *h);
   Int_t    ReadStaticArray(Int_t    *i);
   Int_t    ReadStaticArray(UInt_t   *i);
   Int_t    ReadStaticArray(Long_t   *l);
   Int_t    ReadStaticArray(ULong_t  *l);
   Int_t    ReadStaticArray(Float_t  *f);
   Int_t    ReadStaticArray(Double_t *d);

   void     WriteArray(const Char_t   *c, Int_t n);
   void     WriteArray(const UChar_t  *c, Int_t n);
   void     WriteArray(const Short_t  *h, Int_t n);
   void     WriteArray(const UShort_t *h, Int_t n);
   void     WriteArray(const Int_t    *i, Int_t n);
   void     WriteArray(const UInt_t   *i, Int_t n);
   void     WriteArray(const Long_t   *l, Int_t n);
   void     WriteArray(const ULong_t  *l, Int_t n);
   void     WriteArray(const Float_t  *f, Int_t n);
   void     WriteArray(const Double_t *d, Int_t n);

   void     ReadFastArray(Char_t   *c, Int_t n);
   void     ReadFastArray(UChar_t  *c, Int_t n);
   void     ReadFastArray(Short_t  *h, Int_t n);
   void     ReadFastArray(UShort_t *h, Int_t n);
   void     ReadFastArray(Int_t    *i, Int_t n);
   void     ReadFastArray(UInt_t   *i, Int_t n);
   void     ReadFastArray(Long_t   *l, Int_t n);
   void     ReadFastArray(ULong_t  *l, Int_t n);
   void     ReadFastArray(Float_t  *f, Int_t n);
   void     ReadFastArray(Double_t *d, Int_t n);

   void     WriteFastArray(const Char_t   *c, Int_t n);
   void     WriteFastArray(const UChar_t  *c, Int_t n);
   void     WriteFastArray(const Short_t  *h, Int_t n);
   void     WriteFastArray(const UShort_t *h, Int_t n);
   void     WriteFastArray(const Int_t    *i, Int_t n);
   void     WriteFastArray(const UInt_t   *i, Int_t n);
   void     WriteFastArray(const Long_t   *l, Int_t n);
   void     WriteFastArray(const ULong_t  *l, Int_t n);
   void     WriteFastArray(const Float_t  *f, Int_t n);
   void     WriteFastArray(const Double_t *d, Int_t n);

   TBuffer  &operator>>(Char_t   &c);
   TBuffer  &operator>>(UChar_t  &c);
   TBuffer  &operator>>(Short_t  &h);
   TBuffer  &operator>>(UShort_t &h);
   TBuffer  &operator>>(Int_t    &i);
   TBuffer  &operator>>(UInt_t   &i);
   TBuffer  &operator>>(Long_t   &l);
   TBuffer  &operator>>(ULong_t  &l);
   TBuffer  &operator>>(Float_t  &f);
   TBuffer  &operator>>(Double_t &d);
   TBuffer  &operator>>(Char_t   *c);

   TBuffer  &operator<<(Char_t   c);
   TBuffer  &operator<<(UChar_t  c);
   TBuffer  &operator<<(Short_t  h);
   TBuffer  &operator<<(UShort_t h);
   TBuffer  &operator<<(Int_t    i);
   TBuffer  &operator<<(UInt_t   i);
   TBuffer  &operator<<(Long_t   l);
   TBuffer  &operator<<(ULong_t  l);
   TBuffer  &operator<<(Float_t  f);
   TBuffer  &operator<<(Double_t d);
   TBuffer  &operator<<(const Char_t  *c);

   //friend TBuffer  &operator>>(TBuffer &b, TObject *&obj);
   friend TBuffer  &operator>>(TBuffer &b, const TObject *&obj);
   friend TBuffer  &operator<<(TBuffer &b, const TObject *obj);

   static void  SetGlobalReadParam(Int_t mapsize);
   static void  SetGlobalWriteParam(Int_t mapsize);
   static Int_t GetGlobalReadParam();
   static Int_t GetGlobalWriteParam();

   ClassDef(TBuffer,0)  //Buffer base class used for serializing objects
};

//---------------------- TBuffer inlines ---------------------------------------

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Char_t c)
{
   if (fBufCur + sizeof(Char_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Short_t h)
{
   if (fBufCur + sizeof(Short_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, h);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Int_t i)
{
   if (fBufCur + sizeof(Int_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, i);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Long_t l)
{
   if (fBufCur + sizeof(Long_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, l);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Float_t f)
{
   if (fBufCur + sizeof(Float_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, f);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Double_t d)
{
   if (fBufCur + sizeof(Double_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, d);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(const Char_t *c)
{
   WriteString(c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Char_t &c)
{
   frombuf(fBufCur, &c);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Short_t &h)
{
   frombuf(fBufCur, &h);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Int_t &i)
{
   frombuf(fBufCur, &i);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Long_t &l)
{
   frombuf(fBufCur, &l);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Float_t &f)
{
   frombuf(fBufCur, &f);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Double_t &d)
{
   frombuf(fBufCur, &d);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Char_t *c)
{
   ReadString(c, -1);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UChar_t c)
   { return TBuffer::operator<<((Char_t)c); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UShort_t h)
   { return TBuffer::operator<<((Short_t)h); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(UInt_t i)
   { return TBuffer::operator<<((Int_t)i); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(ULong_t l)
   { return TBuffer::operator<<((Long_t)l); }

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UChar_t &c)
   { return TBuffer::operator>>((Char_t&)c); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UShort_t &h)
   { return TBuffer::operator>>((Short_t&)h); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(UInt_t &i)
   { return TBuffer::operator>>((Int_t&)i); }
//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(ULong_t &l)
   { return TBuffer::operator>>((Long_t&)l); }

//______________________________________________________________________________
inline TBuffer &operator<<(TBuffer &buf, const TObject *obj)
   { buf.WriteObject(obj); return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }
//______________________________________________________________________________
inline TBuffer &operator>>(TBuffer &buf, const TObject *&obj)
   { obj = buf.ReadObject(0); return buf; }

//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UChar_t *&c)
   { return TBuffer::ReadArray((Char_t *&)c); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UShort_t *&h)
   { return TBuffer::ReadArray((Short_t *&)h); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(UInt_t *&i)
   { return TBuffer::ReadArray((Int_t *&)i); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadArray(ULong_t *&l)
   { return TBuffer::ReadArray((Long_t *&)l); }

//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UChar_t *c)
   { return TBuffer::ReadStaticArray((Char_t *)c); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UShort_t *h)
   { return TBuffer::ReadStaticArray((Short_t *)h); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(UInt_t *i)
   { return TBuffer::ReadStaticArray((Int_t *)i); }
//______________________________________________________________________________
inline Int_t TBuffer::ReadStaticArray(ULong_t *l)
   { return TBuffer::ReadStaticArray((Long_t *)l); }

//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UChar_t *c, Int_t n)
   { TBuffer::ReadFastArray((Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UShort_t *h, Int_t n)
   { TBuffer::ReadFastArray((Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(UInt_t *i, Int_t n)
   { TBuffer::ReadFastArray((Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::ReadFastArray(ULong_t *l, Int_t n)
   { TBuffer::ReadFastArray((Long_t *)l, n); }

//______________________________________________________________________________
inline void TBuffer::WriteArray(const UChar_t *c, Int_t n)
   { TBuffer::WriteArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const UShort_t *h, Int_t n)
   { TBuffer::WriteArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const UInt_t *i, Int_t n)
   { TBuffer::WriteArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::WriteArray(const ULong_t *l, Int_t n)
   { TBuffer::WriteArray((const Long_t *)l, n); }

//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UChar_t *c, Int_t n)
   { TBuffer::WriteFastArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UShort_t *h, Int_t n)
   { TBuffer::WriteFastArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const UInt_t *i, Int_t n)
   { TBuffer::WriteFastArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBuffer::WriteFastArray(const ULong_t *l, Int_t n)
   { TBuffer::WriteFastArray((const Long_t *)l, n); }

#endif
