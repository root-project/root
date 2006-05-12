// @(#)root/base:$Name:  $:$Id: TBuffer.h,v 1.54 2006/01/24 21:23:20 pcanal Exp $
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

#include <vector>

#ifdef R__OLDHPACC
namespace std {
   using ::string;
   using ::vector;
}
#endif

class TStreamerInfo;
class TStreamerElement;
class TClass;
class TExMap;

class TBuffer : public TObject {

protected:
   typedef std::vector<TStreamerInfo*> InfoList_t;

   Bool_t          fMode;          //Read or write mode
   Int_t           fVersion;       //Buffer format version
   Int_t           fBufSize;       //Size of buffer
   char           *fBuffer;        //Buffer used to store objects
   char           *fBufCur;        //Current position in buffer
   char           *fBufMax;        //End of buffer
   Int_t           fMapCount;      //Number of objects or classes in map
   Int_t           fMapSize;       //Default size of map
   Int_t           fDisplacement;  //Value to be added to the map offsets
   TExMap         *fMap;           //Map containing object,offset pairs for reading/writing
   TExMap         *fClassMap;      //Map containing object,class pairs for reading
   TObject        *fParent;        //Pointer to the buffer parent (file) where buffer is read/written
   TStreamerInfo  *fInfo;          //Pointer to TStreamerInfo object writing/reading the buffer
   InfoList_t      fInfos;         //Stack of pointers to the TStreamerInfos
   UShort_t        fPidOffset;     //Offset to be added to the pid index in this key/buffer.

   static Int_t fgMapSize; //Default map size for all TBuffer objects

   // Default ctor
   TBuffer() : TObject(), fMode(0), fVersion(0), fBufSize(0), fBuffer(0),
               fBufCur(0), fBufMax(0), fMapCount(0), fMapSize(0),
               fDisplacement(0), fMap(0), fClassMap(0), fParent(0),
               fInfo(0), fInfos(), fPidOffset(0) {}

   // TBuffer objects cannot be copied or assigned
   TBuffer(const TBuffer &);           // not implemented
   void operator=(const TBuffer &);    // not implemented

   Int_t  CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss, const char* classname);
   void   CheckCount(UInt_t offset);
   UInt_t CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass = kFALSE);

   void Expand(Int_t newsize);  // expand buffer to newsize

   Int_t Read(const char *name) { return TObject::Read(name); }
   Int_t Write(const char *name, Int_t opt, Int_t bufs)
                              { return TObject::Write(name, opt, bufs); }
   Int_t Write(const char *name, Int_t opt, Int_t bufs) const
                              { return TObject::Write(name, opt, bufs); }

   virtual  void     WriteObject(const void *actualObjStart, const TClass *actualClass);

public:
   enum EMode { kRead = 0, kWrite = 1 };
   enum { kInitialSize = 1024, kMinimalSize = 128 };
   enum { kMapSize = 503 };
   enum { kStreamedMemberWise = BIT(14) }; //added to version number to know if a collection has been stored member-wise
   enum { kNotDecompressed = BIT(15) }; //indicates a weird buffer, used by TBasket
   enum { kIsOwner = BIT(16) };  //if set TBuffer owns fBuffer
   enum { kCannotHandleMemberWiseStreaming = BIT(17), //if set TClonesArray should not use memeber wise streaming
          kTextBasedStreaming = BIT(18) };            // indicates if buffer used for XML/SQL object streaming
   enum { kUser1 = BIT(21), kUser2 = BIT(22), kUser3 = BIT(23)}; //free for user

   TBuffer(EMode mode);
   TBuffer(EMode mode, Int_t bufsiz);
   TBuffer(EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE);
   virtual ~TBuffer();

   Int_t    GetMapCount() const { return fMapCount; }
   Int_t    GetBufferVersion() const { return fVersion; }
   void     GetMappedObject(UInt_t tag, void* &ptr, TClass* &ClassPtr) const;
   void     MapObject(const TObject *obj, UInt_t offset = 1);
   void     MapObject(const void *obj, const TClass *cl, UInt_t offset = 1);
   virtual void Reset() { SetBufferOffset(); ResetMap(); }
   void     InitMap();
   void     ResetMap();
   void     SetReadMode();
   void     SetReadParam(Int_t mapsize);
   void     SetWriteMode();
   void     SetWriteParam(Int_t mapsize);
   void     SetBuffer(void *buf, UInt_t bufsiz = 0, Bool_t adopt = kTRUE);
   void     SetBufferOffset(Int_t offset = 0) { fBufCur = fBuffer+offset; }
   void     SetParent(TObject *parent);
   TObject *GetParent() const;
   char    *Buffer() const { return fBuffer; }
   Int_t    BufferSize() const { return fBufSize; }
   void     DetachBuffer() { fBuffer = 0; }
   Int_t    Length() const { return (Int_t)(fBufCur - fBuffer); }

   Bool_t   CheckObject(const TObject *obj);
   Bool_t   CheckObject(const void *obj, const TClass *ptrClass);

   virtual   Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss);
   virtual   Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const char *classname);
   virtual   void     SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);

   virtual Version_t  ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0, const TClass *cl = 0);
   virtual UInt_t     WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);
   virtual UInt_t     WriteVersionMemberWise(const TClass *cl, Bool_t useBcnt = kFALSE);

   virtual void      *ReadObjectAny(const TClass* cast);
   virtual void       SkipObjectAny();

   virtual void       IncrementLevel(TStreamerInfo* info);
   virtual void       SetStreamerElementNumber(Int_t) {}
   virtual void       DecrementLevel(TStreamerInfo*);
   TStreamerInfo     *GetInfo() {return fInfo;}
   
   virtual void       ClassBegin(const TClass*, Version_t = -1) {}
   virtual void       ClassEnd(const TClass*) {}
   virtual void       ClassMember(const char*, const char* = 0, Int_t = -1, Int_t = -1) {}

   Bool_t   IsReading() const { return (fMode & kWrite) == 0; }
   Bool_t   IsWriting() const { return (fMode & kWrite) != 0; }

   Int_t    ReadBuf(void *buf, Int_t max);
   void     WriteBuf(const void *buf, Int_t max);

   char    *ReadString(char *s, Int_t max);
   void     WriteString(const char *s);

   virtual TClass  *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   virtual void     WriteClass(const TClass *cl);

   virtual TObject *ReadObject(const TClass *cl);
   virtual void     WriteObject(const TObject *obj);

   virtual Int_t    WriteObjectAny(const void *obj, const TClass *ptrClass);

   UShort_t GetPidOffset() const {
      // See comment in TBuffer::SetPidOffset
      return fPidOffset;
   }
   void     SetPidOffset(UShort_t offset);
   Int_t    GetBufferDisplacement() const { return fDisplacement; }
   void     SetBufferDisplacement() { fDisplacement = 0; }
   void     SetBufferDisplacement(Int_t skipped)
            { fDisplacement =  (Int_t)(Length() - skipped); }

   virtual   void     ReadDouble32 (Double_t *d, TStreamerElement *ele=0);
   virtual   void     WriteDouble32(Double_t *d, TStreamerElement *ele=0);

   virtual   Int_t    ReadArray(Bool_t    *&b);
   virtual   Int_t    ReadArray(Char_t    *&c);
   virtual   Int_t    ReadArray(UChar_t   *&c);
   virtual   Int_t    ReadArray(Short_t   *&h);
   virtual   Int_t    ReadArray(UShort_t  *&h);
   virtual   Int_t    ReadArray(Int_t     *&i);
   virtual   Int_t    ReadArray(UInt_t    *&i);
   virtual   Int_t    ReadArray(Long_t    *&l);
   virtual   Int_t    ReadArray(ULong_t   *&l);
   virtual   Int_t    ReadArray(Long64_t  *&l);
   virtual   Int_t    ReadArray(ULong64_t *&l);
   virtual   Int_t    ReadArray(Float_t   *&f);
   virtual   Int_t    ReadArray(Double_t  *&d);
   virtual   Int_t    ReadArrayDouble32(Double_t  *&d, TStreamerElement *ele=0);

   virtual   Int_t    ReadStaticArray(Bool_t    *b);
   virtual   Int_t    ReadStaticArray(Char_t    *c);
   virtual   Int_t    ReadStaticArray(UChar_t   *c);
   virtual   Int_t    ReadStaticArray(Short_t   *h);
   virtual   Int_t    ReadStaticArray(UShort_t  *h);
   virtual   Int_t    ReadStaticArray(Int_t     *i);
   virtual   Int_t    ReadStaticArray(UInt_t    *i);
   virtual   Int_t    ReadStaticArray(Long_t    *l);
   virtual   Int_t    ReadStaticArray(ULong_t   *l);
   virtual   Int_t    ReadStaticArray(Long64_t  *l);
   virtual   Int_t    ReadStaticArray(ULong64_t *l);
   virtual   Int_t    ReadStaticArray(Float_t   *f);
   virtual   Int_t    ReadStaticArray(Double_t  *d);
   virtual   Int_t    ReadStaticArrayDouble32(Double_t  *d, TStreamerElement *ele=0);

   virtual   void     ReadFastArray(Bool_t    *b, Int_t n);
   virtual   void     ReadFastArray(Char_t    *c, Int_t n);
   virtual   void     ReadFastArrayString(Char_t    *c, Int_t n);
   virtual   void     ReadFastArray(UChar_t   *c, Int_t n);
   virtual   void     ReadFastArray(Short_t   *h, Int_t n);
   virtual   void     ReadFastArray(UShort_t  *h, Int_t n);
   virtual   void     ReadFastArray(Int_t     *i, Int_t n);
   virtual   void     ReadFastArray(UInt_t    *i, Int_t n);
   virtual   void     ReadFastArray(Long_t    *l, Int_t n);
   virtual   void     ReadFastArray(ULong_t   *l, Int_t n);
   virtual   void     ReadFastArray(Long64_t  *l, Int_t n);
   virtual   void     ReadFastArray(ULong64_t *l, Int_t n);
   virtual   void     ReadFastArray(Float_t   *f, Int_t n);
   virtual   void     ReadFastArray(Double_t  *d, Int_t n);
   virtual   void     ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement *ele=0);
   virtual   void     ReadFastArray(void  *start , const TClass *cl, Int_t n=1, TMemberStreamer *s=0);
   virtual   void     ReadFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);

   virtual   void     WriteArray(const Bool_t    *b, Int_t n);
   virtual   void     WriteArray(const Char_t    *c, Int_t n);
   virtual   void     WriteArray(const UChar_t   *c, Int_t n);
   virtual   void     WriteArray(const Short_t   *h, Int_t n);
   virtual   void     WriteArray(const UShort_t  *h, Int_t n);
   virtual   void     WriteArray(const Int_t     *i, Int_t n);
   virtual   void     WriteArray(const UInt_t    *i, Int_t n);
   virtual   void     WriteArray(const Long_t    *l, Int_t n);
   virtual   void     WriteArray(const ULong_t   *l, Int_t n);
   virtual   void     WriteArray(const Long64_t  *l, Int_t n);
   virtual   void     WriteArray(const ULong64_t *l, Int_t n);
   virtual   void     WriteArray(const Float_t   *f, Int_t n);
   virtual   void     WriteArray(const Double_t  *d, Int_t n);
   virtual   void     WriteArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele=0);

   virtual   void     WriteFastArray(const Bool_t    *b, Int_t n);
   virtual   void     WriteFastArray(const Char_t    *c, Int_t n);
   virtual   void     WriteFastArrayString(const Char_t    *c, Int_t n);
   virtual   void     WriteFastArray(const UChar_t   *c, Int_t n);
   virtual   void     WriteFastArray(const Short_t   *h, Int_t n);
   virtual   void     WriteFastArray(const UShort_t  *h, Int_t n);
   virtual   void     WriteFastArray(const Int_t     *i, Int_t n);
   virtual   void     WriteFastArray(const UInt_t    *i, Int_t n);
   virtual   void     WriteFastArray(const Long_t    *l, Int_t n);
   virtual   void     WriteFastArray(const ULong_t   *l, Int_t n);
   virtual   void     WriteFastArray(const Long64_t  *l, Int_t n);
   virtual   void     WriteFastArray(const ULong64_t *l, Int_t n);
   virtual   void     WriteFastArray(const Float_t   *f, Int_t n);
   virtual   void     WriteFastArray(const Double_t  *d, Int_t n);
   virtual   void     WriteFastArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele=0);
   virtual   void     WriteFastArray(void  *start,  const TClass *cl, Int_t n=1, TMemberStreamer *s=0);
   virtual   Int_t    WriteFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);

   virtual   void     StreamObject(void *obj, const type_info &typeinfo);
   virtual   void     StreamObject(void *obj, const char *className);
   virtual   void     StreamObject(void *obj, const TClass *cl);
   virtual   void     StreamObject(TObject *obj);

   virtual   TBuffer  &operator>>(Bool_t    &b);
   virtual   TBuffer  &operator>>(Char_t    &c);
   virtual   TBuffer  &operator>>(UChar_t   &c);
   virtual   TBuffer  &operator>>(Short_t   &h);
   virtual   TBuffer  &operator>>(UShort_t  &h);
   virtual   TBuffer  &operator>>(Int_t     &i);
   virtual   TBuffer  &operator>>(UInt_t    &i);
   virtual   TBuffer  &operator>>(Long_t    &l);
   virtual   TBuffer  &operator>>(ULong_t   &l);
   virtual   TBuffer  &operator>>(Long64_t  &l);
   virtual   TBuffer  &operator>>(ULong64_t &l);
   virtual   TBuffer  &operator>>(Float_t   &f);
   virtual   TBuffer  &operator>>(Double_t  &d);
   virtual   TBuffer  &operator>>(Char_t    *c);

   virtual   TBuffer  &operator<<(Bool_t    b);
   virtual   TBuffer  &operator<<(Char_t    c);
   virtual   TBuffer  &operator<<(UChar_t   c);
   virtual   TBuffer  &operator<<(Short_t   h);
   virtual   TBuffer  &operator<<(UShort_t  h);
   virtual   TBuffer  &operator<<(Int_t     i);
   virtual   TBuffer  &operator<<(UInt_t    i);
   virtual   TBuffer  &operator<<(Long_t    l);
   virtual   TBuffer  &operator<<(ULong_t   l);
   virtual   TBuffer  &operator<<(Long64_t  l);
   virtual   TBuffer  &operator<<(ULong64_t l);
   virtual   TBuffer  &operator<<(Float_t   f);
   virtual   TBuffer  &operator<<(Double_t  d);
   virtual   TBuffer  &operator<<(const Char_t *c);

   //friend TBuffer  &operator>>(TBuffer &b, TObject *&obj);
   //friend TBuffer  &operator>>(TBuffer &b, const TObject *&obj);
   //friend TBuffer  &operator<<(TBuffer &b, const TObject *obj);

   static void    SetGlobalReadParam(Int_t mapsize);
   static void    SetGlobalWriteParam(Int_t mapsize);
   static Int_t   GetGlobalReadParam();
   static Int_t   GetGlobalWriteParam();
   static TClass *GetClass(const type_info &typeinfo);
   static TClass *GetClass(const char *className);

   ClassDef(TBuffer,0)  //Buffer base class used for serializing objects
};

//---------------------- TBuffer default external operators --------------------

#if !defined(R__CONCRETE_INPUT_OPERATOR)
#ifndef __CINT__

#if defined(R__SOLARIS) && defined(R__GNU)
#include <typeinfo>
#endif

template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj)
{
   // Read TObject derived classes from a TBuffer. Need to provide
   // custom version for non-TObject derived classes.

   // This operator has to be a templated and/or automatically
   // generated if we want to be able to check the type of the
   // incoming object. I.e. a operator>>(TBuffer &buf, TObject *&)
   // would not be sufficient to pass the information 'which class do we want'
   // since the pointer could be zero (so typeid(*obj) is not usable).

   TClass *cl = TBuffer::GetClass(typeid(Tmpl));
   obj = (Tmpl *) ( (void*) buf.ReadObjectAny(cl) );
   return buf;
}

template <class Tmpl> TBuffer &operator<<(TBuffer &buf, const Tmpl *obj)
{
   TClass *cl = (obj) ? TBuffer::GetClass(typeid(*obj)) : 0;
   buf.WriteObjectAny(obj, cl);
   return buf;
}
#else
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
template <class Tmpl> TBuffer &operator<<(TBuffer &buf, Tmpl *&obj);
#endif
#endif


//---------------------- TBuffer inlines ---------------------------------------

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Bool_t b)
{
   if (fBufCur + sizeof(UChar_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, b);
   return *this;
}

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
inline TBuffer &TBuffer::operator<<(ULong_t l)
{
   if (fBufCur + sizeof(ULong_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, l);
   return *this;
}

//______________________________________________________________________________
inline TBuffer &TBuffer::operator<<(Long64_t ll)
{
   if (fBufCur + sizeof(Long64_t) > fBufMax) Expand(2*fBufSize);

   tobuf(fBufCur, ll);
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
inline TBuffer &TBuffer::operator>>(Bool_t &b)
{
   frombuf(fBufCur, &b);
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

// Version for Long_t and ULong_t are in TBuffer.cxx

//______________________________________________________________________________
inline TBuffer &TBuffer::operator>>(Long64_t &ll)
{
   frombuf(fBufCur, &ll);
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
inline TBuffer &TBuffer::operator<<(ULong64_t ll)
   { return TBuffer::operator<<((Long64_t)ll); }

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
inline TBuffer &TBuffer::operator>>(ULong64_t &ll)
   { return TBuffer::operator>>((Long64_t&)ll); }

//______________________________________________________________________________
#if defined(R__TEMPLATE_OVERLOAD_BUG)
template <>
#endif
inline TBuffer &operator<<(TBuffer &buf, const TObject *obj)
   { buf.WriteObjectAny(obj, TObject::Class()); return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }
//______________________________________________________________________________
//inline TBuffer &operator>>(TBuffer &buf, const TObject *&obj)
//   { obj = buf.ReadObject(0); return buf; }

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
inline Int_t TBuffer::ReadArray(ULong64_t *&ll)
   { return TBuffer::ReadArray((Long64_t *&)ll); }

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
inline Int_t TBuffer::ReadStaticArray(ULong64_t *ll)
   { return TBuffer::ReadStaticArray((Long64_t *)ll); }

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
inline void TBuffer::ReadFastArray(ULong64_t *ll, Int_t n)
   { TBuffer::ReadFastArray((Long64_t *)ll, n); }

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
inline void TBuffer::WriteArray(const ULong64_t *ll, Int_t n)
   { TBuffer::WriteArray((const Long64_t *)ll, n); }

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
inline void TBuffer::WriteFastArray(const ULong64_t *ll, Int_t n)
   { TBuffer::WriteFastArray((const Long64_t *)ll, n); }

#endif
