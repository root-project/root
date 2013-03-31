// @(#)root/io:$Id$
// Author: Rene Brun   17/01/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferFile
#define ROOT_TBufferFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferFile                                                          //
//                                                                      //
// The concrete implementation of TBuffer for writing/reading to/from a //
// ROOT file or socket.                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBuffer
#include "TBuffer.h"
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

class TVirtualStreamerInfo;
class TStreamerInfo;
class TStreamerElement;
class TClass;
class TExMap;
class TVirtualArray;
namespace TStreamerInfoActions {
   class TActionSequence;
}

class TBufferFile : public TBuffer {

protected:
   typedef std::vector<TStreamerInfo*> InfoList_t;

   Int_t           fMapCount;      //Number of objects or classes in map
   Int_t           fMapSize;       //Default size of map
   Int_t           fDisplacement;  //Value to be added to the map offsets
   UShort_t        fPidOffset;     //Offset to be added to the pid index in this key/buffer.
   TExMap         *fMap;           //Map containing object,offset pairs for reading/writing
   TExMap         *fClassMap;      //Map containing object,class pairs for reading
   TStreamerInfo  *fInfo;          //Pointer to TStreamerInfo object writing/reading the buffer
   InfoList_t      fInfoStack;     //Stack of pointers to the TStreamerInfos

   static Int_t    fgMapSize;      //Default map size for all TBuffer objects

   // Default ctor
   TBufferFile() : TBuffer(), fMapCount(0), fMapSize(0),
               fDisplacement(0),fPidOffset(0), fMap(0), fClassMap(0),
     fInfo(0), fInfoStack() {}

   // TBuffer objects cannot be copied or assigned
   TBufferFile(const TBufferFile &);       // not implemented
   void operator=(const TBufferFile &);    // not implemented

   Int_t  CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss, const char* classname);
   void   CheckCount(UInt_t offset);
   UInt_t CheckObject(UInt_t offset, const TClass *cl, Bool_t readClass = kFALSE);

   virtual  void  WriteObjectClass(const void *actualObjStart, const TClass *actualClass);

public:
   enum { kMapSize = 503 };
   enum { kStreamedMemberWise = BIT(14) }; //added to version number to know if a collection has been stored member-wise
   enum { kNotDecompressed = BIT(15) };    //indicates a weird buffer, used by TBasket
   enum { kTextBasedStreaming = BIT(18) }; //indicates if buffer used for XML/SQL object streaming
   enum { kUser1 = BIT(21), kUser2 = BIT(22), kUser3 = BIT(23)}; //free for user

   TBufferFile(TBuffer::EMode mode);
   TBufferFile(TBuffer::EMode mode, Int_t bufsiz);
   TBufferFile(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE, ReAllocCharFun_t reallocfunc = 0);
   virtual ~TBufferFile();

   Int_t    GetMapCount() const { return fMapCount; }
   void     GetMappedObject(UInt_t tag, void* &ptr, TClass* &ClassPtr) const;
   void     MapObject(const TObject *obj, UInt_t offset = 1);
   void     MapObject(const void *obj, const TClass *cl, UInt_t offset = 1);
   void     Reset() { SetBufferOffset(); ResetMap(); }
   void     InitMap();
   void     ResetMap();
   void     SetReadParam(Int_t mapsize);
   void     SetWriteParam(Int_t mapsize);

   Bool_t   CheckObject(const TObject *obj);
   Bool_t   CheckObject(const void *obj, const TClass *ptrClass);

   virtual Int_t      GetVersionOwner() const;
   virtual Int_t      CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss);
   virtual Int_t      CheckByteCount(UInt_t startpos, UInt_t bcnt, const char *classname);
   virtual void       SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);

   virtual void       SkipVersion(const TClass *cl = 0);
   virtual Version_t  ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0, const TClass *cl = 0);
   virtual Version_t  ReadVersionForMemberWise(const TClass *cl = 0);
   virtual UInt_t     WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);
   virtual UInt_t     WriteVersionMemberWise(const TClass *cl, Bool_t useBcnt = kFALSE);

   virtual void      *ReadObjectAny(const TClass* cast);
   virtual void       SkipObjectAny();

   virtual void       TagStreamerInfo(TVirtualStreamerInfo* info);
   virtual void       IncrementLevel(TVirtualStreamerInfo* info);
   virtual void       SetStreamerElementNumber(Int_t) {}
   virtual void       DecrementLevel(TVirtualStreamerInfo*);
   TVirtualStreamerInfo  *GetInfo() {return (TVirtualStreamerInfo*)fInfo;}
   virtual void       ClassBegin(const TClass*, Version_t = -1) {}
   virtual void       ClassEnd(const TClass*) {}
   virtual void       ClassMember(const char*, const char* = 0, Int_t = -1, Int_t = -1) {}

   virtual Int_t      ReadBuf(void *buf, Int_t max);
   virtual void       WriteBuf(const void *buf, Int_t max);

   virtual char      *ReadString(char *s, Int_t max);
   virtual void       WriteString(const char *s);

   virtual TClass    *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   virtual void       WriteClass(const TClass *cl);

   virtual TObject   *ReadObject(const TClass *cl);
   virtual void       WriteObject(const TObject *obj);

   virtual Int_t      WriteObjectAny(const void *obj, const TClass *ptrClass);

   UShort_t GetPidOffset() const {
      // See comment in TBuffer::SetPidOffset
      return fPidOffset;
   }
   void     SetPidOffset(UShort_t offset);
   Int_t    GetBufferDisplacement() const { return fDisplacement; }
   void     SetBufferDisplacement() { fDisplacement = 0; }
   void     SetBufferDisplacement(Int_t skipped)
            { fDisplacement =  (Int_t)(Length() - skipped); }

   // basic types and arrays of basic types
   virtual   void     ReadFloat16 (Float_t *f, TStreamerElement *ele=0);
   virtual   void     WriteFloat16(Float_t *f, TStreamerElement *ele=0);
   virtual   void     ReadDouble32 (Double_t *d, TStreamerElement *ele=0);
   virtual   void     WriteDouble32(Double_t *d, TStreamerElement *ele=0);
   virtual   void     ReadWithFactor(Float_t *ptr, Double_t factor, Double_t minvalue);
   virtual   void     ReadWithNbits(Float_t *ptr, Int_t nbits);
   virtual   void     ReadWithFactor(Double_t *ptr, Double_t factor, Double_t minvalue);
   virtual   void     ReadWithNbits(Double_t *ptr, Int_t nbits);
   
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
   virtual   Int_t    ReadArrayFloat16(Float_t  *&f, TStreamerElement *ele=0);
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
   virtual   Int_t    ReadStaticArrayFloat16(Float_t  *f, TStreamerElement *ele=0);
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
   virtual   void     ReadFastArrayFloat16(Float_t  *f, Int_t n, TStreamerElement *ele=0);
   virtual   void     ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement *ele=0);
   virtual   void     ReadFastArrayWithFactor(Float_t *ptr, Int_t n, Double_t factor, Double_t minvalue) ;
   virtual   void     ReadFastArrayWithNbits(Float_t *ptr, Int_t n, Int_t nbits);
   virtual   void     ReadFastArrayWithFactor(Double_t *ptr, Int_t n, Double_t factor, Double_t minvalue);
   virtual   void     ReadFastArrayWithNbits(Double_t *ptr, Int_t n, Int_t nbits) ;
   virtual   void     ReadFastArray(void  *start , const TClass *cl, Int_t n=1, TMemberStreamer *s=0, const TClass* onFileClass=0 );
   virtual   void     ReadFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0, const TClass* onFileClass=0);

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
   virtual   void     WriteArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement *ele=0);
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
   virtual   void     WriteFastArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement *ele=0);
   virtual   void     WriteFastArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele=0);
   virtual   void     WriteFastArray(void  *start,  const TClass *cl, Int_t n=1, TMemberStreamer *s=0);
   virtual   Int_t    WriteFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);

   virtual   void     StreamObject(void *obj, const type_info &typeinfo, const TClass* onFileClass = 0 );
   virtual   void     StreamObject(void *obj, const char *className, const TClass* onFileClass = 0 );
   virtual   void     StreamObject(void *obj, const TClass *cl, const TClass* onFileClass = 0 );
   virtual   void     StreamObject(TObject *obj);

   virtual   void     ReadBool(Bool_t       &b);
   virtual   void     ReadChar(Char_t       &c);
   virtual   void     ReadUChar(UChar_t     &c);
   virtual   void     ReadShort(Short_t     &s);
   virtual   void     ReadUShort(UShort_t   &s);
   virtual   void     ReadInt(Int_t         &i);
   virtual   void     ReadUInt(UInt_t       &i);
   virtual   void     ReadLong(Long_t       &l);
   virtual   void     ReadULong(ULong_t     &l);
   virtual   void     ReadLong64(Long64_t   &l);
   virtual   void     ReadULong64(ULong64_t &l);
   virtual   void     ReadFloat(Float_t     &f);
   virtual   void     ReadDouble(Double_t   &d);
   virtual   void     ReadCharP(Char_t      *c);
   virtual   void     ReadTString(TString   &s);

   virtual   void     WriteBool(Bool_t       b);
   virtual   void     WriteChar(Char_t       c);
   virtual   void     WriteUChar(UChar_t     c);
   virtual   void     WriteShort(Short_t     s);
   virtual   void     WriteUShort(UShort_t   s);
   virtual   void     WriteInt(Int_t         i);
   virtual   void     WriteUInt(UInt_t       i);
   virtual   void     WriteLong(Long_t       l);
   virtual   void     WriteULong(ULong_t     l);
   virtual   void     WriteLong64(Long64_t   l);
   virtual   void     WriteULong64(ULong64_t l);
   virtual   void     WriteFloat(Float_t     f);
   virtual   void     WriteDouble(Double_t   d);
   virtual   void     WriteCharP(const Char_t *c);
   virtual   void     WriteTString(const TString &s);

   // Special basic ROOT objects and collections
   virtual   TProcessID *GetLastProcessID(TRefTable *reftable) const;
   virtual   UInt_t      GetTRefExecId();
   virtual   TProcessID *ReadProcessID(UShort_t pidf);
   virtual   UShort_t    WriteProcessID(TProcessID *pid);

   // Utilities for TStreamerInfo
   virtual   void   ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force);
   virtual   void   ForceWriteInfoClones(TClonesArray *a);
   virtual   Int_t  ReadClones (TClonesArray *a, Int_t nobjects, Version_t objvers);
   virtual   Int_t  WriteClones(TClonesArray *a, Int_t nobjects);

   // Utilities for TClass
   virtual   Int_t  ReadClassEmulated(const TClass *cl, void *object, const TClass *onfile_class);
   virtual   Int_t  ReadClassBuffer(const TClass *cl, void *pointer, const TClass *onfile_class);
   virtual   Int_t  ReadClassBuffer(const TClass *cl, void *pointer, Int_t version, UInt_t start, UInt_t count, const TClass *onfile_class);
   virtual   Int_t  WriteClassBuffer(const TClass *cl, void *pointer);
   
   // Utilites to streamer using sequences.
   Int_t ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *object);      
   Int_t ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection);      
   Int_t ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection);

   static void    SetGlobalReadParam(Int_t mapsize);
   static void    SetGlobalWriteParam(Int_t mapsize);
   static Int_t   GetGlobalReadParam();
   static Int_t   GetGlobalWriteParam();

   ClassDef(TBufferFile,0)  //concrete implementation of TBuffer for writing/reading to/from a ROOT file or socket.
};


//---------------------- TBufferFile inlines ---------------------------------------

//______________________________________________________________________________
inline void TBufferFile::WriteBool(Bool_t b)
{
   if (fBufCur + sizeof(UChar_t) > fBufMax) AutoExpand(fBufSize+sizeof(UChar_t));
   tobuf(fBufCur, b);
}

//______________________________________________________________________________
inline void TBufferFile::WriteChar(Char_t c)
{
   if (fBufCur + sizeof(Char_t) > fBufMax) AutoExpand(fBufSize+sizeof(Char_t));
   tobuf(fBufCur, c);
}

//______________________________________________________________________________
inline void TBufferFile::WriteUChar(UChar_t c)
{
   if (fBufCur + sizeof(UChar_t) > fBufMax) AutoExpand(fBufSize+sizeof(UChar_t));
   tobuf(fBufCur, (Char_t)c);
}

//______________________________________________________________________________
inline void TBufferFile::WriteShort(Short_t h)
{
   if (fBufCur + sizeof(Short_t) > fBufMax) AutoExpand(fBufSize+sizeof(Short_t));
   tobuf(fBufCur, h);
}

//______________________________________________________________________________
inline void TBufferFile::WriteUShort(UShort_t h)
{
   if (fBufCur + sizeof(UShort_t) > fBufMax) AutoExpand(fBufSize+sizeof(UShort_t));
   tobuf(fBufCur, (Short_t)h);
}

//______________________________________________________________________________
inline void TBufferFile::WriteInt(Int_t i)
{
   if (fBufCur + sizeof(Int_t) > fBufMax) AutoExpand(fBufSize+sizeof(Int_t));
   tobuf(fBufCur, i);
}

//______________________________________________________________________________
inline void TBufferFile::WriteUInt(UInt_t i)
{
   if (fBufCur + sizeof(UInt_t) > fBufMax) AutoExpand(fBufSize+sizeof(UInt_t));
   tobuf(fBufCur, (Int_t)i);
}

//______________________________________________________________________________
inline void TBufferFile::WriteLong(Long_t l)
{
   if (fBufCur + sizeof(Long_t) > fBufMax) AutoExpand(fBufSize+sizeof(Long_t));
   tobuf(fBufCur, l);
}

//______________________________________________________________________________
inline void TBufferFile::WriteULong(ULong_t l)
{
   if (fBufCur + sizeof(ULong_t) > fBufMax) AutoExpand(fBufSize+sizeof(ULong_t));
   tobuf(fBufCur, (Long_t)l);
}

//______________________________________________________________________________
inline void TBufferFile::WriteLong64(Long64_t ll)
{
   if (fBufCur + sizeof(Long64_t) > fBufMax) AutoExpand(fBufSize+sizeof(Long64_t));
   tobuf(fBufCur, ll);
}

//______________________________________________________________________________
inline void TBufferFile::WriteULong64(ULong64_t ll)
{
   if (fBufCur + sizeof(ULong64_t) > fBufMax) AutoExpand(fBufSize+sizeof(ULong64_t));
   tobuf(fBufCur, (Long64_t)ll);
}

//______________________________________________________________________________
inline void TBufferFile::WriteFloat(Float_t f)
{
   if (fBufCur + sizeof(Float_t) > fBufMax) AutoExpand(fBufSize+sizeof(Float_t));
   tobuf(fBufCur, f);
}

//______________________________________________________________________________
inline void TBufferFile::WriteDouble(Double_t d)
{
   if (fBufCur + sizeof(Double_t) > fBufMax) AutoExpand(fBufSize+sizeof(Double_t));
   tobuf(fBufCur, d);
}

//______________________________________________________________________________
inline void TBufferFile::WriteCharP(const Char_t *c)
{
   WriteString(c);
}

//______________________________________________________________________________
inline void TBufferFile::ReadBool(Bool_t &b)
{
   frombuf(fBufCur, &b);
}

//______________________________________________________________________________
inline void TBufferFile::ReadChar(Char_t &c)
{
   frombuf(fBufCur, &c);
}

//______________________________________________________________________________
inline void TBufferFile::ReadUChar(UChar_t &c)
{
   TBufferFile::ReadChar((Char_t &)c);
}

//______________________________________________________________________________
inline void TBufferFile::ReadShort(Short_t &h)
{
   frombuf(fBufCur, &h);
}

//______________________________________________________________________________
inline void TBufferFile::ReadUShort(UShort_t &h)
{
   TBufferFile::ReadShort((Short_t &)h);
}

//______________________________________________________________________________
inline void TBufferFile::ReadInt(Int_t &i)
{
   frombuf(fBufCur, &i);
}

//______________________________________________________________________________
inline void TBufferFile::ReadUInt(UInt_t &i)
{
   TBufferFile::ReadInt((Int_t &)i);
}


// in implementation file because special case with old version
//______________________________________________________________________________
//inline void TBufferFile::ReadLong(Long_t &ll)
//{
//   frombuf(fBufCur, &ll);
//}

//______________________________________________________________________________
inline void TBufferFile::ReadULong(ULong_t &ll)
{
   TBufferFile::ReadLong((Long_t&)ll);
}


//______________________________________________________________________________
inline void TBufferFile::ReadLong64(Long64_t &ll)
{
   frombuf(fBufCur, &ll);
}

//______________________________________________________________________________
inline void TBufferFile::ReadULong64(ULong64_t &ll)
{
   TBufferFile::ReadLong64((Long64_t &)ll);
}

//______________________________________________________________________________
inline void TBufferFile::ReadFloat(Float_t &f)
{
   frombuf(fBufCur, &f);
}

//______________________________________________________________________________
inline void TBufferFile::ReadDouble(Double_t &d)
{
   frombuf(fBufCur, &d);
}

//______________________________________________________________________________
inline void TBufferFile::ReadCharP(Char_t *c)
{
   ReadString(c, -1);
}

//______________________________________________________________________________
inline Int_t TBufferFile::ReadArray(UChar_t *&c)
   {  return TBufferFile::ReadArray((Char_t *&)c); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadArray(UShort_t *&h)
   {  return TBufferFile::ReadArray((Short_t *&)h); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadArray(UInt_t *&i)
   {  return TBufferFile::ReadArray((Int_t *&)i); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadArray(ULong_t *&l)
   {  return TBufferFile::ReadArray((Long_t *&)l); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadArray(ULong64_t *&ll)
   {  return TBufferFile::ReadArray((Long64_t *&)ll); }

//______________________________________________________________________________
inline Int_t TBufferFile::ReadStaticArray(UChar_t *c)
   {  return TBufferFile::ReadStaticArray((Char_t *)c); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadStaticArray(UShort_t *h)
   {  return TBufferFile::ReadStaticArray((Short_t *)h); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadStaticArray(UInt_t *i)
   {  return TBufferFile::ReadStaticArray((Int_t *)i); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadStaticArray(ULong_t *l)
   {  return TBufferFile::ReadStaticArray((Long_t *)l); }
//______________________________________________________________________________
inline Int_t TBufferFile::ReadStaticArray(ULong64_t *ll)
   {  return TBufferFile::ReadStaticArray((Long64_t *)ll); }

//______________________________________________________________________________
inline void TBufferFile::ReadFastArray(UChar_t *c, Int_t n)
   {        TBufferFile::ReadFastArray((Char_t *)c, n); }
//______________________________________________________________________________
inline void TBufferFile::ReadFastArray(UShort_t *h, Int_t n)
   {        TBufferFile::ReadFastArray((Short_t *)h, n); }
//______________________________________________________________________________
inline void TBufferFile::ReadFastArray(UInt_t *i, Int_t n)
   {        TBufferFile::ReadFastArray((Int_t *)i, n); }
//______________________________________________________________________________
inline void TBufferFile::ReadFastArray(ULong_t *l, Int_t n)
   {        TBufferFile::ReadFastArray((Long_t *)l, n); }
//______________________________________________________________________________
inline void TBufferFile::ReadFastArray(ULong64_t *ll, Int_t n)
   {        TBufferFile::ReadFastArray((Long64_t *)ll, n); }

//______________________________________________________________________________
inline void TBufferFile::WriteArray(const UChar_t *c, Int_t n)
   {        TBufferFile::WriteArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBufferFile::WriteArray(const UShort_t *h, Int_t n)
   {        TBufferFile::WriteArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBufferFile::WriteArray(const UInt_t *i, Int_t n)
   {        TBufferFile::WriteArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBufferFile::WriteArray(const ULong64_t *ll, Int_t n)
   {        TBufferFile::WriteArray((const Long64_t *)ll, n); }

//______________________________________________________________________________
inline void TBufferFile::WriteFastArray(const UChar_t *c, Int_t n)
   {        TBufferFile::WriteFastArray((const Char_t *)c, n); }
//______________________________________________________________________________
inline void TBufferFile::WriteFastArray(const UShort_t *h, Int_t n)
   {        TBufferFile::WriteFastArray((const Short_t *)h, n); }
//______________________________________________________________________________
inline void TBufferFile::WriteFastArray(const UInt_t *i, Int_t n)
   {        TBufferFile::WriteFastArray((const Int_t *)i, n); }
//______________________________________________________________________________
inline void TBufferFile::WriteFastArray(const ULong64_t *ll, Int_t n)
   {        TBufferFile::WriteFastArray((const Long64_t *)ll, n); }

#endif
