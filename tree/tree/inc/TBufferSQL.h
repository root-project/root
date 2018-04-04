// @(#)root/tree:$Id$
// Author: Philippe Canal 2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferSQL
#define ROOT_TBufferSQL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferSQL                                                           //
//                                                                      //
// Implement TBuffer for a SQL backend                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBufferText.h"
#include "TString.h"

class TSQLResult;
class TSQLRow;

class TBufferSQL : public TBufferText {

private:
   std::vector<Int_t>::const_iterator fIter;

   std::vector<Int_t>  *fColumnVec{nullptr};   //!
   TString             *fInsertQuery{nullptr}; //!
   TSQLRow            **fRowPtr{nullptr};      //!

   // TBuffer objects cannot be copied or assigned
   TBufferSQL(const TBufferSQL &);        // not implemented
   void operator=(const TBufferSQL &);    // not implemented

protected:

   virtual void WriteObjectClass(const void *actualObjStart, const TClass *actualClass, Bool_t cacheReuse);

   template <typename T>
   R__ALWAYS_INLINE void SqlWriteArray(const T *vname, Int_t n)
   {
      WriteInt(n);
      WriteFastArray(vname, n);
   }

   template <typename T>
   R__ALWAYS_INLINE Int_t SqlReadArray(T *&vname, Bool_t st = kFALSE)
   {
      Int_t n = 0;
      ReadInt(n);
      if (n && !vname && !st) vname = new T[n];
      ReadFastArray(vname, n);
      return n;
   }

public:
   TBufferSQL() = default;
   TBufferSQL(TBuffer::EMode mode, Int_t bufsiz, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr);
   ~TBufferSQL();

   // suppress TBuffer not used TBuffer methods

   virtual TClass *ReadClass(const TClass *cl = nullptr, UInt_t *objTag = nullptr);
   virtual void WriteClass(const TClass *cl);
   virtual Version_t ReadVersion(UInt_t *start = nullptr, UInt_t *bcnt = nullptr, const TClass *cl = nullptr);
   virtual UInt_t WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);
   virtual void *ReadObjectAny(const TClass *clCast);
   virtual void SkipObjectAny();
   virtual void IncrementLevel(TVirtualStreamerInfo *);
   virtual TVirtualStreamerInfo *GetInfo();
   virtual void SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type);
   virtual void DecrementLevel(TVirtualStreamerInfo *);
   virtual void ClassBegin(const TClass *, Version_t = -1);
   virtual void ClassEnd(const TClass *);
   virtual void ClassMember(const char *name, const char *typeName = nullptr, Int_t arrsize1 = -1, Int_t arrsize2 = -1);
   virtual void StreamObject(void *obj, const TClass *cl, const TClass *onFileClass = nullptr);
   using TBufferText::StreamObject;


   void ResetOffset();

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
   virtual   void     ReadStdString(std::string *s);
   using              TBuffer::ReadStdString;
   virtual   void     ReadCharStar(char* &s);

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
   virtual   void     WriteTString(const TString  &s);
   virtual   void     WriteStdString(const std::string *s);
   using              TBuffer::WriteStdString;
   virtual   void     WriteCharStar(char *s);

   virtual   void     WriteArray(const Bool_t *b, Int_t n) { SqlWriteArray(b, n); }
   virtual   void     WriteArray(const Char_t *c, Int_t n) { SqlWriteArray(c, n); }
   virtual   void     WriteArray(const UChar_t *c, Int_t n) { SqlWriteArray(c, n); }
   virtual   void     WriteArray(const Short_t *h, Int_t n) { SqlWriteArray(h, n); }
   virtual   void     WriteArray(const UShort_t *h, Int_t n)  { SqlWriteArray(h, n); }
   virtual   void     WriteArray(const Int_t *i, Int_t n)  { SqlWriteArray(i, n); }
   virtual   void     WriteArray(const UInt_t *i, Int_t n)  { SqlWriteArray(i, n); }
   virtual   void     WriteArray(const Long_t *l, Int_t n)  { SqlWriteArray(l, n); }
   virtual   void     WriteArray(const ULong_t *l, Int_t n)  { SqlWriteArray(l, n); }
   virtual   void     WriteArray(const Long64_t *l, Int_t n)  { SqlWriteArray(l, n); }
   virtual   void     WriteArray(const ULong64_t *l, Int_t n)  { SqlWriteArray(l, n); }
   virtual   void     WriteArray(const Float_t *f, Int_t n)  { SqlWriteArray(f, n); }
   virtual   void     WriteArray(const Double_t *d, Int_t n)  { SqlWriteArray(d, n); }

   virtual   Int_t    ReadArray(Bool_t *&b) { return SqlReadArray(b); }
   virtual   Int_t    ReadArray(Char_t *&c) { return SqlReadArray(c); }
   virtual   Int_t    ReadArray(UChar_t *&c)  { return SqlReadArray(c); }
   virtual   Int_t    ReadArray(Short_t *&h)  { return SqlReadArray(h); }
   virtual   Int_t    ReadArray(UShort_t *&h)  { return SqlReadArray(h); }
   virtual   Int_t    ReadArray(Int_t *&i)  { return SqlReadArray(i); }
   virtual   Int_t    ReadArray(UInt_t *&i)  { return SqlReadArray(i); }
   virtual   Int_t    ReadArray(Long_t *&l)  { return SqlReadArray(l); }
   virtual   Int_t    ReadArray(ULong_t *&l)  { return SqlReadArray(l); }
   virtual   Int_t    ReadArray(Long64_t *&l)  { return SqlReadArray(l); }
   virtual   Int_t    ReadArray(ULong64_t *&l)  { return SqlReadArray(l); }
   virtual   Int_t    ReadArray(Float_t *&f)  { return SqlReadArray(f); }
   virtual   Int_t    ReadArray(Double_t *&d)  { return SqlReadArray(d); }

   virtual   Int_t    ReadStaticArray(Bool_t *b) { return SqlReadArray(b, kTRUE); }
   virtual   Int_t    ReadStaticArray(Char_t *c) { return SqlReadArray(c, kTRUE); }
   virtual   Int_t    ReadStaticArray(UChar_t *c) { return SqlReadArray(c, kTRUE); }
   virtual   Int_t    ReadStaticArray(Short_t *h) { return SqlReadArray(h, kTRUE); }
   virtual   Int_t    ReadStaticArray(UShort_t *h) { return SqlReadArray(h, kTRUE); }
   virtual   Int_t    ReadStaticArray(Int_t *i) { return SqlReadArray(i, kTRUE); }
   virtual   Int_t    ReadStaticArray(UInt_t *i) { return SqlReadArray(i, kTRUE); }
   virtual   Int_t    ReadStaticArray(Long_t *l) { return SqlReadArray(l, kTRUE); }
   virtual   Int_t    ReadStaticArray(ULong_t *l) { return SqlReadArray(l, kTRUE); }
   virtual   Int_t    ReadStaticArray(Long64_t *l) { return SqlReadArray(l, kTRUE); }
   virtual   Int_t    ReadStaticArray(ULong64_t *l) { return SqlReadArray(l, kTRUE); }
   virtual   Int_t    ReadStaticArray(Float_t *f) { return SqlReadArray(f, kTRUE); }
   virtual   Int_t    ReadStaticArray(Double_t *d) { return SqlReadArray(d, kTRUE); }

   virtual   void     WriteFastArray(const Bool_t    *b, Int_t n);
   virtual   void     WriteFastArray(const Char_t    *c, Int_t n);
   virtual   void     WriteFastArrayString(const Char_t   *c, Int_t n);
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
   virtual   void     WriteFastArray(void  *start,  const TClass *cl, Int_t n=1, TMemberStreamer *s=nullptr);
   virtual   Int_t    WriteFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=nullptr);

   virtual   void     ReadFastArray(Bool_t    *, Int_t );
   virtual   void     ReadFastArray(Char_t    *, Int_t );
   virtual   void     ReadFastArrayString(Char_t   *, Int_t );
   virtual   void     ReadFastArray(UChar_t   *, Int_t );
   virtual   void     ReadFastArray(Short_t   *, Int_t );
   virtual   void     ReadFastArray(UShort_t  *, Int_t );
   virtual   void     ReadFastArray(Int_t     *, Int_t );
   virtual   void     ReadFastArray(UInt_t    *, Int_t );
   virtual   void     ReadFastArray(Long_t    *, Int_t );
   virtual   void     ReadFastArray(ULong_t   *, Int_t );
   virtual   void     ReadFastArray(Long64_t  *, Int_t );
   virtual   void     ReadFastArray(ULong64_t *, Int_t );
   virtual   void     ReadFastArray(Float_t   *, Int_t );
   virtual   void     ReadFastArray(Double_t  *, Int_t );
   virtual   void     ReadFastArray(void  *, const TClass *, Int_t n=1, TMemberStreamer *s=nullptr, const TClass *onFileClass=nullptr);
   virtual   void     ReadFastArray(void **, const TClass *, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=nullptr, const TClass *onFileClass=nullptr);

   ClassDef(TBufferSQL, 0); // Implementation of TBuffer to load and write to a SQL database

};

#endif


