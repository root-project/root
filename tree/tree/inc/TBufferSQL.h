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

#ifndef ROOT_TBufferFile
#include "TBufferFile.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TSQLResult;
class TSQLRow;

class TBufferSQL : public TBufferFile {

private:
   std::vector<Int_t>::const_iterator fIter;

   std::vector<Int_t>  *fColumnVec;   //!
   TString             *fInsertQuery; //!
   TSQLRow            **fRowPtr;      //!

   // TBuffer objects cannot be copied or assigned
   TBufferSQL(const TBufferSQL &);        // not implemented
   void operator=(const TBufferSQL &);    // not implemented

public:
   TBufferSQL();
   TBufferSQL(TBuffer::EMode mode, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr);
   TBufferSQL(TBuffer::EMode mode, Int_t bufsiz, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr);
   TBufferSQL(TBuffer::EMode mode, Int_t bufsiz, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr,void *buf, Bool_t adopt = kTRUE);
   ~TBufferSQL();

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
   virtual   void     WriteFastArray(void  *start,  const TClass *cl, Int_t n=1, TMemberStreamer *s=0);
   virtual   Int_t    WriteFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);

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
   virtual   void     ReadFastArrayFloat16(Float_t  *f, Int_t n, TStreamerElement *ele=0);
   virtual   void     ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement *ele=0);
   virtual   void     ReadFastArrayWithFactor(Float_t *ptr, Int_t n, Double_t factor, Double_t minvalue) ;
   virtual   void     ReadFastArrayWithNbits(Float_t *ptr, Int_t n, Int_t nbits);
   virtual   void     ReadFastArrayWithFactor(Double_t *ptr, Int_t n, Double_t factor, Double_t minvalue);
   virtual   void     ReadFastArrayWithNbits(Double_t *ptr, Int_t n, Int_t nbits) ;
   virtual   void     ReadFastArray(void  *, const TClass *, Int_t n=1, TMemberStreamer *s=0, const TClass *onFileClass=0);
   virtual   void     ReadFastArray(void **, const TClass *, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0, const TClass *onFileClass=0);

   ClassDef(TBufferSQL, 1); // Implementation of TBuffer to load and write to a SQL database

};

#endif


