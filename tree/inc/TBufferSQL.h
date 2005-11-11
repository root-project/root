// @(#)root/tree:$Name:  $:$Id: TBufferSQL.h,v 1.2 2005/09/03 02:21:32 pcanal Exp $
// Author: Philippe Canal 2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBUFFERSQL
#define ROOT_TBUFFERSQL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferSQL                                                           //
//                                                                      //
// Implement TBuffer for a SQL backend                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBuffer.h"
#include "TString.h"

class TSQLResult;
class TSQLRow;

class TBufferSQL : public TBuffer {

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
   TBufferSQL(EMode mode, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr);
   TBufferSQL(EMode mode, Int_t bufsiz, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr);
   TBufferSQL(EMode mode, Int_t bufsiz, std::vector<Int_t> *vc, TString *insert_query, TSQLRow **rowPtr,void *buf, Bool_t adopt = kTRUE);
   ~TBufferSQL();
   
   void ResetOffset();

   virtual TBuffer    &operator>>(Bool_t    &);
   virtual TBuffer    &operator>>(Char_t    &);
   virtual TBuffer    &operator>>(UChar_t   &);
   virtual TBuffer    &operator>>(Short_t   &);
   virtual TBuffer    &operator>>(UShort_t  &);
   virtual TBuffer    &operator>>(Int_t     &);
   virtual TBuffer    &operator>>(UInt_t    &);
   virtual TBuffer    &operator>>(Float_t   &);
   virtual TBuffer    &operator>>(Long_t    &);
   virtual TBuffer    &operator>>(ULong_t   &);
   virtual TBuffer    &operator>>(Long64_t  &);
   virtual TBuffer    &operator>>(ULong64_t &);
   virtual TBuffer    &operator>>(Double_t  &);
   virtual TBuffer    &operator>>(Char_t    *);
  

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
   virtual   void     ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement *ele=0);
   virtual   void     ReadFastArray(void  *, const TClass *, Int_t n=1, TMemberStreamer *s=0);
   virtual   void     ReadFastArray(void **, const TClass *, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);
   
   ClassDef(TBufferSQL, 1); // Implementation of TBuffer to load and write to a SQL database

};

#endif


