// @(#)root/tree:$Name:  $:$Id: Exp $
// Author: Philippe Canal and al. 08/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferSQL                                                           //
//                                                                      //
// Implement TBuffer for a SQL backend                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "Riostream.h"
#include "TError.h"

#include "TBasketSQL.h"
#include "TBufferSQL.h"
#include "TSQLResult.h"
#include "TSQLRow.h"

ClassImp(TBufferSQL);

//________________________________________________________________________
TBufferSQL::TBufferSQL(EMode mode, vector<Int_t> *vc, 
                       TString *insert_query, TSQLRow ** r) : 
   TBuffer(mode),
   fColumnVec(vc), fInsertQuery(insert_query), fRowPtr(r) 
{
   int vc_size = (vc)?vc->size():0;

   fIter = fColumnVec->begin();
}

//________________________________________________________________________
TBufferSQL::TBufferSQL(EMode mode, Int_t bufsiz, vector<Int_t> *vc, 
                       TString *insert_query, TSQLRow ** r) : 
   TBuffer(mode,bufsiz), 
   fColumnVec(vc), fInsertQuery(insert_query), fRowPtr(r) 
{
   int vc_size = (vc)?vc->size():0;

   fIter = fColumnVec->begin();
}

//________________________________________________________________________
TBufferSQL::TBufferSQL(EMode mode, Int_t bufsiz, vector<Int_t> *vc, 
                       TString *insert_query, TSQLRow ** r,
                       void *buf, Bool_t adopt) : 
   TBuffer(mode,bufsiz,buf,adopt),
   fColumnVec(vc), fInsertQuery(insert_query), fRowPtr(r) 
{
   int vc_size = (vc)?vc->size():0;

   fIter = fColumnVec->begin();
}

//________________________________________________________________________
TBufferSQL::TBufferSQL() : fColumnVec(0),fRowPtr(0)
{
}

//________________________________________________________________________
TBufferSQL::~TBufferSQL() 
{
   delete fColumnVec;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Bool_t &b) 
{
   b = (Bool_t)atoi((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Char_t &c)
{
   c = (Char_t)atoi((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Short_t &h)
{
   h = (Short_t)atoi((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Int_t &i)
{
   i = atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Float_t &f)
{
   f = atof((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Long_t &l)
{
   l = atol((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Double_t &d)
{
   d = atof((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}


//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Bool_t    b)
{
   (*fInsertQuery) += b;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Char_t    c)
{
   (*fInsertQuery) += c;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Short_t   h)
{
   (*fInsertQuery) += h;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Int_t     i)
{
   (*fInsertQuery) += i;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Long_t    l)
{
   (*fInsertQuery) += l;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Float_t   f)
{
   (*fInsertQuery) += f;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Double_t  d)
{
   (*fInsertQuery) += d;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}
 
//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(UChar_t& uc)
{
   uc = (UChar_t)atoi((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(UShort_t& us)
{
   us = (UShort_t)atoi((*fRowPtr)->GetField(*fIter));
   
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(UInt_t& ui)
{
   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%u",&ui);
   if(code == 0) Error("operator>>(UInt_t&)","Error reading UInt_t");

   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(ULong_t& ul)
{
   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%lu",&ul);
   if(code == 0) Error("operator>>(ULong_t&)","Error reading ULong_t");

   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Long64_t &ll)
{
   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%lld",&ll);
   if(code == 0) Error("operator>>(ULong_t&)","Error reading Long64_t");

   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(ULong64_t &ull)
{
   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%llu",&ull);
   if(code == 0) Error("operator>>(ULong_t&)","Error reading ULong64_t");

   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator>>(Char_t *str)
{
   strcpy(str,(*fRowPtr)->GetField(*fIter));
   if (fIter != fColumnVec->end()) ++fIter;
   return *this;
}

// Method to send to database.

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(UChar_t uc)
{
   (*fInsertQuery) += uc;
   (*fInsertQuery) += ",";
   ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(UShort_t us)
{
   (*fInsertQuery) += us;
   (*fInsertQuery) += ",";
   ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(UInt_t ui)
{
   (*fInsertQuery) += ui;
   (*fInsertQuery) += ",";
   ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(ULong_t ul)
{
   (*fInsertQuery) += ul;
   (*fInsertQuery) += ",";
   ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(Long64_t ll)
{
   (*fInsertQuery) += ll;
   (*fInsertQuery) += ",";
   ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(ULong64_t ull)
{
   (*fInsertQuery) += ull;
   (*fInsertQuery) += ",";
   ++fIter;
   return *this;
}

//________________________________________________________________________
TBuffer& TBufferSQL::operator<<(const Char_t *str)
{
   (*fInsertQuery) += "\"";
   (*fInsertQuery) += str;
   (*fInsertQuery) += "\",";
    ++fIter;
   return *this;
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Bool_t *b, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += b[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Char_t *c, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += (Short_t)c[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArrayString(const Char_t *c, Int_t n)
{
   (*fInsertQuery) += "\"";
   (*fInsertQuery) += c;
   (*fInsertQuery) += "\",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const UChar_t *uc, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += uc[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Short_t *h, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += h[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const UShort_t *us, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += us[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Int_t     *ii, Int_t n)
{
    //   cerr << "Column: " <<*fIter << "   i:" << *ii << endl;
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ii[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const UInt_t *ui, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ui[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Long_t    *l, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery)+= l[i];
      (*fInsertQuery)+= ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const ULong_t   *ul, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ul[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Long64_t  *l, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += l[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const ULong64_t *ul, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ul[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Float_t   *f, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += f[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Double_t  *d, Int_t n)
{
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += d[i];
      (*fInsertQuery )+= ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(void*, const TClass*, Int_t, TMemberStreamer *)
{
   Fatal("riteFastArray(void*, const TClass*, Int_t, TMemberStreamer *)","Not implemented yet");
}

//________________________________________________________________________
Int_t TBufferSQL::WriteFastArray(void **, const TClass*, Int_t, Bool_t, TMemberStreamer*)
{
   Fatal("WriteFastArray(void **, const TClass*, Int_t, Bool_t, TMemberStreamer*)","Not implemented yet");
   return 0;
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Bool_t *b, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      b[i] = (Bool_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

/*
//________________________________________________________________________
void     TBufferSQL::ReadFastArray(Char_t    *c, Int_t n)
{
   if (n <= 0) return;
   Int_t l = sizeof(Char_t)*n;
   TString temp = (*fRowPtrs)->GetString(Collumn);
   Int_t min = l < temp.Length() ? l : temp.Length();
   memcpy(c,temp.Data(),min);
}
*/

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Char_t *c, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      c[i] = (Char_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArrayString(Char_t *c, Int_t n)
{
   strcpy(c,((*fRowPtr)->GetField(*fIter)));
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(UChar_t *uc, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      uc[i] = (UChar_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Short_t *s, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      s[i] = (Short_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(UShort_t *us, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      us[i] = (UShort_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(Int_t *in, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      in[i] = atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(UInt_t *ui, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      ui[i] = atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Long_t *l, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      l[i] = atol((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(ULong_t   *ul, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      (*this) >> ul[i];
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Long64_t  *ll, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      (*this) >> ll[i];
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(ULong64_t *ull, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      (*this) >> ull[i];
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Float_t   *f, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      f[i] = atof((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Double_t *d, Int_t n)
{
   for(int i=0; i<n; ++i) {  
      d[i] = atof((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArrayDouble32(Double_t  *, Int_t , TStreamerElement *)
{
   Fatal("ReadFastArrayDouble32(Double_t  *, Int_t , TStreamerElement *)","Not implemented yet");
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(void  *, const TClass *, Int_t, TMemberStreamer *)
{
   Fatal("ReadFastArray(void  *, const TClass *, Int_t, TMemberStreamer *)","Not implemented yet");
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(void **, const TClass *, Int_t, Bool_t, TMemberStreamer *)
{
   Fatal("ReadFastArray(void **, const TClass *, Int_t, Bool_t, TMemberStreamer *)","Not implemented yet");
}

//________________________________________________________________________
void TBufferSQL::ResetOffset() 
{
   fIter = fColumnVec->begin();
}

#if 0
//________________________________________________________________________
void TBufferSQL::insert_test(const Text_t* dsn, const Text_t* usr, 
                             const Text_t* pwd, const TString& tblname) 
{
   TString str;
   TString select = "select * from ";
   TString sql;
   TSQLStatement* stmt; 
   sql = select + "ins";
   
   con = gSQLDriverManager->GetConnection(dsn,usr,pwd);
   
   if(!con)
     printf("\n\n\nConnection NOT Successful\n\n\n");
   else
     printf("\n\n\nConnection Sucessful\n\n\n");
   
   
   
   stmt = con->CreateStatement(0, odbc::ResultSet::CONCUR_READ_ONLY);
   
   ptr = stmt->ExecuteQuery(sql.Data()); 
   if(!ptr) printf("No recorSet found!");	
   
   ptr->Next();
   ptr->MoveToInsertRow();
   cerr << "IsAfterLast(): " << ptr->IsAfterLast() << endl;
   ptr->UpdateInt(1, 5555);
   ptr->InsertRow();
   con->Commit();
   
   ptr1 = stmt->ExecuteQuery(sql.Data());
   
}
#endif
