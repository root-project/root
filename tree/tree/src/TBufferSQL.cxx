// @(#)root/tree:$Id$
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
#include <stdlib.h>

ClassImp(TBufferSQL);

//________________________________________________________________________
TBufferSQL::TBufferSQL(TBuffer::EMode mode, vector<Int_t> *vc,
                       TString *insert_query, TSQLRow ** r) :
   TBufferFile(mode),
   fColumnVec(vc), fInsertQuery(insert_query), fRowPtr(r)
{
   // Constructor.

   fIter = fColumnVec->begin();
}

//________________________________________________________________________
TBufferSQL::TBufferSQL(TBuffer::EMode mode, Int_t bufsiz, vector<Int_t> *vc,
                       TString *insert_query, TSQLRow ** r) :
   TBufferFile(mode,bufsiz),
   fColumnVec(vc), fInsertQuery(insert_query), fRowPtr(r)
{
   // Constructor.

   fIter = fColumnVec->begin();
}

//________________________________________________________________________
TBufferSQL::TBufferSQL(TBuffer::EMode mode, Int_t bufsiz, vector<Int_t> *vc,
                       TString *insert_query, TSQLRow ** r,
                       void *buf, Bool_t adopt) :
   TBufferFile(mode,bufsiz,buf,adopt),
   fColumnVec(vc), fInsertQuery(insert_query), fRowPtr(r)
{
   // Constructor.

   fIter = fColumnVec->begin();
}

//________________________________________________________________________
TBufferSQL::TBufferSQL() : TBufferFile(), fColumnVec(0),fInsertQuery(0),fRowPtr(0)
{
   // Constructor.

}

//________________________________________________________________________
TBufferSQL::~TBufferSQL()
{
   // Destructo.

   delete fColumnVec;
}

//________________________________________________________________________
void TBufferSQL::ReadBool(Bool_t &b)
{
   // Operator>>

   b = (Bool_t)atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadChar(Char_t &c)
{
   // Operator>>

   c = (Char_t)atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadShort(Short_t &h)
{
   // Operator>>

   h = (Short_t)atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadInt(Int_t &i)
{
   // Operator>>

   i = atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadFloat(Float_t &f)
{
   // Operator>>

   f = atof((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadLong(Long_t &l)
{
   // Operator>>

   l = atol((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadDouble(Double_t &d)
{
   // Operator>>

   d = atof((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}


//________________________________________________________________________
void TBufferSQL::WriteBool(Bool_t    b)
{
   // Operator<<

   (*fInsertQuery) += b;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteChar(Char_t    c)
{
   // Operator<<

   (*fInsertQuery) += c;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteShort(Short_t   h)
{
   // Operator<<

   (*fInsertQuery) += h;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteInt(Int_t     i)
{
   // Operator<<

   (*fInsertQuery) += i;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteLong(Long_t    l)
{
   // Operator<<

   (*fInsertQuery) += l;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteFloat(Float_t   f)
{
   // Operator<<

   (*fInsertQuery) += f;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteDouble(Double_t  d)
{
   // Operator<<

   (*fInsertQuery) += d;
   (*fInsertQuery) += ",";
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadUChar(UChar_t& uc)
{
   // Operator>>

   uc = (UChar_t)atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadUShort(UShort_t& us)
{
   // Operator>>

   us = (UShort_t)atoi((*fRowPtr)->GetField(*fIter));

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadUInt(UInt_t& ui)
{
   // Operator>>

   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%u",&ui);
   if(code == 0) Error("operator>>(UInt_t&)","Error reading UInt_t");

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadULong(ULong_t& ul)
{
   // Operator>>

   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%lu",&ul);
   if(code == 0) Error("operator>>(ULong_t&)","Error reading ULong_t");

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadLong64(Long64_t &ll)
{
   // Operator>>

   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%lld",&ll);
   if(code == 0) Error("operator>>(ULong_t&)","Error reading Long64_t");

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadULong64(ULong64_t &ull)
{
   // Operator>>

   TString val = (*fRowPtr)->GetField(*fIter);
   Int_t code = sscanf(val.Data(), "%llu",&ull);
   if(code == 0) Error("operator>>(ULong_t&)","Error reading ULong64_t");

   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadCharP(Char_t *str)
{
   // Operator>>

   strcpy(str,(*fRowPtr)->GetField(*fIter));  // Legacy interface, we have no way to know the user's buffer size ....
   if (fIter != fColumnVec->end()) ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadTString(TString   &s)
{
   // Read a TString

   TBufferFile::ReadTString(s);
}

//________________________________________________________________________
void TBufferSQL::WriteTString(const TString   &s)
{
   // Write a TString

   TBufferFile::WriteTString(s);
}

//________________________________________________________________________
void TBufferSQL::ReadStdString(std::string *s)
{
   // Read a std::string

   TBufferFile::ReadStdString(s);
}

//________________________________________________________________________
void TBufferSQL::WriteStdString(const std::string *s)
{
   // Write a std::string

   TBufferFile::WriteStdString(s);
}

//________________________________________________________________________
void TBufferSQL::ReadCharStar(char* &s)
{
   // Read a char* string

   TBufferFile::ReadCharStar(s);
}

//________________________________________________________________________
void TBufferSQL::WriteCharStar(char *s)
{
   // Write a char* string

   TBufferFile::WriteCharStar(s);
}


// Method to send to database.

//________________________________________________________________________
void TBufferSQL::WriteUChar(UChar_t uc)
{
   // Operator<<

   (*fInsertQuery) += uc;
   (*fInsertQuery) += ",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteUShort(UShort_t us)
{
   // Operator<<

   (*fInsertQuery) += us;
   (*fInsertQuery) += ",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteUInt(UInt_t ui)
{
   // Operator<<

   (*fInsertQuery) += ui;
   (*fInsertQuery) += ",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteULong(ULong_t ul)
{
   // Operator<<

   (*fInsertQuery) += ul;
   (*fInsertQuery) += ",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteLong64(Long64_t ll)
{
   // Operator<<

   (*fInsertQuery) += ll;
   (*fInsertQuery) += ",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteULong64(ULong64_t ull)
{
   // Operator<<

   (*fInsertQuery) += ull;
   (*fInsertQuery) += ",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteCharP(const Char_t *str)
{
   // Operator<<

   (*fInsertQuery) += "\"";
   (*fInsertQuery) += str;
   (*fInsertQuery) += "\",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Bool_t *b, Int_t n)
{
   // WriteFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += b[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Char_t *c, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += (Short_t)c[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArrayString(const Char_t *c, Int_t /* n */)
{
   // WriteFastArray SQL implementation.

   (*fInsertQuery) += "\"";
   (*fInsertQuery) += c;
   (*fInsertQuery) += "\",";
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const UChar_t *uc, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += uc[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Short_t *h, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += h[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const UShort_t *us, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += us[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Int_t     *ii, Int_t n)
{
   // WriteFastArray SQL implementation.

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
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ui[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Long_t    *l, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery)+= l[i];
      (*fInsertQuery)+= ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const ULong_t   *ul, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ul[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Long64_t  *l, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += l[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const ULong64_t *ul, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += ul[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Float_t   *f, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += f[i];
      (*fInsertQuery) += ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(const Double_t  *d, Int_t n)
{
   // WriteFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      (*fInsertQuery) += d[i];
      (*fInsertQuery )+= ",";
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::WriteFastArray(void*, const TClass*, Int_t, TMemberStreamer *)
{
   // WriteFastArray SQL implementation.

   Fatal("riteFastArray(void*, const TClass*, Int_t, TMemberStreamer *)","Not implemented yet");
}

//________________________________________________________________________
Int_t TBufferSQL::WriteFastArray(void **, const TClass*, Int_t, Bool_t, TMemberStreamer*)
{
   // WriteFastArray SQL implementation.

   Fatal("WriteFastArray(void **, const TClass*, Int_t, Bool_t, TMemberStreamer*)","Not implemented yet");
   return 0;
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Bool_t *b, Int_t n)
{
   // ReadFastArray SQL implementation.

   for(int i=0; i<n; ++i) {
      b[i] = (Bool_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Char_t *c, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      c[i] = (Char_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArrayString(Char_t *c, Int_t /* n */)
{
   // ReadFastArray SQL implementation.
   strcpy(c,((*fRowPtr)->GetField(*fIter)));
   ++fIter;
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(UChar_t *uc, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      uc[i] = (UChar_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Short_t *s, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      s[i] = (Short_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(UShort_t *us, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      us[i] = (UShort_t)atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(Int_t *in, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      in[i] = atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(UInt_t *ui, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      ui[i] = atoi((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Long_t *l, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      l[i] = atol((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(ULong_t   *ul, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      (*this) >> ul[i];
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Long64_t  *ll, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      (*this) >> ll[i];
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(ULong64_t *ull, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      (*this) >> ull[i];
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Float_t   *f, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      f[i] = atof((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void TBufferSQL::ReadFastArray(Double_t *d, Int_t n)
{
   // ReadFastArray SQL implementation.
   for(int i=0; i<n; ++i) {
      d[i] = atof((*fRowPtr)->GetField(*fIter));
      ++fIter;
   }
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArrayFloat16(Float_t  *, Int_t , TStreamerElement *)
{
   // ReadFastArray SQL implementation.
   Fatal("ReadFastArrayFloat16(Float_t  *, Int_t , TStreamerElement *)","Not implemented yet");
}

//______________________________________________________________________________
void TBufferSQL::ReadFastArrayWithFactor(Float_t  *, Int_t , Double_t /* factor */, Double_t /* minvalue */)
{
   // read array of Float16_t from buffer

   Fatal("ReadFastArrayWithFactor(Float_t  *, Int_t, Double_t, Double_t)","Not implemented yet");
}

//______________________________________________________________________________
void TBufferSQL::ReadFastArrayWithNbits(Float_t  *, Int_t , Int_t /*nbits*/)
{
   // read array of Float16_t from buffer

   Fatal("ReadFastArrayWithNbits(Float_t  *, Int_t , Int_t )","Not implemented yet");
}

//______________________________________________________________________________
void TBufferSQL::ReadFastArrayWithFactor(Double_t  *, Int_t , Double_t /* factor */, Double_t /* minvalue */)
{
   // read array of Double32_t from buffer

   Fatal("ReadFastArrayWithFactor(Double_t  *, Int_t, Double_t, Double_t)","Not implemented yet");
}
//______________________________________________________________________________
void TBufferSQL::ReadFastArrayWithNbits(Double_t  *, Int_t , Int_t /*nbits*/)
{
   // read array of Double32_t from buffer

   Fatal("ReadFastArrayWithNbits(Double_t  *, Int_t , Int_t )","Not implemented yet");
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArrayDouble32(Double_t  *, Int_t , TStreamerElement *)
{
   // ReadFastArray SQL implementation.
   Fatal("ReadFastArrayDouble32(Double_t  *, Int_t , TStreamerElement *)","Not implemented yet");
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(void  *, const TClass *, Int_t, TMemberStreamer *, const TClass *)
{
   // ReadFastArray SQL implementation.
   Fatal("ReadFastArray(void  *, const TClass *, Int_t, TMemberStreamer *, const TClass *)","Not implemented yet");
}

//________________________________________________________________________
void     TBufferSQL::ReadFastArray(void **, const TClass *, Int_t, Bool_t, TMemberStreamer *, const TClass *)
{
   // ReadFastArray SQL implementation.
   Fatal("ReadFastArray(void **, const TClass *, Int_t, Bool_t, TMemberStreamer *, const TClass *)","Not implemented yet");
}

//________________________________________________________________________
void TBufferSQL::ResetOffset()
{
   // Reset Offset.
   fIter = fColumnVec->begin();
}

#if 0
//________________________________________________________________________
void TBufferSQL::insert_test(const char* dsn, const char* usr,
                             const char* pwd, const TString& tblname)
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
