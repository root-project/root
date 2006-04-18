// @(#)root/odbc:$Name:  $:$Id: TODBCStatement.cxx,v 1.1 2006/04/17 14:12:52 rdm Exp $
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//________________________________________________________________________
//
//  SQL statement class for Oracle
//
//  See TSQLStatement class documentation for more details
//
//________________________________________________________________________


#include "TODBCStatement.h"
#include "TDataType.h"
#include "snprintf.h"
#include "Riostream.h"

#include <sqlext.h>


ClassImp(TODBCStatement)

//______________________________________________________________________________
TODBCStatement::TODBCStatement(SQLHSTMT stmt, Int_t rowarrsize) :
   TSQLStatement()
{
   fHstmt = stmt;
   fBufferPreferredSize = rowarrsize;

   fBuffer = 0;
   fStatusBuffer = 0;
   fNumBuffers = 0;
   fBufferLength = 0;
   fBufferCounter = 0;

   fWorkingMode = 0;

   fNumParsProcessed = 0;
   fNumRowsFetched = 0;

   SQLSMALLINT   ParamsCount = 0;
   SQLRETURN retcode = SQLNumParams(fHstmt, &ParamsCount);
   if (ExtractErrors(retcode,"Constructor"))
      ParamsCount = 0;

   cout << "Num parameters = " << ParamsCount << endl;

   if (ParamsCount>0) {

      fWorkingMode = 1; // we are now using buffers for parameters
      fNumParsProcessed = 0;

      SQLSetStmtAttr(fHstmt, SQL_ATTR_PARAM_BIND_TYPE, SQL_PARAM_BIND_BY_COLUMN, 0);

      SQLUINTEGER setsize = fBufferPreferredSize;
      retcode = SQLSetStmtAttr(fHstmt, SQL_ATTR_PARAMSET_SIZE, (SQLPOINTER) setsize, 0);
      ExtractErrors(retcode,"Constructor");

      SQLUINTEGER getsize = 0;

      retcode = SQLGetStmtAttr(fHstmt, SQL_ATTR_PARAMSET_SIZE, &getsize, 0, 0);
      ExtractErrors(retcode,"Constructor");

      Int_t bufferlen = fBufferPreferredSize;

      // MySQL is not yet support array of parameters
      if (getsize<=1) bufferlen=1; else
      if (getsize!=setsize) {
         SQLSetStmtAttr(fHstmt, SQL_ATTR_PARAMSET_SIZE, (SQLPOINTER) 1, 0);
         bufferlen = 1;
      }

      SetNumBuffers(ParamsCount, bufferlen);

      SQLSetStmtAttr(fHstmt, SQL_ATTR_PARAM_STATUS_PTR, fStatusBuffer, 0);
      SQLSetStmtAttr(fHstmt, SQL_ATTR_PARAMS_PROCESSED_PTR, &fNumParsProcessed, 0);

/*
      for (int n=0;n<ParamsCount;n++) {
         SQLSMALLINT   DataType = 0, DecimalDigits = 0, Nullable = 0;
         SQLUINTEGER   ParamSize = 0;
         SQLDescribeParam(fHstmt, n + 1, &DataType, &ParamSize, &DecimalDigits, &Nullable);
         cout << "Par" << n << "  type = " << DataType
              << "  size = " << ParamSize << "  Digits = " <<  DecimalDigits << "  Null = " << Nullable << endl;
      }
*/

      // indicates that we are starting
      fBufferCounter = -1;
   }

   fNumRowsFetched = 0;
}

//______________________________________________________________________________
TODBCStatement::~TODBCStatement()
{
   Close();
}

//______________________________________________________________________________
void TODBCStatement::Close(Option_t *)
{
   SQLFreeHandle(SQL_HANDLE_STMT, fHstmt);

   fHstmt=0;
}

//______________________________________________________________________________
Bool_t TODBCStatement::Process()
{

   SQLRETURN retcode = SQL_SUCCESS;

   if (IsParSettMode()) {

      // check if we start filling buffers, but not complete it
      if (fBufferCounter>=0) {
         // if buffer used not fully, set smaller size of buffer arrays
         if ((fBufferCounter>0) && (fBufferCounter<fBufferLength-1)) {
            SQLUINTEGER setsize = fBufferCounter+1;
            SQLSetStmtAttr(fHstmt, SQL_ATTR_PARAMSET_SIZE, (SQLPOINTER) setsize, 0);
         }
         retcode = SQLExecute(fHstmt);
      }

      // after Process we finish working with parameters data,
      // if necessary, user can try to access resultset of statement
      fWorkingMode = 0;
      FreeBuffers();
      fBufferCounter = -1;
   } else {

      // just execute statement,
      // later one can try to access results of statement
      retcode = SQLExecute(fHstmt);
   }

   return !ExtractErrors(retcode, "Process");
}

//______________________________________________________________________________
Int_t TODBCStatement::GetNumAffectedRows()
{

   ODBCInt_t RowCount;
   SQLRETURN retcode = SQL_SUCCESS;

   retcode = SQLRowCount(fHstmt, &RowCount);

   if (ExtractErrors(retcode, "GetNumAffectedRows")) return -1;

   return RowCount;
}

//______________________________________________________________________________
Bool_t TODBCStatement::StoreResult()
{
   if (IsParSettMode()) {
      Error("StoreResult()","Call Process() method before");
      return kFALSE;
   }

   FreeBuffers();

   SQLSMALLINT ColumnCount = 0;

   SQLRETURN retcode = SQLNumResultCols(fHstmt, &ColumnCount);
   if (ExtractErrors(retcode, "StoreResult()")) return kFALSE;

//   cout << "Num results columns = " << ColumnCount << endl;

   if (ColumnCount==0) return kFALSE;

   SetNumBuffers(ColumnCount, fBufferPreferredSize);

   SQLUINTEGER arrsize = fBufferLength;

   SQLSetStmtAttr(fHstmt, SQL_ATTR_ROW_BIND_TYPE, SQL_BIND_BY_COLUMN, 0);
   SQLSetStmtAttr(fHstmt, SQL_ATTR_ROW_ARRAY_SIZE, (SQLPOINTER) arrsize, 0);
   SQLSetStmtAttr(fHstmt, SQL_ATTR_ROW_STATUS_PTR, fStatusBuffer, 0);
   SQLSetStmtAttr(fHstmt, SQL_ATTR_ROWS_FETCHED_PTR, &fNumRowsFetched, 0);

   for (int n=0;n<fNumBuffers;n++) {
      SQLCHAR ColumnName[1024];
      SQLSMALLINT NameLength;
      SQLSMALLINT DataType;
      ODBCUInt_t  ColumnSize;
      SQLSMALLINT DecimalDigits;
      SQLSMALLINT Nullable;

      retcode = SQLDescribeCol(fHstmt, n+1, ColumnName, 1024,
                               &NameLength, &DataType,
                               &ColumnSize, &DecimalDigits, &Nullable);

      BindColumn(n, DataType, ColumnSize);

      if (NameLength>0) {
         fBuffer[n].namebuffer = new char[NameLength+1];
         strcpy(fBuffer[n].namebuffer, (const char*) ColumnName);
      }
   }

   fNumRowsFetched = 0;

   fWorkingMode = 2;

   return kTRUE;
}

//______________________________________________________________________________
Int_t TODBCStatement::GetNumFields()
{
   return IsResultSet() ? fNumBuffers : -1;
}

//______________________________________________________________________________
const char* TODBCStatement::GetFieldName(Int_t nfield)
{
   if (!IsResultSet() || (nfield<0) || (nfield>=fNumBuffers)) return 0;

   return fBuffer[nfield].namebuffer;
}


//______________________________________________________________________________
Bool_t TODBCStatement::NextResultRow()
{
   if (!IsResultSet()) return kFALSE;

   if ((fNumRowsFetched==0) ||
       (1.*fBufferCounter >= 1.*(fNumRowsFetched-1))) {

      fBufferCounter = 0;
      fNumRowsFetched = 0;

      SQLRETURN retcode = SQLFetchScroll(fHstmt,SQL_FETCH_NEXT,0);

      if (ExtractErrors(retcode,"NextResultRow")) fNumRowsFetched=0;

      if (fNumRowsFetched==0) {
         fWorkingMode = 0;
         FreeBuffers();
      }

   } else
      fBufferCounter++;

   return IsResultSet();
}

//______________________________________________________________________________
Bool_t TODBCStatement::ExtractErrors(SQLRETURN retcode, const char* method)
{
   if ((retcode== SQL_SUCCESS) || (retcode == SQL_SUCCESS_WITH_INFO)) return kFALSE;

    SQLINTEGER i = 0;
    SQLINTEGER native;
    SQLCHAR state[ 7 ];
    SQLCHAR text[256];
    SQLSMALLINT len;
    SQLRETURN ret;
    do
    {
      ret = SQLGetDiagRec(SQL_HANDLE_STMT, fHstmt, ++i, state, &native, text,
                          sizeof(text), &len );
      if (ret == SQL_SUCCESS)
         Error(method, "%s:%ld:%ld:%s\n", state, i, native, text);
    }
    while( ret == SQL_SUCCESS );
    return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::NextIteration()
{
   if (!IsParSettMode() || (fBuffer==0) || (fBufferLength<=0)) return kFALSE;

   if (fBufferCounter>=fBufferLength-1) {
      SQLRETURN retcode = SQLExecute(fHstmt);
      if (ExtractErrors(retcode,"NextIteration")) return kFALSE;
      fBufferCounter = 0;
   } else
      fBufferCounter++;

   // probably, we do not need it, but anyway
   fStatusBuffer[fBufferCounter] = SQL_ROW_SUCCESS;

   return kTRUE;
}

//______________________________________________________________________________
Int_t TODBCStatement::GetNumParameters()
{
   return IsParSettMode() ? fNumBuffers : 0;
}

//______________________________________________________________________________
void TODBCStatement::SetNumBuffers(Int_t isize, Int_t ilen)
{
   FreeBuffers();

   fNumBuffers = isize;
   fBufferLength = ilen;
   fBufferCounter = 0;

   fBuffer = new ODBCBufferRec_t[fNumBuffers];
   for (Int_t n=0;n<fNumBuffers;n++) {
      fBuffer[n].roottype = 0;
      fBuffer[n].sqltype = 0;
      fBuffer[n].sqlctype = 0;
      fBuffer[n].buffer = 0;
      fBuffer[n].elementsize = 0;
      fBuffer[n].lenarray = 0;
      fBuffer[n].strbuffer = 0;
      fBuffer[n].namebuffer = 0;
   }

   fStatusBuffer = new SQLUSMALLINT[fBufferLength];

}


//______________________________________________________________________________
void TODBCStatement::FreeBuffers()
{
   if (fBuffer==0) return;
   for (Int_t n=0;n<fNumBuffers;n++) {
      if (fBuffer[n].buffer!=0)
        free(fBuffer[n].buffer);
      delete[] fBuffer[n].lenarray;
      delete[] fBuffer[n].strbuffer;
      delete[] fBuffer[n].namebuffer;
   }

   delete[] fStatusBuffer;
   delete[] fBuffer;
   fBuffer = 0;
   fNumBuffers = 0;
   fBufferLength = 0;
   fStatusBuffer = 0;
}

//______________________________________________________________________________
Bool_t TODBCStatement::BindColumn(Int_t ncol, SQLSMALLINT sqltype, SQLUINTEGER size)
{
   if ((ncol<0) || (ncol>=fNumBuffers)) return kFALSE;

   if (fBuffer[ncol].sqltype!=0) {
      Error("BindColumn","Column %d already binded", ncol);
      return kFALSE;
   }

   SQLSMALLINT sqlctype = 0;
   switch (sqltype) {
      case SQL_CHAR:
      case SQL_VARCHAR: sqlctype = SQL_C_CHAR; break;
      case SQL_LONGVARCHAR: Info("BindColumn","BIG VARCHAR not supported yet"); return kFALSE; break;

      case SQL_DECIMAL:   sqlctype = SQL_C_DOUBLE; break;
      case SQL_NUMERIC:   sqlctype = SQL_C_DOUBLE; break;
      case SQL_SMALLINT:  sqlctype = SQL_C_SLONG; break;
      case SQL_INTEGER:   sqlctype = SQL_C_SLONG; break;
      case SQL_FLOAT:     sqlctype = SQL_C_FLOAT; break;
      case SQL_REAL:
      case SQL_DOUBLE:    sqlctype = SQL_C_DOUBLE; break;
      case SQL_TINYINT:   sqlctype = SQL_C_STINYINT; break;
      case SQL_BIGINT:    sqlctype = SQL_C_SBIGINT; break;
      default: {
         Error("BindColumn","SQL type %d not supported",sqltype);
         return kFALSE;
      }

   }

   int elemsize = 0;

   switch (sqlctype) {
      case SQL_C_ULONG:    elemsize = sizeof(unsigned long int); break;
      case SQL_C_SLONG:    elemsize = sizeof(long int); break;
      case SQL_C_UBIGINT:  elemsize = sizeof(ULong64_t); break;
      case SQL_C_SBIGINT:  elemsize = sizeof(Long64_t); break;
      case SQL_C_USHORT:   elemsize = sizeof(unsigned short int); break;
      case SQL_C_SSHORT:   elemsize = sizeof(short int); break;
      case SQL_C_UTINYINT: elemsize = sizeof(unsigned char); break;
      case SQL_C_STINYINT: elemsize = sizeof(signed char); break;
      case SQL_C_FLOAT:    elemsize = sizeof(float); break;
      case SQL_C_DOUBLE:   elemsize = sizeof(double); break;
      case SQL_C_CHAR:     elemsize = size; break;

      default: {
         Error("BindColumn","SQL C Type %d is not supported",sqlctype);
         return kFALSE;
     }
   }

   fBuffer[ncol].roottype    = 0;
   fBuffer[ncol].sqltype     = sqltype;
   fBuffer[ncol].sqlctype    = sqlctype;
   fBuffer[ncol].buffer      = malloc(elemsize * fBufferLength);
   fBuffer[ncol].elementsize = elemsize;
   fBuffer[ncol].lenarray    = new ODBCInt_t[fBufferLength];

   SQLRETURN retcode =
      SQLBindCol(fHstmt, ncol+1, sqlctype, fBuffer[ncol].buffer,
                 elemsize,
                 fBuffer[ncol].lenarray);

   return !ExtractErrors(retcode, "BindColumn");
}

//______________________________________________________________________________
Bool_t TODBCStatement::BindParam(Int_t npar, Int_t roottype, Int_t size)
{
   if ((npar<0) || (npar>=fNumBuffers)) return kFALSE;

   if (fBuffer[npar].roottype!=0) {
      Error("SetParameterType","ParameterType for par %d already specified", npar);
      return kFALSE;
   }

   SQLSMALLINT sqltype = 0, sqlctype = 0;
   int elemsize = 0;

   switch (roottype) {
      case kUInt_t:     sqltype = SQL_INTEGER; sqlctype = SQL_C_ULONG; elemsize = sizeof(unsigned long int); break;
      case kInt_t:      sqltype = SQL_INTEGER; sqlctype = SQL_C_SLONG; elemsize = sizeof(long int); break;
      case kULong_t:    sqltype = SQL_INTEGER; sqlctype = SQL_C_ULONG; elemsize = sizeof(unsigned long int); break;
      case kLong_t:     sqltype = SQL_INTEGER; sqlctype = SQL_C_SLONG; elemsize = sizeof(long int); break;
      case kULong64_t:  sqltype = SQL_BIGINT;  sqlctype = SQL_C_UBIGINT; elemsize = sizeof(ULong64_t); break;
      case kLong64_t:   sqltype = SQL_BIGINT;  sqlctype = SQL_C_SBIGINT; elemsize = sizeof(Long64_t); break;
      case kUShort_t:   sqltype = SQL_SMALLINT;sqlctype = SQL_C_USHORT; elemsize = sizeof(unsigned short int); break;
      case kShort_t:    sqltype = SQL_SMALLINT;sqlctype = SQL_C_SSHORT; elemsize = sizeof(short int); break;
      case kUChar_t:    sqltype = SQL_TINYINT; sqlctype = SQL_C_UTINYINT; elemsize = sizeof(unsigned char); break;
      case kChar_t:     sqltype = SQL_TINYINT; sqlctype = SQL_C_STINYINT; elemsize = sizeof(signed char); break;
      case kBool_t:     sqltype = SQL_TINYINT; sqlctype = SQL_C_UTINYINT; elemsize = sizeof(unsigned char); break;
      case kFloat_t:    sqltype = SQL_FLOAT;   sqlctype = SQL_C_FLOAT; elemsize = sizeof(float); break;
      case kDouble_t:   sqltype = SQL_DOUBLE;  sqlctype = SQL_C_DOUBLE; elemsize = sizeof(double); break;
      case kDouble32_t: sqltype = SQL_DOUBLE;  sqlctype = SQL_C_DOUBLE; elemsize = sizeof(double); break;
      case kCharStar:   sqltype = SQL_CHAR;    sqlctype = SQL_C_CHAR; elemsize = size; break;
      default: {
         Error("SetParameterValue","Root type %d is not supported", roottype);
         return kFALSE;
     }
   }

   void* buffer = malloc(elemsize * fBufferLength);
   ODBCInt_t *lenarray = new ODBCInt_t[fBufferLength];

   SQLRETURN retcode =
      SQLBindParameter(fHstmt, npar+1, SQL_PARAM_INPUT,
                       sqlctype, sqltype, 0, 0,
                       buffer, elemsize, lenarray);

   if (ExtractErrors(retcode,"SetParameterType")) {
      free(buffer);
      delete[] lenarray;
      return kFALSE;
   }

   fBuffer[npar].roottype = roottype;
   fBuffer[npar].sqlctype = sqlctype;
   fBuffer[npar].sqltype = sqltype;
   fBuffer[npar].buffer = buffer;
   fBuffer[npar].elementsize = elemsize;
   fBuffer[npar].lenarray = lenarray;

   return kTRUE;
}

//______________________________________________________________________________
void* TODBCStatement::GetParAddr(Int_t npar, Int_t roottype, Int_t length)
{
   if ((fBuffer==0) || (npar<0) || (npar>=fNumBuffers) || (fBufferCounter<0)) return 0;

   if (fBuffer[npar].buffer==0) {
      if (IsParSettMode() && (roottype!=0) && (fBufferCounter==0))
         if (!BindParam(npar, roottype, length)) return 0;

      if (fBuffer[npar].buffer==0) return 0;
   }

   if (roottype!=0)
      if (fBuffer[npar].roottype!=roottype) return 0;

   return (char*)fBuffer[npar].buffer + fBufferCounter*fBuffer[npar].elementsize;
}


//______________________________________________________________________________
long double TODBCStatement::ConvertToNumeric(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   switch (fBuffer[npar].sqlctype) {
      case SQL_C_ULONG:    return *((unsigned long int*) addr); break;
      case SQL_C_SLONG:    return *((long int*) addr); break;
      case SQL_C_UBIGINT:  return *((ULong64_t*) addr); break;
      case SQL_C_SBIGINT:  return *((Long64_t*) addr); break;
      case SQL_C_USHORT:   return *((unsigned short int*) addr); break;
      case SQL_C_SSHORT:   return *((short int*) addr); break;
      case SQL_C_UTINYINT: return *((unsigned char*) addr); break;
      case SQL_C_STINYINT: return *((signed char*) addr); break;
      case SQL_C_FLOAT:    return *((float*) addr); break;
      case SQL_C_DOUBLE:   return *((double*) addr); break;
   }
   return 0;
}

//______________________________________________________________________________
const char* TODBCStatement::ConvertToString(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;
   if (fBuffer[npar].strbuffer==0)
      fBuffer[npar].strbuffer = new char[100];

   char* buf = fBuffer[npar].strbuffer;

   switch(fBuffer[npar].sqlctype) {
      case SQL_C_SLONG:   snprintf(buf,100,"%ld",*((long*) addr)); break;
      case SQL_C_ULONG:   snprintf(buf,100,"%lu",*((unsigned long*) addr)); break;
      case SQL_C_SBIGINT: snprintf(buf,100,"%lld",*((long long*) addr)); break;
      case SQL_C_UBIGINT: snprintf(buf,100,"%llu",*((unsigned long long*) addr)); break;
      case SQL_C_SSHORT:  snprintf(buf,100,"%hd",*((short*) addr)); break;
      case SQL_C_USHORT:  snprintf(buf,100,"%hu",*((unsigned short*) addr)); break;
      case SQL_C_STINYINT:snprintf(buf,100,"%d",*((char*) addr)); break;
      case SQL_C_UTINYINT:snprintf(buf,100,"%u",*((unsigned char*) addr)); break;
      case SQL_C_FLOAT:   snprintf(buf,100,"%f",*((float*) addr)); break;
      case SQL_C_DOUBLE:  snprintf(buf,100,"%f",*((double*) addr)); break;
      default: return 0;
   }

   return buf;
}

//______________________________________________________________________________
Int_t TODBCStatement::GetInt(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_SLONG)
      return (Int_t) *((long int*) addr);

   return (Int_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
UInt_t TODBCStatement::GetUInt(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_ULONG)
      return (UInt_t) *((unsigned long int*) addr);

   return (UInt_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Long_t TODBCStatement::GetLong(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_SLONG)
     return (Long_t) *((long int*) addr);

   return (Long_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Long64_t TODBCStatement::GetLong64(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_SBIGINT)
     return *((Long64_t*) addr);

   return (Long64_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
ULong64_t TODBCStatement::GetULong64(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_UBIGINT)
     return *((ULong64_t*) addr);

   return (ULong64_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Double_t TODBCStatement::GetDouble(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_DOUBLE)
     return *((double*) addr);

   return (Double_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
const char* TODBCStatement::GetString(Int_t npar)
{
   void* addr = GetParAddr(npar);
   if (addr==0) return 0;

   if (fBuffer[npar].sqlctype==SQL_C_CHAR) {
       // first check if string is null

       int len = fBuffer[npar].lenarray[fBufferCounter];

       if ((len == SQL_NULL_DATA) || (len==0)) return 0;

       char* res = (char*) addr;
       if (len < fBuffer[npar].elementsize) {
          *(res + len) = 0;
          return res;
       }

       if (len > fBuffer[npar].elementsize) {
          Error("getString","Problems with string size %d", len);
          return 0;
       }

       if (fBuffer[npar].strbuffer==0)
         fBuffer[npar].strbuffer = new char[len+1];

       strncpy(fBuffer[npar].strbuffer, res, len);

       res = fBuffer[npar].strbuffer;
       *(res + len) = 0;
       return res;
   }

   return ConvertToString(npar);
}


//______________________________________________________________________________
Bool_t TODBCStatement::SetInt(Int_t npar, Int_t value)
{
   void* addr = GetParAddr(npar, kInt_t);
   if (addr==0) return kFALSE;

   *((long int*) addr) = value;

   fBuffer[npar].lenarray[fBufferCounter] = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::SetUInt(Int_t npar, UInt_t value)
{
   void* addr = GetParAddr(npar, kUInt_t);
   if (addr==0) return kFALSE;

   *((unsigned long int*) addr) = value;

   fBuffer[npar].lenarray[fBufferCounter] = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::SetLong(Int_t npar, Long_t value)
{
   void* addr = GetParAddr(npar, kLong_t);
   if (addr==0) return kFALSE;

   *((long int*) addr) = value;

   fBuffer[npar].lenarray[fBufferCounter] = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::SetLong64(Int_t npar, Long64_t value)
{
   void* addr = GetParAddr(npar, kLong64_t);
   if (addr==0) return kFALSE;

   *((Long64_t*) addr) = value;

   fBuffer[npar].lenarray[fBufferCounter] = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::SetULong64(Int_t npar, ULong64_t value)
{
   void* addr = GetParAddr(npar, kULong64_t);
   if (addr==0) return kFALSE;

   *((ULong64_t*) addr) = value;

   fBuffer[npar].lenarray[fBufferCounter] = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::SetDouble(Int_t npar, Double_t value)
{
   void* addr = GetParAddr(npar, kDouble_t);
   if (addr==0) return kFALSE;

   *((double*) addr) = value;

   fBuffer[npar].lenarray[fBufferCounter] = 0;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TODBCStatement::SetString(Int_t npar, const char* value, Int_t maxsize)
{
   void* addr = GetParAddr(npar, kCharStar, maxsize);

   if (addr==0) return kFALSE;

   int len = value==0 ? 0 : strlen(value);

   if (len>=fBuffer[npar].elementsize) {
      len = fBuffer[npar].elementsize;
      strncpy((char*) addr, value, len);
      fBuffer[npar].lenarray[fBufferCounter] = len;
   } else
   if (len>0) {
      strcpy((char*) addr, value);
      fBuffer[npar].lenarray[fBufferCounter] = SQL_NTS;
   } else {
      *((char*) addr) = 0;
      fBuffer[npar].lenarray[fBufferCounter] = SQL_NTS;
   }

   return kTRUE;
}
