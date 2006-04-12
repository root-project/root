// @(#)root/mysql:$Name:  $:$Id: TMySQLStatement.cxx,v 1.1 2006/02/6 10:00:44 rdm Exp $
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  SQL statement class for MySQL                                       //
//                                                                      //
//  See TSQLStatement class documentation for more details.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMySQLStatement.h"
#include "TDataType.h"
#include "snprintf.h"

ClassImp(TMySQLStatement)

//______________________________________________________________________________
TMySQLStatement::TMySQLStatement(MYSQL_STMT* stmt) :
   TSQLStatement(),
   fStmt(stmt),
   fNumBuffers(0),
   fBind(0),
   fBuffer(0),
   fWorkingMode(0)
{
   unsigned long paramcount = mysql_stmt_param_count(fStmt);

   if (paramcount>0) {
       fWorkingMode = 1;
       SetBuffersNumber(paramcount);
       fNeedParBind = kTRUE;
       fIterationCount = -1;
   }
}

//______________________________________________________________________________
TMySQLStatement::~TMySQLStatement()
{
   Close();
}

//______________________________________________________________________________
void TMySQLStatement::Close(Option_t *)
{
   // Close query result.

   if (fStmt)
      mysql_stmt_close(fStmt);

   fStmt = 0;

   FreeBuffers();

}

//______________________________________________________________________________
Bool_t TMySQLStatement::Process()
{
   if (fStmt==0) return kFALSE;

   // if parameters was set, processing just means of closing parameters and variables
   if (IsSetParsMode()) {
       if (fIterationCount>=0)
         if (!NextIteration()) return kFALSE;
       fWorkingMode = 0;
       fIterationCount = -1;
       FreeBuffers();
       return kTRUE;
   }


   if (mysql_stmt_execute(fStmt)) {
       Error("Process"," mysql_stmt_execute() failed %s", mysql_stmt_error(fStmt));
       return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumAffectedRows()
{
   if (fStmt==0) return -1;
   my_ulonglong res = mysql_stmt_affected_rows(fStmt);

   if (res == (my_ulonglong) -1) {
      Error("GetNumAffectedRows", mysql_stmt_error(fStmt));
      return -1;
   }

   return (Int_t) res;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumParameters()
{
   if (fStmt==0) return -1;

   return mysql_stmt_param_count(fStmt);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::StoreResult()
{
   if ((fStmt==0) || (fWorkingMode!=0)) return kFALSE;

   if (mysql_stmt_store_result(fStmt)) {
      Error("StoreResult","mysql_stmt_store_result() failed %s", mysql_stmt_error(fStmt));
      return kFALSE;
   }

   // allocate memeory for data reading from query
   MYSQL_RES* meta = mysql_stmt_result_metadata(fStmt);
   if (meta) {
      int count = mysql_num_fields(meta);

      SetBuffersNumber(count);

      MYSQL_FIELD *fields = mysql_fetch_fields(meta);

      for (int n=0;n<count;n++) {
         SetSQLParamType(n, fields[n].type, true, fields[n].length);
         if (fields[n].name!=0) {
            fBuffer[n].fFieldName = new char[strlen(fields[n].name)+1];
            strcpy(fBuffer[n].fFieldName, fields[n].name);
         }
      }

      mysql_free_result(meta);
   }

   if (fBind==0) return kFALSE;

   /* Bind the buffers */
   if (mysql_stmt_bind_result(fStmt, fBind)) {
      Error("StoreResult"," mysql_stmt_bind_result() failed %s", mysql_stmt_error(fStmt));
      return kFALSE;
   }

   fWorkingMode = 2;

   return kTRUE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumFields()
{
   return IsResultSetMode() ? fNumBuffers : -1;
}

//______________________________________________________________________________
const char* TMySQLStatement::GetFieldName(Int_t nfield)
{
   if (!IsResultSetMode() || (nfield<0) || (nfield>=fNumBuffers)) return 0;

   return fBuffer[nfield].fFieldName;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::NextResultRow()
{
   if ((fStmt==0) || !IsResultSetMode()) return kFALSE;

   Bool_t res = !mysql_stmt_fetch(fStmt);

   if (!res) {
      fWorkingMode = 0;
      FreeBuffers();
   }

   return res;
}


//______________________________________________________________________________
Bool_t TMySQLStatement::NextIteration()
{
   if (!IsSetParsMode() || (fBind==0)) return kFALSE;

   fIterationCount++;

   if (fIterationCount==0) return kTRUE;

   if (fNeedParBind) {
      if (mysql_stmt_bind_param(fStmt, fBind)) {
         Error("NextIteration","Cannot bind parameter structures to the statement");
         return kFALSE;
      }
      fNeedParBind = kFALSE;
   }

   if (mysql_stmt_execute(fStmt)) {
      Error("NextIteration", mysql_stmt_error(fStmt));
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TMySQLStatement::FreeBuffers()
{
   if (fBuffer) {
     for (Int_t n=0; n<fNumBuffers;n++) {
       free(fBuffer[n].buffer);
       if (fBuffer[n].fStrBuffer)
          delete[] fBuffer[n].fStrBuffer;
       if (fBuffer[n].fFieldName)
          delete[] fBuffer[n].fFieldName;
     }
     delete[] fBuffer;
   }

   if (fBind)
     delete[] fBind;

   fBuffer = 0;
   fBind = 0;
   fNumBuffers = 0;
}


//______________________________________________________________________________
void TMySQLStatement::SetBuffersNumber(Int_t numpars)
{
   FreeBuffers();
   if (numpars<=0) return;

   fNumBuffers = numpars;

   fBind = new MYSQL_BIND[fNumBuffers];
   memset(fBind, 0, sizeof(MYSQL_BIND)*fNumBuffers);

   fBuffer = new TParamData[fNumBuffers];
   memset(fBuffer, 0, sizeof(fNumBuffers)*fNumBuffers);
}

//______________________________________________________________________________
const char* TMySQLStatement::ConvertToString(Int_t npar)
{
   if (fBuffer[npar].fResNull) return 0;

   void* addr = fBuffer[npar].buffer;
   bool sig = fBuffer[npar].sign;

   if (addr==0) return 0;

   if ((fBind[npar].buffer_type==MYSQL_TYPE_STRING) ||
      (fBind[npar].buffer_type==MYSQL_TYPE_VAR_STRING))
     return (const char*) addr;

   if (fBuffer[npar].fStrBuffer==0)
     fBuffer[npar].fStrBuffer = new char[100];

   char* buf = fBuffer[npar].fStrBuffer;

   switch(fBind[npar].buffer_type) {
      case MYSQL_TYPE_LONG:
         if (sig) snprintf(buf,100,"%ld",*((long*) addr));
             else snprintf(buf,100,"%lu",*((unsigned long*) addr));
         break;
      case MYSQL_TYPE_LONGLONG:
         if (sig) snprintf(buf,100,"%lld",*((long long*) addr)); else
                  snprintf(buf,100,"%llu",*((unsigned long long*) addr));
         break;
      case MYSQL_TYPE_SHORT:
         if (sig) snprintf(buf,100,"%hd",*((short*) addr)); else
                  snprintf(buf,100,"%hu",*((unsigned short*) addr));
         break;
      case MYSQL_TYPE_TINY:
         if (sig) snprintf(buf,100,"%d",*((char*) addr)); else
                  snprintf(buf,100,"%u",*((unsigned char*) addr));
         break;
      case MYSQL_TYPE_FLOAT:
         snprintf(buf,100,"%f",*((float*) addr));
         break;
      case MYSQL_TYPE_DOUBLE:
         snprintf(buf,100,"%f",*((double*) addr));
         break;
      default:
         return 0;
   }
   return buf;
}

//______________________________________________________________________________
long double TMySQLStatement::ConvertToNumeric(Int_t npar)
{
   if (fBuffer[npar].fResNull) return 0;

   void* addr = fBuffer[npar].buffer;
   bool sig = fBuffer[npar].sign;

   if (addr==0) return 0;

   switch(fBind[npar].buffer_type) {
      case MYSQL_TYPE_LONG:
         if (sig) return *((long*) addr); else
                  return *((unsigned long*) addr);
         break;
      case MYSQL_TYPE_LONGLONG:
         if (sig) return *((long long*) addr); else
                  return *((unsigned long long*) addr);
         break;
      case MYSQL_TYPE_SHORT:
         if (sig) return *((short*) addr); else
                  return *((unsigned short*) addr);
         break;
      case MYSQL_TYPE_TINY:
         if (sig) return *((char*) addr); else
                  return *((unsigned char*) addr);
         break;
      case MYSQL_TYPE_FLOAT:
         return *((float*) addr);
         break;
      case MYSQL_TYPE_DOUBLE:
         return *((double*) addr);
         break;
      case MYSQL_TYPE_STRING:
      case MYSQL_TYPE_VAR_STRING: {
         const char* str = (const char*) addr;
         if ((str==0) || (*str==0)) return 0;
         long double buf = 0;
         sscanf(str,"%Lf",&buf);
         return buf;
         break;
      }
      default:
         return 0;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetInt(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fBuffer[npar].sqltype==MYSQL_TYPE_LONG) && fBuffer[npar].sign)
     return (Int_t) *((long*) fBuffer[npar].buffer);

   return (Int_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
UInt_t TMySQLStatement::GetUInt(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fBuffer[npar].sqltype==MYSQL_TYPE_LONG) && !fBuffer[npar].sign)
     return (UInt_t) *((unsigned long*) fBuffer[npar].buffer);

   return (UInt_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Long_t TMySQLStatement::GetLong(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fBuffer[npar].sqltype==MYSQL_TYPE_LONG) && fBuffer[npar].sign)
     return (Long_t) *((long*) fBuffer[npar].buffer);

   return (Long_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Long64_t TMySQLStatement::GetLong64(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fBuffer[npar].sqltype==MYSQL_TYPE_LONGLONG) && fBuffer[npar].sign)
     return (Long64_t) *((long long*) fBuffer[npar].buffer);

   return (Long64_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
ULong64_t TMySQLStatement::GetULong64(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fBuffer[npar].sqltype==MYSQL_TYPE_LONGLONG) && !fBuffer[npar].sign)
     return (ULong64_t) *((unsigned long long*) fBuffer[npar].buffer);

   return (ULong64_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Double_t TMySQLStatement::GetDouble(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0.;

   if (fBuffer[npar].sqltype==MYSQL_TYPE_DOUBLE)
     return (Double_t) *((double*) fBuffer[npar].buffer);

   return (Double_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
const char *TMySQLStatement::GetString(Int_t npar)
{
   if (!IsResultSetMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fBind[npar].buffer_type==MYSQL_TYPE_STRING) ||
      (fBind[npar].buffer_type==MYSQL_TYPE_VAR_STRING))
     return (const char*) fBuffer[npar].buffer;

   return ConvertToString(npar);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetSQLParamType(Int_t npar, int sqltype, bool sig, int sqlsize)
{
   if ((npar<0) || (npar>=fNumBuffers)) return kFALSE;

   fBuffer[npar].buffer = 0;
   fBuffer[npar].fSize = 0;
   fBuffer[npar].fResLength = 0;
   fBuffer[npar].fResNull = false;
   fBuffer[npar].fStrBuffer = 0;

   int allocsize = 0;

   switch (sqltype) {
      case MYSQL_TYPE_LONG:     allocsize = sizeof(long);  break;
      case MYSQL_TYPE_LONGLONG: allocsize = sizeof(long long); break;
      case MYSQL_TYPE_SHORT:    allocsize = sizeof(short); break;
      case MYSQL_TYPE_TINY:     allocsize = sizeof(char); break;
      case MYSQL_TYPE_FLOAT:    allocsize = sizeof(float); break;
      case MYSQL_TYPE_DOUBLE:   allocsize = sizeof(double); break;
      case MYSQL_TYPE_STRING:   allocsize = sqlsize > 256 ? sqlsize : 256; break;
      case MYSQL_TYPE_VAR_STRING: allocsize = sqlsize > 256 ? sqlsize : 256; break;
      default: printf("???? \n"); return kFALSE;
   }

   fBuffer[npar].buffer = malloc(allocsize);
   fBuffer[npar].fSize = allocsize;
   fBuffer[npar].sqltype = sqltype;
   fBuffer[npar].sign = sig;

   fBind[npar].buffer_type = enum_field_types(sqltype);
   fBind[npar].buffer = fBuffer[npar].buffer;
   fBind[npar].buffer_length = allocsize;
   fBind[npar].is_null= &(fBuffer[npar].fResNull);
   fBind[npar].length = &(fBuffer[npar].fResLength);
   fBind[npar].is_unsigned = !sig;

   return kTRUE;
}

//______________________________________________________________________________
void *TMySQLStatement::BeforeSet(Int_t npar, Int_t sqltype, Bool_t sig, Int_t size)
{
   if (!IsSetParsMode() || (npar<0) || (npar>=fNumBuffers)) return 0;

   if ((fIterationCount==0) && (fBuffer[npar].sqltype==0))
      if (!SetSQLParamType(npar, sqltype, sig, size)) return 0;

   if ((fBuffer[npar].sqltype!=sqltype) ||
      (fBuffer[npar].sign != sig)) return 0;

   return fBuffer[npar].buffer;
}


//______________________________________________________________________________
Bool_t TMySQLStatement::SetInt(Int_t npar, Int_t value)
{
   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG);

   if (addr!=0)
      *((long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetUInt(Int_t npar, UInt_t value)
{
   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG, kFALSE);

   if (addr!=0)
      *((unsigned long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetLong(Int_t npar, Long_t value)
{
   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG);

   if (addr!=0)
      *((long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetLong64(Int_t npar, Long64_t value)
{
   void* addr = BeforeSet(npar, MYSQL_TYPE_LONGLONG);

   if (addr!=0)
      *((long long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetULong64(Int_t npar, ULong64_t value)
{
   void* addr = BeforeSet(npar, MYSQL_TYPE_LONGLONG, kFALSE);

   if (addr!=0)
      *((unsigned long long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetDouble(Int_t npar, Double_t value)
{
   void* addr = BeforeSet(npar, MYSQL_TYPE_DOUBLE, kFALSE);

   if (addr!=0)
      *((double*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetString(Int_t npar, const char* value, Int_t maxsize)
{
   Int_t len = value ? strlen(value) : 0;

   void* addr = BeforeSet(npar, MYSQL_TYPE_STRING, true, maxsize);

   if (addr==0) return kFALSE;

   if (len >= fBuffer[npar].fSize) {
       free(fBuffer[npar].buffer);

       fBuffer[npar].buffer = malloc(len+1);
       fBuffer[npar].fSize = len + 1;

       fBind[npar].buffer = fBuffer[npar].buffer;
       fBind[npar].buffer_length = fBuffer[npar].fSize;

       addr = fBuffer[npar].buffer;

       fNeedParBind = kTRUE;
   }

   strcpy((char*) addr, value);

   fBuffer[npar].fResLength = len;

   return kTRUE;
}
