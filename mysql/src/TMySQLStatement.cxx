// @(#)root/mysql:$Name:  $:$Id: TMySQLStatement.cxx,v 1.6 2006/06/30 06:36:35 brun Exp $
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


#if MYSQL_VERSION_ID >= 40100

//______________________________________________________________________________
TMySQLStatement::TMySQLStatement(MYSQL_STMT* stmt, Bool_t errout) :
   TSQLStatement(errout),
   fStmt(stmt),
   fNumBuffers(0),
   fBind(0),
   fBuffer(0),
   fWorkingMode(0)
{
   // Normal constructor 
   // Checks if statement contains parameters tags 
   
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
   // destructor
    
   Close();
}

//______________________________________________________________________________
void TMySQLStatement::Close(Option_t *)
{
   // Close statement

   if (fStmt)
      mysql_stmt_close(fStmt);

   fStmt = 0;

   FreeBuffers();
}


// Reset error and check that statement exists
#define CheckStmt(method, res)                          \
   {                                                    \
      ClearError();                                     \
      if (fStmt==0) {                                   \
         SetError(-1,"Statement handle is 0",method);   \
         return res;                                    \
      }                                                 \
   }

// check last mysql statement error code
#define CheckErrNo(method, force, res)                  \
   {                                                    \
      unsigned int stmterrno = mysql_stmt_errno(fStmt);     \
      if ((stmterrno!=0) || force) {                        \
         const char* stmterrmsg = mysql_stmt_error(fStmt);  \
         if (stmterrno==0) { stmterrno = 11111; stmterrmsg = "MySQL statement error"; } \
         SetError(stmterrno, stmterrmsg, method);               \
         return res;                                    \
      }                                                 \
   }


// check last mysql statement error code
#define CheckGetField(method, res)                      \
   {                                                    \
      ClearError();                                     \
      if (!IsResultSetMode()) {                         \
         SetError(-1,"Cannot get statement parameters",method); \
         return res;                                    \
      }                                                 \
      if ((npar<0) || (npar>=fNumBuffers)) {            \
         SetError(-1,Form("Invalid parameter number %d", npar),method); \
         return res;                                    \
      }                                                 \
   }

//______________________________________________________________________________
Bool_t TMySQLStatement::Process()
{
   // Process statement 
    
   CheckStmt("Process",kFALSE); 

   // if parameters was set, processing just means of closing parameters and variables
   if (IsSetParsMode()) {
      if (fIterationCount>=0)
         if (!NextIteration()) return kFALSE;
      fWorkingMode = 0;
      fIterationCount = -1;
      FreeBuffers();
      return kTRUE;
   }


   if (mysql_stmt_execute(fStmt)) 
      CheckErrNo("Process",kTRUE, kFALSE);

   return kTRUE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumAffectedRows()
{
   // Return number of affected rows after statement is processed 
    
   CheckStmt("Process", -1); 
   
   my_ulonglong res = mysql_stmt_affected_rows(fStmt);

   if (res == (my_ulonglong) -1)
      CheckErrNo("GetNumAffectedRows", kTRUE, -1);

   return (Int_t) res;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumParameters()
{
   // Return number of statement parameters 
    
   CheckStmt("GetNumParameters", -1); 

   Int_t res = mysql_stmt_param_count(fStmt);
   
   CheckErrNo("GetNumParameters", kFALSE, -1);
   
   return res;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::StoreResult()
{
   // Store result of statement processing to access them 
   // via GetInt(), GetDouble() and so on methods.
    
   CheckStmt("StoreResult", kFALSE); 
   if (fWorkingMode!=0) {
      SetError(-1,"Cannot store result for that statement","StoreResult");
      return kFALSE;
   }

   if (mysql_stmt_store_result(fStmt)) 
      CheckErrNo("StoreResult",kTRUE, kFALSE);

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
   if (mysql_stmt_bind_result(fStmt, fBind)) 
      CheckErrNo("StoreResult",kTRUE, kFALSE);

   fWorkingMode = 2;

   return kTRUE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumFields()
{
   // Return number of fields in result set 
    
   return IsResultSetMode() ? fNumBuffers : -1;
}

//______________________________________________________________________________
const char* TMySQLStatement::GetFieldName(Int_t nfield)
{
   // Returns field name in result set 
    
   if (!IsResultSetMode() || (nfield<0) || (nfield>=fNumBuffers)) return 0;

   return fBuffer[nfield].fFieldName;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::NextResultRow()
{
   // Shift cursor to nect row in result set
    
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
   // Increment iteration counter for statement, where parameter can be set.
   // Statement with parameters of previous iteration 
   // automatically will be applied to database

   ClearError(); 
    
   if (!IsSetParsMode() || (fBind==0)) {
      SetError(-1,"Cannot call for that statement","NextIteration");
      return kFALSE;
   }

   fIterationCount++;

   if (fIterationCount==0) return kTRUE;

   if (fNeedParBind) {
      fNeedParBind = kFALSE; 
      if (mysql_stmt_bind_param(fStmt, fBind)) 
         CheckErrNo("NextIteration",kTRUE, kFALSE);
   }

   if (mysql_stmt_execute(fStmt))
      CheckErrNo("NextIteration", kTRUE, kFALSE);

   return kTRUE;
}

//______________________________________________________________________________
void TMySQLStatement::FreeBuffers()
{
   // Release all buffers, used by statement 
    
   if (fBuffer) {
      for (Int_t n=0; n<fNumBuffers;n++) {
         free(fBuffer[n].fMem);
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
   // Allocate buffers for statement parameters/ result fields 
    
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
   // Convert field value to string 
    
   if (fBuffer[npar].fResNull) return 0;

   void* addr = fBuffer[npar].fMem;
   bool sig = fBuffer[npar].fSign;

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
   // Convert field to numeric value
    
   if (fBuffer[npar].fResNull) return 0;

   void* addr = fBuffer[npar].fMem;
   bool sig = fBuffer[npar].fSign;

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
Bool_t TMySQLStatement::IsNull(Int_t npar)
{
   // Checks if field value is null 
    
   CheckGetField("IsNull", kTRUE);
   
   return fBuffer[npar].fResNull;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetInt(Int_t npar)
{
   // Return field value as integer 
    
   CheckGetField("GetInt", 0);

   if ((fBuffer[npar].fSqlType==MYSQL_TYPE_LONG) && fBuffer[npar].fSign)
     return (Int_t) *((long*) fBuffer[npar].fMem);

   return (Int_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
UInt_t TMySQLStatement::GetUInt(Int_t npar)
{
   // Return field value as unsigned integer 

   CheckGetField("GetUInt", 0);

   if ((fBuffer[npar].fSqlType==MYSQL_TYPE_LONG) && !fBuffer[npar].fSign)
     return (UInt_t) *((unsigned long*) fBuffer[npar].fMem);

   return (UInt_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Long_t TMySQLStatement::GetLong(Int_t npar)
{
   // Return field value as long integer 

   CheckGetField("GetLong", 0);

   if ((fBuffer[npar].fSqlType==MYSQL_TYPE_LONG) && fBuffer[npar].fSign)
     return (Long_t) *((long*) fBuffer[npar].fMem);

   return (Long_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Long64_t TMySQLStatement::GetLong64(Int_t npar)
{
   // Return field value as 64-bit integer 

   CheckGetField("GetLong64", 0);

   if ((fBuffer[npar].fSqlType==MYSQL_TYPE_LONGLONG) && fBuffer[npar].fSign)
     return (Long64_t) *((long long*) fBuffer[npar].fMem);

   return (Long64_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
ULong64_t TMySQLStatement::GetULong64(Int_t npar)
{
   // Return field value as unsigned 64-bit integer 

   CheckGetField("GetULong64", 0);

   if ((fBuffer[npar].fSqlType==MYSQL_TYPE_LONGLONG) && !fBuffer[npar].fSign)
     return (ULong64_t) *((unsigned long long*) fBuffer[npar].fMem);

   return (ULong64_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
Double_t TMySQLStatement::GetDouble(Int_t npar)
{
   // Return field value as double 

   CheckGetField("GetDouble", 0);

   if (fBuffer[npar].fSqlType==MYSQL_TYPE_DOUBLE)
     return (Double_t) *((double*) fBuffer[npar].fMem);

   return (Double_t) ConvertToNumeric(npar);
}

//______________________________________________________________________________
const char *TMySQLStatement::GetString(Int_t npar)
{
   // Return field value as string 

   CheckGetField("GetString", 0);

   if ((fBind[npar].buffer_type==MYSQL_TYPE_STRING) ||
       (fBind[npar].buffer_type==MYSQL_TYPE_VAR_STRING))
      return (const char*) fBuffer[npar].fMem;

   return ConvertToString(npar);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetSQLParamType(Int_t npar, int sqltype, bool sig, int sqlsize)
{
   // Set parameter type to be used as buffer
   // Used in both setting data to database and retriving data from data base
   // Initialize proper MYSQL_BIND structure and allocate required buffers

   if ((npar<0) || (npar>=fNumBuffers)) return kFALSE;

   fBuffer[npar].fMem = 0;
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
      default: SetError(-1,"Nonsupported SQL type","SetSQLParamType"); return kFALSE;
   }

   fBuffer[npar].fMem = malloc(allocsize);
   fBuffer[npar].fSize = allocsize;
   fBuffer[npar].fSqlType = sqltype;
   fBuffer[npar].fSign = sig;

   fBind[npar].buffer_type = enum_field_types(sqltype);
   fBind[npar].buffer = fBuffer[npar].fMem;
   fBind[npar].buffer_length = allocsize;
   fBind[npar].is_null= &(fBuffer[npar].fResNull);
   fBind[npar].length = &(fBuffer[npar].fResLength);
   fBind[npar].is_unsigned = !sig;

   return kTRUE;
}

//______________________________________________________________________________
void *TMySQLStatement::BeforeSet(Int_t npar, Int_t sqltype, Bool_t sig, Int_t size)
{
   // Check boundary condition before setting value of parameter
   // Return address of parameter buffer
    
   ClearError(); 
    
   if (!IsSetParsMode()) {
      SetError(-1,"Cannot set parameter for statement","Set***");
      return 0;   
   }
   
   if ((npar<0) || (npar>=fNumBuffers)) {
      SetError(-1,Form("Invalid parameter number %d",npar), "Set***");
      return 0;
   }

   if ((fIterationCount==0) && (fBuffer[npar].fSqlType==0))
      if (!SetSQLParamType(npar, sqltype, sig, size)) {
         SetError(-1,"Cannot initialize parameter buffer","BeforeSet");
         return 0;
      }

   if ((fBuffer[npar].fSqlType!=sqltype) ||
      (fBuffer[npar].fSign != sig)) return 0;
      
   fBuffer[npar].fResNull = false;

   return fBuffer[npar].fMem;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetNull(Int_t npar)
{
   // Set NULL as parameter value
   // If NULL should be set for statement parameter during first iteration,
   // one should call before proper Set... method to identify type of argument for
   // the future. For instance, if one suppose to have double as type of parameter,
   // code should look like:
   //    stmt->SetDouble(2, 0.);
   //    stmt->SetNull(2);

   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG);

   if (addr!=0) 
      *((long*) addr) = 0;
      
   if ((npar>=0) && (npar<fNumBuffers)) 
      fBuffer[npar].fResNull = true;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetInt(Int_t npar, Int_t value)
{
   // Set parameter value as integer 
    
   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG);

   if (addr!=0)
      *((long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetUInt(Int_t npar, UInt_t value)
{
   // Set parameter value as unsigned integer 

   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG, kFALSE);

   if (addr!=0)
      *((unsigned long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetLong(Int_t npar, Long_t value)
{
   // Set parameter value as long integer 

   void* addr = BeforeSet(npar, MYSQL_TYPE_LONG);

   if (addr!=0)
      *((long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetLong64(Int_t npar, Long64_t value)
{
   // Set parameter value as 64-bit integer 

   void* addr = BeforeSet(npar, MYSQL_TYPE_LONGLONG);

   if (addr!=0)
      *((long long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetULong64(Int_t npar, ULong64_t value)
{
   // Set parameter value as unsigned 64-bit integer 

   void* addr = BeforeSet(npar, MYSQL_TYPE_LONGLONG, kFALSE);

   if (addr!=0)
      *((unsigned long long*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetDouble(Int_t npar, Double_t value)
{
   // Set parameter value as double

   void* addr = BeforeSet(npar, MYSQL_TYPE_DOUBLE, kFALSE);

   if (addr!=0)
      *((double*) addr) = value;

   return (addr!=0);
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetString(Int_t npar, const char* value, Int_t maxsize)
{
   // Set parameter value as string

   Int_t len = value ? strlen(value) : 0;

   void* addr = BeforeSet(npar, MYSQL_TYPE_STRING, true, maxsize);

   if (addr==0) return kFALSE;

   if (len >= fBuffer[npar].fSize) {
      free(fBuffer[npar].fMem);

      fBuffer[npar].fMem = malloc(len+1);
      fBuffer[npar].fSize = len + 1;

      fBind[npar].buffer = fBuffer[npar].fMem;
      fBind[npar].buffer_length = fBuffer[npar].fSize;

      addr = fBuffer[npar].fMem;
      fNeedParBind = kTRUE;
   }

   strcpy((char*) addr, value);

   fBuffer[npar].fResLength = len;

   return kTRUE;
}


#else

//______________________________________________________________________________
TMySQLStatement::TMySQLStatement(MYSQL_STMT*, Bool_t)
{
   // Normal constructor 
   // For MySQL version < 4.1 no statement is supported
}

//______________________________________________________________________________
TMySQLStatement::~TMySQLStatement()
{
   // destructor
}

//______________________________________________________________________________
void TMySQLStatement::Close(Option_t *)
{
   // Close statement
}

//______________________________________________________________________________
Bool_t TMySQLStatement::Process()
{
   // Process statement 
   
   return kFALSE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumAffectedRows()
{
   // Return number of affected rows after statement is processed 
   
   return 0;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumParameters()
{
   // Return number of statement parameters 
    
   return 0;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::StoreResult()
{
   // Store result of statement processing to access them 
   // via GetInt(), GetDouble() and so on methods.
    
   return kFALSE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetNumFields()
{
   // Return number of fields in result set 
    
   return 0;
}

//______________________________________________________________________________
const char* TMySQLStatement::GetFieldName(Int_t)
{
   // Returns field name in result set 
    
   return 0;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::NextResultRow()
{
   // Shift cursor to nect row in result set
    
   return kFALSE;
}


//______________________________________________________________________________
Bool_t TMySQLStatement::NextIteration()
{
   // Increment iteration counter for statement, where parameter can be set.
   // Statement with parameters of previous iteration 
   // automatically will be applied to database

   return kFALSE;
}

//______________________________________________________________________________
void TMySQLStatement::FreeBuffers()
{
   // Release all buffers, used by statement 
}

//______________________________________________________________________________
void TMySQLStatement::SetBuffersNumber(Int_t)
{
   // Allocate buffers for statement parameters/ result fields 
}

//______________________________________________________________________________
const char* TMySQLStatement::ConvertToString(Int_t)
{
   // Convert field value to string 
   
   return 0;
}

//______________________________________________________________________________
long double TMySQLStatement::ConvertToNumeric(Int_t)
{
   // Convert field to numeric value
    
   return 0; 
}

//______________________________________________________________________________
Bool_t TMySQLStatement::IsNull(Int_t)
{
   // Checks if field value is null 
   
   return kTRUE;
}

//______________________________________________________________________________
Int_t TMySQLStatement::GetInt(Int_t)
{
   // Return field value as integer 

   return 0;    
}

//______________________________________________________________________________
UInt_t TMySQLStatement::GetUInt(Int_t)
{
   // Return field value as unsigned integer 

   return 0;
}

//______________________________________________________________________________
Long_t TMySQLStatement::GetLong(Int_t)
{
   // Return field value as long integer 
   
   return 0;
}

//______________________________________________________________________________
Long64_t TMySQLStatement::GetLong64(Int_t)
{
   // Return field value as 64-bit integer 

   return 0;
}

//______________________________________________________________________________
ULong64_t TMySQLStatement::GetULong64(Int_t)
{
   // Return field value as unsigned 64-bit integer 

   return 0;
}

//______________________________________________________________________________
Double_t TMySQLStatement::GetDouble(Int_t)
{
   // Return field value as double 
   
   return 0.;
}

//______________________________________________________________________________
const char *TMySQLStatement::GetString(Int_t)
{
   // Return field value as string 

   return 0;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetSQLParamType(Int_t, int, bool, int)
{
   // Set parameter type to be used as buffer
   // Used in both setting data to database and retriving data from data base
   // Initialize proper MYSQL_BIND structure and allocate required buffers
   
   return kFALSE;
}

//______________________________________________________________________________
void *TMySQLStatement::BeforeSet(Int_t, Int_t, Bool_t, Int_t)
{
   // Check boundary condition before setting value of parameter
   // Return address of parameter buffer
   
   return 0;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetNull(Int_t)
{
   // Set NULL as parameter value
   // If NULL should be set for statement parameter during first iteration,
   // one should call before proper Set... method to identify type of argument for
   // the future. For instance, if one suppose to have double as type of parameter,
   // code should look like:
   //    stmt->SetDouble(2, 0.);
   //    stmt->SetNull(2);

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetInt(Int_t, Int_t)
{
   // Set parameter value as integer 

   return kFALSE;    
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetUInt(Int_t, UInt_t)
{
   // Set parameter value as unsigned integer 
   
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetLong(Int_t, Long_t)
{
   // Set parameter value as long integer 

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetLong64(Int_t, Long64_t)
{
   // Set parameter value as 64-bit integer 

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetULong64(Int_t, ULong64_t)
{
   // Set parameter value as unsigned 64-bit integer 

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetDouble(Int_t, Double_t)
{
   // Set parameter value as double

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TMySQLStatement::SetString(Int_t, const char*, Int_t)
{
   // Set parameter value as string

   return kFALSE;
}


#endif // MYSQL_VERSION_ID > 40100
