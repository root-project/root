// @(#)root/sqlite:$Id$
// Author: o.freyermuth <o.f@cern.ch>, 01/06/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  SQL statement class for SQLite.                                     //
//                                                                      //
//  See TSQLStatement class documentation for more details.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSQLiteStatement.h"
#include "TDataType.h"
#include "TDatime.h"
#include "TTimeStamp.h"

#include <stdlib.h>

ClassImp(TSQLiteStatement)

//______________________________________________________________________________
TSQLiteStatement::TSQLiteStatement(SQLite3_Stmt_t* stmt, Bool_t errout):
      TSQLStatement(errout),
      fStmt(stmt),
      fWorkingMode(0),
      fNumPars(0),
      fIterationCount(0)
{
   // Normal constructor.
   // Checks if statement contains parameters tags.

   unsigned long bindParamcount = sqlite3_bind_parameter_count(fStmt->fRes);

   if (bindParamcount > 0) {
      fWorkingMode = 1;
      fNumPars = bindParamcount;
   } else {
      fWorkingMode = 2;
      fNumPars = sqlite3_column_count(fStmt->fRes);
   }
}

//______________________________________________________________________________
TSQLiteStatement::~TSQLiteStatement()
{
   // Destructor.

   Close();
}

//______________________________________________________________________________
void TSQLiteStatement::Close(Option_t *)
{
   // Close statement.

   if (fStmt->fRes) {
      sqlite3_finalize(fStmt->fRes);
   }

   fStmt->fRes = 0;
   fStmt->fConn = 0;
   delete fStmt;
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

#define CheckErrNo(method, force, res)                  \
   {                                                    \
      int stmterrno = sqlite3_errcode(fStmt->fConn);    \
      if ((stmterrno!=0) || force) {                    \
         const char* stmterrmsg = sqlite3_errmsg(fStmt->fConn);  \
         if (stmterrno==0) { stmterrno = -1; stmterrmsg = "SQLite statement error"; } \
         SetError(stmterrno, stmterrmsg, method);       \
         return res;                                    \
      }                                                 \
   }

#define CheckGetField(method, res)                      \
   {                                                    \
      ClearError();                                     \
      if (!IsResultSetMode()) {                         \
         SetError(-1,"Cannot get statement parameters",method); \
         return res;                                    \
      }                                                 \
      if ((npar<0) || (npar>=fNumPars)) {     \
         SetError(-1,Form("Invalid parameter number %d", npar),method); \
         return res;                                    \
      }                                                 \
   }


Bool_t TSQLiteStatement::CheckBindError(const char *method, int res)
{
   if (res == SQLITE_RANGE) {
      SetError(-1, Form("SQLite parameter out of bounds, error: %d %s", res, sqlite3_errmsg(fStmt->fConn)), method);
      return kFALSE;
   }
   if (res != SQLITE_OK) {
      SetError(-1, Form("SQLite error code during parameter binding, error: %d %s", res, sqlite3_errmsg(fStmt->fConn)), method);
      return kFALSE;
   }
   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLiteStatement::Process()
{
   // Process statement.

   CheckStmt("Process", kFALSE);

   int res = sqlite3_step(fStmt->fRes);
   if ((res != SQLITE_DONE) && (res != SQLITE_ROW)) {
      SetError(-1, Form("SQLite error code during statement-stepping: %d %s", res, sqlite3_errmsg(fStmt->fConn)), "Process");
      return kFALSE;
   }

   // After a DONE-step, we have to reset, note this still KEEPS the parameters bound in SQLite,
   // real reset happens in finalize, but user can still reuse the query!
   if (res == SQLITE_DONE) {
      sqlite3_reset(fStmt->fRes);

      // If IsResultSetMode then this means we are done and should return kFALSE:
      if (IsResultSetMode()) {
         return kFALSE;
      }

      // If IsSetParsMode then this means we just stepped and should return kTRUE:
      if (IsSetParsMode()) {
         return kTRUE;
      }
   }

   if (res == SQLITE_ROW) {
      // Next row data retrieved, return kTRUE.
      return kTRUE;
   }

   return kFALSE;
}

//________________________________________________________________________
Int_t TSQLiteStatement::GetNumAffectedRows()
{
   // Return number of affected rows after statement is processed.
   // Indirect changes e.g. by triggers are not counted, only direct changes
   // from last completed statement are taken into account.

   CheckStmt("GetNumAffectedRows", kFALSE);

   return (Int_t) sqlite3_changes(fStmt->fConn);
}

//______________________________________________________________________________
Int_t TSQLiteStatement::GetNumParameters()
{
   // Return number of statement parameters.

   CheckStmt("GetNumParameters", -1);

   Int_t res = sqlite3_bind_parameter_count(fStmt->fRes);

   CheckErrNo("GetNumParameters", kFALSE, -1);

   return res;
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::StoreResult()
{
   // Store result of statement processing to access them
   // via GetInt(), GetDouble() and so on methods.
   // For SQLite, this is a NO-OP.

   fWorkingMode = 2;

   CheckStmt("StoreResult", kFALSE);

   return kTRUE;
}

//______________________________________________________________________________
Int_t TSQLiteStatement::GetNumFields()
{
   // Return number of fields in result set.

   return fNumPars;
}

//______________________________________________________________________________
const char* TSQLiteStatement::GetFieldName(Int_t nfield)
{
   // Returns field name in result set.

   if (!IsResultSetMode() || (nfield < 0) || (nfield >= sqlite3_column_count(fStmt->fRes))) {
      return 0;
   }

   return sqlite3_column_name(fStmt->fRes, nfield);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::NextResultRow()
{
   // Shift cursor to next row in result set.

   ClearError();

   if ((fStmt == 0) || !IsResultSetMode()) return kFALSE;

   if (fIterationCount == 0) {
      // The interface says user should call NextResultRow() before getting any data,
      // this makes no sense at least for SQLite.
      // We just return kTRUE here and only do something on second request.
      fIterationCount++;
      return kTRUE;
   }

   return Process();
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::NextIteration()
{
   // Increment iteration counter for statement, where parameter can be set.
   // Statement with parameters of previous iteration
   // automatically will be applied to database.
   // Actually a NO-OP for SQLite, as parameters stay bound when step-ping.

   ClearError();

   if (!IsSetParsMode()) {
      SetError(-1, "Cannot call for that statement", "NextIteration");
      return kFALSE;
   }

   if (fIterationCount == 0) {
      // The interface says user should call NextIteration() before binding any parameters,
      // this makes no sense at least for SQLite.
      // We just return kTRUE here and wait for data to really do something.
      fIterationCount++;
      return kTRUE;
   }

   fIterationCount++;

   return Process();
}

//______________________________________________________________________________
const char* TSQLiteStatement::ConvertToString(Int_t npar)
{
   // Convert field value to string.

   CheckGetField("ConvertToString", "");

   return reinterpret_cast<const char *>(sqlite3_column_text(fStmt->fRes, npar));
}

//______________________________________________________________________________
long double TSQLiteStatement::ConvertToNumeric(Int_t npar)
{
   // Convert field to numeric.

   CheckGetField("ConvertToNumeric", -1);

   return (long double) sqlite3_column_double(fStmt->fRes, npar);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::IsNull(Int_t npar)
{
   // Checks if field value is null.

   CheckGetField("IsNull", kFALSE);

   return (sqlite3_column_type(fStmt->fRes, npar) == SQLITE_NULL);
}

//______________________________________________________________________________
Int_t TSQLiteStatement::GetInt(Int_t npar)
{
   // Get integer.

   CheckGetField("GetInt", -1);

   return (Int_t) sqlite3_column_int(fStmt->fRes, npar);
}

//______________________________________________________________________________
UInt_t TSQLiteStatement::GetUInt(Int_t npar)
{
   // Get unsigned integer.

   CheckGetField("GetUInt", 0);

   return (UInt_t) sqlite3_column_int(fStmt->fRes, npar);
}

//______________________________________________________________________________
Long_t TSQLiteStatement::GetLong(Int_t npar)
{
   // Get long.

   CheckGetField("GetLong", -1);

   return (Long_t) sqlite3_column_int64(fStmt->fRes, npar);
}

//______________________________________________________________________________
Long64_t TSQLiteStatement::GetLong64(Int_t npar)
{
   // Get long64.

   CheckGetField("GetLong64", -1);

   return (Long64_t) sqlite3_column_int64(fStmt->fRes, npar);
}

//______________________________________________________________________________
ULong64_t TSQLiteStatement::GetULong64(Int_t npar)
{
   // Return field value as unsigned 64-bit integer

   CheckGetField("GetULong64", 0);

   return (ULong64_t) sqlite3_column_int64(fStmt->fRes, npar);
}

//______________________________________________________________________________
Double_t TSQLiteStatement::GetDouble(Int_t npar)
{
   // Return field value as double.

   CheckGetField("GetDouble", -1);

   return (Double_t) sqlite3_column_double(fStmt->fRes, npar);
}

//______________________________________________________________________________
const char *TSQLiteStatement::GetString(Int_t npar)
{
   // Return field value as string.

   CheckGetField("GetString", "");

   return reinterpret_cast<const char *>(sqlite3_column_text(fStmt->fRes, npar));
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::GetBinary(Int_t npar, void* &mem, Long_t& size)
{
   // Return field value as binary array.
   // Memory at 'mem' will be reallocated and size updated
   // to fit the data if not large enough.

   CheckGetField("GetBinary", kFALSE);

   // As we retrieve "as blob", we do NOT call sqlite3_column_text() before
   // sqlite3_column_bytes(), which might leave us with a non-zero terminated
   // data struture, but this should be okay for BLOB.
   size_t sz = sqlite3_column_bytes(fStmt->fRes, npar);
   if ((Long_t)sz > size) {
      delete [](unsigned char*) mem;
      mem = (void*) new unsigned char[sz];
   }
   size = sz;

   memcpy(mem, sqlite3_column_blob(fStmt->fRes, npar), sz);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::GetDate(Int_t npar, Int_t& year, Int_t& month, Int_t& day)
{
   // Return field value as date.

   CheckGetField("GetDate", kFALSE);

   TString val = reinterpret_cast<const char*>(sqlite3_column_text(fStmt->fRes, npar));
   TDatime d = TDatime(val.Data());
   year = d.GetYear();
   month = d.GetMonth();
   day = d.GetDay();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::GetTime(Int_t npar, Int_t& hour, Int_t& min, Int_t& sec)
{
   // Return field as time.

   CheckGetField("GetTime", kFALSE);

   TString val = reinterpret_cast<const char*>(sqlite3_column_text(fStmt->fRes, npar));
   TDatime d = TDatime(val.Data());
   hour = d.GetHour();
   min = d.GetMinute();
   sec = d.GetSecond();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::GetDatime(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec)
{
   // Return field value as date & time.

   CheckGetField("GetDatime", kFALSE);

   TString val = reinterpret_cast<const char*>(sqlite3_column_text(fStmt->fRes, npar));
   TDatime d = TDatime(val.Data());
   year = d.GetYear();
   month = d.GetMonth();
   day = d.GetDay();
   hour = d.GetHour();
   min = d.GetMinute();
   sec = d.GetSecond();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::GetTimestamp(Int_t npar, Int_t& year, Int_t& month, Int_t& day, Int_t& hour, Int_t& min, Int_t& sec, Int_t& frac)
{
   // Return field as timestamp.
   // Second fraction is in milliseconds, which is also the precision all date and time functions of sqlite use.

   CheckGetField("GetTimestamp", kFALSE);

   TString val = reinterpret_cast<const char*>(sqlite3_column_text(fStmt->fRes, npar));

   Ssiz_t p = val.Last('.');
   TSubString ts_part = val(0, p);

   TDatime d(ts_part.Data());
   year = d.GetYear();
   month = d.GetMonth();
   day = d.GetDay();
   hour = d.GetHour();
   min = d.GetMinute();
   sec = d.GetSecond();

   TSubString s_frac = val(p, val.Length() - p+1);
   frac=(Int_t) (atof(s_frac.Data())*1.E3);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetNull(Int_t npar)
{
   // Set NULL as parameter value.

   int res = sqlite3_bind_null(fStmt->fRes, npar + 1);

   return CheckBindError("SetNull", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetInt(Int_t npar, Int_t value)
{
   // Set parameter value as integer.

   int res = sqlite3_bind_int(fStmt->fRes, npar + 1, value);

   return CheckBindError("SetInt", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetUInt(Int_t npar, UInt_t value)
{
   // Set parameter value as unsigned integer.
   // Actually casted to signed integer, has to be re-casted upon read!

   int res = sqlite3_bind_int(fStmt->fRes, npar + 1, (Int_t)value);

   return CheckBindError("SetUInt", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetLong(Int_t npar, Long_t value)
{
   // Set parameter value as long.

   int res = sqlite3_bind_int64(fStmt->fRes, npar + 1, value);

   return CheckBindError("SetLong", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetLong64(Int_t npar, Long64_t value)
{
   // Set parameter value as 64-bit integer.

   int res = sqlite3_bind_int64(fStmt->fRes, npar + 1, value);

   return CheckBindError("SetLong64", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetULong64(Int_t npar, ULong64_t value)
{
   // Set parameter value as unsigned 64-bit integer.
   // Actually casted to signed integer, has to be re-casted upon read!

   int res = sqlite3_bind_int64(fStmt->fRes, npar + 1, (Long64_t)value);

   return CheckBindError("SetULong64", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetDouble(Int_t npar, Double_t value)
{
   // Set parameter value as double value.

   int res = sqlite3_bind_double(fStmt->fRes, npar + 1, value);

   return CheckBindError("SetDouble", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetString(Int_t npar, const char* value, Int_t maxsize)
{
   // Set parameter value as string.

   int res = sqlite3_bind_text(fStmt->fRes, npar + 1, value, maxsize, SQLITE_TRANSIENT);

   return CheckBindError("SetString", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetBinary(Int_t npar, void* mem, Long_t size, Long_t /*maxsize*/)
{
   // Set parameter value as binary data.
   // Maxsize is ignored for SQLite, we directly insert BLOB of size 'size'.
   // Negative size would cause undefined behaviour, so we refuse that.

   if (size < 0) {
      SetError(-1, "Passing negative value to size for BLOB to SQLite would cause undefined behaviour, refusing it!", "SetBinary");
      return kFALSE;
   }

   int res = sqlite3_bind_blob(fStmt->fRes, npar + 1, mem, (size_t)size, SQLITE_TRANSIENT);

   return CheckBindError("SetBinary", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetDate(Int_t npar, Int_t year, Int_t month, Int_t day)
{
   // Set parameter value as date.

   TDatime d = TDatime(year, month, day, 0, 0, 0);
   int res = sqlite3_bind_text(fStmt->fRes, npar + 1, (char*)d.AsSQLString(), -1, SQLITE_TRANSIENT);

   return CheckBindError("SetDate", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetTime(Int_t npar, Int_t hour, Int_t min, Int_t sec)
{
   // Set parameter value as time.

   TDatime d = TDatime(2000, 1, 1, hour, min, sec);

   int res = sqlite3_bind_text(fStmt->fRes, npar + 1, (char*)d.AsSQLString(), -1, SQLITE_TRANSIENT);

   return CheckBindError("SetTime", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetDatime(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec)
{
   // Set parameter value as date & time.

   TDatime d = TDatime(year, month, day, hour, min, sec);

   int res = sqlite3_bind_text(fStmt->fRes, npar + 1, (char*)d.AsSQLString(), -1, SQLITE_TRANSIENT);

   return CheckBindError("SetDatime", res);
}

//______________________________________________________________________________
Bool_t TSQLiteStatement::SetTimestamp(Int_t npar, Int_t year, Int_t month, Int_t day, Int_t hour, Int_t min, Int_t sec, Int_t frac)
{
   // Set parameter value as timestamp.
   // The second fraction has to be in milliseconds,
   // as all SQLite functions for date and time assume 3 significant digits.

   TDatime d(year,month,day,hour,min,sec);
   TString value;
   value.Form("%s.%03d", (char*)d.AsSQLString(), frac);

   int res = sqlite3_bind_text(fStmt->fRes, npar + 1, value.Data(), -1, SQLITE_TRANSIENT);

   return CheckBindError("SetTimestamp", res);
}
