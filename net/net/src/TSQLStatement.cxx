// @(#)root/net:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                                                                      
// TSQLStatement                                                        
//                                                                      
// Abstract base class defining SQL statements, which can be submitted
// in bulk to DB server.
//                                                                      
// This is alternative to TSQLServer::Query() method, which allows only pure
// text queries and pure text result in TSQLResult classes.
// TSQLStatement is designed to support following features:
//   - usage of basic data type (like int or double) as parameters 
//     in SQL statements
//   - bulk operation when inserting/updating/selecting data in data base
//   - uasge of basic data types when accessing result set of executed query
//
//
// 1. Creation of statement
// ======================================
// To create instance of TSQLStatement class, TSQLServer::Statement() method
// should be used. Depending of the driver, used for connection to ODBC,
// appropriate object instance will be created. For the moment there are
// three different implementation of TSQLStatement class: for MySQL, 
// Oracle and ODBC. Hopefully, support of ODBC will allows usage of
// statements for most existing RDBMS.
//
//   // first connect to data base
//   TSQLServer* serv = TSQLServer::Connect("mysql://hostname.domain:3306/test",
//                                          "user", "pass");
//   // check if connection is ok
//   if ((serv!=0) && serv->IsConnected()) {
//       // create statement instance     
//       TSQLStatement* stmt = serv->Statement("CREATE TABLE TESTTABLE (ID1 INT, ID2 INT, FFIELD VARCHAR(255), FVALUE VARCHAR(255))";
//       // process statement
//       stmt->Process();
//       // destroy object
//       delete stmt;
//   }
//   delete serv;
//
//
// 2. Insert data to data base
// ===============================================
// There is a special syntax of SQL queries, which allow to use values,
// provided as parameters. For instance, insert one row in TESTTABLE, created
// with previous example, one can simply execute query like: 
// 
//    serv->Query("INSERT INTO TESTTABLE VALUES (1, 2, \"name1\", \"value1\"");
//
// But when many (100-1000) rows should be inserted, each call of 
// TSQLServer::Query() method will cause communication loop with database
// server. As a result, insertion of data will takes too much time.
//
// TSQLStatement provides a mechanism to insert many rows at once. First of all,
// appropriate statement should be created:
//
//    TSQLStatement* stmt = serv->Statement("INSERT INTO TESTTABLE (ID1, ID2, FFIELD, FVALUE) VALUES (?, ?, ?, ?)", 100);
//
// Here question marks "?" indicates where statement parameters can be inserted.
// To specify values of parameters, SetInt(), SetDouble(), SetString() and other
// methods of TSQLStatement class should be used. Before parameters values
// can be specified, NextIteration() method of statement class should be called.
// For each new row first, NextIteration() called, that parameters values are
// specified. There is one limitation - once parameter set as integer via
// SetInt(), for all other rows should be specified as integer. At the end,
// TSQLStatement::Process() should be called. Here a small example:
//
//    // first, create statement  
//    TSQLStatement* stmt = serv->Statement("INSERT INTO TESTTABLE (ID1, ID2, FFIELD, FVALUE) VALUES (?, ?, ?, ?)", 100);
//
//    for (int n=0;n<357;n++) 
//       if (stmt->NextIteration()) {
//          stmt->SetInt(0, 123);
//          stmt->SetUInt(1, n+10);
//          stmt->SetString(2, Form("name %d",n), 200);
//          stmt->SetString(3, Form("value %d", n+10), 200);
//      }
//   
//     stmt->Process();
//     delete stmt;
//
// Second argument in TSQLServer::Statement() method specifies depth of 
// of buffers, used to keep parameter values (100 in example). It is not
// a limitation of rows number, which can be inserted with the statement.
// When buffers are filled, they will be submitted to database and can be
// reused again. This happens transparent to the user in NextIteration()
// method.
//
// Oracle and some ODBC drivers support buffering of parameter values and,
// as a result, bulk insert (update) operation. MySQL (native driver and
// MyODBC 3)  does not support such mode of operation, therefore adding
// new rows will result in communication loop to database.
//
// One should also mention difference between Oracle and ODBC SQL syntax for
// parameters. ODBC (and MySQL) uses question marks to specify position,
// where parameters should be inserted (as shown in the example). Oracle uses
// :1, :2 and so on marks for specify position of parameter 0, 1, and so on.
// Therefore, similar to example query will look like:
//
//    TSQLStatement* stmt = serv->Statement("INSERT INTO TESTTABLE (ID1, ID2, FFIELD, FVALUE) VALUES (:1, :2, :3, :4)", 100);
//  
// There is a possibility to set parameter value to NULL with SetNull() method.
// If this method called for first iteration, before one should call other Set...
// to identify actual parameter type, which will be used for parameter later.
//
//
// 3. Getting data from database
// =============================
// To request data from data base, SELECT statement should be used.
// After SELECT statement is created, it must be processed
// (with TSQLStatement::Process()) method and result of statement
// should be stored in internal buffers with TSQLStatement::StoreResult()
// method. Information about selected fields (columns)
// can be obtained with GetNumFields() and GetFieldName() methods. 
// To recieve data for next result row, NextResultRow() method should be called.
// Value from each column can be taken with the GetInt(), GetDouble(),
// GetString() and other methods. 
// 
// There are no strict limitation which method should be used
// to get column values. GetString() can be used as generic method, 
// which should always return correct result, but also convertion between most 
// basic data types are supported. For instance, if column contains integer
// values, GetInt(), GetLong64(), GetDouble() and GetString() methods can be used.
// If column has float point format, GetDouble() and GetString() methods can
// be used without loss of precision while GetInt() or GetLong64() will return
// integer part of the value. One also can test, if value is NULL with IsNull()
// method.
//
// Buffer length, specified for statement in TSQLServer::Statement() call,
// will also be used to allocate buffers for column values. Usage of these
// buffers is transparent for users and does not limit number of rows,
// which can be accessed with  one statement. Example of select query:
//
//    stmt = serv->Statement("SELECT * FROM TESTTABLE", 100);
//    // process statement
//    if (stmt->Process()) {
//       // store result of statement in buffer    
//       stmt->StoreResult();
//         
//       // display info about selected field 
//       cout << "NumFields = " << stmt->GetNumFields() << endl;
//       for (int n=0;n<stmt->GetNumFields();n++) 
//          cout << "Field " << n << "  = " << stmt->GetFieldName(n) << endl;
//
//       // extract rows one after another
//       while (stmt->NextResultRow()) {
//          Double_t id1 = stmt->GetDouble(0);
//          UInt_t id2 = stmt->GetUInt(1);
//          const char* name1 = stmt->GetString(2);
//          const char* name2 = stmt->GetString(3);
//          cout << id1 << " - " << id2 << "  " << name1 << "  " << name2 << endl;
//       }
//    }    
//
// 4. Working with date/time parameters
// ====================================
// Current implementation supports date, time, date&time and timestamp 
// data (all time intervals not supported yet). To set or get date/time values,
// following methods should be used:
//   SetTime()/GetTime() - only time (hour:min:sec), 
//   SetDate()/GetDate() - only date (year-month-day), 
//   SetDatime()/GetDatime() - date and time 
//   SetTimestamp()/GetTimestamp() - timestamp with seconds fraction
// For some of these methods TDatime type can be used as parameter / return value.
// Be aware, that TDatime supports only dates after 1995-01-01.
// There are also methods to get separately year, month, day, hour, minutes and seconds.
//
// Different SQL databases has different treatement of date/time types.
// For instance, MySQL has all correspondent types (TIME, DATE, DATETIME and TIMESTAMP),
// Oracle native driver supports only DATE (which is actually date and time) and TIMESTAMP
// ODBC interface provides access for time, date and timestamps.
// Therefore, one should use correct methods to access such data.
// For instance, in MySQL SQL type 'DATE' is only date (one should use GetDate() to 
// access such data), while in Oracle it is date and time. Therefore, 
// to get complete data from 'DATE' column in Oracle, one should use GetDatime() method.
//
// The only difference of timestamp from date/time, that it has fractional
// seconds part. Be aware, that fractional part can has different meaning 
// (actual value) in different SQL plugins.
//
// 5. Binary data
// ==============
// Most of modern data bases support just binary data, which is
// typically has SQL type name 'BLOB'. To access data in such
// columns, GetBinary()/SetBinary() methods should be used. 
// Current implementation supposed, that complete content of the 
// column must be retrieved at once. Therefore very big data of 
// gigabytes size may cause a problem.
//
////////////////////////////////////////////////////////////////////////////////

#include "TSQLStatement.h"

ClassImp(TSQLStatement)

//______________________________________________________________________________
Int_t TSQLStatement::GetErrorCode() const
{
   // returns error code of last operation
   // if res==0, no error
   // Each specific implementation of TSQLStatement provides its own error coding
   
   return fErrorCode;
}

//______________________________________________________________________________
const char* TSQLStatement::GetErrorMsg() const
{
   //  returns error message of last operation
   // if no errors, return 0
   // Each specific implementation of TSQLStatement provides its own error messages
   
   return GetErrorCode()==0 ? 0 : fErrorMsg.Data();
}

//______________________________________________________________________________
void TSQLStatement::ClearError()
{
   // reset error fields
   
   fErrorCode = 0;
   fErrorMsg = "";
}

//______________________________________________________________________________
void TSQLStatement::SetError(Int_t code, const char* msg, const char* method)
{
   // set new values for error fields
   // if method specified, displays error message
   
   fErrorCode = code;
   fErrorMsg = msg;
   if ((method!=0) && fErrorOut)
      Error(method,"Code: %d  Msg: %s", code, (msg ? msg : "No message"));
}

//______________________________________________________________________________
Bool_t TSQLStatement::SetDate(Int_t npar, const TDatime& tm)
{
   // set only date value for specified parameter from TDatime object
   
   return SetDate(npar, tm.GetYear(), tm.GetMonth(), tm.GetDay());
}

//______________________________________________________________________________
Bool_t TSQLStatement::SetTime(Int_t npar, const TDatime& tm)
{
   // set only time value for specified parameter from TDatime object
   
   return SetTime(npar, tm.GetHour(), tm.GetMinute(), tm.GetSecond());
}

//______________________________________________________________________________
Bool_t TSQLStatement::SetDatime(Int_t npar, const TDatime& tm)
{
   // set date & time value for specified parameter from TDatime object
   
   return SetDatime(npar, tm.GetYear(), tm.GetMonth(), tm.GetDay(),
                          tm.GetHour(), tm.GetMinute(), tm.GetSecond());
}

//______________________________________________________________________________
Bool_t TSQLStatement::SetTimestamp(Int_t npar, const TDatime& tm)
{
   // set timestamp value for specified parameter from TDatime object
   
   return SetTimestamp(npar, tm.GetYear(), tm.GetMonth(), tm.GetDay(),
                             tm.GetHour(), tm.GetMinute(), tm.GetSecond(), 0);
}

//______________________________________________________________________________
TDatime TSQLStatement::GetDatime(Int_t npar)
{
   // return value of parameter in form of TDatime
   // Be aware, that TDatime does not allow dates before 1995-01-01 
   
   Int_t year, month, day, hour, min, sec;
   
   if (!GetDatime(npar, year, month, day, hour, min, sec))
     return TDatime();
   
   if (year<1995) {
      SetError(-1, "Date before year 1995 does not supported by TDatime type", "GetDatime");
      return TDatime();
   }
   
   return TDatime(year, month, day, hour, min, sec);
}

//______________________________________________________________________________
Int_t TSQLStatement::GetYear(Int_t npar)
{
   // return year value for parameter (if applicable) 
   
   Int_t year, month, day, hour, min, sec, frac;
   if (GetDate(npar, year, month, day)) return year;
   if (GetTimestamp(npar, year, month, day, hour, min, sec, frac)) return year;
   return 0;
}

//______________________________________________________________________________
Int_t TSQLStatement::GetMonth(Int_t npar)
{
   // return month value for parameter (if applicable) 

   Int_t year, month, day, hour, min, sec, frac;
   if (GetDate(npar, year, month, day)) return month;
   if (GetTimestamp(npar, year, month, day, hour, min, sec, frac)) return month;
   return 0;
}

//______________________________________________________________________________
Int_t TSQLStatement::GetDay(Int_t npar)
{
   // return day value for parameter (if applicable) 

   Int_t year, month, day, hour, min, sec, frac;
   if (GetDate(npar, year, month, day)) return day;
   if (GetTimestamp(npar, year, month, day, hour, min, sec, frac)) return day;
   return 0;
}

//______________________________________________________________________________
Int_t TSQLStatement::GetHour(Int_t npar)
{
   // return hours value for parameter (if applicable) 

   Int_t year, month, day, hour, min, sec, frac;
   if (GetTime(npar, hour, min, sec)) return hour;
   if (GetTimestamp(npar, year, month, day, hour, min, sec, frac)) return hour;
   return 0;
}

//______________________________________________________________________________
Int_t TSQLStatement::GetMinute(Int_t npar)
{
   // return minutes value for parameter (if applicable) 

   Int_t year, month, day, hour, min, sec, frac;
   if (GetTime(npar, hour, min, sec)) return min;
   if (GetTimestamp(npar, year, month, day, hour, min, sec, frac)) return min;
   return 0;
}

//______________________________________________________________________________
Int_t TSQLStatement::GetSecond(Int_t npar)
{
   // return seconds value for parameter (if applicable) 

   Int_t year, month, day, hour, min, sec, frac;
   if (GetTime(npar, hour, min, sec)) return sec;
   if (GetTimestamp(npar, year, month, day, hour, min, sec, frac)) return sec;
   return 0;
}

//______________________________________________________________________________
TDatime TSQLStatement::GetTimestamp(Int_t npar)
{
   // return value of parameter in form of TDatime
   // Be aware, that TDatime does not allow dates before 1995-01-01 
   
   Int_t year, month, day, hour, min, sec, frac;
   
   if (!GetTimestamp(npar, year, month, day, hour, min, sec, frac))
     return TDatime();
   
   if (year<1995) {
      SetError(-1, "Date before year 1995 does not supported by TDatime type", "GetTimestamp");
      return TDatime();
   }
   
   return TDatime(year, month, day, hour, min, sec);
}

