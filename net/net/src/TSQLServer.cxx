// @(#)root/net:$Id$
// Author: Fons Rademakers   25/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLServer                                                           //
//                                                                      //
// Abstract base class defining interface to a SQL server.              //
//                                                                      //
// To open a connection to a server use the static method Connect().    //
// The db argument of Connect() is of the form:                         //
//    <dbms>://<host>[:<port>][/<database>], e.g.                       //
// mysql://pcroot.cern.ch:3456/test, oracle://srv1.cern.ch/main, ...    //
// Depending on the <dbms> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
// For SQLite, the syntax is slightly different:                        //
//   sqlite://<database>                                                //
// The string 'database' is directly passed to sqlite3_open(_v2),       //
// so e.g. a filename or ":memory:" are possible values.                //
// For SQLite versions >= 3.7.7, SQLITE_OPEN_URI is activated to also   //
// allow URI-parameters if needed.                                      //
//                                                                      //
// Related classes are TSQLResult and TSQLRow.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TSQLTableInfo.h"
#include "TSQLColumnInfo.h"
#include "TROOT.h"
#include "TList.h"
#include "TObjString.h"
#include "TPluginManager.h"
#include "TVirtualMutex.h"

ClassImp(TSQLServer)


const char* TSQLServer::fgFloatFmt = "%e";


//______________________________________________________________________________
TSQLServer *TSQLServer::Connect(const char *db, const char *uid, const char *pw)
{
   // The db should be of the form:  <dbms>://<host>[:<port>][/<database>],
   // e.g.:  mysql://pcroot.cern.ch:3456/test, oracle://srv1.cern.ch/main,
   // pgsql://... or sapdb://...
   // The uid is the username and pw the password that should be used for
   // the connection. Depending on the <dbms> the shared library (plugin)
   // for the selected system will be loaded. When the connection could not
   // be opened 0 is returned.

   TPluginHandler *h;
   TSQLServer *serv = 0;

   if ((h = gROOT->GetPluginManager()->FindHandler("TSQLServer", db))) {
      if (h->LoadPlugin() == -1)
         return 0;
      serv = (TSQLServer *) h->ExecPlugin(3, db, uid, pw);
   }

   if (serv && serv->IsZombie()) {
      delete serv;
      serv = 0;
   }

   return serv;
}

//______________________________________________________________________________
Bool_t TSQLServer::Exec(const char* sql)
{
   // Execute sql query.
   // Usefull for commands like DROP TABLE or INSERT, where result set
   // is not interested. Return kTRUE if no error

   TSQLResult* res = Query(sql);
   if (res==0) return kFALSE;

   delete res;

   return !IsError();
}


//______________________________________________________________________________
Int_t TSQLServer::GetErrorCode() const
{
   // returns error code of last operation
   // if res==0, no error
   // Each specific implementation of TSQLServer provides its own error coding

   return fErrorCode;
}

//______________________________________________________________________________
const char* TSQLServer::GetErrorMsg() const
{
   // returns error message of last operation
   // if no errors, return 0
   // Each specific implementation of TSQLServer provides its own error messages

   return GetErrorCode()==0 ? 0 : fErrorMsg.Data();
}

//______________________________________________________________________________
void TSQLServer::ClearError()
{
   // reset error fields

   fErrorCode = 0;
   fErrorMsg = "";
}

//______________________________________________________________________________
void TSQLServer::SetError(Int_t code, const char* msg, const char* method)
{
   // set new values for error fields
   // if method is specified, displays error message

   fErrorCode = code;
   fErrorMsg = msg;
   if ((method!=0) && fErrorOut)
      Error(method,"Code: %d  Msg: %s", code, (msg ? msg : "No message"));
}

//______________________________________________________________________________
Bool_t TSQLServer::StartTransaction()
{
   // submit "START TRANSACTION" query to database
   // return kTRUE, if successful

   return Exec("START TRANSACTION");
}

//______________________________________________________________________________
Bool_t TSQLServer::Commit()
{
   // submit "COMMIT" query to database
   // return kTRUE, if successful

   return Exec("COMMIT");
}

//______________________________________________________________________________
Bool_t TSQLServer::Rollback()
{
   // submit "ROLLBACK" query to database
   // return kTRUE, if successful

   return Exec("ROLLBACK");
}

//______________________________________________________________________________
TList* TSQLServer::GetTablesList(const char* wild)
{
   // Return list of user tables
   // Parameter wild specifies wildcard for table names.
   // It either contains exact table name to verify that table is exists or
   // wildcard with "%" (any number of symbols) and "_" (exactly one symbol).
   // Example of vaild wildcards: "%", "%name","___user__".
   // If wild=="", list of all available tables will be produced.
   // List contain just tables names in the TObjString.
   // List must be deleted by the user.
   // Example code of method usage:
   //
   // TList* lst = serv->GetTablesList();
   // TIter next(lst);
   // TObject* obj;
   // while (obj = next())
   //   std::cout << "Table: " << obj->GetName() << std::endl;
   // delete lst;

   TSQLResult* res = GetTables(fDB.Data(), wild);
   if (res==0) return 0;

   TList* lst = 0;
   TSQLRow* row = 0;
   while ((row = res->Next())!=0) {
      const char* tablename = row->GetField(0);
      if (lst==0) {
         lst = new TList;
         lst->SetOwner(kTRUE);
      }
      lst->Add(new TObjString(tablename));
      delete row;
   }

   delete res;

   return lst;
}

//______________________________________________________________________________
Bool_t TSQLServer::HasTable(const char* tablename)
{
   // Tests if table of that name exists in database
   // Return kTRUE, if table exists

   if ((tablename==0) || (strlen(tablename)==0)) return kFALSE;

   TList* lst = GetTablesList(tablename);
   if (lst==0) return kFALSE;

   Bool_t res = kFALSE;

   TObject* obj = 0;
   TIter iter(lst);

   // Can be, that tablename contains "_" or "%" symbols, which are wildcards in SQL,
   // therefore more than one table can be returned as result.
   // One should check that exactly same name is appears

   while ((obj = iter()) != 0)
      if (strcmp(tablename, obj->GetName())==0) res = kTRUE;

   delete lst;
   return res;
}

//______________________________________________________________________________
TSQLTableInfo* TSQLServer::GetTableInfo(const char* tablename)
{
   // Producec TSQLTableInfo object, which contain info about
   // table itself and each table column
   // Object must be deleted by user.

   if ((tablename==0) || (*tablename==0)) return 0;

   TSQLResult* res = GetColumns(fDB.Data(), tablename);
   if (res==0) return 0;

   TList* lst = 0;
   TSQLRow* row = 0;
   while ((row = res->Next())!=0) {
      const char* columnname = row->GetField(0);
      if (lst==0) lst = new TList;
      lst->Add(new TSQLColumnInfo(columnname));
      delete row;
   }

   delete res;

   return new TSQLTableInfo(tablename, lst);
}

//______________________________________________________________________________
void TSQLServer::SetFloatFormat(const char* fmt)
{
   // set printf format for float/double members, default "%e"

   if (fmt==0) fmt = "%e";
   fgFloatFmt = fmt;
}

//______________________________________________________________________________
const char* TSQLServer::GetFloatFormat()
{
   // return current printf format for float/double members, default "%e"

   return fgFloatFmt;
}
