// @(#)root/sapdb:$Id$
// Author: Mark Hemberger & Fons Rademakers   03/08/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TSapDBServer.h"
#include "TSapDBResult.h"
#include "TSapDBRow.h"
#include "TUrl.h"
#include <ctype.h>


ClassImp(TSapDBServer)

//______________________________________________________________________________
TSapDBServer::TSapDBServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a SapDB DB server. The db arguments should be
   // of the form "sapdb://<host>[:<port>][/<database>]", e.g.:
   // "sapdb://pcroot.cern.ch:3456/test". The uid is the username and pw
   // the password that should be used for the connection.

   fSapDB     = 0;
   fEnv       = 0;
   fStmt      = 0;
   fStmtCnt   = 0;

   TUrl url(db);

   if (!url.IsValid()) {
      Error("TSapDBServer", "malformed db argument %s", db);
      MakeZombie();
      return;
   }

   if (strncmp(url.GetProtocol(), "sapdb", 5)) {
      Error("TSapDBServer", "protocol in db argument should be sapdb it is %s",
            url.GetProtocol());
      MakeZombie();
      return;
   }

   const char *dbase = url.GetFile();

   // Allocate environment, connection, and statement handle
   RETCODE rc = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &fEnv);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      Error("TSapDBServer", "allocation of environment failed");
      MakeZombie();
      return;
   }

   rc = SQLAllocHandle(SQL_HANDLE_DBC, fEnv, &fSapDB);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, SQL_NULL_HSTMT);
      Error("TSapDBServer", "allocation of db failed");
      MakeZombie();
      return;
   }

   // Connect to data source
   const char *dbnam = Form("%s:%s", url.GetHost(), dbase);
   rc = SQLConnect(fSapDB, (SQLCHAR*) dbnam, SQL_NTS,
                   (SQLCHAR*) uid, SQL_NTS, (SQLCHAR*) pw, SQL_NTS);

   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, SQL_NULL_HSTMT);
      Error("TSapDBServer", "connection to %s:%s failed", url.GetHost(), dbase);
      MakeZombie();
      return;
   }

   rc = SQLAllocHandle(SQL_HANDLE_STMT, fSapDB, &fStmt);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, fStmt);
      Error("TSapDBServer", "allocation of statement handle failed");
      MakeZombie();
      return;
   }
   rc = SQLAllocHandle(SQL_HANDLE_STMT, fSapDB, &fStmtCnt);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, fStmtCnt);
      Error("TSapDBServer", "allocation of count statement handle failed");
      MakeZombie();
      return;
   }

   fType = "SapDB";
   fHost = url.GetHost();
   fDB   = dbase;
   fPort = url.GetPort();
}

//______________________________________________________________________________
TSapDBServer::~TSapDBServer()
{
   // Close connection to SapDB DB server.

   if (IsConnected())
      Close();
}

//______________________________________________________________________________
void TSapDBServer::Close(Option_t *)
{
   // Close connection to SapDB DB server.

   // Disconnect from the data source and free all handles
   RETCODE rc = SQLDisconnect(fSapDB);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, SQL_NULL_HSTMT);
      Error("TSapDBServer", "disconnect during close failed");
   }

   rc = SQLFreeHandle(SQL_HANDLE_STMT, fStmt);
   if (rc != SQL_SUCCESS) {
      //Error("TSapDBServer", "free statement handle during close failed");
   }

   rc = SQLFreeHandle(SQL_HANDLE_STMT, fStmtCnt);
   if (rc != SQL_SUCCESS) {
      //Error("TSapDBServer", "free count statement handle during close failed");
   }

   rc = SQLFreeHandle(SQL_HANDLE_DBC, fSapDB);
   if (rc != SQL_SUCCESS) {
      printSQLError(fSapDB, SQL_NULL_HSTMT);
      Error("TSapDBServer", "free database handle during close failed");
   }

   rc = SQLFreeHandle(SQL_HANDLE_ENV, fEnv);
   if (rc != SQL_SUCCESS) {
      Error("TSapDBServer", "free environment handle during close failed");
   }

  fPort = -1;
}

//______________________________________________________________________________
TSQLResult *TSapDBServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("Query", "not connected");
      return 0;
   }

   RETCODE rc = SQLFreeHandle(SQL_HANDLE_STMT, fStmt);
   if (rc != SQL_SUCCESS) {
      printSQLError(fSapDB, fStmt);
      Error("TSapDBServer", "free statement handle failed");
   }

   rc = SQLAllocHandle(SQL_HANDLE_STMT, fSapDB, &fStmt);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, fStmt);
      Error("TSapDBServer", "allocation statement handle failed");
   }

   rc = SQLFreeHandle(SQL_HANDLE_STMT, fStmtCnt);
   if (rc != SQL_SUCCESS) {
      printSQLError(fSapDB, fStmtCnt);
      Error("TSapDBServer", "free count statement handle failed");
   }

   rc = SQLAllocHandle(SQL_HANDLE_STMT, fSapDB, &fStmtCnt);
   if (rc != SQL_SUCCESS && rc != SQL_SUCCESS_WITH_INFO) {
      printSQLError(fSapDB, fStmtCnt);
      Error("TSapDBServer", "allocation count statement handle failed");
   }

   SDWORD slRowCount;
   TString sqlcnt = "SELECT COUNT(*) ";
   TString sqlt = sql;
   sqlt = sqlt.Strip(TString::kBoth);

   if (sqlt.BeginsWith("SELECT", TString::kIgnoreCase)) {
      Ssiz_t i = sqlt.Index("FROM", 0, TString::kIgnoreCase);
      if (i != kNPOS)
         sqlcnt += sqlt(i, sqlt.Length());

      if (SQLExecDirect(fStmtCnt, (SQLCHAR*)sqlcnt.Data(), SQL_NTS) !=
          SQL_SUCCESS) {
         printSQLError(fSapDB, fStmtCnt);
         return 0;
      }

      SQLBindCol(fStmtCnt, 1, SQL_C_LONG, &slRowCount, 0, 0);
      rc = SQLFetch(fStmtCnt);
      //if (rc == SQL_SUCCESS)
      //   printf("RowCount: %ld\n", slRowCount);
   }

   if (SQLPrepare(fStmt, (SQLCHAR*)sqlt.Data(), SQL_NTS) != SQL_SUCCESS) {
      printSQLError(fSapDB, fStmt);
      return 0;
   }

   if (SQLExecute(fStmt) != SQL_SUCCESS) {
      printSQLError(fSapDB, fStmt);
      return 0;
   }
   if (SQLEndTran(SQL_HANDLE_DBC, fSapDB, SQL_COMMIT) != SQL_SUCCESS) {
      printSQLError(fSapDB, fStmt);
      return 0;
   }

   return new TSapDBResult(fStmt, slRowCount);
}

//______________________________________________________________________________
Int_t TSapDBServer::SelectDataBase(const char *dbname)
{
   // Select a database. Returns 0 if successful, non-zero otherwise.
   // For SapDB: only to be used to check the dbname.

   if (!IsConnected()) {
      Error("SelectDataBase", "not connected");
      return -1;
   }

   if (fDB != dbname) {
      Error("SelectDataBase", "no such database");
      return -1;
   }

   return 0;
}

//______________________________________________________________________________
TSQLResult *TSapDBServer::GetDataBases(const char *wild)
{
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.
   // For SapDB: you are connected to a certain database, so give me a
   // list of tables

   if (!IsConnected()) {
      Error("GetDataBases", "not connected");
      return 0;
   }

   return GetTables(fDB, wild);
}

//______________________________________________________________________________
TSQLResult *TSapDBServer::GetTables(const char * /*dbname*/, const char *wild)
{
   // List all tables in the specified database. Wild is for wildcarding
   // "t%" list all tables starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetTables", "not connected");
      return 0;
   }

   TString sql = "SELECT TABLENAME FROM TABLES";
   if (wild)
      sql += Form(" WHERE TABLENAME LIKE '%s'", wild);

   return Query(sql);
}

//______________________________________________________________________________
TSQLResult *TSapDBServer::GetColumns(const char *dbname, const char *table,
                                     const char *wild)
{
   // List all columns in specified table in the specified database.
   // Wild is for wildcarding "t%" list all columns starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetColumns", "not connected");
      return 0;
   }

   if (SelectDataBase(dbname) == 0) {
      Error("GetColumns", "no such database %s", dbname);
      return 0;
   }

   char *sql;
   if (wild)
      sql = Form("SELECT COLUMNNAME FROM COLUMNS WHERE TABLENAME LIKE '%s' AND COLUMNNAME LIKE '%s'", table, wild);
   else
      sql = Form("SELECT COLUMNNAME FROM COLUMNS WHERE TABLENAME LIKE '%s'", table);

   return Query(sql);
}

//______________________________________________________________________________
Int_t TSapDBServer::CreateDataBase(const char * /*dbname*/)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.
   // For SapDB: do nothing

   if (!IsConnected()) {
      Error("CreateDataBase", "not connected");
      return -1;
   }

   Error("CreateDataBase", "not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TSapDBServer::DropDataBase(const char * /*dbname*/)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.
   // For SapDB: do nothing

   if (!IsConnected()) {
      Error("DropDataBase", "not connected");
      return -1;
   }

   Error("DropDataBase", "not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TSapDBServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.
   // For SapDB: do nothing

   if (!IsConnected()) {
      Error("Reload", "not connected");
      return -1;
   }

   Error("Reload", "not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TSapDBServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.
   // for SapDB: do nothing

   if (!IsConnected()) {
      Error("Shutdown", "not connected");
      return -1;
   }

   Error("Shutdown", "not implemented");
   return 0;
}

//______________________________________________________________________________
const char *TSapDBServer::ServerInfo()
{
   // Return server info.

   if (!IsConnected()) {
      Error("ServerInfo", "not connected");
      return 0;
   }

   TString sql = "SELECT KERNEL,RUNTIMEENVIRONMENT FROM DOMAIN.VERSIONS";
   TSQLResult *res_info = Query(sql);

   TSQLRow *row_info = res_info->Next();

   TString info;
   while (row_info) {
      info  = row_info->GetField(0);
      info += " ";
      info += row_info->GetField(1);
      row_info = res_info->Next();
   }

   delete res_info;
   delete row_info;

   return info;
}

//______________________________________________________________________________
Int_t TSapDBServer::printSQLError(SQLHDBC hdbc, SQLHSTMT hstmt)
{
   // Print SapDB error message.

   UCHAR  sqlstate[10];
   SDWORD sqlcode;
   UCHAR  errortxt[512+1];
   SWORD  usederrortxt;

   SQLError(SQL_NULL_HENV, hdbc, hstmt, sqlstate, &sqlcode, errortxt,
            512, &usederrortxt);

   printf ("SQL state: %s\n", sqlstate);
   printf ("SQL code:  %ld\n", long(sqlcode));
   printf ("SQL Errortext:\n%s\n\n", errortxt);

   return 0;
}
