// @(#)root/oracle:$Name:  $:$Id: TOracleServer.cxx,v 1.3 2005/03/03 11:45:06 rdm Exp $
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TOracleServer.h"
#include "TOracleResult.h"
#include "TUrl.h"


ClassImp(TOracleServer)

//______________________________________________________________________________
TOracleServer::TOracleServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a Oracle DB server. The db arguments should be
   // of the form "oracle://connection_identifier][/<database>]", e.g.:
   // "oracle://cmscald.fnal.gov/test". The uid is the username and pw
   // the password that should be used for the connection.

   fEnv = 0;
   fConn = 0;
   fStmt = 0;

   TUrl url(db);

   if (!url.IsValid()) {
      Error("TOracleServer", "malformed db argument %s", db);
      MakeZombie();
      return;
   }

   if (strncmp(url.GetProtocol(), "oracle", 6)) {
      Error("TOracleServer", "protocol in db argument should be oracle it is %s",
            url.GetProtocol());
      MakeZombie();
      return;
   }

   const char *conn_str = 0;
   if (strcmp(url.GetFile(), "/"))
      conn_str = url.GetFile()+1;

   try {
      fEnv = Environment::createEnvironment();
      fConn = fEnv->createConnection(uid, pw, conn_str);

      fType = "Oracle";
      fHost = url.GetHost();
      fDB   = conn_str;
      fPort = url.GetPort();
      fPort = (fPort) ? fPort : 1521;
      //fPort = 1521;
   } catch (SQLException &oraex) {
      Error("TOracleServer", "connection to Oracle database %s failed (error: %s)",conn_str, (oraex.getMessage()).c_str());
      MakeZombie();
   }
}

//______________________________________________________________________________
TOracleServer::~TOracleServer()
{
   // Close connection to Oracle DB server.

   if (IsConnected())
      Close();
}

//______________________________________________________________________________
void TOracleServer::Close(Option_t *)
{
   // Close connection to Oracle DB server.

   try {
      if (fStmt)
         fConn->terminateStatement(fStmt);
      if (fConn)
         fEnv->terminateConnection(fConn);
      if (fEnv)
         Environment::terminateEnvironment(fEnv);
   } catch (SQLException &oraex)  {
      Error("TOracleServer", "close connection failed: (error: %s)", (oraex.getMessage()).c_str());
      //MakeZombie();
   }

   fPort = -1;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("Query", "not connected");
      return 0;
   }

   try {
      if (!fStmt)
         fStmt = fConn->createStatement();
      fStmt->setSQL(sql);
      fStmt->execute();
      return new TOracleResult(fStmt);
   } catch (SQLException &oraex)  {
         Error("TOracleServer", "query failed: (error: %s)", (oraex.getMessage()).c_str());
      //MakeZombie();
   }

   return 0;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::GetTables(const char *dbname, const char * /*wild*/)
{
   // List all tables in the specified database. Wild is for wildcarding
   // "t%" list all tables starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.
   //
   // In Oracle 9 and above, table is accessed in schema.table format.
   // GetTables returns tables in all schemas accessible for the user.
   // Assumption: table ALL_OBJECTS is accessible for the user, which is
   // true in Oracle 10g. The returned TSQLResult has two columns: schema_name,
   // table_name.
   // "dbname": if specified, returns table list of this schema, or return
   //           all tables
   // "wild":   is not used in this implementation

   if (!IsConnected()) {
      Error("GetTables", "not connected");
      return 0;
   }

   TString sqlstr("SELECT owner, object_name FROM ALL_OBJECTS WHERE object_type='TABLE'");
   if (dbname)
      sqlstr = sqlstr + " AND owner='" + dbname + "'";
   TSQLResult *tabRs;
   tabRs = Query(sqlstr.Data());
   return tabRs;

}

//______________________________________________________________________________
TSQLResult *TOracleServer::GetColumns(const char *dbname, const char *table,
                                      const char * /*wild*/)
{
   // List all columns in specified table in the specified database.
   // Wild is for wildcarding "t%" list all columns starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetColumns", "not connected");
      return 0;
   }

   if (SelectDataBase(dbname) != 0) {
      Error("GetColumns", "no such database %s", dbname);
      return 0;
   }
   return new TOracleResult(fConn, table);

}

//______________________________________________________________________________
Int_t TOracleServer::SelectDataBase(const char * /*dbname*/)
{
   // NOT IMPLEMENTED
   // Select a database. Returns 0 if successful, non-zero otherwise.

   if (!IsConnected()) {
      Error("SelectDataBase", "not connected");
      return -1;
   }

   // do nothing and return success code
   return 0;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::GetDataBases(const char * /*wild*/)
{
   // NOT IMPLEMENTED
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetDataBases", "not connected");
      return 0;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TOracleServer::CreateDataBase(const char * /*dbname*/)
{
   // NOT IMPLEMENTED
   // Create a database. Returns 0 if successful, non-zero otherwise.

   if (!IsConnected()) {
      Error("CreateDataBase", "not connected");
      return -1;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::DropDataBase(const char * /*dbname*/)
{
   // NOT IMPLEMENTED
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.

   if (!IsConnected()) {
      Error("DropDataBase", "not connected");
      return -1;
   }

   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Reload()
{
   // NOT IMPLEMENTED
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.

   if (!IsConnected()) {
      Error("Reload", "not connected");
      return -1;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Shutdown()
{
   // NOT IMPLEMENTED
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.

   if (!IsConnected()) {
      Error("Shutdown", "not connected");
      return -1;
   }
   return -1;
}

//______________________________________________________________________________
const char *TOracleServer::ServerInfo()
{
   // NOT IMPLEMENTED
   // Return server info.

   if (!IsConnected()) {
      Error("ServerInfo", "not connected");
      return 0;
   }
   return "Oracle";
}
