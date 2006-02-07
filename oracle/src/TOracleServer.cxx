// @(#)root/oracle:$Name:  $:$Id: TOracleServer.cxx,v 1.6 2005/06/01 16:12:16 rdm Exp $
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
   if (!sql || !*sql) {
      Error("Query", "no query string specified");
      return 0;
   }

   try {
      if (!fStmt) {
         fStmt = fConn->createStatement();
         fStmt->setPrefetchRowCount(1000);
         fStmt->setPrefetchMemorySize(1024*1024);
      }

      // count the number of rows of the resultset of this select
      // NOTE: Oracle doesn't provide a way through OCI or OCCI to count
      //       the number of rows. The reason is the memory concern of client
      //       application. Consider a select statment with 1 million rows
      //       returned. In Oracle, user can set prefetch size to repeatedly
      //       retrieve all rows by calling next(#rows_to_fetch). By default,
      //       prefetch size is set to 1, meaning client only fetch 1 row each
      //       time it contacts db server.
      // The best-so-far way to count the number of rows is to traverse the
      // resultset (count(query)). This method is neither efficient, fast
      // nor 100% accurate. Please see OCI/OCCI discussion forum for details
      // So the only purpose to count rows is follow TSQL specification.

      // TODO: We should change TSQL spec on GetRowCount(). Not every db server
      // provides natural way to do so like mysql. User can loop over resultset
      // and get the count after the last next() unless he must know the count
      // before next()

      // NOTE: sql should not end with ";" !!!
      int row_count = -1;
      char *str, *sql_chars = new char[strlen(sql)+1];
      strcpy(sql_chars, sql);
      str = sql_chars;
      // skip space and newline chars
      while ( *str == '\n' || *str == '\t' || *str == ' ' || *str == '\r' )
         str ++;
      string sql_string = sql;
      if (strncasecmp(str, "SELECT",6)==0) {
         string count_sql = "select COUNT(*) from ( " + sql_string + " )";
         fStmt->setSQL(count_sql.c_str());
         fStmt->execute();

         if (fStmt->status() == Statement::RESULT_SET_AVAILABLE) {
            ResultSet *count_rs = fStmt->getResultSet();
            if (count_rs->next())
               row_count = count_rs->getInt(1);
            fStmt->closeResultSet(count_rs);
         }
      }
      // NOTE: between above and below execute(), if there is any DDL operated
      // on target tables, the row_count may not be accurate
      fStmt->setSQL(sql);
      fStmt->execute();

      TOracleResult *res = new TOracleResult(fStmt, row_count);
      delete [] sql_chars;
      return res;
   } catch (SQLException &oraex)  {
      Error("TOracleServer", "query failed: (error: %s)", (oraex.getMessage()).c_str());
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

   // In Oracle 9 and above, table is accessed in schema.table format.
   // GetTables returns tables in all schemas accessible for the user.
   // Assumption: table ALL_OBJECTS is accessible for the user, which is true in Oracle 10g
   // The returned TSQLResult has two columns: schema_name, table_name
   // "dbname": if specified, return table list of this schema, or return all tables
   // "wild" is not used in this implementation

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
   // Select a database. Returns 0 if successful, non-zero otherwise.
   // NOT IMPLEMENTED.

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
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.
   // NOT IMPLEMENTED.

   if (!IsConnected()) {
      Error("GetDataBases", "not connected");
      return 0;
   }

   return 0;
}

//______________________________________________________________________________
Int_t TOracleServer::CreateDataBase(const char * /*dbname*/)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.
   // NOT IMPLEMENTED.

   if (!IsConnected()) {
      Error("CreateDataBase", "not connected");
      return -1;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::DropDataBase(const char * /*dbname*/)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.
   // NOT IMPLEMENTED.

   if (!IsConnected()) {
      Error("DropDataBase", "not connected");
      return -1;
   }

   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.
   // NOT IMPLEMENTED.

   if (!IsConnected()) {
      Error("Reload", "not connected");
      return -1;
   }
   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.
   // NOT IMPLEMENTED.

   if (!IsConnected()) {
      Error("Shutdown", "not connected");
      return -1;
   }
   return -1;
}

//______________________________________________________________________________
const char *TOracleServer::ServerInfo()
{
   // Return server info.
   // NOT IMPLEMENTED.

   if (!IsConnected()) {
      Error("ServerInfo", "not connected");
      return 0;
   }
   return "Oracle";
}
