// @(#)root/mysql:$Name$:$Id$
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMySQLServer.h"
#include "TMySQLResult.h"
#include "TUrl.h"


ClassImp(TMySQLServer)

//______________________________________________________________________________
TMySQLServer::TMySQLServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a MySQL DB server. The db arguments should be
   // of the form "mysql://<host>[:<port>][/<database>]", e.g.:
   // "mysql://pcroot.cern.ch:3456/test". The uid is the username and pw
   // the password that should be used for the connection.

   fMySQL = 0;

   TUrl url(db);
   
   if (!url.IsValid()) {
      Error("TMySQLServer", "malformed db argument %s", db);
      MakeZombie();
      return;
   }
   
   if (strncmp(url.GetProtocol(), "mysql", 5)) {
      Error("TMySQLServer", "protocol in db argument should be mysql it is %s",
            url.GetProtocol());
      MakeZombie();
      return;
   }
   
   const char *dbase = 0;
   if (strcmp(url.GetFile(), "/"))
      dbase = url.GetFile()+1;   //skip leading /

   fMySQL = new MYSQL;
   mysql_init(fMySQL);
   
   if (mysql_real_connect(fMySQL, url.GetHost(), uid, pw, dbase,
                          url.GetPort(), 0, 0)) {
      fType = "MySQL";
      fHost = url.GetHost();
      fDB   = dbase;
      fPort = url.GetPort();
   } else {
      Error("TMySQLServer", "connection to %s failed", url.GetHost());
      MakeZombie();
   }
}

//______________________________________________________________________________
TMySQLServer::~TMySQLServer()
{
   // Close connection to MySQL DB server.
   
   if (IsConnected())
      Close();
   delete fMySQL;
}

//______________________________________________________________________________
void TMySQLServer::Close(Option_t *)
{
   // Close connection to MySQL DB server.

   if (!fMySQL)
      return;

   mysql_close(fMySQL);
   fPort = -1;
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.
   
   if (!IsConnected()) {
      Error("Query", "not connected");
      return 0;
   }

   if (mysql_query(fMySQL, sql) < 0) {
      Error("Query", mysql_error(fMySQL));
      return 0;
   }

   MYSQL_RES *res = mysql_store_result(fMySQL);
   return new TMySQLResult(res);
}

//______________________________________________________________________________
Int_t TMySQLServer::SelectDataBase(const char *dbname)
{
   // Select a database. Returns 0 if successful, non-zero otherwise.
   
   if (!IsConnected()) {
      Error("SelectDataBase", "not connected");
      return 0;
   }

   Int_t res;
   if ((res = mysql_select_db(fMySQL, dbname)) == 0) {
      fDB = dbname;
      return 0;
   }
   return res;
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::GetDataBases(const char *wild)
{
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.
   
   if (!IsConnected()) {
      Error("GetDataBases", "not connected");
      return 0;
   }

   MYSQL_RES *res = mysql_list_dbs(fMySQL, wild);
   return new TMySQLResult(res);
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::GetTables(const char *dbname, const char *wild)
{
   // List all tables in the specified database. Wild is for wildcarding 
   // "t%" list all tables starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetTables", "not connected");
      return 0;
   }
   
   if (SelectDataBase(dbname) != 0) {
      Error("GetTables", "no such database %s", dbname);
      return 0;
   }

   MYSQL_RES *res = mysql_list_tables(fMySQL, wild);
   return new TMySQLResult(res);
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::GetColumns(const char *dbname, const char *table,
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

   if (SelectDataBase(dbname) != 0) {
      Error("GetColumns", "no such database %s", dbname);
      return 0;
   }
   
   char *sql;
   if (wild)
      sql = Form("SHOW COLUMNS FROM %s LIKE %s", table, wild);
   else
      sql = Form("SHOW COLUMNS FROM %s", table);

   return Query(sql);
}

//______________________________________________________________________________
Int_t TMySQLServer::CreateDataBase(const char *dbname)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.
   
   if (!IsConnected()) {
      Error("CreateDataBase", "not connected");
      return 0;
   }
   return mysql_create_db(fMySQL, dbname);
}

//______________________________________________________________________________
Int_t TMySQLServer::DropDataBase(const char *dbname)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.
   
   if (!IsConnected()) {
      Error("DropDataBase", "not connected");
      return 0;
   }
   return mysql_drop_db(fMySQL, dbname);
}

//______________________________________________________________________________
Int_t TMySQLServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.
   
   if (!IsConnected()) {
      Error("Reload", "not connected");
      return 0;
   }
   return mysql_reload(fMySQL);
}

//______________________________________________________________________________
Int_t TMySQLServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.

   if (!IsConnected()) {
      Error("Shutdown", "not connected");
      return 0;
   }
   return mysql_shutdown(fMySQL);
}

//______________________________________________________________________________
const char *TMySQLServer::ServerInfo()
{
   // Return server info.

   if (!IsConnected()) {
      Error("ServerInfo", "not connected");
      return 0;
   }
   return mysql_get_server_info(fMySQL);
}
